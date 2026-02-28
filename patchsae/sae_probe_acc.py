#!/usr/bin/env python3
import os, gc, argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
from tqdm import tqdm

# patchsae imports (same ones you already use)
from tasks.utils import load_sae, load_hooked_vit
from src.sae_training.hooked_vit import Hook

# ---------- defaults (adjust if your paths differ) ----------
SAE_PATH_DEFAULT   = "/home/sunayana/Documents/Concept_LoRA/patchsae/data/sae_weight/base/out.pt"
NPZ_PATH_DEFAULT   = "/home/sunayana/Documents/Concept_LoRA/data/pathmnist.npz"
TRAIN_ROOT_DEFAULT = "/home/sunayana/Documents/Concept_LoRA/data/pathmnist_imagefolder"
BACKBONE_DEFAULT   = "openai/clip-vit-base-patch16"

MEDMNIST_CLASSES = [
    "adipose", "background", "debris", "lymphocytes", "mucus",
    "smooth muscle", "normal colon mucosa",
    "cancer-associated stroma", "colorectal adenocarcinoma epithelium",
]

# ---------- helpers ----------
def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def build_npz_to_if_mapping(train_root):
    ds = datasets.ImageFolder(root=train_root)
    if_map = {n.replace('_', ' ').lower(): i for n, i in ds.class_to_idx.items()}
    return {i: if_map.get(c.lower(), i) for i, c in enumerate(MEDMNIST_CLASSES)}

class NpzSplit(Dataset):
    def __init__(self, npz_path, split, mapping, transform):
        d = np.load(npz_path)
        # expected keys: train_images/train_labels/test_images/test_labels
        self.imgs   = d[f"{split}_images"]
        self.labels = d[f"{split}_labels"].flatten()
        self.mapping = mapping
        self.transform = transform

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        x = Image.fromarray(self.imgs[i])
        y = self.mapping[int(self.labels[i])]
        return self.transform(x), y

@torch.no_grad()
def collect_acts(vit, loader, block_layer, module_name, device, max_batches=None):
    """HF CLIP hook: expects act as [B, S, D] OR [S, B, D] depending on is_custom."""
    to_pil = transforms.ToPILImage()
    captured = {}

    # robust: accept either layout and convert to [B,S,D]
    def capture_fn(act):
        if act.dim() == 3 and act.shape[0] != loader.batch_size and act.shape[1] == loader.batch_size:
            # likely [S,B,D]
            captured["act"] = act.transpose(0, 1).detach().float().cpu()
            return act
        captured["act"] = act.detach().float().cpu()
        return (act,)  # safe for standard Hook path

    hook = Hook(block_layer, module_name, capture_fn,
                return_module_output=False, is_custom=False)

    all_acts, all_y = [], []
    for bi, (images, y) in enumerate(tqdm(loader, desc="collect", leave=False)):
        if max_batches is not None and bi >= max_batches:
            break
        inputs = vit.processor(
            images=[to_pil(img) for img in images],
            text="",
            return_tensors="pt",
            padding=True
        ).to(device)
        vit.run_with_hooks([hook], return_type="output", **inputs)
        all_acts.append(captured["act"])
        all_y.append(y)
    return torch.cat(all_acts, 0), torch.cat(all_y, 0)

@torch.no_grad()
def sae_features(sae, acts_cpu, device, use_cls=True, batch_size=8):
    """
    acts_cpu: [N,S,D] on CPU
    Returns: [N, F] float32 on CPU (SAE hidden codes aggregated)
    """
    feats = []
    for i in tqdm(range(0, acts_cpu.shape[0], batch_size), desc="sae->feat", leave=False):
        a = acts_cpu[i:i+batch_size].to(device)  # [B,S,D]
        out = sae(a)
        if len(out) < 2:
            raise RuntimeError("SAE forward didn't return hidden activations. Expected (recon, h, ...).")
        h = out[1]  # [B,S,F] (typical)
        if use_cls:
            z = h[:, 0, :]
        else:
            z = h.mean(dim=1)
        feats.append(z.detach().float().cpu())
    return torch.cat(feats, 0)

def train_linear_probe(Xtr, ytr, Xte, yte, n_classes, device, epochs=5, lr=1e-2, wd=0.0):
    Xtr = Xtr.to(device); ytr = ytr.to(device)
    Xte = Xte.to(device); yte = yte.to(device)

    clf = nn.Linear(Xtr.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, epochs+1):
        clf.train()
        logits = clf(Xtr)
        loss = loss_fn(logits, ytr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        clf.eval()
        with torch.no_grad():
            pred = clf(Xte).argmax(dim=1)
            acc = (pred == yte).float().mean().item() * 100
        print(f"epoch {ep:02d} | loss {loss.item():.4f} | test acc {acc:.2f}%")
    return acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae_path", default=SAE_PATH_DEFAULT)
    ap.add_argument("--npz_path", default=NPZ_PATH_DEFAULT)
    ap.add_argument("--train_root", default=TRAIN_ROOT_DEFAULT)
    ap.add_argument("--backbone", default=BACKBONE_DEFAULT)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_batches", type=int, default=20, help="sanity run limit (None=all)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--use_cls", action="store_true", help="use CLS token codes (default: mean pool)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    # load SAE (cfg tells us which layer/module to hook)
    sae, cfg = load_sae(args.sae_path, device)
    print(f"[SAE] layer={cfg.block_layer} module={cfg.module_name}")
    print(f"[SAE] ckpt={args.sae_path}")

    # build datasets
    mapping = build_npz_to_if_mapping(args.train_root)
    tfm = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

    ds_tr = NpzSplit(args.npz_path, "train", mapping, tfm)
    ds_te = NpzSplit(args.npz_path, "test",  mapping, tfm)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # load base hooked vit (matches your earlier usage)
    vit = load_hooked_vit(cfg, "base", args.backbone, device)
    vit.eval()

    mb = args.max_batches if args.max_batches > 0 else None

    print("[collect] train acts")
    tr_acts, tr_y = collect_acts(vit, dl_tr, cfg.block_layer, cfg.module_name, device, max_batches=mb)
    print("[collect] test acts")
    te_acts, te_y = collect_acts(vit, dl_te, cfg.block_layer, cfg.module_name, device, max_batches=mb)

    print(f"[acts] train {tuple(tr_acts.shape)} | test {tuple(te_acts.shape)}")

    # SAE -> features
    print("[feat] SAE codes")
    tr_X = sae_features(sae, tr_acts, device, use_cls=args.use_cls, batch_size=8)
    te_X = sae_features(sae, te_acts, device, use_cls=args.use_cls, batch_size=8)

    n_classes = int(max(tr_y.max().item(), te_y.max().item()) + 1)
    print(f"[probe] Xtr {tuple(tr_X.shape)} Xte {tuple(te_X.shape)} classes={n_classes}")

    # train probe
    acc = train_linear_probe(
        tr_X, tr_y.long(),
        te_X, te_y.long(),
        n_classes=n_classes,
        device=device,
        epochs=args.epochs,
        lr=1e-2
    )
    print(f"\nFINAL test accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()