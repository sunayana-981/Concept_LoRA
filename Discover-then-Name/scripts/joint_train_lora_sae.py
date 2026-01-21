#!/usr/bin/env python3
import os, math, argparse, torch, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import clip
import torch.nn as nn
import torch.nn.functional as F

# ====== LoRA wrappers (import your own if you have them) ======
from layers import PlainMultiheadAttentionLoRA  # you already use this

INDEX_POSITIONS_TEXT = {
    'top1': [11], 'top2': [10, 11], 'top3': [9, 10, 11],
    'bottom': [0, 1, 2, 3], 'mid': [4, 5, 6, 7], 'up': [8, 9, 10, 11],
    'half-up': [6, 7, 8, 9, 10, 11], 'half-bottom': [0, 1, 2, 3, 4, 5],
    'all': list(range(12))
}
INDEX_POSITIONS_VISION = {
    'ViT-B/16': {
        'top': [11], 'top3': [9, 10, 11], 'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7], 'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11], 'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': list(range(12))
    },
    'ViT-B/32': {'bottom': [0, 1, 2, 3], 'mid': [4, 5, 6, 7], 'up': [8, 9, 10, 11],
                 'half-up': [6, 7, 8, 9, 10, 11], 'half-bottom': [0, 1, 2, 3, 4, 5],
                 'all': list(range(12))},
    'ViT-L/14': {'half-up': list(range(12,24)), 'half-bottom': list(range(12)),
                 'all': list(range(24))}
}

def attach_lora(clip_model, backbone="ViT-B/16", which="top3", params=('q','v'), r=4, alpha=2, dropout=0.0):
    layers = []
    vis = clip_model.visual.transformer
    for i, block in enumerate(vis.resblocks):
        if i in INDEX_POSITIONS_VISION[backbone][which]:
            for name, submodule in block.named_children():
                if isinstance(submodule, nn.MultiheadAttention):
                    l = PlainMultiheadAttentionLoRA(submodule, enable_lora=list(params), r=r, lora_alpha=alpha, dropout_rate=dropout)
                    setattr(block, name, l)
                    layers.append(l)
    return layers

@torch.no_grad()
def merge_lora_inplace(layers, scale):
    # Optional: fold LoRA (A@B) into base weights for faster export
    for l in layers:
        for attr in ("q_proj","k_proj","v_proj","proj"):
            if not hasattr(l, attr): continue
            proj = getattr(l, attr)
            A = getattr(proj, "w_lora_A", None); B = getattr(proj, "w_lora_B", None)
            W = getattr(proj, "weight", None)
            if A is None or B is None or W is None: continue
            if A.numel()==0 or B.numel()==0: continue
            delta = (B @ A)
            if delta.shape != W.data.shape and delta.T.shape == W.data.shape:
                delta = delta.T
            W.data.add_(scale * delta)
            if isinstance(A, torch.nn.Parameter): A.detach_(); A.requires_grad_(False); A.data.zero_()
            if isinstance(B, torch.nn.Parameter): B.detach_(); B.requires_grad_(False); B.data.zero_()

# ====== Dataset & text prototypes ======
def build_cub_imagefolder(root):
    # Accept root=.../cub2002011/images or .../train|val|test/<class>
    if os.path.isdir(os.path.join(root, "images")):
        return ImageFolder(os.path.join(root, "images"))
    for split in ("train","val","test"):
        if os.path.isdir(os.path.join(root, split)):
            # concat via single folder with subdirs
            return ImageFolder(root)
    raise FileNotFoundError(f"CUB images not found under {root}")

def build_prompts_and_text_features(clip_model, classnames, device):
    templates = [ "a photo of a {}.", "a bird that is a {}.", "a photo of the bird {}." ]
    with torch.no_grad():
        text_feats = []
        for c in classnames:
            toks = [clip.tokenize(t.format(c)) for t in templates]
            tok = torch.cat(toks).to(device)
            feat = clip_model.encode_text(tok)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            text_feats.append(feat.mean(dim=0))
        text_feats = torch.stack(text_feats, dim=0)   # [C, D]
    return text_feats

# ====== LoRA training step (image-text CE) ======
def lora_train_epoch(clip_model, dataloader, text_feats, optimizer, device, logit_scale=100.0):
    clip_model.train()  # only LoRA params have requires_grad=True
    total, correct, loss_sum = 0, 0, 0.0
    for imgs, y in dataloader:
        imgs, y = imgs.to(device, non_blocking=True), y.to(device)
        img_f = clip_model.encode_image(imgs)              # [B,D]
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        logits = logit_scale * img_f @ text_feats.t()      # [B,C]
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_sum += float(loss) * imgs.size(0)
        pred = logits.argmax(dim=-1)
        correct += (pred==y).sum().item()
        total += imgs.size(0)
    return loss_sum/total, correct/total

# ====== Export current features into SAE shard folder ======
@torch.no_grad()
def export_features(clip_model, ds, preprocess, out_dir, device="cuda", batch_size=128, shard_size=10000, train_ratio=0.9, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    # deterministic split
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(ds))
    rng.shuffle(idxs)
    n_train = int(len(ds) * train_ratio)
    splits = [("train", idxs[:n_train]), ("train_val", idxs[n_train:])]

    # create dataloader wrapper that applies preprocess on the fly
    from torchvision.datasets.folder import default_loader
    def collate(batch):
        xs, ys = [], []
        for (path, y) in batch:
            im = default_loader(path)
            xs.append(preprocess(im))
            ys.append(y)
        return torch.stack(xs), torch.tensor(ys)

    for split_name, sel in splits:
        if len(sel)==0: continue
        subset = torch.utils.data.Subset(ds, sel.tolist())
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate)

        buf, row_count, shard_id = [], 0, 0
        path_f = open(Path(out_dir)/f"{split_name}_{shard_id}.txt","w")

        def flush():
            nonlocal buf, shard_id, path_f
            if not buf: return
            X = torch.cat(buf, dim=0).to(dtype=torch.float16, device="cpu")  # [M,512]
            torch.save(X, Path(out_dir)/f"{split_name}_{shard_id}.pt")
            path_f.close()
            shard_id += 1
            buf.clear()
            path_f = open(Path(out_dir)/f"{split_name}_{shard_id}.txt","w")

        for b, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device)
            feats = clip_model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            buf.append(feats.float().cpu())
            # write paths
            start = b*loader.batch_size
            for orig_idx in sel[start:start+imgs.size(0)]:
                path_f.write(ds.samples[int(orig_idx)][0]+"\n")
            row_count += imgs.size(0)
            if row_count >= shard_size:
                flush(); row_count = 0
        flush()

def build_cub_imagefolder(root):
    """
    Returns (ds, classnames, samples_paths)
    Accepts:
      A) <root>/images/<class>/*.jpg
      B) <root>/{train,val,test}/<class>/*.jpg   (concat)
    """
    img_dir = os.path.join(root, "images")
    if os.path.isdir(img_dir):
        ds = ImageFolder(img_dir)                          # ✅ correct for CUB
        classnames = ds.classes
        samples = [p for p, _ in ds.samples]
        return ds, classnames, samples

    # fallback: concat train/val/test
    split_dirs = [os.path.join(root, s) for s in ("train", "val", "test")]
    if all(os.path.isdir(d) for d in split_dirs):
        parts = [ImageFolder(d) for d in split_dirs]
        # sanity: same class order across splits
        for p in parts[1:]:
            assert p.classes == parts[0].classes, "CUB class lists differ across splits"
        from torch.utils.data import ConcatDataset
        ds = ConcatDataset(parts)
        classnames = parts[0].classes
        samples = []
        for p in parts:
            samples.extend([q for q, _ in p.samples])
        return ds, classnames, samples

    raise FileNotFoundError(
        f"Expected {root}/images or {root}/train,val,test with class subfolders."
    )


# ====== Main alternator ======
def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--cub_root", default="/home/sunayana/Documents/Concept_LoRA/datasets/cub2002011/")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=128)
    # lora
    ap.add_argument("--backbone", default="ViT-B/16")
    ap.add_argument("--blocks", default="top3")      # 'top3' or 'all'
    ap.add_argument("--lora_params", nargs="+", default=["q","v"])
    ap.add_argument("--r", type=int, default=4)
    ap.add_argument("--alpha", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs_lora", type=int, default=2)
    ap.add_argument("--cycles", type=int, default=3)  # [LoRA->export->SAE] cycles
    # sae export
    ap.add_argument("--export_dir", default="/home/sunayana/Documents/Concept_LoRA/datasets/cub_feats_vitb16_lora_combined")
    ap.add_argument("--train_ratio", type=float, default=0.9)
    # sae script hook
    ap.add_argument("--sae_script", default="scripts/train_sae_cub.py")
    ap.add_argument("--sae_epochs", type=int, default=2)     # how many epochs per cycle
    ap.add_argument("--val_freq", type=int, default=1)       # propagated in your script
    args = ap.parse_args()

    device = "cuda" if (args.device=="cuda" and torch.cuda.is_available()) else "cpu"

    # 1) Load dataset (paths only; transforms applied on-the-fly)
    ds_raw = build_cub_imagefolder(args.cub_root)
    classnames = ds_raw.classes

    # 2) Load CLIP & preprocess
    clip_model, preprocess = clip.load(args.backbone, device=device)
    clip_model.eval()

    # 3) Attach LoRA (vision)
    layers = attach_lora(clip_model, backbone=args.backbone, which=args.blocks, params=args.lora_params, r=args.r, alpha=args.alpha, dropout=0.0)

    # 4) Build text prototypes (frozen text)
    text_feats = build_prompts_and_text_features(clip_model, classnames, device)

    # Trainable params = LoRA only
    trainable = [p for p in clip_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    # Image dataloader (uses preprocess)
    tfm = preprocess
    ds = ImageFolder(ds_raw.root, transform=tfm)  # same paths/classes
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    for c in range(args.cycles):
        # ---- LoRA mini-train ----
        for e in range(args.epochs_lora):
            loss, acc = lora_train_epoch(clip_model, loader, text_feats, opt, device)
            print(f"[LoRA] cycle {c+1}/{args.cycles} epoch {e+1}/{args.epochs_lora}  loss={loss:.4f} acc={acc:.3f}")

        # ---- (Optional) fold LoRA for faster export ----
        merge_lora_inplace(layers, scale=float(args.alpha)/float(args.r))
        # NOTE: post-export you can re-attach LoRA if you want to keep training deltas; for simplicity we keep training in un-merged mode next cycle by reloading CLIP+LoRA from a checkpoint if needed.

        # ---- Export features for SAE ----
        export_features(clip_model, ds_raw, preprocess, args.export_dir, device=device, batch_size=args.batch_size, train_ratio=args.train_ratio)

        # ---- Call your SAE trainer ----
        # Your train_sae.py already reads from args.data_dir_activations; we invoke it via subprocess with HACK override pointing to args.export_dir
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" if device=="cuda" else ""
        cmd = f'python {args.sae_script} --resample_freq 10 --ckpt_freq 0 --val_freq {args.val_freq} --num_epochs {args.sae_epochs}'
        print(f"[SAE] launching: {cmd} (reads shards from {args.export_dir})")
        os.system(cmd)

        print(f"[cycle {c+1}] done.")
    print("[joint] finished.")

if __name__ == "__main__":
    main()
