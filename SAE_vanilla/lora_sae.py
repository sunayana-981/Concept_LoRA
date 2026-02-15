# CLIP_LoRA/lora/run_lora_kld.py
import os, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torchvision.transforms as transforms

from CLIP_LoRA import clip as clip_pkg
from CLIP_LoRA import clip                             # CLIP loader
from CLIP_LoRA.datasets import build_dataset
from CLIP_LoRA.datasets.utils import build_data_loader
from CLIP_LoRA.utils import set_random_seed            # only this (no get_arguments!)
from CLIP_LoRA.run_utils import inject_lora, mark_only_lora_as_trainable

# ====================== SAE loader (robust) ======================
from model import SparseAutoencoder

def _strip_prefix(sd, prefixes=("module.", "autoencoder.")):
    out = {}
    for k, v in sd.items():
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
                break
        out[k] = v
    return out

def _infer_dims(sd):
    if "encoder._weight" in sd:
        w = sd["encoder._weight"]
        if w.ndim == 3:  # [C,F,D]
            C, F, D = map(int, w.shape);  return D, F, (None if C == 1 else C)
        elif w.ndim == 2:                  # [F,D]
            F, D = map(int, w.shape);      return D, F, None
    if "decoder._weight" in sd:
        w = sd["decoder._weight"]
        if w.ndim == 3:  # [C,D,F]
            C, D, F = map(int, w.shape);   return D, F, (None if C == 1 else C)
        elif w.ndim == 2:                  # [D,F]
            D, F = map(int, w.shape);      return D, F, None
    tb = sd.get("tied_bias") or sd.get("pre_encoder_bias._bias_reference")
    eb = sd.get("encoder._bias")
    D = F = C = None
    if tb is not None:
        if tb.ndim == 2:  C, D = map(int, tb.shape); C = None if C == 1 else C
        elif tb.ndim == 1: D = int(tb.shape[0])
    if eb is not None:
        if eb.ndim == 2:  C2, F = map(int, eb.shape);  C = C or (None if C2 == 1 else C2)
        elif eb.ndim == 1: F = int(eb.shape[0])
    if D is None or F is None:
        raise ValueError("Could not infer (D,F,C) from SAE checkpoint.")
    return D, F, C

def _align_C_axis(sd, want_components: bool):
    out = {}
    for k, v in sd.items():
        if k in ("encoder._weight","decoder._weight"):
            if want_components and v.ndim == 2: v = v.unsqueeze(0)
            if (not want_components) and v.ndim == 3 and v.shape[0] == 1: v = v.squeeze(0)
        elif k in ("encoder._bias","tied_bias","pre_encoder_bias._bias_reference","post_decoder_bias._bias_reference"):
            if want_components and v.ndim == 1: v = v.unsqueeze(0)
            if (not want_components) and v.ndim == 2 and v.shape[0] == 1: v = v.squeeze(0)
        out[k] = v
    return out

def load_sae(path, device="cuda"):
    pkg = torch.load(path, map_location=device)
    if isinstance(pkg, dict) and "state_dict" in pkg:
        sd, meta = pkg["state_dict"], pkg.get("meta", {})
    elif isinstance(pkg, dict):
        sd, meta = pkg, {}
    else:  # torch.nn.Module
        sd, meta = pkg.state_dict(), {}
    sd = _strip_prefix(sd)

    D = meta.get("n_input_features"); F = meta.get("n_learned_features"); C = meta.get("n_components", None)
    if D is None or F is None:
        D, F, C2 = _infer_dims(sd)
        if C is None: C = C2
    want_components = (C is not None and C != 1)

    model = SparseAutoencoder(
        n_input_features=int(D),
        n_learned_features=int(F),
        n_components=(int(C) if want_components else None)
    ).to(device)

    sd = _align_C_axis(sd, want_components)
    model.load_state_dict(sd, strict=False)
    for p in model.parameters(): p.requires_grad = False
    model.eval()
    return model, int(D), int(F), (int(C) if want_components else 1)

# ================== Zero-shot teacher ==================
def build_zeroshot_classifier(clip_model, classnames, templates, device, normalize=True):
    with torch.no_grad():
        zs = []
        for cname in classnames:
            texts = [t.format(cname.replace("_", " ")) for t in templates]
            tokens = clip_pkg.tokenize(texts).to(device)
            text_feats = clip_model.encode_text(tokens)  # [T, d]
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            class_feat = text_feats.mean(dim=0)
            class_feat = class_feat / class_feat.norm()
            zs.append(class_feat)
        W = torch.stack(zs, dim=1).to(device)  # [d, K]
        if normalize: W = W / W.norm(dim=0, keepdim=True)
    return W

# ================== KD training loop ==================
def run_lora_kld(args, clip_model, logit_scale_init, dataset, train_loader, val_loader, test_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Inject LoRA (CLIP_LoRA native)
    r = args.lora_r
    lora_alpha = args.lora_alpha
    target_modules = args.lora_targets
    inject_lora(clip_model.visual, r=r, lora_alpha=lora_alpha, target_modules=target_modules)

    # 2) Freeze base; train only LoRA + head
    mark_only_lora_as_trainable(clip_model)

    # 3) SAE (frozen)
    assert os.path.isfile(args.sae_ckpt), f"--sae_ckpt not found: {args.sae_ckpt}"
    sae, D, F, C = load_sae(args.sae_ckpt, device=device)

    # sanity check CLIP image dim vs SAE
    with torch.no_grad():
        d_clip = clip_model.encode_image(torch.randn(1,3,224,224).to(device)).shape[-1]
    if d_clip != D:
        print(f"[warn] CLIP image dim ({d_clip}) != SAE input dim ({D}). Ensure SAE matches backbone.")

    head = nn.Linear(F, dataset.num_classes).to(device)

    # 4) Teacher (zero-shot text)
    classnames = getattr(dataset, "classnames", None)
    if not classnames:
        raise ValueError("Dataset must expose .classnames (list of class names).")
    templates = getattr(dataset, "templates", ["a photo of a {}."])
    Wz = build_zeroshot_classifier(clip_model, classnames, templates, device)  # [d, K]

    # 5) Optimizer (LoRA + head)
    params = list(filter(lambda p: p.requires_grad, clip_model.parameters())) + list(head.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # 6) KD hyperparams
    T = args.kd_temp
    alpha = args.kd_alpha
    ce_weight = args.ce_weight
    logit_scale = float(getattr(args, "logit_scale", logit_scale_init))
    kl = nn.KLDivLoss(reduction="batchmean")

    @torch.no_grad()
    def encode_image_base(imgs):
        x = clip_model.encode_image(imgs)               # [B, d]
        return x / x.norm(dim=-1, keepdim=True)

    def sae_encode(x):
        la, _ = sae(x.unsqueeze(1))                     # [B,1,F] works for C=None/1
        return la.squeeze(1)                            # [B,F]

    def forward_student(imgs):
        x = clip_model.encode_image(imgs)               # [B,D]
        x = x / x.norm(dim=-1, keepdim=True)
        f = sae_encode(x)                               # [B,F]
        logits_s = head(f)                              # [B,K]
        return logits_s

    def teacher_logits(imgs):
        with torch.no_grad():
            x = encode_image_base(imgs)                 # [B, d]
            logits_t = (x @ Wz) * logit_scale           # [B, K]
        return logits_t

    def eval_loader(loader):
        clip_model.eval(); head.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                s = forward_student(imgs)
                pred = s.argmax(dim=-1)
                correct += (pred == labels).sum().item()
                total += labels.numel()
        return correct / max(1, total)

    # 7) Train
    epochs = args.epochs
    clip_model.train(); head.train()
    if train_loader is None:
        train_loader = val_loader

    for ep in range(1, epochs + 1):
        running = []
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            logits_t = teacher_logits(imgs)
            logits_s = forward_student(imgs)

            pt = F.softmax(logits_t / T, dim=-1)
            log_ps = F.log_softmax(logits_s / T, dim=-1)
            loss_kd = kl(log_ps, pt) * (T * T)

            if ce_weight > 0:
                loss_ce = F.cross_entropy(logits_s, labels)
                loss = alpha * loss_kd + ce_weight * loss_ce
            else:
                loss = alpha * loss_kd

            optim.zero_grad()
            loss.backward()
            optim.step()
            running.append(float(loss.item()))

        acc_val = eval_loader(val_loader)
        print(f"[ep {ep:03d}] loss={np.mean(running):.4f}  val@1={acc_val*100:.2f}")

    test_acc = eval_loader(test_loader)
    print(f"[test] top1 = {test_acc*100:.2f}")

    return {"model": clip_model, "head": head, "zeroshot_W": Wz, "sae": sae}

# ================== CLI & launcher ==================
def build_parser():
    p = argparse.ArgumentParser()
    # dataset / paths
    p.add_argument("--dataset", type=str, required=True)          # e.g., cub / pets / dtd
    p.add_argument("--root_path", type=str, required=True)        # dataset root
    p.add_argument("--shots", type=int, default=16)
    # model
    p.add_argument("--backbone", type=str, default="ViT-B/16")
    # LoRA
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16)
    p.add_argument("--lora_targets", nargs="+",
                   default=["q_proj","k_proj","v_proj","out_proj","fc1","fc2"])
    # training
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--eval_only", action="store_true")
    # KD w/ SAE
    p.add_argument("--sae_ckpt", type=str, required=True)
    p.add_argument("--kd_temp", type=float, default=2.0)
    p.add_argument("--kd_alpha", type=float, default=1.0)
    p.add_argument("--ce_weight", type=float, default=0.0)
    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    set_random_seed(args.seed)

    # CLIP
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()
    logit_scale = 100.0

    # Dataset (+ loaders)
    print("Preparing dataset.")
    dataset = build_dataset(args.dataset, args.root_path, args.shots, preprocess)

    if args.dataset == 'imagenet':
        val_loader  = torch.utils.data.DataLoader(dataset.val,  batch_size=256, num_workers=8, shuffle=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=256, num_workers=8, shuffle=False, pin_memory=True)
    else:
        val_loader  = build_data_loader(data_source=dataset.val,  batch_size=256, is_train=False, tfm=preprocess, shuffle=False, num_workers=8)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=256, is_train=False, tfm=preprocess, shuffle=False, num_workers=8)

    train_loader = None
    if not args.eval_only:
        train_tfm = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711))
        ])
        if args.dataset == 'imagenet':
            train_loader = torch.utils.data.DataLoader(dataset.train_x, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
        else:
            train_loader = build_data_loader(data_source=dataset.train_x, batch_size=args.batch_size, tfm=train_tfm, is_train=True, shuffle=True, num_workers=8)

    # Train
    run_lora_kld(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader)

if __name__ == "__main__":
    main()
