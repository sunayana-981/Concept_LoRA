# lora_finetune_sae.py
import os
import math
import argparse
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Optional: only needed if you want on-the-fly feature extraction
import clip
from torchvision import transforms
from torchvision.datasets import ImageFolder
import pandas as pd
from torch.utils.data import Subset
from tqdm import tqdm

from sparse_autoencoder import SparseAutoencoder

# -----------------------
# LoRA for nn.Linear (module-level)
# -----------------------
class LoRALinear(nn.Module):
    """
    y = x W^T + scale * x A B^T + b
    A: (in, r), B: (out, r), scale = alpha / r
    """
    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: float = 8.0, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")

        self.base = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        self.A = nn.Parameter(torch.empty(self.in_features, r))
        self.B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        x_d = self.drop(x)
        lora_out = (x_d @ self.A) @ self.B.t()  # (B, in)->(B, r)->(B, out)
        return base_out + self.scale * lora_out

    def merge_into_base(self):
        with torch.no_grad():
            self.base.weight.add_(self.scale * self.B @ self.A.t())
        nn.init.zeros_(self.A); nn.init.zeros_(self.B)

def lora_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    for m in module.modules():
        if isinstance(m, LoRALinear):
            yield m.A; yield m.B

def merge_all_lora(module: nn.Module):
    for m in module.modules():
        if isinstance(m, LoRALinear):
            m.merge_into_base()

# -----------------------
# Param-level LoRA (broadcast) for N-D weights (..., out, in)
# -----------------------
import torch.nn.utils.parametrize as parametrize

class LoRABroadcastParametrization(nn.Module):
    """
    For weight W with shape (..., out, in):
    W_eff = W + (alpha/r) * (B @ A^T), broadcast over leading dims.
    A: (in, r), B: (out, r)
    """
    def __init__(self, in_features: int, out_features: int, r: int, alpha: float):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")
        self.r = r
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(in_features, r))
        self.B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        delta = self.B @ self.A.t()  # (out, in)
        # Broadcast to (..., out, in)
        for _ in range(W.ndim - 2):
            delta = delta.unsqueeze(0)
        delta = delta.expand_as(W)
        return W + self.scale * delta

def lora_param_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    # Walk all modules; if they hold parametrizations, pull A/B
    for mod in module.modules():
        if hasattr(mod, "parametrizations"):
            for _, plist in mod.parametrizations.items():
                for pmod in plist:
                    if isinstance(pmod, LoRABroadcastParametrization):
                        yield pmod.A; yield pmod.B

def merge_param_lora_inplace(model: nn.Module):
    """
    Materialize effective weights and remove parametrizations (merging LoRA into base).
    """
    # Iterate a snapshot of modules to avoid mutation during traversal
    for mod in list(model.modules()):
        if hasattr(mod, "parametrizations"):
            for pname in list(mod.parametrizations.keys()):
                try:
                    # leave_parametrized=False => replace with effective tensor
                    parametrize.remove_parametrizations(mod, pname, leave_parametrized=False)
                except Exception:
                    pass

# -----------------------
# Robust subtree attachment for nn.Linear
# -----------------------
def _replace_linears_subtree(root: nn.Module, r: int, alpha: float, dropout: float, prefix: str = "") -> List[str]:
    adapted = []
    for child_name, child in list(root.named_children()):
        path = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear):
            lora = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
            setattr(root, child_name, lora)
            adapted.append(path)
        else:
            adapted.extend(_replace_linears_subtree(child, r, alpha, dropout, path))
    return adapted

# -----------------------
# Param-level attachment helpers
# -----------------------
def _looks_linearish(W: torch.Tensor) -> bool:
    return (W.ndim >= 2) and (W.shape[-2] > 0) and (W.shape[-1] > 0)

def _name_matches_target(name: str, target: str) -> bool:
    n = name.lower()
    if target == "both":
        return any(k in n for k in ("enc", "encoder", "dec", "decoder"))
    if target == "encoder":
        return any(k in n for k in ("enc", "encoder"))
    if target == "decoder":
        return any(k in n for k in ("dec", "decoder"))
    return True

def _attach_param_lora(sae: nn.Module, r: int, alpha: float, target: str) -> List[str]:
    adapted = []
    # Walk parameters and register parametrizations on their owning modules
    for full_name, param in list(sae.named_parameters(recurse=True)):
        if not _looks_linearish(param):
            continue
        if not _name_matches_target(full_name, target):
            continue

        # Descend to owner module and local name
        owner = sae
        parts = full_name.split(".")
        for p in parts[:-1]:
            owner = getattr(owner, p)
        local = parts[-1]

        out_features, in_features = param.shape[-2], param.shape[-1]
        try:
            parametrize.register_parametrization(
                owner, local, LoRABroadcastParametrization(in_features, out_features, r=r, alpha=alpha)
            )
            adapted.append(full_name)
        except Exception:
            pass

    # Freeze base; unfreeze only LoRA params
    for p in sae.parameters():
        p.requires_grad = False
    any_lora = False
    for p in lora_param_parameters(sae):
        p.requires_grad = True
        any_lora = True
    if not any_lora:
        raise RuntimeError("Param-level LoRA attachment failed: found 0 LoRA parameters.")

    print(f"Param-level LoRA attached to {len(adapted)} weight(s):")
    for n in adapted:
        print(f"  - {n}")
    print("Trainable LoRA params:", sum(p.numel() for p in lora_param_parameters(sae)))
    return adapted

# -----------------------
# Unified apply: try module-level first, then param-level fallback
# -----------------------
def apply_lora_to_sae(
    sae: nn.Module,
    r: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target: str = "encoder"
) -> List[str]:
    assert target in {"encoder", "decoder", "both"}
    adapted: List[str] = []

    # Try module-level (nn.Linear) under encoder/decoder subtrees
    def maybe_adapt(attr_name: str):
        sub = getattr(sae, attr_name, None)
        if isinstance(sub, nn.Module):
            return _replace_linears_subtree(sub, r=r, alpha=alpha, dropout=dropout, prefix=attr_name)
        return []

    if target in {"encoder", "both"}:
        adapted += maybe_adapt("encoder")
    if target in {"decoder", "both"}:
        adapted += maybe_adapt("decoder")

    if adapted:
        for p in sae.parameters():
            p.requires_grad = False
        for p in lora_parameters(sae):
            p.requires_grad = True
        num = sum(p.numel() for p in lora_parameters(sae))
        print(f"Module-level LoRA attached to {len(adapted)} Linear layer(s). Trainable LoRA params: {num}")
        for n in adapted:
            print(f"  - {n}")
        return adapted

    # Fallback: parameter-level (handles your 3D weights)
    return _attach_param_lora(sae, r=r, alpha=alpha, target=target)

# -----------------------
# Minimal CUB feature pipeline (optional)
# -----------------------
def get_cub_dataloader(data_root, split='train', batch_size=64, image_size=224, num_workers=4):
    image_dir = os.path.join(data_root, "images")
    image_txt = os.path.join(data_root, "images.txt")
    split_txt = os.path.join(data_root, "train_test_split.txt")

    image_df = pd.read_csv(image_txt, sep=' ', header=None, names=['img_id', 'img_path'])
    split_df = pd.read_csv(split_txt, sep=' ', header=None, names=['img_id', 'is_train'])

    is_train = int(split == 'train')
    split_ids = split_df[split_df['is_train'] == is_train]['img_id'].values
    split_img_paths = image_df[image_df['img_id'].isin(split_ids)]['img_path'].tolist()

    full_dataset = ImageFolder(image_dir, transform=transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]))

    img_path_to_idx = {os.path.relpath(path, image_dir): idx for idx, (path, _) in enumerate(full_dataset.samples)}
    selected_indices, missing = [], []
    for rel_path in split_img_paths:
        norm_path = os.path.normpath(rel_path.strip())
        if norm_path in img_path_to_idx:
            selected_indices.append(img_path_to_idx[norm_path])
        else:
            missing.append(norm_path)

    if missing:
        raise KeyError(f"{len(missing)} metadata paths not found in ImageFolder. First: {missing[0]}")

    subset = Subset(full_dataset, selected_indices)
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers,
        pin_memory=True, persistent_workers=(num_workers > 0)
    )
    return loader

def extract_clip_features(clip_model, loader, device, l2_normalize: bool = False) -> torch.Tensor:
    feats = []
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Extracting CLIP feats"):
            x = x.to(device, non_blocking=True)
            f = clip_model.encode_image(x)
            if l2_normalize:
                f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            feats.append(f.detach().cpu())
    return torch.cat(feats, dim=0)

# -----------------------
# LoRA fine-tune loop
# -----------------------
def train_lora_on_features(
    sae: SparseAutoencoder,
    features: torch.Tensor,
    device: str = "cuda",
    epochs: int = 5,
    batch_size: int = 1024,
    lr: float = 1e-3,
    l1_coeff: float = 0.0
):
    if features.dtype != torch.float32:
        features = features.float()
    sae = sae.to(device).train()

    dataset = TensorDataset(features)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True)

    # Collect both kinds of LoRA params
    params = list(lora_parameters(sae)) + list(lora_param_parameters(sae))
    if len(params) == 0:
        raise RuntimeError("No LoRA parameters found. Did you call apply_lora_to_sae()?")

    opt = optim.AdamW(params, lr=lr)
    mse = nn.MSELoss()

    for ep in range(1, epochs + 1):
        total, n = 0.0, 0
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            # SAE forward must return (z, x_hat). Adjust if different.
            z, recon = sae(xb)
            loss = mse(recon, xb)
            if l1_coeff > 0:
                loss = loss + l1_coeff * z.abs().mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += loss.item() * xb.size(0)
            n += xb.size(0)

        print(f"[LoRA SAE] Epoch {ep}/{epochs}  Recon+Sparsity Loss: {total / max(n,1):.6f}")

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    # Data/feature options
    ap.add_argument("--mode", choices=["from_tensors", "from_cub"], default="from_cub",
                    help="Use precomputed tensors or extract CLIP features from CUB.")
    ap.add_argument("--features_pt", type=str, default="",
                    help="Path to a torch .pt tensor of features (N, D) or dict with key 'features'.")
    ap.add_argument("--cub_root", type=str, default="/home/sunayana/Documents/Concept_LoRA/datasets/cub2002011")
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--norm_features", action="store_true", help="L2-normalize CLIP features when extracting.")

    # SAE + checkpoint
    ap.add_argument("--sae_ckpt", type=str,
                    default="/home/sunayana/Documents/Concept_LoRA/Discover-then-Name/pretrained/Checkpoints/clip_ViT-B:16_sparse_autoencoder_final.pt",
                    help="Path to pretrained SAE checkpoint (state_dict).")
    ap.add_argument("--input_dim", type=int, default=512, help="Input feature dim.")
    ap.add_argument("--latent_dim", type=int, default=4096, help="Latent size.")
    ap.add_argument("--n_components", type=int, default=1)

    # LoRA config
    ap.add_argument("--lora_target", choices=["encoder", "decoder", "both"], default="encoder")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=8.0)
    ap.add_argument("--lora_dropout", type=float, default=0.0)

    # Train config
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l1_coeff", type=float, default=0.0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Save/merge
    ap.add_argument("--save_dir", type=str, default="saved_models")
    ap.add_argument("--save_lora_only", action="store_true", help="Save only LoRA adapter params.")
    ap.add_argument("--merge_and_save_full", action="store_true", help="Merge LoRA into SAE and save full state_dict.")

    # Optional on-the-fly CLIP
    ap.add_argument("--clip_model", type=str, default="ViT-B/16")  # type: ignore

    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # 1) Build SAE and load base weights
    sae = SparseAutoencoder(
        n_input_features=args.input_dim,
        n_learned_features=args.latent_dim,
        n_components=args.n_components
    )
    print(f"Loading SAE from: {args.sae_ckpt}")
    state = torch.load(args.sae_ckpt, map_location="cpu")
    sae.load_state_dict(state, strict=True)

    # 2) Apply LoRA to chosen parts (module-level if possible, else param-level for N-D weights)
    apply_lora_to_sae(
        sae,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target=args.lora_target
    )

    # 3) Prepare features
    if args.mode == "from_tensors":
        if not args.features_pt:
            raise ValueError("--features_pt is required when --mode from_tensors")
        feats = torch.load(args.features_pt, map_location="cpu")
        if isinstance(feats, dict) and "features" in feats:
            feats = feats["features"]
        if feats.dim() != 2 or feats.size(1) != args.input_dim:
            raise ValueError(f"features must be (N, {args.input_dim})")
        feats = feats.contiguous()
    else:
        print("Extracting CLIP features from CUB (train split)...")
        clip_model, _ = clip.load(args.clip_model, device=args.device)
        clip_model.eval()
        loader = get_cub_dataloader(args.cub_root, split='train', batch_size=256, image_size=224, num_workers=4)
        feats = extract_clip_features(clip_model, loader, device=args.device, l2_normalize=args.norm_features)

    # 4) Train LoRA
    train_lora_on_features(
        sae=sae,
        features=feats,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        l1_coeff=args.l1_coeff
    )

    # 5) Save
    if args.save_lora_only:
        lora_sd = {}
        # Module-level LoRA
        for name, m in sae.named_modules():
            if isinstance(m, LoRALinear):
                lora_sd[f"{name}.A"] = m.A.detach().cpu()
                lora_sd[f"{name}.B"] = m.B.detach().cpu()
                lora_sd[f"{name}.scale"] = torch.tensor(m.scale)
        # Param-level LoRA
        for mod in sae.modules():
            if hasattr(mod, "parametrizations"):
                for pname, plist in mod.parametrizations.items():
                    for pmod in plist:
                        if isinstance(pmod, LoRABroadcastParametrization):
                            key = f"{type(mod).__name__}.{id(mod)}/{pname}"
                            lora_sd[f"{key}.A"] = pmod.A.detach().cpu()
                            lora_sd[f"{key}.B"] = pmod.B.detach().cpu()
                            lora_sd[f"{key}.scale"] = torch.tensor(pmod.scale)
        out = os.path.join(args.save_dir, "sae_lora_only.pt")
        torch.save(lora_sd, out)
        print(f"✔ Saved LoRA-only adapters to: {out}")

    if args.merge_and_save_full:
        print("Merging LoRA adapters into base weights...")
        merge_all_lora(sae)           # merge module-level LoRA
        merge_param_lora_inplace(sae) # merge param-level LoRA
        out = os.path.join(args.save_dir, "sae_merged_lora_finetuned.pt")
        torch.save(sae.state_dict(), out)
        print(f"✔ Saved merged SAE to: {out}")

    if not args.save_lora_only and not args.merge_and_save_full:
        out = os.path.join(args.save_dir, "sae_with_lora_params.pt")
        torch.save(sae.state_dict(), out)
        print(f"✔ Saved SAE (with LoRA modules/parametrizations) to: {out}")

if __name__ == "__main__":
    main()
