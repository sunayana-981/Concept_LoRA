#!/usr/bin/env python3
import os, math, argparse, torch, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torch.utils.data import ConcatDataset
import torch.nn as nn
import clip

# === from your project ===
from layers import PlainMultiheadAttentionLoRA

# ----------------- Index maps -----------------
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

# ----------------- LoRA attach -----------------
def apply_lora(args, clip_model):
    list_lora_layers = []
    if args.encoder in ('text', 'both'):
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_mha = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=args.params, r=args.r,
                            lora_alpha=args.alpha, dropout_rate=args.dropout_rate
                        )
                        setattr(block, name, new_mha)
                        list_lora_layers.append(new_mha)
    if args.encoder in ('vision', 'both'):
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_mha = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=args.params, r=args.r,
                            lora_alpha=args.alpha, dropout_rate=args.dropout_rate
                        )
                        setattr(block, name, new_mha)
                        list_lora_layers.append(new_mha)
    return list_lora_layers

# ----------------- Load LoRA weights -----------------
def load_lora(args, list_lora_layers):
    # If you really want a hardcoded path, keep it here:
    load_path = "/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16/cub2002011/16shots/seed1/lora_weights.pt"
    if not os.path.exists(load_path):
        raise FileNotFoundError(f'File {load_path} does not exist.')
    loaded = torch.load(load_path, map_location="cpu")
    metadata = loaded['metadata']

    # Accept list or string for params
    def _as_set(x):
        if isinstance(x, str): return set(list(x))
        if isinstance(x, (list, tuple)): return set(x)
        return set(x)
    if metadata['r'] != args.r:
        raise ValueError(f"r mismatch: expected {args.r}, found {metadata['r']}")
    if metadata['alpha'] != args.alpha:
        raise ValueError(f"alpha mismatch: expected {args.alpha}, found {metadata['alpha']}")
    if metadata['encoder'] != args.encoder:
        raise ValueError(f"Encoder mismatch: expected {args.encoder}, found {metadata['encoder']}")
    if _as_set(metadata['params']) != _as_set(args.params):
        raise ValueError(f"Params mismatch: expected {args.params}, found {metadata['params']}")
    if metadata['position'] != args.position:
        raise ValueError(f"Position mismatch: expected {args.position}, found {metadata['position']}")

    weights = loaded['weights']
    for i, layer in enumerate(list_lora_layers):
        lw = weights[f'layer_{i}']
        if 'q' in args.params and 'q_proj' in lw:
            layer.q_proj.w_lora_A.data.copy_(lw['q_proj']['w_lora_A'])
            layer.q_proj.w_lora_B.data.copy_(lw['q_proj']['w_lora_B'])
        if 'k' in args.params and 'k_proj' in lw:
            layer.k_proj.w_lora_A.data.copy_(lw['k_proj']['w_lora_A'])
            layer.k_proj.w_lora_B.data.copy_(lw['k_proj']['w_lora_B'])
        if 'v' in args.params and 'v_proj' in lw:
            layer.v_proj.w_lora_A.data.copy_(lw['v_proj']['w_lora_A'])
            layer.v_proj.w_lora_B.data.copy_(lw['v_proj']['w_lora_B'])
        if ('o' in args.params or 'proj' in lw) and 'proj' in lw:
            layer.proj.w_lora_A.data.copy_(lw['proj']['w_lora_A'])
            layer.proj.w_lora_B.data.copy_(lw['proj']['w_lora_B'])
    print(f'LoRA weights loaded from {load_path}')

# ----------------- Merge utilities -----------------
def _merge_one_proj(proj, scale: float) -> bool:
    W = getattr(proj, "weight", None)
    A = getattr(proj, "w_lora_A", None)
    B = getattr(proj, "w_lora_B", None)
    if W is None or A is None or B is None: return False
    Wt = W.data if hasattr(W, "data") else W
    At = A.data if hasattr(A, "data") else A
    Bt = B.data if hasattr(B, "data") else B
    if not (torch.is_tensor(Wt) and torch.is_tensor(At) and torch.is_tensor(Bt)): return False

    delta = None
    if Bt.ndim == 2 and At.ndim == 2:
        if Bt.shape[1] == At.shape[0]: delta = Bt @ At
        elif At.shape[1] == Bt.shape[0]: delta = At @ Bt
        if delta is not None and delta.shape != Wt.shape and delta.t().shape == Wt.shape:
            delta = delta.t()
    if delta is None or delta.shape != Wt.shape: return False

    with torch.no_grad():
        # cast delta to W dtype (handles fp16 base + fp32 LoRA)
        if delta.dtype != Wt.dtype:
            delta = delta.to(dtype=Wt.dtype)
        Wt.add_(scale * delta)
        for pname in ("w_lora_A", "w_lora_B"):
            p = getattr(proj, pname, None)
            if p is None: continue
            if isinstance(p, torch.nn.Parameter):
                p.detach_(); p.requires_grad_(False); p.data.zero_()
            else:
                (p.data if hasattr(p, "data") else p).zero_()
        if hasattr(proj, "merged"): proj.merged = True
    return True

def merge_lora_into_list(list_lora_layers, lora_alpha: float, r: int) -> int:
    scale = float(lora_alpha) / float(r)
    merged = 0
    for layer in list_lora_layers:
        for attr in ("q_proj", "k_proj", "v_proj", "proj"):
            if hasattr(layer, attr):
                merged += int(_merge_one_proj(getattr(layer, attr), scale))
    return merged

# ----------------- Dataset helpers -----------------
def build_dataset(root, preprocess):
    p = os.path.join(root, "cub2002011", "images")
    if os.path.isdir(p): return ImageFolder(p, transform=preprocess), p
    p = os.path.join(root, "images")
    if os.path.isdir(p): return ImageFolder(p, transform=preprocess), p
    base = os.path.join(root, "cub2002011")
    if all(os.path.isdir(os.path.join(base, s)) for s in ("train", "val", "test")):
        parts = [ImageFolder(os.path.join(base, s), transform=preprocess) for s in ("train", "val", "test")]
        return ConcatDataset(parts), base
    if all(os.path.isdir(os.path.join(root, s)) for s in ("train", "val", "test")):
        parts = [ImageFolder(os.path.join(root, s), transform=preprocess) for s in ("train", "val", "test")]
        return ConcatDataset(parts), root
    raise FileNotFoundError(f"Could not find CUB images under {root}.")

def collect_paths(ds):
    if hasattr(ds, "samples"):
        return [p for p, _ in ds.samples]
    if isinstance(ds, ConcatDataset):
        paths = []
        for sub in ds.datasets:
            if not hasattr(sub, "samples"):
                raise TypeError("ConcatDataset contains non-ImageFolder subset.")
            paths.extend([p for p, _ in sub.samples])
        return paths
    raise TypeError(f"Unsupported dataset type: {type(ds)}")

# ----------------- Encode with merged model -----------------
@torch.no_grad()
def encode_ds_with_model(model, preprocess, ds, out_dir, device="cuda",
                         batch_size=128, shard_size=10000, train_ratio=0.9, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    rng = np.random.default_rng(seed)
    idxs = np.arange(len(ds)); rng.shuffle(idxs)
    n_train = int(len(ds) * train_ratio)
    splits = [("train", idxs[:n_train]), ("train_val", idxs[n_train:])]

    print(f"Dataset size={len(ds)} → train={n_train}, val={len(ds)-n_train}")
    base_paths = collect_paths(ds); assert len(base_paths) == len(ds)

    for split_name, sel in splits:
        if len(sel) == 0: continue
        subset = Subset(ds, sel)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

        buf, row_count, shard_id = [], 0, 0
        paths_f = open(Path(out_dir) / f"{split_name}_{shard_id}.txt", "w")

        def flush():
            nonlocal buf, shard_id, paths_f
            if not buf: return
            X = torch.cat(buf, dim=0).to(dtype=torch.float16, device="cpu")
            out_pt = Path(out_dir) / f"{split_name}_{shard_id}.pt"
            torch.save(X, out_pt)
            paths_f.close()
            print(f"wrote {out_pt}  ({X.shape[0]} rows)")
            shard_id += 1; buf = []
            paths_f = open(Path(out_dir) / f"{split_name}_{shard_id}.txt", "w")

        for b, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            feats = model.encode_image(imgs)                  # [B,512], fp16 on CUDA typically
            feats = feats / feats.norm(dim=-1, keepdim=True)  # cosine-normalized
            buf.append(feats.float().cpu())                   # keep float32 in RAM; save fp16
            start = b * loader.batch_size; end = start + imgs.size(0)
            for orig_idx in sel[start:end]:
                paths_f.write(base_paths[int(orig_idx)] + "\n")
            row_count += imgs.size(0)
            if row_count >= shard_size:
                flush(); row_count = 0
        flush()

# ----------------- CLI -----------------
def build_args():
    p = argparse.ArgumentParser("Merge LoRA into CLIP and save activations.")
    # data
    p.add_argument("--root", default="/home/sunayana/Documents/Concept_LoRA/datasets/cub2002011/")
    p.add_argument("--out",  default="/home/sunayana/Documents/Concept_LoRA/datasets/cub_features_merged_lora/")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--shard_size", type=int, default=10000)
    p.add_argument("--train_ratio", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=1)
    # LoRA/CLIP
    p.add_argument("--backbone", type=str, default="ViT-B/16")
    p.add_argument("--encoder", type=str, choices=["text", "vision", "both"], default="both")
    p.add_argument('--params', type=str, nargs='+', default=['q','k','v'])
    p.add_argument('--position', type=str, default='all',
                   choices=['bottom','mid','up','half-up','half-bottom','all','top3'])
    p.add_argument('--r', default=2, type=int)
    p.add_argument('--alpha', default=1, type=int)
    p.add_argument('--dropout_rate', default=0.25, type=float)
    # (kept for metadata checks / path building if you un-hardcode later)
    p.add_argument("--save_path", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--shots", type=int, default=16)
    p.add_argument("--lora_seed", type=int, default=1)
    return p.parse_args()

# ----------------- main -----------------
def main():
    args = build_args()
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    # 1) Load CLIP
    clip_model, preprocess = clip.load(args.backbone, device=device)
    clip_model.eval()

    # 2) Attach LoRA wrappers
    list_lora_layers = apply_lora(args, clip_model)

    # 3) Ensure new modules match CLIP device+dtype
    ref = next(clip_model.parameters())
    model_device, model_dtype = ref.device, ref.dtype
    for m in list_lora_layers:
        for attr in ("q_proj", "k_proj", "v_proj", "proj"):
            if hasattr(m, attr):
                getattr(m, attr).to(device=model_device, dtype=model_dtype)
    clip_model.to(device=model_device, dtype=model_dtype)

    # 4) Load LoRA weights (they might be fp32) → recast to model dtype/device
    load_lora(args, list_lora_layers)
    for m in list_lora_layers:
        for attr in ("q_proj", "k_proj", "v_proj", "proj"):
            if hasattr(m, attr):
                getattr(m, attr).to(device=model_device, dtype=model_dtype)

    # 5) Merge LoRA → base CLIP
    with torch.no_grad():
        merged = merge_lora_into_list(list_lora_layers, lora_alpha=args.alpha, r=args.r)
    print(f"[merge] folded LoRA into {merged} projection weights. Proceeding with merged model.")

    # 6) Build dataset & encode with merged model
    ds, _ = build_dataset(args.root, preprocess)
    encode_ds_with_model(
        model=clip_model, preprocess=preprocess, ds=ds, out_dir=args.out,
        device=device, batch_size=args.batch_size, shard_size=args.shard_size,
        train_ratio=args.train_ratio, seed=args.seed,
    )
    print("[done] activations saved.")

if __name__ == "__main__":
    main()
