#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import os
from collections import Counter
from functools import lru_cache

import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# -------------------------
# Shard discovery / loading
# -------------------------
def find_activation_files(d: Path) -> List[Path]:
    d = d.expanduser().resolve()
    if not d.is_dir():
        raise FileNotFoundError(f"activations_dir not found: {d}")
    pts = sorted(
        p for p in d.iterdir()
        if p.is_file() and p.suffix == ".pt" and p.name.startswith(("train_", "train_val_"))
    )
    if not pts:
        raise RuntimeError(f"No .pt shards (train_*.pt or train_val_*.pt) in {d}")
    return pts

def load_activations(shards: List[Path]) -> Tuple[torch.Tensor, List[str]]:
    tensors, names = [], []
    for s in shards:
        t = torch.load(s, map_location="cpu")
        if isinstance(t, torch.Tensor):
            arr = t
        elif isinstance(t, dict):
            keys = [k for k, v in t.items() if isinstance(v, torch.Tensor) and v.ndim == 2]
            if not keys:
                keys = [k for k, v in t.items() if isinstance(v, torch.Tensor)]
            if not keys:
                raise RuntimeError(f"No tensor found in shard: {s}")
            arr = t[keys[0]]
        else:
            raise RuntimeError(f"Unsupported shard format for {s}: {type(t)}")
        if arr.ndim != 2:
            raise RuntimeError(f"Shard {s} has shape {tuple(arr.shape)}; expected 2D (N,D)")
        tensors.append(arr)

        names_file = s.with_suffix(".txt")
        if names_file.exists():
            with open(names_file, "r") as fh:
                file_names = [ln.strip() for ln in fh if ln.strip()]
            if len(file_names) < arr.shape[0]:
                file_names += [f"{s.name}#{i}" for i in range(len(file_names), arr.shape[0])]
            else:
                file_names = file_names[:arr.shape[0]]
            names.extend(file_names)
        else:
            names.extend([f"{s.name}#{i}" for i in range(arr.shape[0])])

    activations = torch.cat(tensors, dim=0).contiguous()
    if len(names) != activations.shape[0]:
        raise RuntimeError(f"Names length ({len(names)}) != activations rows ({activations.shape[0]}).")
    return activations, names

# -------------------------
# Checkpoint utilities
# -------------------------
def load_state_dict(checkpoint_path: Path) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    return ckpt

def _normalize_to_2d(W: torch.Tensor):
    if not isinstance(W, torch.Tensor):
        return None
    orig = tuple(W.shape)
    if W.ndim > 2 and any(s == 1 for s in W.shape):
        W = W.squeeze()
    if W.ndim > 2:
        try:
            Wm = W.mean(dim=0)
            if Wm.ndim == 2:
                print(f"DEBUG: normalized tensor from {orig} -> mean -> {tuple(Wm.shape)}")
                return Wm
        except Exception:
            return None
    if W.ndim == 2:
        if tuple(W.shape) != orig:
            print(f"DEBUG: normalized tensor from {orig} -> squeezed -> {tuple(W.shape)}")
        return W
    return None

def _orient_to_CxD(W: torch.Tensor, D: int):
    W2 = _normalize_to_2d(W)
    if W2 is None:
        return None
    if W2.shape[1] == D:   # (C, D)
        return W2
    if W2.shape[0] == D:   # (D, C) -> (C, D)
        return W2.t()
    return None

def pick_component_matrix(state_dict: dict, input_dim: int):
    """Prefer decoder weights oriented to (C,D) with D==input_dim; fallback to encoder."""
    likely_keys = [
        "decoder._weight","decoder.weight","linear_decoder.weight",
        "reconstruction.weight","components.weight","dict.weight",
        "encoder._weight","encoder.weight","linear_encoder.weight","features.weight",
    ]
    present = [(k, tuple(state_dict[k].shape))
               for k in likely_keys
               if k in state_dict and isinstance(state_dict[k], torch.Tensor)]
    print("DEBUG: candidate weight shapes:", present if present else "none")

    for k in ["decoder._weight","decoder.weight","linear_decoder.weight",
              "reconstruction.weight","components.weight","dict.weight"]:
        if k in state_dict:
            M = _orient_to_CxD(state_dict[k], input_dim)
            if M is not None and 1 not in M.shape:
                print(f"DEBUG: selected decoder '{k}' -> (C,D)={tuple(M.shape)}")
                return k, M, "decoder"

    dec_cands = []
    for k, v in state_dict.items():
        k0 = k.lower()
        if "bias" in k0: continue
        if any(t in k0 for t in ["decoder","recon","component","dict","linear_decoder"]):
            M = _orient_to_CxD(v, input_dim)
            if M is not None and 1 not in M.shape:
                dec_cands.append((M.shape[0], k, M))
    if dec_cands:
        dec_cands.sort(key=lambda x: x[0], reverse=True)
        C, k, M = dec_cands[0]
        print(f"DEBUG: selected decoder '{k}' -> (C,D)={tuple(M.shape)}")
        return k, M, "decoder"

    for k in ["encoder._weight","encoder.weight","linear_encoder.weight","features.weight"]:
        if k in state_dict:
            M = _orient_to_CxD(state_dict[k], input_dim)
            if M is not None and 1 not in M.shape:
                print(f"DEBUG: FALLBACK encoder '{k}' -> (C,D)={tuple(M.shape)}")
                return k, M, "encoder"

    enc_cands = []
    for k, v in state_dict.items():
        if "bias" in k.lower(): continue
        M = _orient_to_CxD(v, input_dim)
        if M is not None and 1 not in M.shape:
            enc_cands.append((M.shape[0], k, M))
    if enc_cands:
        enc_cands.sort(key=lambda x: x[0], reverse=True)
        C, k, M = enc_cands[0]
        print(f"DEBUG: FALLBACK encoder '{k}' -> (C,D)={tuple(M.shape)}")
        return k, M, "encoder"

    raise RuntimeError(
        f"No matrix with an axis equal to input_dim D={input_dim} found. "
        f"Use a matching checkpoint or re-extract features to the checkpoint's D."
    )

# -------------------------
# Image indexing + smart resolver
# -------------------------
def _norm_key(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())

@lru_cache(maxsize=1)
def _index_images_multi(images_root: Path) -> Dict[str, List[str]]:
    """Multi-key index: basename, stem, super-normalized stem (all lowercase)."""
    idx: Dict[str, List[str]] = {}
    root_str = str(images_root)
    total = 0
    for r, _, files in os.walk(root_str):
        for f in files:
            fl = f.lower()
            if not fl.endswith(IMG_EXTS):
                continue
            full = str(Path(r) / f)
            base = fl
            stem = Path(fl).stem
            norm = _norm_key(stem)
            for key in (base, stem, norm):
                idx.setdefault(key, []).append(full)
            total += 1
    print(f"DEBUG: indexed {total} images under {images_root}")
    return idx

def _guess_roots_from_names(names: List[str]) -> List[Path]:
    """
    Find the most common directory prefixes from .txt names and produce plausible roots:
      - the exact common dir if it exists,
      - if it starts with '/Documents', rewrite to '~/Documents/...'
    """
    dirs = []
    for nm in names[: min(5000, len(names))]:  # sample up to 5k
        if "/" in nm:
            try:
                d = str(Path(nm).parent)
                dirs.append(d)
            except Exception:
                pass
    roots = []
    if dirs:
        [(common_dir, _)] = Counter(dirs).most_common(1)
        p = Path(common_dir)
        # exact directory as a root (if exists)
        if p.exists():
            roots.append(p)
        # heuristic: /Documents/... -> ~/Documents/...
        if common_dir.startswith("/Documents"):
            home_rewrite = Path.home() / common_dir.lstrip("/")
            roots.append(home_rewrite)
        # also try ~/ + common_dir if it starts with a slash and doesn't exist
        if common_dir.startswith("/") and not p.exists():
            home_try = Path.home() / common_dir.lstrip("/")
            roots.append(home_try)
    # de-dup while preserving order
    seen = set()
    uniq_roots = []
    for r in roots:
        r = r.expanduser().resolve()
        if r not in seen:
            seen.add(r)
            uniq_roots.append(r)
    if uniq_roots:
        print("DEBUG: guessed extra roots from names:", [str(r) for r in uniq_roots])
    return uniq_roots

def resolve_paths(names: List[str], images_root: Path, extra_roots: List[Path]) -> List[str]:
    """
    Resolution order:
      1) absolute path as-is
      2) images_root / raw
      3) for each extra_root: extra_root / raw
      4) multi-key lookup: basename, stem, normalized stem under images_root index
    """
    idx = _index_images_multi(images_root)
    out = []
    for nm in names:
        raw = nm.strip()
        p = Path(raw)

        # 1) absolute
        if p.is_absolute() and p.exists():
            out.append(str(p)); continue

        # 2) relative to images_root
        cand = images_root / raw
        if cand.exists():
            out.append(str(cand)); continue

        # 3) try extra roots inferred from names
        hit = None
        for root in extra_roots:
            cand2 = root / raw.lstrip("/")  # support absolute-looking raw
            if cand2.exists():
                hit = str(cand2); break
        if hit:
            out.append(hit); continue

        # 4) index-based matching (basename/stem/normalized)
        base = p.name.lower()
        stem = Path(base).stem
        norm = _norm_key(stem)
        hits = idx.get(base) or idx.get(stem) or idx.get(norm) or []
        out.append(hits[0] if hits else str(cand))  # may not exist; filtered before plotting
    return out

def safe_open_img(p: Path):
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None

def plot_topk_images(image_paths: List[str], outpath: Path, topk: int, title: str = None):
    topk = max(1, topk)
    cols = topk
    rows = 1
    fig = plt.figure(figsize=(cols * 2, rows * 2))
    for i, p in enumerate(image_paths[:topk]):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.axis("off")
        im = safe_open_img(Path(p))
        if im is None:
            ax.text(0.5, 0.5, "MISSING", ha="center", va="center")
        else:
            ax.imshow(im)
    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close(fig)

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations_dir", required=True,
                    help="Dir with train_*.pt and train_val_*.pt shards (features)")
    ap.add_argument("--checkpoint", required=True,
                    help="SAE checkpoint (.pt) containing decoder/encoder weights")
    ap.add_argument("--images_root", required=True,
                    help="Root of structured dataset (class subfolders) to resolve image paths")
    ap.add_argument("--topk", type=int, default=8, help="Top-k images per neuron")
    ap.add_argument("--max_components", type=int, default=256, help="Cap on neurons to visualize")
    ap.add_argument("--out_dir", default="./sae_concepts", help="Output dir for PNG/TXT")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    args = ap.parse_args()

    activations_dir = Path(args.activations_dir)
    images_root = Path(args.images_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load activations + names
    shards = find_activation_files(activations_dir)
    print("DEBUG: shards =", [s.name for s in shards])
    activations, names = load_activations(shards)
    print(f"DEBUG: activations shape = {tuple(activations.shape)}; names = {len(names)}")
    if any("#" in n for n in names[:50]):
        print("WARNING: Some names look like placeholders (e.g., 'train_0.pt#i'). "
              "Make sure *.txt sidecars are present next to the shards.")

    # 2) Device + D
    device = torch.device(args.device)
    activations = activations.to(device).float()
    N, D = activations.shape
    print(f"DEBUG: N={N}, D={D}")

    # 3) Load checkpoint + pick matrix
    state_dict = load_state_dict(Path(args.checkpoint))
    print("DEBUG: keys (sample):", list(state_dict.keys())[:20])
    key, comp_mat, source = pick_component_matrix(state_dict, D)  # comp_mat: (C, D)
    comp_mat = comp_mat.to(device).float()
    C = comp_mat.shape[0]
    print(f"DEBUG: using {source} '{key}' -> (C,D)={tuple(comp_mat.shape)}")

    # 4) Scores = A @ W^T  → (N, C)
    scores = (activations @ comp_mat.t()).detach().cpu().numpy()

    # 5) Build image index once + guess extra roots from names (handles moved datasets)
    _ = _index_images_multi(images_root)
    extra_roots = _guess_roots_from_names(names)

    # 6) Visualize per neuron
    max_c = min(C, args.max_components)
    for c in range(max_c):
        sc = scores[:, c]
        order = np.argsort(-np.abs(sc))  # use -sc for positive-only
        top_idx = order[:args.topk]
        top_names = [names[i] for i in top_idx]

        # TXT
        txt_path = out_dir / f"component_{c:04d}_topk.txt"
        with open(txt_path, "w") as fh:
            fh.write(f"matrix_key={key} source={source} component={c}\n")
            for rank, i in enumerate(top_idx, 1):
                fh.write(f"{rank}\tidx={i}\tscore={float(sc[i]):.6f}\tname={names[i]}\n")

        # Resolve → filter existing → plot
        resolved = resolve_paths(top_names, images_root, extra_roots)
        existing = [p for p in resolved if Path(p).exists()]
        hit = len(existing)

        png_path = out_dir / f"component_{c:04d}_topk.png"
        if hit:
            plot_topk_images(existing, png_path, topk=min(args.topk, hit), title=f"component {c}")
            print(f"[{c+1}/{max_c}] images found {hit}/{args.topk} → {png_path.name}")
        else:
            plot_topk_images([], png_path, topk=1, title=f"component {c} (no images)")
            print(f"[{c+1}/{max_c}] images found 0/{args.topk} (all missing) → {png_path.name}")

    print("DONE: visualized", max_c, "components →", out_dir)

if __name__ == "__main__":
    main()
