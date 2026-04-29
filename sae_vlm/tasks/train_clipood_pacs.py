#!/usr/bin/env python3
"""
CLIPood fine-tuning on PACS (leave-one-out protocol).

Implements two CLIPood ingredients on top of standard CLIP fine-tuning:
  1. Margin metric softmax: per-class margin derived from semantic distances
     between class text embeddings, subtracted from logits before cross-entropy.
  2. Beta Moving Average (BMA): stores checkpoint snapshots at even intervals
     during training and returns a Beta-distribution-weighted average of all
     snapshots as the final model.

Output: a full HF CLIPModel state dict saved to
    {output_dir}/clipood_pacs_held_{held_out}.pt

Usage:
    # Fine-tune with photo held out (~30 min on 3090):
    python tasks/train_clipood_pacs.py \
        --held_out photo \
        --pacs_root /path/to/PACS \
        --output_dir out/clipood_pacs

    # All four splits:
    for domain in photo art_painting cartoon sketch; do
        python tasks/train_clipood_pacs.py --held_out $domain \
            --pacs_root /path/to/PACS --output_dir out/clipood_pacs
    done
"""

import argparse
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parents[1])
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from transformers import CLIPModel, CLIPTokenizerFast

PACS_DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
PACS_CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
BACKBONE = "openai/clip-vit-base-patch16"

# CLIP's normalization constants
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


# ── Data ──────────────────────────────────────────────────────────────────────

def build_source_loader(pacs_root: str, source_domains: list, batch_size: int) -> DataLoader:
    """Concatenate source domains into one shuffled DataLoader."""
    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
    ])
    parts = []
    for domain in source_domains:
        p = Path(pacs_root) / domain
        if not p.is_dir():
            raise FileNotFoundError(
                f"PACS domain not found: {p}\n"
                f"Expected structure: {pacs_root}/art_painting/, {pacs_root}/cartoon/, etc."
            )
        ds = datasets.ImageFolder(str(p), transform=tfm)
        # Remap to PACS_CLASSES order (alphabetical sort matches PACS_CLASSES)
        parts.append(ds)

    dataset = ConcatDataset(parts)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=4, pin_memory=True, drop_last=True)


# ── Beta Moving Average ────────────────────────────────────────────────────────

class BMATracker:
    """
    Collects model snapshots at evenly-spaced steps and returns a
    Beta(alpha, beta)-weighted average of all snapshots as the final model.

    The Beta PDF is evaluated at each snapshot's fractional position in training
    (0, 1/n, 2/n, ..., 1).  Beta(2, 5) peaks at t≈0.2, giving slightly more
    weight to early/mid snapshots — stable but still adapted.
    """

    def __init__(self, alpha: float = 2.0, beta: float = 5.0):
        self.alpha = alpha
        self.beta  = beta
        self.snapshots: list[dict] = []

    @torch.no_grad()
    def snapshot(self, model: torch.nn.Module):
        self.snapshots.append(
            {k: v.detach().clone().cpu().float() for k, v in model.state_dict().items()}
        )

    def compute(self) -> dict:
        n = len(self.snapshots)
        assert n > 0, "No snapshots recorded."
        positions = torch.linspace(1 / n, 1.0, n)
        dist = torch.distributions.Beta(
            torch.tensor(float(self.alpha)), torch.tensor(float(self.beta))
        )
        # Clamp away from 0/1 where Beta PDF can be 0 or inf
        log_weights = dist.log_prob(positions.clamp(1e-4, 1 - 1e-4))
        weights = torch.softmax(log_weights, dim=0)  # [n], sum to 1

        bma_state = {}
        for key in self.snapshots[0]:
            stacked = torch.stack([s[key].float() for s in self.snapshots], dim=0)
            w = weights.view(-1, *([1] * (stacked.ndim - 1)))
            bma_state[key] = (w * stacked).sum(0)
        return bma_state


# ── Training ──────────────────────────────────────────────────────────────────

def train_clipood(args):
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable — falling back to CPU.")
        device = "cpu"

    held         = args.held_out
    source_doms  = [d for d in PACS_DOMAINS if d != held]
    print(f"[INFO] Held-out: {held}  |  Sources: {source_doms}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"[INFO] Loading CLIP ({BACKBONE})...")
    model     = CLIPModel.from_pretrained(BACKBONE).to(device)
    tokenizer = CLIPTokenizerFast.from_pretrained(BACKBONE)
    model.train()

    # ── Pre-compute initial text embeddings + class margins ────────────────────
    text_inputs = tokenizer(PACS_CLASSES, return_tensors="pt", padding=True).to(device)

    @torch.no_grad()
    def get_text_feats_and_margins():
        t_feats = model.get_text_features(**text_inputs)          # [C, d]
        t_feats = F.normalize(t_feats.float(), dim=-1)
        C       = len(PACS_CLASSES)
        sim     = t_feats @ t_feats.T                            # [C, C]
        # Mean cosine similarity to all other classes
        off_diag = (sim.sum(1) - sim.diagonal()) / (C - 1)      # [C]
        logit_scale = model.logit_scale.exp().item()
        margins = logit_scale * off_diag                          # [C], in logit space
        return t_feats, margins

    t_feats, margins = get_text_feats_and_margins()
    print(f"  logit_scale={model.logit_scale.exp().item():.2f} | "
          f"margins min={margins.min().item():.3f} max={margins.max().item():.3f}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("[INFO] Building dataloader...")
    loader = build_source_loader(args.pacs_root, source_doms, args.batch_size)
    print(f"  Source images: {len(loader.dataset):,} | batches/epoch: {len(loader)}")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # ── BMA tracker ───────────────────────────────────────────────────────────
    bma = BMATracker(alpha=args.bma_alpha, beta=args.bma_beta)
    snap_interval = max(1, args.steps // args.n_bma_snapshots)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"[INFO] Training for {args.steps} steps "
          f"(snap every {snap_interval} steps, {args.n_bma_snapshots} total)...")
    step = 0
    running_loss = 0.0

    while step < args.steps:
        for pixel_values, labels in loader:
            if step >= args.steps:
                break

            pixel_values = pixel_values.to(device)
            labels       = labels.to(device)

            # Refresh text features + margins every 200 steps (both encoders update)
            if step % 200 == 0 and step > 0:
                t_feats, margins = get_text_feats_and_margins()

            img_feats = model.get_image_features(pixel_values=pixel_values).float()
            img_feats = F.normalize(img_feats, dim=-1)           # [B, d]

            logit_scale = model.logit_scale.exp()
            logits = logit_scale * (img_feats @ t_feats.T)       # [B, C]
            logits = logits - margins.unsqueeze(0)                # subtract per-class margin

            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            step += 1

            if step % snap_interval == 0:
                bma.snapshot(model)

            if step % 100 == 0:
                avg = running_loss / 100
                running_loss = 0.0
                print(f"  step {step:5d}/{args.steps} | loss {avg:.4f} | "
                      f"snapshots {len(bma.snapshots)}")

    # Ensure we have at least one snapshot
    if not bma.snapshots:
        bma.snapshot(model)

    # ── Apply BMA ─────────────────────────────────────────────────────────────
    print(f"[INFO] Computing BMA over {len(bma.snapshots)} snapshots...")
    bma_state = bma.compute()
    model.load_state_dict(bma_state, strict=True)
    model.eval()

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"clipood_pacs_held_{held}.pt"
    torch.save(model.state_dict(), str(out_path))
    print(f"[INFO] Saved: {out_path}  ({out_path.stat().st_size / 1e6:.0f} MB)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CLIPood fine-tuning on PACS (leave-one-out)")
    parser.add_argument("--held_out", required=True, choices=PACS_DOMAINS,
                        help="Domain to hold out as target")
    parser.add_argument("--pacs_root", required=True,
                        help="Path whose subdirectories are art_painting/, cartoon/, photo/, sketch/")
    parser.add_argument("--output_dir", default="out/clipood_pacs")
    parser.add_argument("--steps", type=int, default=4000,
                        help="Gradient steps (CLIPood paper uses 4000 for PACS)")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate (CLIPood paper: 5e-6 for PACS)")
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--bma_alpha", type=float, default=2.0,
                        help="Beta(alpha, beta) shape for BMA checkpoint weighting")
    parser.add_argument("--bma_beta", type=float, default=5.0)
    parser.add_argument("--n_bma_snapshots", type=int, default=20,
                        help="How many model snapshots to accumulate for BMA")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train_clipood(args)


if __name__ == "__main__":
    main()
