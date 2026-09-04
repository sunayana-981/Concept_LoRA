"""
One-off migration: strip existing MaPLe checkpoints saved by the old (bloated)
train_maple.py down to just prompt_learner's state, matching the new save
format. Safe because the downstream loader (get_adapted_clip ->
load_state_dict_without_prompt_learner) always calls
model.load_state_dict(..., strict=False) -- dropping the frozen
image_encoder/text_encoder keys changes nothing it actually uses.

Usage: python3 strip_maple_checkpoints.py [maple_weights_dir]
"""

import sys
from pathlib import Path

import torch

root = Path(sys.argv[1] if len(sys.argv) > 1 else "maple_weights")
total_before = 0
total_after = 0
n_stripped = 0
n_already_slim = 0

for f in sorted(root.rglob("model.pth.tar-*")):
    before = f.stat().st_size
    ckpt = torch.load(f, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    stripped = {k: v for k, v in sd.items() if k.startswith("prompt_learner.")}

    if len(stripped) == len(sd):
        n_already_slim += 1
        total_before += before
        total_after += before
        continue

    ckpt["state_dict"] = stripped
    torch.save(ckpt, f)
    after = f.stat().st_size
    total_before += before
    total_after += after
    n_stripped += 1
    print(f"{f}: {before / 1e6:.1f}MB -> {after / 1e6:.1f}MB")

print(f"\nStripped {n_stripped} checkpoints, {n_already_slim} already slim.")
print(f"TOTAL: {total_before / 1e9:.2f}GB -> {total_after / 1e9:.2f}GB "
      f"(freed {(total_before - total_after) / 1e9:.2f}GB)")
