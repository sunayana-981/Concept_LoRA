# scripts/train_sae.py
from dncbm.custom_pipeline import Pipeline
from dncbm.arg_parser import get_common_parser
from dncbm.utils import common_init

import os
from pathlib import Path
from time import time
import datetime
import math
import numpy as np
import torch
import wandb

from sparse_autoencoder import (
    ActivationResampler,
    AdamWithReset,
    L2ReconstructionLoss,
    LearnedActivationsL1Loss,
    LossReducer,
    SparseAutoencoder,
)

# ----------------------------- main -----------------------------
# (Logic moved to global scope per reference)

parser = get_common_parser()
args = parser.parse_args()

common_init(args)
start_time = time()

# --- HACK: OVERRIDE CONFIG VALUES ---
# !! UPDATE THIS PATH TO YOUR CUB ACTIVATIONS !!
args.data_dir_activations = {
    'img': '/home/sunayana/Documents/Concept_LoRA/datasets/cub_feats_vitb16' 
}
print(f"[HACK] Overriding data_dir_activations to: {args.data_dir_activations['img']}")
# --- END HACK ---

print("DEBUG: Parsed CLI args (partial):", {
    "modality": args.modality,
    "embeddings_path": getattr(args, "embeddings_path", None),
    "data_dir_activations": args.data_dir_activations,
    "ae_input_dim": getattr(args, "ae_input_dim", None),
    "use_wandb": args.use_wandb,
    "img_enc_name_for_saving": args.img_enc_name_for_saving,
})

# Vocab embeddings (simplified load per reference)
# Note: The reference script uses a hardcoded path structure.
# We will use the reference script's logic directly.

# --- HACK: FIX EMBEDDINGS PATH ---
# Use the logic from your original script to find embeddings
default_root = "/home/sunayana/Documents/Concept_LoRA/Discover-then-Name/vocab"
cands = [
    getattr(args, "embeddings_path", None), # From CLI (if you add it back)
    os.path.join(default_root, f"embeddings_{args.img_enc_name_for_saving}_20k.pth"),
    os.path.join(default_root, f"embeddings_{args.img_enc_name_for_saving}_clipdissect_20k.pth")
]
embeddings_path = next((p for p in cands if isinstance(p, str) and os.path.isfile(p)), None)

if embeddings_path is None:
    print(f"[warn] No embedding file found for {args.img_enc_name_for_saving} in {default_root}; skipping vocab_specific_embedding.")
# --- END HACK ---

if embeddings_path:
    print(f"DEBUG: Using embeddings_path = {embeddings_path}")
    try:
        args.vocab_specific_embedding = torch.load(embeddings_path, map_location="cpu").to(args.device).float()
        print(f"DEBUG: Loaded embeddings tensor shape: {args.vocab_specific_embedding.shape}")
    except Exception as e:
        print(f"[error] Failed to load embeddings from {embeddings_path}: {e}")
        args.vocab_specific_embedding = None
else:
    args.vocab_specific_embedding = None


# --- HACK: OVERRIDE INPUT DIM ---
# !! VERIFY THIS IS CORRECT FOR YOUR CUB FEATURES !!
autoencoder_input_dim: int = 512
print(f"[HACK] Overriding autoencoder_input_dim to: {autoencoder_input_dim}")
# --- END HACK ---


# Model (use component axis always, per reference)
n_learned_features = int(autoencoder_input_dim * args.expansion_factor)
n_components = len(args.hook_points) # Reference always uses len
autoencoder = SparseAutoencoder(
    n_input_features=autoencoder_input_dim,
    n_learned_features=n_learned_features,
    n_components=n_components
).to(args.device)
print(f"DEBUG: Autoencoder created D={autoencoder_input_dim}, F={n_learned_features}, C={n_components} at {time()-start_time:.2f}s")
print(f"------------Getting Image activations from directory: {args.data_dir_activations[args.modality]}")
print(f"------------Getting Image activations from model: {args.img_enc_name}")
print("all the training args:", args)

# Loss / Optimizer
loss = LossReducer(
    LearnedActivationsL1Loss(l1_coefficient=float(args.l1_coeff)),
    L2ReconstructionLoss()
)
print(f"DEBUG: Loss created at {time()-start_time:.2f}s")

optimizer = AdamWithReset(
    params=autoencoder.parameters(),
    named_parameters=autoencoder.named_parameters(),
    lr=float(args.lr),
    betas=(float(args.adam_beta_1), float(args.adam_beta_2)),
    eps=float(args.adam_epsilon),
    weight_decay=float(args.adam_weight_decay),
    has_components_dim=True,  # Reference script hardcodes this to True
)
print(f"DEBUG: Optimizer created at {time()-start_time:.2f}s")

# Resampler (per reference)
actual_resample_interval = 1 # From reference
activation_resampler = ActivationResampler(
    resample_interval=actual_resample_interval,
    n_activations_activity_collate=actual_resample_interval,
    max_n_resamples=math.inf, # From reference
    n_learned_features=n_learned_features,
    resample_epoch_freq=int(args.resample_freq),
    resample_dataset_size=int(args.resample_dataset_size),
)
print(f"DEBUG: Activation resampler created at {time()-start_time:.2f}s")

# W&B (optional; per reference)
if args.use_wandb:
    print("DEBUG: wandb requested; initializing")
    wandb_project_name = f"SAEImg_{args.sae_dataset}_{args.img_enc_name_for_saving}_{args.hook_points[0]}_{datetime.datetime.now().strftime('%Y-%m-%d')}{args.save_suffix}"
    print(f"DEBUG: wandb project name = {wandb_project_name}")
    
    wandb_dir = os.path.join(args.save_dir[args.modality], ".cache/")
    wandb_path = Path(wandb_dir) # Use 'wandb_path' per reference
    wandb_path.mkdir(exist_ok=True)
    
    wandb.init(
        project=wandb_project_name,
        dir=wandb_dir,
        name=args.config_name,
        config=args, # Pass 'args' directly per reference
        entity="text_concept_explanations"
    )
    wandb.define_metric("custom_steps")
    wandb.define_metric("train/loss_instability_across_batches", step_metric="custom_steps")
    # Reference script does not include the extra debug wandb.log()
    print(f"DEBUG: Wandb initialized at {time()-start_time:.2f}s")

# Build shard lists (per reference)
root = args.data_dir_activations[args.modality]
assert isinstance(root, str), f"Bad data_dir_activations for modality '{args.modality}': {args.data_dir_activations}"
assert os.path.isdir(root), f"Activation dir does not exist: {root}"
print(f"DEBUG: Getting fnames from {root}")

fnames = sorted(os.listdir(root)) # Sort for consistency
train_fnames = []
train_val_fnames = []
abs_root = os.path.abspath(root)

for fname in fnames:
    if not fname.endswith(".pt"): # Ensure we only process .pt files
        continue
        
    # Reference logic: 'train_val' check must come first
    if fname.startswith("train_val"):
        train_val_fnames.append(os.path.join(abs_root, fname))
    elif fname.startswith("train"):
        train_fnames.append(os.path.join(abs_root, fname))

if args.val_freq == 0:
    train_fnames = train_fnames + train_val_fnames
    train_val_fnames = None
    
print(f"DEBUG: Train pieces: {len(train_fnames)}; Val pieces: {len(train_val_fnames) if train_val_fnames is not None else 0} at {time()-start_time:.2f}s")
assert len(train_fnames) > 0, "No .pt train* shards found."
if args.val_freq > 0 and train_val_fnames is not None:
    assert len(train_val_fnames) > 0, "Validation requested (val_freq > 0) but no .pt train_val* shards found."

# Pipeline
checkpoint_dir = Path(f"{args.save_dir_sae_ckpts[args.modality]}{args.save_suffix}")
pipeline = Pipeline(
    activation_resampler=activation_resampler,
    autoencoder=autoencoder,
    checkpoint_directory=checkpoint_dir, # Use Path object
    loss=loss,
    optimizer=optimizer,
    device=args.device,
    args=args,
)
print(f"DEBUG: Pipeline created at {time()-start_time:.2f}s")
print("DEBUG: checkpoint_directory =", checkpoint_dir)

# Train
pipeline.run_pipeline(
    train_batch_size=int(args.train_sae_bs),
    checkpoint_frequency=int(args.ckpt_freq),
    val_frequency=int(args.val_freq),
    num_epochs=args.num_epochs,
    train_fnames=train_fnames,
    train_val_fnames=train_val_fnames,
    start_time=start_time,
    resample_epoch_freq=int(args.resample_freq) # Pass args.resample_freq directly per reference
)

# Standardized save (REMOVED per reference script)
# The reference script does not perform the _standard_save call at the end.
# _standard_save(autoencoder, checkpoint_dir, args) 

print(f"DEBUG: -------total time taken------ {np.round(time()-start_time,3)}")

# Removed `if __name__ == "__main__":` block per reference