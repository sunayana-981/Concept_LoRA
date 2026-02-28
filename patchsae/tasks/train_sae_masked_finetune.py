"""
Masked fine-tune of a pre-trained ImageNet SAE on LoRA-adapted CLIP activations.

Pipeline:
  1. Load base CLIP ViT + apply LoRA weights  →  LoRA-CLIP
  2. Load pre-trained ImageNet SAE from checkpoint
  3. Build activation store over the NEW dataset (e.g. MedMNIST) using LoRA-CLIP
  4. Estimate per-unit activity on a reference distribution to find "protected" units
  5. Fine-tune the SAE with gradient masking: protected units are frozen,
     only unused / low-activity capacity adapts to new concepts
  6. Decoder rows are re-normalised to unit norm after every optimizer step

Usage:
    python train_sae_masked_finetune.py \
        --sae_checkpoint_path /path/to/imagenet_sae/out.pt \
        --lora_checkpoint_path /path/to/lora_weights.pt \
        --dataset medmnist \
        --protect_frac 0.2 \
        --activity_n_batches 50 \
        --l1_coefficient 0.00008 \
        --batch_size 16 \
        --total_training_tokens 2621440 \
        --log_to_wandb
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import wandb
from datasets import load_dataset

# --- Project imports ---
print("[DEBUG] Importing project modules...")
try:
    from src.sae_training.config import ViTSAERunnerConfig
    from src.sae_training.masked_sae_trainer import (
        MaskedSAETrainer,
        compute_protected_mask,
    )
    from src.sae_training.sparse_autoencoder import SparseAutoencoder
    from src.sae_training.utils import get_scheduler
    from src.sae_training.vit_activations_store import ViTActivationsStore
    from tasks.utils import DATASET_INFO, get_classnames, load_hooked_vit, load_sae
    print("[DEBUG] All project imports successful.")
except ImportError as e:
    print(f"[FATAL] Import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Re-use utilities from the companion LoRA script (optional for MaPLe mode)
try:
    from tasks.train_sae_lora_clip import (
        load_lora_weights,
        verify_activations,
        verify_model_loaded,
    )
except ImportError:
    load_lora_weights = None
    verify_model_loaded = None

    def verify_activations(activation_store, block_layer, device, n_samples=2):
        """Lightweight fallback when train_sae_lora_clip is unavailable."""
        print("[DEBUG] Activation verification (fallback)...")
        for i in range(n_samples):
            batch = activation_store.get_batch_activations()
            print(f"  Sample {i}: shape={batch.shape}, dtype={batch.dtype}, "
                  f"mean={batch.float().mean():.4f}, std={batch.float().std():.4f}")
            if torch.isnan(batch).any():
                print("[FATAL] NaN in activations!")
                sys.exit(1)
        print("[DEBUG] Activation verification PASSED.\n")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Masked fine-tune of a pre-trained ImageNet SAE on LoRA-CLIP activations"
    )

    # --- Pre-trained SAE ---
    parser.add_argument(
        "--sae_checkpoint_path", type=str, required=True,
        help="Path to pre-trained ImageNet SAE checkpoint (.pt)"
    )

    # --- LoRA (required when vit_type != maple) ---
    parser.add_argument(
        "--lora_checkpoint_path", type=str, default=None,
        help="Path to saved LoRA weights (.pt, .bin, or .safetensors). "
             "Required when --vit_type is not 'maple'."
    )

    # --- Model ---
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch16")
    parser.add_argument("--module_name", type=str, default="resid")
    parser.add_argument("--block_layer", type=int, default=-3,
                        help="Transformer block layer (negative = from end)")
    parser.add_argument("--clip_dim", type=int, default=768)
    parser.add_argument("--class_token", action="store_true", default=None)
    parser.add_argument("--image_width", type=int, default=224)
    parser.add_argument("--image_height", type=int, default=224)
    parser.add_argument("--vit_type", type=str, default="base")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)

    # --- Dataset ---
    parser.add_argument("--dataset", type=str, default="medmnist",
                        help="Dataset to fine-tune on (must be in DATASET_INFO)")

    # --- Masking ---
    parser.add_argument("--protect_frac", type=float, default=0.2,
                        help="Fraction of SAE units to protect (top activity)")
    parser.add_argument("--activity_n_batches", type=int, default=50,
                        help="Number of batches to estimate unit activity")
    parser.add_argument(
        "--activity_dataset", type=str, default="imagenet",
        help="Dataset to estimate activity on. Defaults to same as --dataset. "
             "Use 'imagenet' to protect units active on ImageNet."
    )

    # --- SAE architecture (must match pre-trained SAE) ---
    parser.add_argument("--expansion_factor", type=int, default=64)
    parser.add_argument("--gated_sae", action="store_true", default=None)

    # --- Training ---
    parser.add_argument("--lr", type=float, default=0.0004)
    parser.add_argument("--l1_coefficient", type=float, default=0.00008)
    parser.add_argument("--lr_scheduler_name", type=str, default="constantwithwarmup")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_warm_up_steps", type=int, default=500)
    parser.add_argument("--total_training_tokens", type=int, default=2_621_440)
    parser.add_argument("--n_batches_in_store", type=int, default=15)
    parser.add_argument("--mse_cls_coefficient", type=float, default=1.0)

    # --- Dead Neurons ---
    parser.add_argument("--use_ghost_grads", action="store_true", default=True)
    parser.add_argument("--feature_sampling_method", type=str, default=None)
    parser.add_argument("--feature_sampling_window", type=int, default=64)
    parser.add_argument("--dead_feature_window", type=int, default=64)
    parser.add_argument("--dead_feature_threshold", type=float, default=1e-6)

    # --- W&B ---
    parser.add_argument("--log_to_wandb", action="store_true", default=None)
    parser.add_argument("--wandb_project", type=str, default="masked_sae_finetune")
    parser.add_argument("--wandb_entity", type=str, default="test")
    parser.add_argument("--wandb_log_frequency", type=int, default=20)

    # --- Misc ---
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_checkpoints", type=int, default=1)
    parser.add_argument("--checkpoint_path", type=str, default="out/checkpoints/masked_finetune")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run_name", type=str, default="masked_sae_finetune")
    parser.add_argument("--b_dec_init_method", type=str, default="geometric_median")

    # --- Early stopping ---
    parser.add_argument("--early_stop", action="store_true",
                        help="Enable early stopping when loss plateaus")
    parser.add_argument("--early_stop_patience", type=int, default=50,
                        help="Stop after this many steps without improvement (default: 50)")
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-5,
                        help="Minimum loss improvement to count as progress (default: 1e-5)")
    parser.add_argument("--early_stop_warmup", type=int, default=100,
                        help="Don't check early stopping for this many initial steps (default: 100)")

    # --- Debugging ---
    parser.add_argument("--skip_activation_verify", action="store_true",
                        help="Skip activation sanity check")
    parser.add_argument("--dry_run", action="store_true",
                        help="Load everything but don't train")

    args = parser.parse_args()

    # =====================================================================
    # Environment
    # =====================================================================
    print("=" * 60)
    print("MASKED SAE FINE-TUNE — ENVIRONMENT")
    print("=" * 60)
    print(f"  Python:    {sys.version}")
    print(f"  PyTorch:   {torch.__version__}")
    print(f"  CUDA:      {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:       {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:      {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Device:    {args.device}")
    print()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU.")
        args.device = "cpu"

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # =====================================================================
    # Validate paths
    # =====================================================================
    use_maple = (args.vit_type == "maple")

    sae_path = Path(args.sae_checkpoint_path)
    if not sae_path.exists():
        print(f"[FATAL] SAE checkpoint not found: {sae_path}")
        sys.exit(1)

    if use_maple:
        # MaPLe mode: validate model_path and config_path instead of lora
        if not args.model_path or not Path(args.model_path).exists():
            print(f"[FATAL] MaPLe model checkpoint not found: {args.model_path}")
            sys.exit(1)
        if not args.config_path or not Path(args.config_path).exists():
            print(f"[FATAL] MaPLe config not found: {args.config_path}")
            sys.exit(1)
        print(f"[INFO] SAE checkpoint:    {sae_path}")
        print(f"[INFO] MaPLe checkpoint:  {args.model_path}")
        print(f"[INFO] MaPLe config:      {args.config_path}")
    else:
        # LoRA mode: validate lora checkpoint
        if not args.lora_checkpoint_path:
            print("[FATAL] --lora_checkpoint_path is required when --vit_type is not 'maple'")
            sys.exit(1)
        lora_path = Path(args.lora_checkpoint_path)
        if not lora_path.exists():
            print(f"[FATAL] LoRA checkpoint not found: {lora_path}")
            sys.exit(1)
        print(f"[INFO] SAE checkpoint:  {sae_path}")
        print(f"[INFO] LoRA checkpoint: {lora_path}")

    print(f"[INFO] VIT type:        {args.vit_type}")
    print(f"[INFO] Dataset:         {args.dataset}")
    print(f"[INFO] protect_frac:    {args.protect_frac}")
    print()

    t_start = time.time()

    # =====================================================================
    # Step 1: Load pre-trained SAE
    # =====================================================================
    print("=" * 60)
    print("STEP 1: LOAD PRE-TRAINED SAE")
    print("=" * 60)

    sae, sae_cfg = load_sae(str(sae_path), args.device)
    print(f"[DEBUG] Loaded SAE: d_in={sae.d_in}, d_sae={sae.d_sae}, "
          f"params={sum(p.numel() for p in sae.parameters()):,}")

    # We need a proper ViTSAERunnerConfig for the trainer.
    # Build one from args, overriding d_in / expansion_factor to match loaded SAE.
    cfg = ViTSAERunnerConfig(
        class_token=args.class_token,
        image_width=args.image_width,
        image_height=args.image_height,
        model_name=args.model_name,
        module_name=args.module_name,
        block_layer=args.block_layer,
        dataset_path=DATASET_INFO[args.dataset]["path"],
        image_key="image",
        label_key="label",
        d_in=sae.d_in,
        expansion_factor=sae.d_sae // sae.d_in,  # match loaded SAE
        b_dec_init_method=args.b_dec_init_method,
        gated_sae=args.gated_sae,
        lr=args.lr,
        l1_coefficient=args.l1_coefficient,
        lr_scheduler_name=args.lr_scheduler_name,
        batch_size=args.batch_size,
        lr_warm_up_steps=args.lr_warm_up_steps,
        total_training_tokens=args.total_training_tokens,
        n_batches_in_store=args.n_batches_in_store,
        mse_cls_coefficient=args.mse_cls_coefficient,
        use_ghost_grads=args.use_ghost_grads,
        feature_sampling_method=args.feature_sampling_method,
        feature_sampling_window=args.feature_sampling_window,
        dead_feature_window=args.dead_feature_window,
        dead_feature_threshold=args.dead_feature_threshold,
        log_to_wandb=args.log_to_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_log_frequency=args.wandb_log_frequency,
        device=args.device,
        seed=args.seed,
        n_checkpoints=args.n_checkpoints,
        checkpoint_path=args.checkpoint_path,
        dtype=torch.float32,
    )

    # Update SAE config to use our training config (l1, lr, etc.)
    sae.cfg = cfg
    sae.l1_coefficient = cfg.l1_coefficient

    # =====================================================================
    # Step 2: Load ViT (MaPLe or base+LoRA)
    # =====================================================================
    print("\n" + "=" * 60)
    if use_maple:
        print("STEP 2: LOAD MaPLe-ADAPTED ViT")
    else:
        print("STEP 2: LOAD ViT + APPLY LoRA")
    print("=" * 60)

    dataset = load_dataset(**DATASET_INFO[args.dataset])
    classnames = get_classnames(args.dataset, dataset)

    vit = load_hooked_vit(
        cfg, args.vit_type, args.model_name, args.device,
        args.model_path, args.config_path, classnames,
    )

    if use_maple:
        vit.eval()
        print("[DEBUG] MaPLe-adapted ViT loaded. Eval mode.")
    else:
        print("[DEBUG] Base ViT loaded.")
        vit = load_lora_weights(vit, args.lora_checkpoint_path, args.device)
        vit.eval()
        print("[DEBUG] LoRA applied. ViT in eval mode.")
        verify_model_loaded(vit, args.lora_checkpoint_path, args.device)

    # =====================================================================
    # Step 3: Build activation store for target dataset
    # =====================================================================
    print("\n" + "=" * 60)
    print("STEP 3: BUILD ACTIVATION STORE")
    print("=" * 60)

    activation_store = ViTActivationsStore(
        dataset, args.batch_size, args.device, args.seed,
        vit, args.block_layer, cfg.module_name, args.class_token,
    )
    print("[DEBUG] ActivationStore ready.")

    if not args.skip_activation_verify:
        verify_activations(activation_store, args.block_layer, args.device)

    # =====================================================================
    # Step 4: Compute protected mask (activity-based)
    # =====================================================================
    print("\n" + "=" * 60)
    print("STEP 4: COMPUTE PROTECTED MASK")
    print("=" * 60)

    # Optionally use a different dataset for activity estimation
    if args.activity_dataset and args.activity_dataset != args.dataset:
        print(f"[INFO] Using '{args.activity_dataset}' to estimate activity (protect old concepts).")
        act_dataset = load_dataset(**DATASET_INFO[args.activity_dataset])
        act_store = ViTActivationsStore(
            act_dataset, args.batch_size, args.device, args.seed,
            vit, args.block_layer, cfg.module_name, args.class_token,
        )
    else:
        print(f"[INFO] Using '{args.dataset}' to estimate activity.")
        act_store = activation_store

    protected_mask = compute_protected_mask(
        sae, act_store,
        protect_frac=args.protect_frac,
        n_batches=args.activity_n_batches,
        device=args.device,
    )

    # Clean up activity estimation store if separate
    if args.activity_dataset and args.activity_dataset != args.dataset:
        del act_store, act_dataset
        torch.cuda.empty_cache()

    # =====================================================================
    # Step 5: Set up optimizer & train
    # =====================================================================
    print("\n" + "=" * 60)
    print("STEP 5: MASKED FINE-TUNE")
    print("=" * 60)

    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    scheduler = get_scheduler(args.lr_scheduler_name, optimizer=optimizer)
    print(f"[DEBUG] Optimizer: Adam (lr={cfg.lr}), Scheduler: {args.lr_scheduler_name}")

    sae.train()

    if args.dry_run:
        print("\n[INFO] DRY RUN complete. Pipeline validated.")
        print(f"[INFO] Elapsed: {time.time() - t_start:.1f}s")
        return

    # --- W&B ---
    if cfg.log_to_wandb:
        run_name = f"{args.run_name}_pf{args.protect_frac}_layer{args.block_layer}"
        wandb.init(
            project=cfg.wandb_project,
            config={
                **vars(args),
                "sae_d_in": sae.d_in,
                "sae_d_sae": sae.d_sae,
                "n_protected": protected_mask.sum().item(),
                "n_free": sae.d_sae - protected_mask.sum().item(),
            },
            name=run_name,
        )
        print(f"[DEBUG] W&B: {cfg.wandb_project}/{run_name}")

    # --- Train ---
    try:
        trainer = MaskedSAETrainer(
            sae, vit, activation_store, cfg, optimizer, scheduler,
            args.device, protected_mask,
            early_stop=args.early_stop,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            early_stop_warmup=args.early_stop_warmup,
        )
        trainer.fit()
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        traceback.print_exc()
        if cfg.log_to_wandb:
            wandb.finish(exit_code=1)
        sys.exit(1)

    if cfg.log_to_wandb:
        wandb.finish()

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Elapsed:       {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  Checkpoints:   {args.checkpoint_path}")
    print(f"  Protected:     {protected_mask.sum().item()}/{sae.d_sae}")
    print("=" * 60)


if __name__ == "__main__":
    main()
