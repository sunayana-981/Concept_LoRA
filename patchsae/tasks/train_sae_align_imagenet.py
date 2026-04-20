"""
Train Sparse Autoencoder on activations from the ALIGN VLM vision encoder on ImageNet.

ALIGN (Jia et al., 2021) uses an EfficientNet-B7-like vision encoder (not a ViT).
HuggingFace: kakaobrain/align-base

Architecture:
  vision_model.encoder.blocks  — 55 AlignVisionBlocks
  Last block [54] output:       [B, 640, 10, 10]   (for 346×346 input)
  Pooler (AvgPool2d) output:    [B, 640]

Two activation modes:
  --class_token   (default) — global average pool → [B, 640]; analogous to CLS token
  --no_class_token          — all 10×10=100 spatial positions → [B*100, 640]

Calibrated hyperparameters (see comments in argparse for derivation):
  expansion_factor = 32   → d_sae = 20,480  (vs 64→49,152 for CLIP; avoids 87% dead features)
  l1_coefficient   = 0.00025               (CLIP×3.1 to match l1/MSE ratio: larger MSE scale)
  batch_size       = 32                    (346×346 fits on 24 GB; better gradient)
  total_tokens     = 10,240,000            (500 tok/feature × 20,480; ~18h on RTX 3090)
  dead_window      = 1024                  (features fire every ~410 steps at L0=50)
  use_ghost_grads  = True                  (critical to revive dead features)
  lr_warm_up_steps = 1000                  (EfficientNet GAP features need longer warmup)

Usage:
    # Standard run (calibrated defaults):
    python tasks/train_sae_align_imagenet.py --log_to_wandb

    # Spatial (patch-like) mode — 10×10=100 spatial positions per image:
    python tasks/train_sae_align_imagenet.py --no_class_token

    # Multiple blocks (one SAE per block, d_in must match block channel dim):
    python tasks/train_sae_align_imagenet.py --block_layers -3 -2 -1 --d_in 640

    # Quick sanity check:
    python tasks/train_sae_align_imagenet.py --dry_run --skip_activation_verify

Block channel dims for reference (kakaobrain/align-base, 346×346 input):
  blocks[ 0]:  32 ch, 173×173
  blocks[10]:  48 ch,  86×86
  blocks[20]: 160 ch,  21×21
  blocks[30]: 224 ch,  21×21
  blocks[40]: 384 ch,  10×10
  blocks[50]: 384 ch,  10×10
  blocks[54]: 640 ch,  10×10   ← default (block_layer=-1)
"""

import argparse
import sys
import traceback
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Tuple

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from transformers import AlignModel, AlignProcessor

print("[DEBUG] Importing project modules...")
try:
    from src.sae_training.config import ViTSAERunnerConfig
    from src.sae_training.sae_trainer import SAETrainer
    from src.sae_training.sparse_autoencoder import SparseAutoencoder
    from src.sae_training.utils import get_scheduler
    from tasks.utils import DATASET_INFO
    print("[DEBUG] All project imports successful.")
except ImportError as e:
    print(f"[FATAL] Import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_ID   = "kakaobrain/align-base"
N_BLOCKS   = 55     # total AlignVisionBlocks in kakaobrain/align-base
IMAGE_SIZE = 346    # AlignProcessor default


# ── Hook + Hooked wrapper ──────────────────────────────────────────────────────

class ALIGNHook:
    """
    Forward hook for AlignVisionBlock.

    EfficientNet blocks output spatial feature maps [B, C, H, W],
    not sequential tokens like transformers.

    class_token=True  → global average pool → [B, C]  (CLS-like)
    class_token=False → flatten spatial     → [B*H*W, C]
    """

    def __init__(self, block_idx: int, hook_fn: Callable, class_token: bool):
        self.block_idx   = block_idx
        self.class_token = class_token
        self.function    = self._make_hook(hook_fn)

    def _make_hook(self, hook_fn: Callable):
        def _hook(module, module_input, module_output):
            feat = module_output  # [B, C, H, W]
            if self.class_token:
                # Global average pool: [B, C, H, W] → [B, C]
                pooled = feat.mean(dim=[-2, -1])
                hook_fn(pooled)
            else:
                # Spatial tokens: [B, C, H, W] → [B, H*W, C] → [B*H*W, C]
                B, C, H, W = feat.shape
                spatial = feat.permute(0, 2, 3, 1).reshape(B * H * W, C)
                hook_fn(spatial)
            return module_output  # always pass through unchanged
        return _hook

    def get_module(self, model):
        return model.vision_model.encoder.blocks[self.block_idx]


class HookedALIGNVision:
    """
    Thin wrapper around AlignModel that supports hook-based activation caching.
    Matches the interface expected by ALIGNActivationsStore.
    """

    def __init__(self, model: AlignModel, processor: AlignProcessor, device: str):
        self.model     = model.to(device)
        self.processor = processor
        self.device    = device

    def run_with_cache(
        self,
        block_indices: List[int],
        pixel_values: torch.Tensor,
        class_token: bool,
    ) -> Tuple[object, dict]:
        cache_dict = {}

        def save_fn(name, acts):
            cache_dict[name] = acts.detach()

        hooks_objs = []
        for idx in block_indices:
            fn = partial(save_fn, idx)
            h  = ALIGNHook(idx, fn, class_token)
            hooks_objs.append(h)

        with self._register_hooks(hooks_objs):
            with torch.no_grad():
                out = self.model.vision_model(pixel_values=pixel_values)

        return out, cache_dict

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def to(self, device):
        self.model = self.model.to(device)

    @contextmanager
    def _register_hooks(self, hooks: List[ALIGNHook]):
        handles = []
        try:
            for hook in hooks:
                module = hook.get_module(self.model)
                handle = module.register_forward_hook(hook.function)
                handles.append(handle)
            yield self.model
        finally:
            for h in handles:
                h.remove()


# ── Activations Store ──────────────────────────────────────────────────────────

class ALIGNActivationsStore:
    """
    Streams ImageNet images through ALIGN's vision encoder and provides
    activations for SAE training.  Drop-in replacement for DinoActivationsStore.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        device: str,
        seed: int,
        model: HookedALIGNVision,
        block_layer: int,   # signed int (e.g. -1 = last block)
        class_token: bool,
    ):
        self.device      = device
        self.model       = model
        self.batch_size  = batch_size
        self.class_token = class_token

        # Resolve negative block index
        self.block_idx = block_layer if block_layer >= 0 else N_BLOCKS + block_layer

        self.dataset      = dataset.shuffle(seed=seed)
        self.dataset_iter = iter(self.dataset)

    # ── Public interface ───────────────────────────────────────────────────────

    def get_next_batch(self) -> torch.Tensor:
        return self.get_batch_activations()

    def get_batch_activations(self) -> torch.Tensor:
        pv = self._get_batch_pixel_values()
        return self._extract_activations(pv)

    def get_batch_model_inputs(self, process_labels: bool = False) -> dict:
        return {"pixel_values": self._get_batch_pixel_values()}

    def initialize_b_dec(self, sae):
        sae.initialize_b_dec(self)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _get_batch_pixel_values(self) -> torch.Tensor:
        images = []
        for _ in range(self.batch_size):
            try:
                item = next(self.dataset_iter)
            except StopIteration:
                self.dataset_iter = iter(self.dataset)
                item = next(self.dataset_iter)
            img = item["image"]
            if hasattr(img, "mode") and img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)

        inputs = self.model.processor(images=images, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)

    def _extract_activations(self, pixel_values: torch.Tensor) -> torch.Tensor:
        _, cache = self.model.run_with_cache(
            [self.block_idx], pixel_values, class_token=self.class_token
        )
        return cache[self.block_idx]  # [B, C] or [B*H*W, C]


# ── Model Loading ──────────────────────────────────────────────────────────────

def load_align_model(device: str) -> HookedALIGNVision:
    print(f"[DEBUG] Loading ALIGN from HuggingFace: {MODEL_ID}")
    proc  = AlignProcessor.from_pretrained(MODEL_ID)
    model = AlignModel.from_pretrained(MODEL_ID)
    model.eval()
    total  = sum(p.numel() for p in model.parameters())
    vision = sum(p.numel() for p in model.vision_model.parameters())
    print(f"[DEBUG] ALIGN loaded. Total params: {total:,}  Vision params: {vision:,}")
    return HookedALIGNVision(model, proc, device=device)


# ── Verification Helpers ───────────────────────────────────────────────────────

def verify_activations(store: ALIGNActivationsStore, block_idx: int, n_samples: int = 2):
    print("\n" + "=" * 60)
    print("[DEBUG] ACTIVATION VERIFICATION")
    print("=" * 60)
    for i in range(n_samples):
        batch = store.get_next_batch()
        print(f"  Sample {i}: shape={batch.shape}, dtype={batch.dtype}, "
              f"mean={batch.float().mean():.4f}, std={batch.float().std():.4f}, "
              f"min={batch.float().min():.4f}, max={batch.float().max():.4f}")
        if batch.float().std() < 1e-8:
            print(f"  [WARN] Near-zero variance at block {block_idx}!")
        if torch.isnan(batch).any():
            print(f"  [FATAL] NaN in activations at block {block_idx}!")
            sys.exit(1)
    print("[DEBUG] Activation verification PASSED.\n")


def verify_model_loaded(model: HookedALIGNVision):
    print("\n" + "=" * 60)
    print("[DEBUG] MODEL WEIGHT VERIFICATION")
    print("=" * 60)
    inner = model.model
    stats = []
    for name, param in inner.named_parameters():
        if "weight" in name:
            stats.append((name, param.float().mean().item(), param.float().std().item()))
            if len(stats) >= 3:
                break
    for name, mean, std in stats:
        print(f"  {name}: mean={mean:.6f}, std={std:.6f}")
    total     = sum(p.numel() for p in inner.parameters())
    trainable = sum(p.numel() for p in inner.parameters() if p.requires_grad)
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}")
    print(f"  Device:           {next(inner.parameters()).device}")
    print("[DEBUG] Model verification complete.\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train SAE on ALIGN vision encoder activations from ImageNet"
    )

    # --- Architecture ---
    parser.add_argument(
        "--block_layers", type=int, nargs="+", default=[-1],
        help="AlignVisionBlock indices to train SAEs for. "
             "Negative indices count from the end. "
             "E.g. -1=last (blocks[54], 640ch), -5=blocks[50] (384ch). "
             "Pass multiple: --block_layers -3 -2 -1"
    )
    parser.add_argument(
        "--d_in", type=int, default=640,
        help="Channel dim of the hooked block. "
             "blocks[-1]=640, blocks[40..50]=384, blocks[30]=224, "
             "blocks[20]=160, blocks[10]=48, blocks[0]=32"
    )
    parser.add_argument(
        "--class_token", action="store_true", default=True,
        help="Global average pool block output → [B, d_in]  (default, CLS-like)"
    )
    parser.add_argument(
        "--no_class_token", dest="class_token", action="store_false",
        help="Use all 10×10 spatial positions → [B*100, d_in]  (patch-like)"
    )

    # --- Dataset ---
    parser.add_argument("--dataset", type=str, default="imagenet")

    # --- SAE ---
    # expansion=32 → d_sae=20,480.  CLIP/DINOv2 used 64→49,152 but that left 87% dead features;
    # 32 gives more gradient signal per feature while still providing a large dictionary.
    parser.add_argument("--expansion_factor",   type=int,   default=32)
    parser.add_argument("--b_dec_init_method",  type=str,   default="geometric_median")
    parser.add_argument("--gated_sae",          action="store_true", default=None)

    # --- Training ---
    parser.add_argument("--lr",                type=float, default=0.0004)
    # l1=0.00025: calibrated to match CLIP's l1/MSE ratio at initialisation.
    # ALIGN acts have L2 norm~8.6 vs CLIP's~6 → init MSE is 2×larger → l1 scales up ~3×.
    # Empirical formula: l1_align = l1_clip × (MSE_align/MSE_clip) × (d_sae_clip/d_sae_align)
    #                             = 0.00008 × (74/36) × (49152/20480) ≈ 0.00025
    # Monitor W&B 'l0' metric: target L0=30–80. Increase l1 if L0 > 100, decrease if L0 < 20.
    parser.add_argument("--l1_coefficient",        type=float, default=0.00025)
    parser.add_argument("--lr_scheduler_name",     type=str,   default="constantwithwarmup")
    # batch=32: ALIGN uses 346×346 images which fit on 24 GB; larger batch reduces gradient noise.
    parser.add_argument("--batch_size",            type=int,   default=32)
    # Longer warmup because EfficientNet GAP features have wider activation spread than ViT CLS.
    parser.add_argument("--lr_warm_up_steps",      type=int,   default=1000)
    # 10.24M tokens ≈ 500 tokens/feature × 20,480 features; ~8 passes over ImageNet (1.28M imgs).
    # At batch=32 on RTX 3090: ~320k steps ≈ 18 hours.
    parser.add_argument("--total_training_tokens", type=int,   default=10_240_000)
    # --max_steps overrides total_training_tokens when set (steps = tokens / batch_size).
    # E.g. --max_steps 50000 with batch_size=32 → 1,600,000 tokens.
    parser.add_argument("--max_steps",             type=int,   default=None,
                        help="Stop after this many optimiser steps regardless of token count. "
                             "Overrides --total_training_tokens.")
    parser.add_argument("--n_batches_in_store",    type=int,   default=20)
    parser.add_argument("--mse_cls_coefficient",   type=float, default=1.0)

    # --- Dead Neurons ---
    # ghost_grads: enabled by default to revive dead features (critical for EfficientNet features
    # which can be correlated and cause many features to stay at zero).
    parser.add_argument("--use_ghost_grads",          action="store_true", default=True)
    parser.add_argument("--no_ghost_grads",           dest="use_ghost_grads", action="store_false")
    parser.add_argument("--feature_sampling_method")
    parser.add_argument("--feature_sampling_window",  type=int,   default=1024)
    # dead_feature_window=1024: at target L0=50 out of 20,480, each feature fires every ~410 steps;
    # window must be >> 410 to reliably detect dead features before resampling.
    parser.add_argument("--dead_feature_window",      type=int,   default=1024)
    parser.add_argument("--dead_feature_threshold",   type=float, default=1e-6)

    # --- W&B ---
    parser.add_argument("--log_to_wandb",        action="store_true", default=None)
    parser.add_argument("--wandb_project",        type=str, default="align_imagenet_sae")
    parser.add_argument("--wandb_entity",         type=str, default="test")
    parser.add_argument("--wandb_log_frequency",  type=int, default=20)

    # --- Misc ---
    parser.add_argument("--seed",             type=int, default=42)
    parser.add_argument("--n_checkpoints",    type=int, default=1)
    parser.add_argument("--checkpoint_path",  type=str, default="out/checkpoints")
    parser.add_argument("--device",           type=str, default="cuda")
    parser.add_argument("--run_name",         type=str, default="align_sae_train")
    parser.add_argument("--skip_activation_verify", action="store_true")
    parser.add_argument("--dry_run",          action="store_true",
                        help="Validate pipeline without training.")

    args = parser.parse_args()

    # ── Environment Check ──────────────────────────────────────────────────────
    print("=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)
    print(f"  Python:      {sys.version}")
    print(f"  PyTorch:     {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"  CUDA:        {cuda_avail}"
          + (f" (count={torch.cuda.device_count()})" if cuda_avail else ""))
    if cuda_avail:
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:        {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  Device:      {args.device}")
    print(f"  Model:       {MODEL_ID}")
    print(f"  Block layers:{args.block_layers}  (resolved from {N_BLOCKS} total blocks)")
    print(f"  d_in:        {args.d_in}")
    print(f"  Mode:        {'global_pool (CLS-like)' if args.class_token else 'spatial (patch-like)'}")
    print()

    if args.device == "cuda" and not cuda_avail:
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    torch.manual_seed(args.seed)
    if cuda_avail:
        torch.cuda.manual_seed_all(args.seed)

    # ── Training Loop ──────────────────────────────────────────────────────────
    saes = {}
    print(f"[INFO] Training SAEs for block layers: {args.block_layers}")
    print()

    for layer_idx, block_layer in enumerate(args.block_layers):
        block_idx = block_layer if block_layer >= 0 else N_BLOCKS + block_layer
        print("\n" + "=" * 60)
        print(f"TRAINING SAE FOR BLOCK LAYER {block_layer}  (blocks[{block_idx}])"
              f"  ({layer_idx + 1}/{len(args.block_layers)})")
        print("=" * 60)

        t_start = time.time()

        # --max_steps overrides token budget: convert steps → tokens
        training_tokens = args.total_training_tokens
        if args.max_steps is not None:
            training_tokens = args.max_steps * args.batch_size
            print(f"[INFO] --max_steps={args.max_steps} → "
                  f"total_training_tokens={training_tokens:,}  "
                  f"(overrides --total_training_tokens)")

        # Config — reuse ViTSAERunnerConfig; irrelevant ViT fields are not used
        cfg = ViTSAERunnerConfig(
            class_token=args.class_token,
            image_width=IMAGE_SIZE,
            image_height=IMAGE_SIZE,
            model_name=MODEL_ID,
            module_name="resid",       # placeholder; hooks bypass this
            block_layer=block_layer,
            dataset_path=DATASET_INFO[args.dataset]["path"],
            image_key="image",
            label_key="label",
            use_cached_activations=None,
            cached_activations_path=None,
            d_in=args.d_in,
            expansion_factor=args.expansion_factor,
            b_dec_init_method=args.b_dec_init_method,
            gated_sae=args.gated_sae,
            lr=args.lr,
            l1_coefficient=args.l1_coefficient,
            lr_scheduler_name=args.lr_scheduler_name,
            batch_size=args.batch_size,
            lr_warm_up_steps=args.lr_warm_up_steps,
            total_training_tokens=training_tokens,
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
        print(f"[DEBUG] Config: d_in={cfg.d_in}, "
              f"expansion={cfg.expansion_factor}, "
              f"d_sae={cfg.d_in * cfg.expansion_factor}")

        # Dataset
        print("[DEBUG] Loading dataset...")
        dataset = load_dataset(**DATASET_INFO[args.dataset])
        if hasattr(dataset, "keys"):
            split = DATASET_INFO[args.dataset].get("split", "train")
            dataset = dataset[split]
        print(f"[DEBUG] Dataset: {args.dataset} ({len(dataset)} samples)")

        # SAE
        print("[DEBUG] Initializing SparseAutoencoder...")
        sae = SparseAutoencoder(cfg, args.device)
        print(f"[DEBUG] SAE params: {sum(p.numel() for p in sae.parameters()):,}")

        # ALIGN model
        print("[DEBUG] Loading ALIGN model...")
        align = load_align_model(args.device)
        align.eval()
        verify_model_loaded(align)

        # Activation store
        print("[DEBUG] Initializing ALIGNActivationsStore...")
        activation_store = ALIGNActivationsStore(
            dataset=dataset,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
            model=align,
            block_layer=block_layer,
            class_token=args.class_token,
        )
        print(f"[DEBUG] Store ready. block_idx={activation_store.block_idx}, "
              f"class_token={args.class_token}")

        if not args.skip_activation_verify:
            verify_activations(activation_store, block_idx)

        # Optimizer & Scheduler
        optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
        scheduler = get_scheduler(args.lr_scheduler_name, optimizer=optimizer)
        print(f"[DEBUG] Optimizer: Adam (lr={cfg.lr}), Scheduler: {args.lr_scheduler_name}")

        # Init decoder bias
        print("[DEBUG] Initializing SAE b_dec...")
        sae.initialize_b_dec(activation_store)
        print(f"[DEBUG] b_dec: mean={sae.b_dec.data.float().mean():.4f}, "
              f"std={sae.b_dec.data.float().std():.4f}")

        sae.train()

        if args.dry_run:
            print("\n[INFO] DRY RUN complete. Pipeline validated. Exiting.")
            print(f"[INFO] Time for block {block_layer}: {time.time()-t_start:.1f}s")
            continue

        # W&B
        if cfg.log_to_wandb:
            run_name = f"{args.run_name}_block{block_layer}"
            wandb.init(
                project=cfg.wandb_project,
                config=vars(args),
                name=run_name,
                reinit=True,
            )
            print(f"[DEBUG] W&B: {cfg.wandb_project}/{run_name}")

        # Train
        print(f"\n[INFO] Starting SAE training for block layer {block_layer}...")
        try:
            trainer = SAETrainer(
                sae, align, activation_store, cfg, optimizer, scheduler, args.device
            )
            trainer.fit()
        except Exception as e:
            print(f"[ERROR] Training failed for block {block_layer}: {e}")
            traceback.print_exc()
            if cfg.log_to_wandb:
                wandb.finish(exit_code=1)
            continue

        if cfg.log_to_wandb:
            wandb.finish()

        saes[block_layer] = sae
        elapsed = time.time() - t_start
        print(f"[INFO] Block {block_layer} done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Blocks trained: {list(saes.keys())}")
    print(f"  Checkpoints at: {args.checkpoint_path}")
    if args.dry_run:
        print("  (DRY RUN — no actual training performed)")
    print("=" * 60)


if __name__ == "__main__":
    main()
