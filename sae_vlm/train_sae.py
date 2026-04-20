#!/usr/bin/env python3
"""
Unified SAE training entry point.

Trains a Sparse Autoencoder on activations from any supported backbone
(CLIP, DINOv2, ALIGN) on any dataset registered in configs/datasets/registry.yaml.

All hyperparameters can be set via CLI args or overridden with --sae_config
(a YAML file from configs/sae/).  Per-model defaults live in configs/sae/.

Usage:
    # DINOv2-base on ImageNet (uses configs/sae/default.yaml):
    python train_sae.py --model dinov2_base --dataset imagenet --log_to_wandb

    # ALIGN on ImageNet with calibrated hyperparameters:
    python train_sae.py --model align_base --dataset imagenet \\
        --sae_config configs/sae/align.yaml --log_to_wandb

    # Quick sanity check:
    python train_sae.py --model clip_vit_b16 --dataset imagenet \\
        --max_steps 100 --dry_run

    # Multi-layer (one SAE per layer):
    python train_sae.py --model dinov2_base --dataset imagenet \\
        --layers -3 -2 -1 --log_to_wandb
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

_root = str(Path(__file__).resolve().parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch
import yaml

from src.models.registry import get_backbone, load_model_cfg
from src.data.dataset_registry import get_dataset, get_label_key, load_registry
from src.sae_training.config import ViTSAERunnerConfig
from src.sae_training.sparse_autoencoder import SparseAutoencoder
from src.sae_training.sae_trainer import SAETrainer
from src.sae_training.utils import get_scheduler

_SAE_CFG_DIR = Path(__file__).parent / "configs" / "sae"
_DEFAULT_SAE_CFG = _SAE_CFG_DIR / "default.yaml"


# ── Activation store (backbone-agnostic) ──────────────────────────────────────

class ActivationsStore:
    """Streams images through any backbone and caches activations for SAE training."""

    def __init__(self, backbone, dataset, layer: int, batch_size: int,
                 device: str, seed: int, use_cls: bool = True):
        self.backbone   = backbone
        self.layer      = layer
        self.batch_size = batch_size
        self.device     = device
        self.use_cls    = use_cls
        self.dataset    = dataset.shuffle(seed=seed)
        self._iter      = iter(self.dataset)

    def get_next_batch(self) -> torch.Tensor:
        images = []
        for _ in range(self.batch_size):
            try:
                item = next(self._iter)
            except StopIteration:
                self._iter = iter(self.dataset)
                item = next(self._iter)
            img = item["image"]
            if hasattr(img, "mode") and img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)
        return self.backbone.get_activations(images, layer=self.layer, use_cls=self.use_cls)

    def get_batch_activations(self) -> torch.Tensor:
        return self.get_next_batch()

    def initialize_b_dec(self, sae):
        sae.initialize_b_dec(self)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_sae_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_cfgs(base: dict, override: dict) -> dict:
    merged = dict(base)
    merged.update({k: v for k, v in override.items() if v is not None})
    return merged


def build_vit_runner_cfg(model_cfg: dict, sae_hparams: dict, dataset_path: str,
                          layer: int, args) -> ViTSAERunnerConfig:
    d_in = model_cfg["d_model"]
    return ViTSAERunnerConfig(
        class_token     = True,
        image_width     = model_cfg.get("image_size", 224),
        image_height    = model_cfg.get("image_size", 224),
        model_name      = model_cfg["model_id"],
        module_name     = "resid",
        block_layer     = layer,
        dataset_path    = dataset_path,
        image_key       = "image",
        label_key       = "label",
        d_in            = d_in,
        expansion_factor= sae_hparams["expansion_factor"],
        b_dec_init_method = sae_hparams.get("b_dec_init_method", "geometric_median"),
        lr              = sae_hparams.get("lr", 0.0004),
        l1_coefficient  = sae_hparams["l1_coefficient"],
        lr_scheduler_name = sae_hparams.get("lr_scheduler", "constantwithwarmup"),
        batch_size      = sae_hparams["batch_size"],
        lr_warm_up_steps= sae_hparams.get("lr_warm_up_steps", 500),
        total_training_tokens = sae_hparams["total_training_tokens"],
        n_batches_in_store    = sae_hparams.get("n_batches_in_store", 15),
        use_ghost_grads = sae_hparams.get("use_ghost_grads", False),
        feature_sampling_window = sae_hparams.get("feature_sampling_window", 64),
        dead_feature_window     = sae_hparams.get("dead_feature_window", 64),
        dead_feature_threshold  = sae_hparams.get("dead_feature_threshold", 1e-6),
        log_to_wandb    = args.log_to_wandb,
        wandb_project   = args.wandb_project,
        wandb_entity    = args.wandb_entity,
        wandb_log_frequency = args.wandb_log_frequency,
        device          = args.device,
        seed            = args.seed,
        n_checkpoints   = sae_hparams.get("n_checkpoints", 1),
        checkpoint_path = args.checkpoint_path,
        dtype           = torch.float32,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train SAE on any backbone × dataset")

    parser.add_argument("--model",      required=True,
                        help="Model config name, e.g. clip_vit_b16 / dinov2_base / align_base")
    parser.add_argument("--dataset",    required=True,
                        help="Dataset name from configs/datasets/registry.yaml")
    parser.add_argument("--layers",     type=int, nargs="+", default=[-1],
                        help="Backbone layers to train SAEs for (default: -1 = last)")
    parser.add_argument("--sae_config", default=None,
                        help="YAML file with SAE hyperparameter overrides (e.g. configs/sae/align.yaml)")

    # SAE hyperparameter overrides (all optional; YAML defaults used otherwise)
    parser.add_argument("--expansion_factor",    type=int)
    parser.add_argument("--l1_coefficient",      type=float)
    parser.add_argument("--lr",                  type=float)
    parser.add_argument("--batch_size",          type=int)
    parser.add_argument("--lr_warm_up_steps",    type=int)
    parser.add_argument("--total_training_tokens", type=int)
    parser.add_argument("--max_steps",           type=int, default=None,
                        help="Shorthand: sets total_training_tokens = max_steps * batch_size")
    parser.add_argument("--use_ghost_grads",     action="store_true", default=None)
    parser.add_argument("--no_ghost_grads",      dest="use_ghost_grads", action="store_false")
    parser.add_argument("--dead_feature_window", type=int)

    # W&B
    parser.add_argument("--log_to_wandb",         action="store_true", default=False)
    parser.add_argument("--wandb_project",         default="sae_vlm")
    parser.add_argument("--wandb_entity",          default=None)
    parser.add_argument("--wandb_log_frequency",   type=int, default=20)

    # Misc
    parser.add_argument("--device",          default="cuda")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--max_images",      type=int, default=None,
                        help="Truncate dataset to N images (for quick tests)")
    parser.add_argument("--checkpoint_path", default="out/checkpoints")
    parser.add_argument("--dry_run",         action="store_true",
                        help="Load everything but skip training")
    parser.add_argument("--skip_verify",     action="store_true")

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU.")
        device = "cpu"
    args.device = device

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Load configs ──────────────────────────────────────────────────────────
    model_cfg  = load_model_cfg(args.model)
    base_hps   = load_sae_cfg(_DEFAULT_SAE_CFG)
    sae_hparams = dict(base_hps)

    if args.sae_config:
        sae_hparams.update(load_sae_cfg(args.sae_config))

    # CLI overrides
    for key in ("expansion_factor", "l1_coefficient", "lr", "batch_size",
                "lr_warm_up_steps", "total_training_tokens", "dead_feature_window"):
        val = getattr(args, key, None)
        if val is not None:
            sae_hparams[key] = val

    if args.use_ghost_grads is not None:
        sae_hparams["use_ghost_grads"] = args.use_ghost_grads

    if args.max_steps is not None:
        sae_hparams["total_training_tokens"] = args.max_steps * sae_hparams["batch_size"]

    # ── Load dataset ──────────────────────────────────────────────────────────
    registry = load_registry()
    print(f"[INFO] Loading dataset '{args.dataset}'...")
    dataset  = get_dataset(args.dataset, registry=registry, max_samples=args.max_images)
    print(f"[INFO] Dataset size: {len(dataset):,}")

    dataset_hf_path = registry[args.dataset].get("hf_path", args.dataset)

    # ── Load backbone ─────────────────────────────────────────────────────────
    print(f"[INFO] Loading backbone '{args.model}'...")
    backbone = get_backbone(args.model, device=device).load()
    print(f"[INFO] Backbone loaded: {model_cfg['model_id']}")

    # ── Train one SAE per layer ───────────────────────────────────────────────
    for layer in args.layers:
        print(f"\n{'='*60}")
        print(f"LAYER {layer}  |  model={args.model}  |  dataset={args.dataset}")
        print(f"{'='*60}")

        t0 = time.time()

        cfg = build_vit_runner_cfg(model_cfg, sae_hparams, dataset_hf_path, layer, args)
        sae = SparseAutoencoder(cfg, device)

        store = ActivationsStore(
            backbone=backbone, dataset=dataset, layer=layer,
            batch_size=sae_hparams["batch_size"], device=device, seed=args.seed,
        )

        if not args.skip_verify:
            b = store.get_next_batch()
            print(f"  Activation check: shape={b.shape}, mean={b.mean():.4f}, std={b.std():.4f}")
            assert not torch.isnan(b).any(), "NaN in activations!"

        sae.initialize_b_dec(store)
        sae.train()

        if args.dry_run:
            print(f"  [dry_run] Pipeline OK for layer {layer}. Skipping training.")
            continue

        if cfg.log_to_wandb:
            import wandb
            wandb.init(project=cfg.wandb_project, config={**vars(args), **sae_hparams},
                       name=f"{args.model}_{args.dataset}_layer{layer}", reinit=True)

        optimizer      = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
        training_steps = cfg.total_training_tokens // cfg.batch_size
        scheduler      = get_scheduler(
            cfg.lr_scheduler_name, optimizer=optimizer,
            warm_up_steps=cfg.lr_warm_up_steps,
            training_steps=training_steps,
            lr_end=0.0,
        )

        try:
            trainer = SAETrainer(sae, None, store, cfg, optimizer, scheduler, device)
            trainer.fit()
        except Exception as e:
            print(f"[ERROR] Training failed for layer {layer}: {e}")
            traceback.print_exc()
            if cfg.log_to_wandb:
                import wandb
                wandb.finish(exit_code=1)
            continue

        if cfg.log_to_wandb:
            import wandb
            wandb.finish()

        elapsed = time.time() - t0
        print(f"[INFO] Layer {layer} done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    print(f"\n[INFO] Checkpoints at: {args.checkpoint_path}")


if __name__ == "__main__":
    main()
