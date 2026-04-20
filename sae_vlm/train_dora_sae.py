#!/usr/bin/env python3
"""
Train SAEs on DoRA fine-tuned CLIP models.

Trains one SAE per (dataset × SAE-type) combination for 50k steps each.
Supports three SAE architectures:
  - standard   : ReLU + L1 penalty  (baseline)
  - jumprelu   : JumpReLU threshold gate + L0 proxy loss
  - topk       : Top-K activation + aux-k dead-neuron loss

DoRA weights are loaded from dora_finetune.py checkpoints and merged
directly into CLIP before activation extraction.

Usage:
    # Single dataset, all three SAE types:
    python train_dora_sae.py --dataset eurosat --sae_types standard jumprelu topk

    # All datasets, single SAE type:
    python train_dora_sae.py --all_datasets --sae_types topk

    # Full sweep (all datasets × all types):
    python train_dora_sae.py --all_datasets --sae_types standard jumprelu topk

    # Quick sanity check:
    python train_dora_sae.py --dataset eurosat --sae_types standard --max_steps 100 --dry_run
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

from src.sae_training.config import ViTSAERunnerConfig
from src.sae_training.sparse_autoencoder import SparseAutoencoder
from src.sae_training.sae_trainer import SAETrainer
from src.sae_training.utils import get_scheduler

_DEFAULT_SAE_CFG = Path(__file__).parent / "configs" / "sae" / "default.yaml"

# ---------------------------------------------------------------------------
# Dataset → DoRA checkpoint mapping
# ---------------------------------------------------------------------------

DATASET_DORA_MAP = {
    "eurosat":     "vitb16/eurosat/16shots/seed1/dora_weights.pt",
    "dtd":         "vitb16/dtd/16shots/seed1/dora_weights.pt",
    "fgvc":        "vitb16/fgvc/16shots/seed1/dora_weights.pt",
    "oxford_pets": "vitb16/oxford_pets/16shots/seed1/dora_weights.pt",
    "ucf101":      "vitb16/ucf101/16shots/seed1/dora_weights.pt",
    "caltech101":  "vitb16/caltech101/16shots/seed1/dora_weights.pt",
    "cub2002011":  "vitb16/cub2002011/16shots/seed1/dora_weights.pt",
    "medmnist":    "vitb16/medmnist/16shots/seed1/dora_weights.pt",
}

# HuggingFace image datasets used as activation source.
# Using imagenet-1k as the activation distribution (the model was fine-tuned on
# domain-specific data, but we train the SAE on natural image activations so
# features are comparable across datasets).
DATASET_HF_PATH = "evanarlian/imagenet_1k_resized_256"

CLIP_MODEL_ID  = "openai/clip-vit-base-patch16"
BACKBONE_NAME  = "ViT-B/16"
D_MODEL        = 768
N_LAYERS       = 12
STEPS_PER_RUN  = 50_000


# ---------------------------------------------------------------------------
# Activation store
# ---------------------------------------------------------------------------

class ActivationsStore:
    """Streams images through DoRA-CLIP and caches activations for SAE training."""

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


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_cfg(
    dataset_name: str,
    sae_type: str,
    sae_hparams: dict,
    layer: int,
    args,
) -> ViTSAERunnerConfig:
    batch_size  = sae_hparams["batch_size"]
    total_toks  = STEPS_PER_RUN * batch_size

    # "gated" maps to the existing gated_sae flag; others use the sae_type field
    is_gated  = (sae_type == "gated")
    arch_type = "standard" if is_gated else sae_type

    return ViTSAERunnerConfig(
        class_token           = True,
        image_width           = 224,
        image_height          = 224,
        model_name            = f"dora_clip_{dataset_name}",
        module_name           = "resid",
        block_layer           = layer,
        dataset_path          = DATASET_HF_PATH,
        image_key             = "image",
        label_key             = "label",
        d_in                  = D_MODEL,
        expansion_factor      = sae_hparams["expansion_factor"],
        b_dec_init_method     = sae_hparams.get("b_dec_init_method", "geometric_median"),
        lr                    = sae_hparams.get("lr", 4e-4),
        l1_coefficient        = sae_hparams["l1_coefficient"],
        lr_scheduler_name     = sae_hparams.get("lr_scheduler", "constantwithwarmup"),
        batch_size            = batch_size,
        lr_warm_up_steps      = sae_hparams.get("lr_warm_up_steps", 500),
        total_training_tokens = total_toks,
        n_batches_in_store    = sae_hparams.get("n_batches_in_store", 15),
        use_ghost_grads       = sae_hparams.get("use_ghost_grads", False),
        feature_sampling_window = sae_hparams.get("feature_sampling_window", 64),
        dead_feature_window   = sae_hparams.get("dead_feature_window", 64),
        dead_feature_threshold = sae_hparams.get("dead_feature_threshold", 1e-6),
        log_to_wandb          = args.log_to_wandb,
        wandb_project         = args.wandb_project,
        wandb_entity          = getattr(args, "wandb_entity", None),
        wandb_log_frequency   = args.wandb_log_frequency,
        device                = args.device,
        seed                  = args.seed,
        n_checkpoints         = sae_hparams.get("n_checkpoints", 1),
        checkpoint_path       = args.checkpoint_path,
        dtype                 = torch.float32,
        gated_sae             = is_gated,
        sae_type              = arch_type,
        topk_k                = args.topk_k,
        jumprelu_bandwidth    = args.jumprelu_bandwidth,
    )


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def train_one(dataset_name: str, sae_type: str, layer: int,
              backbone, hf_dataset, sae_hparams: dict, args) -> None:
    device = args.device
    print(f"\n{'='*70}")
    print(f"  dataset={dataset_name}  sae_type={sae_type}  layer={layer}")
    print(f"{'='*70}")

    cfg = build_cfg(dataset_name, sae_type, sae_hparams, layer, args)
    sae = SparseAutoencoder(cfg, device)

    store = ActivationsStore(
        backbone=backbone, dataset=hf_dataset, layer=layer,
        batch_size=sae_hparams["batch_size"], device=device, seed=args.seed,
    )

    if not args.skip_verify:
        b = store.get_next_batch()
        print(f"  Activation check: shape={b.shape}, mean={b.mean():.4f}, std={b.std():.4f}")
        assert not torch.isnan(b).any(), "NaN in activations!"

    sae.initialize_b_dec(store)
    sae.train()

    if args.dry_run:
        print(f"  [dry_run] Pipeline OK — skipping training.")
        return

    if cfg.log_to_wandb:
        import wandb
        wandb.init(
            project=cfg.wandb_project,
            config={
                "dataset": dataset_name, "sae_type": sae_type, "layer": layer,
                "steps": STEPS_PER_RUN, **sae_hparams,
            },
            name=f"dora_{dataset_name}_{sae_type}_layer{layer}",
            reinit=True,
        )

    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    training_steps = cfg.total_training_tokens // cfg.batch_size
    scheduler = get_scheduler(
        cfg.lr_scheduler_name, optimizer=optimizer,
        warm_up_steps=cfg.lr_warm_up_steps,
        training_steps=training_steps,
        lr_end=0.0,
    )

    t0 = time.time()
    try:
        trainer = SAETrainer(sae, None, store, cfg, optimizer, scheduler, device)
        trainer.fit()
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        if cfg.log_to_wandb:
            import wandb; wandb.finish(exit_code=1)
        return

    if cfg.log_to_wandb:
        import wandb; wandb.finish()

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train SAEs on DoRA fine-tuned CLIP")

    # Dataset selection
    parser.add_argument("--dataset",      default=None,
                        choices=list(DATASET_DORA_MAP.keys()),
                        help="Single dataset to train on")
    parser.add_argument("--all_datasets", action="store_true",
                        help="Train on all 8 DoRA datasets sequentially")

    # SAE types
    parser.add_argument("--sae_types", nargs="+",
                        default=["standard", "jumprelu", "topk"],
                        choices=["standard", "gated", "jumprelu", "topk"],
                        help="SAE architecture(s) to train")

    # Architecture hyperparams
    parser.add_argument("--topk_k",            type=int,   default=64,
                        help="Number of active features for Top-K SAE")
    parser.add_argument("--jumprelu_bandwidth", type=float, default=0.001,
                        help="STE kernel bandwidth for JumpReLU SAE")

    # Paths
    parser.add_argument("--dora_weights_dir", default="../dora_weights",
                        help="Root directory of DoRA checkpoints (contains vitb16/...)")
    parser.add_argument("--checkpoint_path",  default="out/checkpoints/dora")
    parser.add_argument("--sae_config",       default=None,
                        help="YAML with SAE hyperparameter overrides")

    # Training
    parser.add_argument("--layers",  type=int, nargs="+", default=[-1],
                        help="Vision encoder layers to train SAEs on (default: last)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override total steps (default: 50k)")
    parser.add_argument("--expansion_factor", type=int,   default=None)
    parser.add_argument("--l1_coefficient",   type=float, default=None)
    parser.add_argument("--lr",               type=float, default=None)
    parser.add_argument("--batch_size",       type=int,   default=None)

    # W&B
    parser.add_argument("--log_to_wandb",       action="store_true", default=False)
    parser.add_argument("--wandb_project",       default="sae_dora_clip")
    parser.add_argument("--wandb_entity",        default=None)
    parser.add_argument("--wandb_log_frequency", type=int, default=20)

    # Misc
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--max_images",   type=int, default=None)
    parser.add_argument("--dry_run",      action="store_true")
    parser.add_argument("--skip_verify",  action="store_true")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU.")
        args.device = "cpu"

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Determine dataset list
    if args.all_datasets:
        datasets_to_run = list(DATASET_DORA_MAP.keys())
    elif args.dataset:
        datasets_to_run = [args.dataset]
    else:
        parser.error("Specify --dataset <name> or --all_datasets")

    # Load SAE hyperparameters
    with open(_DEFAULT_SAE_CFG) as f:
        sae_hparams = yaml.safe_load(f)
    if args.sae_config:
        with open(args.sae_config) as f:
            sae_hparams.update(yaml.safe_load(f))

    for key in ("expansion_factor", "l1_coefficient", "lr", "batch_size"):
        val = getattr(args, key, None)
        if val is not None:
            sae_hparams[key] = val

    if args.max_steps is not None:
        global STEPS_PER_RUN
        STEPS_PER_RUN = args.max_steps

    # Load HuggingFace image dataset (shared across all runs)
    from datasets import load_dataset
    print(f"[INFO] Loading image dataset '{DATASET_HF_PATH}'...")
    hf_dataset = load_dataset(DATASET_HF_PATH, split="train", trust_remote_code=True)
    if args.max_images:
        hf_dataset = hf_dataset.select(range(min(args.max_images, len(hf_dataset))))
    print(f"[INFO] Dataset size: {len(hf_dataset):,}")

    # Sweep: dataset × sae_type × layer
    for dataset_name in datasets_to_run:
        dora_ckpt_rel = DATASET_DORA_MAP[dataset_name]
        dora_ckpt = str(Path(args.dora_weights_dir) / dora_ckpt_rel)

        if not Path(dora_ckpt).exists():
            print(f"[WARN] DoRA checkpoint not found: {dora_ckpt} — skipping {dataset_name}")
            continue

        # Load DoRA-CLIP backbone for this dataset
        print(f"\n[INFO] Loading DoRA-CLIP for dataset '{dataset_name}'...")
        from src.models.dora_clip_backbone import DoRACLIPBackbone
        model_cfg = {
            "model_id":      CLIP_MODEL_ID,
            "backbone_name": BACKBONE_NAME,
            "dora_ckpt":     dora_ckpt,
            "d_model":       D_MODEL,
            "n_layers":      N_LAYERS,
            "image_size":    224,
        }
        backbone = DoRACLIPBackbone(model_cfg=model_cfg, device=args.device).load()

        for sae_type in args.sae_types:
            for layer in args.layers:
                try:
                    train_one(
                        dataset_name=dataset_name,
                        sae_type=sae_type,
                        layer=layer,
                        backbone=backbone,
                        hf_dataset=hf_dataset,
                        sae_hparams=sae_hparams,
                        args=args,
                    )
                except Exception as e:
                    print(f"[ERROR] {dataset_name}/{sae_type}/layer{layer}: {e}")
                    traceback.print_exc()

        # Free GPU memory between datasets
        del backbone
        torch.cuda.empty_cache()

    print(f"\n[INFO] All runs complete. Checkpoints at: {args.checkpoint_path}")


if __name__ == "__main__":
    main()
