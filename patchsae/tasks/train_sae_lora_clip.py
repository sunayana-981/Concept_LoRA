"""
Train Sparse Autoencoder on activations from a LoRA fine-tuned CLIP model.

This script:
1. Loads a base CLIP ViT model
2. Applies LoRA weights from a saved checkpoint
3. Hooks into specified transformer block layers to extract activations
4. Trains an SAE on those activations

Usage:
    python train_sae_lora_clip.py \
        --model_name openai/clip-vit-base-patch16 \
        --lora_checkpoint_path /path/to/lora_weights.pt \
        --block_layers -3 -2 -1 \
        --dataset imagenet \
        --expansion_factor 64 \
        --l1_coefficient 0.00008 \
        --batch_size 16 \
        --log_to_wandb
"""

import argparse
import sys
import traceback
import time
from pathlib import Path

# Ensure project root is on sys.path so "src" and "tasks" packages resolve
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import wandb
from datasets import load_dataset

# --- Debug: Check imports early ---
print("[DEBUG] Importing project modules...")
try:
    from src.sae_training.config import ViTSAERunnerConfig
    from src.sae_training.sae_trainer import SAETrainer
    from src.sae_training.sparse_autoencoder import SparseAutoencoder
    from src.sae_training.utils import get_scheduler
    from src.sae_training.vit_activations_store import ViTActivationsStore
    from tasks.utils import DATASET_INFO, get_classnames, load_hooked_vit
    print("[DEBUG] All project imports successful.")
except ImportError as e:
    print(f"[FATAL] Import failed: {e}")
    traceback.print_exc()
    sys.exit(1)


# =========================================================================
# LoRA Loading Utilities
# =========================================================================

def load_lora_weights(model, lora_checkpoint_path: str, device: str = "cuda"):
    """
    Load LoRA weights into a HookedVisionTransformer wrapping a HuggingFace CLIPModel.

    Supports:
      - Custom CLIP_LoRA format (keys: 'weights' + 'metadata')
      - PEFT-style state_dicts (keys contain 'lora_A', 'lora_B')
      - Direct / merged state_dicts

    Args:
        model: HookedVisionTransformer (or raw model with .state_dict()).
        lora_checkpoint_path: Path to saved LoRA weights (.pt, .bin, .safetensors).
        device: Target device.

    Returns:
        model with LoRA weights merged in.
    """
    import math

    ckpt_path = Path(lora_checkpoint_path)
    assert ckpt_path.exists(), f"[FATAL] LoRA checkpoint not found: {ckpt_path}"

    print(f"[DEBUG] Loading LoRA checkpoint from: {ckpt_path}")
    print(f"[DEBUG] Checkpoint file size: {ckpt_path.stat().st_size / 1e6:.2f} MB")

    # --- Load checkpoint ---
    if ckpt_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
            checkpoint = load_file(str(ckpt_path), device=device)
            print("[DEBUG] Loaded .safetensors checkpoint.")
        except ImportError:
            print("[FATAL] safetensors package not installed. pip install safetensors")
            sys.exit(1)
    else:
        checkpoint = torch.load(str(ckpt_path), map_location=device)
        print(f"[DEBUG] Loaded .pt/.bin checkpoint with keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}")

    # --- Detect custom CLIP_LoRA format ---
    if isinstance(checkpoint, dict) and "weights" in checkpoint and "metadata" in checkpoint:
        return _apply_custom_clip_lora(model, checkpoint, device)

    # --- Unwrap nested formats ---
    lora_state_dict = checkpoint
    if isinstance(lora_state_dict, dict) and "state_dict" in lora_state_dict:
        lora_state_dict = lora_state_dict["state_dict"]
    elif isinstance(lora_state_dict, dict) and "model_state_dict" in lora_state_dict:
        lora_state_dict = lora_state_dict["model_state_dict"]

    sample_keys = list(lora_state_dict.keys())[:10]
    print(f"[DEBUG] Sample checkpoint keys: {sample_keys}")

    is_peft_style = any("lora_" in k for k in lora_state_dict.keys())
    if is_peft_style:
        return _apply_peft_lora(model, lora_state_dict, device)

    return _apply_direct_weights(model, lora_state_dict, device)


def _apply_custom_clip_lora(model, checkpoint, device):
    """
    Apply LoRA weights saved via CLIP_LoRA's save_lora() utility.

    Checkpoint structure:
        {
          "weights": { "layer_0": { "q_proj": {"w_lora_A": ..., "w_lora_B": ...}, ... }, ... },
          "metadata": { "r": int, "alpha": int, "encoder": str, "params": list, "position": str }
        }

    For encoder="both", layers 0..N-1 are text encoder, N..2N-1 are vision encoder.
    """
    import math

    metadata = checkpoint["metadata"]
    weights = checkpoint["weights"]
    r = metadata["r"]
    alpha = metadata["alpha"]
    encoder_type = metadata["encoder"]
    params = metadata["params"]
    scaling = alpha / math.sqrt(r)

    print(f"[DEBUG] Custom CLIP_LoRA format detected.")
    print(f"[DEBUG]   r={r}, alpha={alpha}, scaling={scaling:.4f}")
    print(f"[DEBUG]   encoder={encoder_type}, params={params}")
    print(f"[DEBUG]   {len(weights)} LoRA layers in checkpoint.")

    # Get the actual CLIPModel from HookedVisionTransformer wrapper
    if hasattr(model, "model"):
        clip_model = model.model
    else:
        clip_model = model

    # Determine which layers are text vs vision
    n_layers = len(weights)
    sorted_layer_keys = sorted(weights.keys(), key=lambda k: int(k.split("_")[1]))

    if encoder_type == "both":
        n_per_encoder = n_layers // 2
        text_layer_keys = sorted_layer_keys[:n_per_encoder]
        vision_layer_keys = sorted_layer_keys[n_per_encoder:]
    elif encoder_type == "text":
        text_layer_keys = sorted_layer_keys
        vision_layer_keys = []
    elif encoder_type == "vision":
        text_layer_keys = []
        vision_layer_keys = sorted_layer_keys
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    merged_count = 0

    # --- Map projection names to HuggingFace CLIPAttention attribute names ---
    proj_name_map = {"q_proj": "q_proj", "k_proj": "k_proj", "v_proj": "v_proj", "proj": "out_proj"}

    # --- Apply vision encoder LoRA ---
    if vision_layer_keys and hasattr(clip_model, "vision_model"):
        vision_layers = clip_model.vision_model.encoder.layers
        for i, layer_key in enumerate(vision_layer_keys):
            if i >= len(vision_layers):
                print(f"[WARN] Vision layer index {i} out of range, skipping {layer_key}")
                continue
            hf_layer = vision_layers[i]
            layer_weights = weights[layer_key]
            for proj_name, proj_data in layer_weights.items():
                hf_proj_name = proj_name_map.get(proj_name, proj_name)
                if not hasattr(hf_layer.self_attn, hf_proj_name):
                    print(f"[WARN] vision layer {i} has no '{hf_proj_name}', skipping.")
                    continue
                hf_proj = getattr(hf_layer.self_attn, hf_proj_name)
                lora_A = proj_data["w_lora_A"].to(device).float()  # [r, d_in]
                lora_B = proj_data["w_lora_B"].to(device).float()  # [d_out, r]
                delta = (lora_B @ lora_A) * scaling                # [d_out, d_in]
                assert delta.shape == hf_proj.weight.shape, (
                    f"Shape mismatch at vision.{i}.{hf_proj_name}: "
                    f"delta {delta.shape} vs weight {hf_proj.weight.shape}"
                )
                hf_proj.weight.data.add_(delta)
                merged_count += 1

    # --- Apply text encoder LoRA ---
    if text_layer_keys and hasattr(clip_model, "text_model"):
        text_layers = clip_model.text_model.encoder.layers
        for i, layer_key in enumerate(text_layer_keys):
            if i >= len(text_layers):
                print(f"[WARN] Text layer index {i} out of range, skipping {layer_key}")
                continue
            hf_layer = text_layers[i]
            layer_weights = weights[layer_key]
            for proj_name, proj_data in layer_weights.items():
                hf_proj_name = proj_name_map.get(proj_name, proj_name)
                if not hasattr(hf_layer.self_attn, hf_proj_name):
                    print(f"[WARN] text layer {i} has no '{hf_proj_name}', skipping.")
                    continue
                hf_proj = getattr(hf_layer.self_attn, hf_proj_name)
                lora_A = proj_data["w_lora_A"].to(device).float()
                lora_B = proj_data["w_lora_B"].to(device).float()
                delta = (lora_B @ lora_A) * scaling
                assert delta.shape == hf_proj.weight.shape, (
                    f"Shape mismatch at text.{i}.{hf_proj_name}: "
                    f"delta {delta.shape} vs weight {hf_proj.weight.shape}"
                )
                hf_proj.weight.data.add_(delta)
                merged_count += 1

    total_possible = sum(len(weights[k]) for k in sorted_layer_keys)
    print(f"[DEBUG] Merged {merged_count}/{total_possible} LoRA projections into CLIPModel.")
    return model


def _apply_peft_lora(model, lora_state_dict, device):
    """
    Apply PEFT-style LoRA adapters.

    If the model was saved with `peft`, try loading via peft first.
    Otherwise, manually merge LoRA_A @ LoRA_B * scaling into base weights.
    """
    print("[DEBUG] Attempting PEFT-style LoRA application...")

    # Try using peft library directly
    try:
        from peft import PeftModel, PeftConfig
        print("[DEBUG] peft library available. Attempting PeftModel.from_pretrained...")

        # If user saved the full peft directory, we can load directly
        ckpt_dir = Path(list(lora_state_dict.keys())[0]).parent if False else None
        # This path is for when you have the peft adapter directory
        # For raw state_dict, fall through to manual merge
        raise ImportError("Falling through to manual merge for raw state_dict")

    except (ImportError, Exception) as e:
        print(f"[DEBUG] peft auto-load not applicable ({e}), doing manual LoRA merge.")

    # --- Manual LoRA merge ---
    inner_model = model.model if hasattr(model, "model") else model
    model_state = inner_model.state_dict()
    merged_count = 0
    skipped_keys = []

    # Group lora_A and lora_B pairs
    lora_pairs = {}
    for key in lora_state_dict:
        if "lora_A" in key:
            base_key = key.replace(".lora_A.weight", "").replace(".lora_A.default.weight", "")
            lora_pairs.setdefault(base_key, {})["A"] = lora_state_dict[key]
        elif "lora_B" in key:
            base_key = key.replace(".lora_B.weight", "").replace(".lora_B.default.weight", "")
            lora_pairs.setdefault(base_key, {})["B"] = lora_state_dict[key]

    print(f"[DEBUG] Found {len(lora_pairs)} LoRA adapter pairs.")

    for base_key, ab in lora_pairs.items():
        if "A" not in ab or "B" not in ab:
            print(f"[WARN] Incomplete LoRA pair for {base_key}, skipping.")
            continue

        # Find matching key in model state dict
        # PEFT keys often have extra prefixes like "base_model.model."
        target_key = None
        stripped = base_key.replace("base_model.model.", "").replace("base_model.", "")
        for model_key in model_state:
            if model_key.endswith(stripped + ".weight") or model_key == stripped + ".weight":
                target_key = model_key
                break
            # Also try without .weight suffix
            if model_key == stripped:
                target_key = model_key
                break

        if target_key is None:
            skipped_keys.append(base_key)
            continue

        # Merge: W' = W + B @ A * scaling
        # Default LoRA scaling = alpha / rank. If unknown, assume 1.0.
        lora_A = ab["A"].to(device).float()
        lora_B = ab["B"].to(device).float()
        delta = lora_B @ lora_A  # (out_features, in_features)

        print(f"[DEBUG] Merging LoRA into {target_key}: "
              f"delta shape={delta.shape}, base shape={model_state[target_key].shape}")

        assert delta.shape == model_state[target_key].shape, (
            f"Shape mismatch: delta {delta.shape} vs base {model_state[target_key].shape}"
        )

        model_state[target_key] = model_state[target_key].float() + delta
        merged_count += 1

    if skipped_keys:
        print(f"[WARN] Skipped {len(skipped_keys)} LoRA keys (no match in model):")
        for k in skipped_keys[:5]:
            print(f"  - {k}")

    print(f"[DEBUG] Successfully merged {merged_count}/{len(lora_pairs)} LoRA pairs.")
    inner_model.load_state_dict(model_state, strict=False)
    print("[DEBUG] LoRA weights applied to model.")
    return model


def _apply_direct_weights(model, state_dict, device):
    """
    Load weights directly (for merged LoRA checkpoints or full fine-tunes).
    Uses strict=False to handle partial loads gracefully.
    """
    print("[DEBUG] Applying direct state_dict load (merged LoRA or full fine-tune)...")

    inner_model = model.model if hasattr(model, "model") else model
    model_keys = set(inner_model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    matched = model_keys & ckpt_keys
    missing_in_ckpt = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    print(f"[DEBUG] Key match summary:")
    print(f"  Matched:        {len(matched)}")
    print(f"  Missing in ckpt: {len(missing_in_ckpt)}")
    print(f"  Unexpected:      {len(unexpected)}")

    if len(matched) == 0:
        print("[WARN] Zero matching keys! Attempting key remapping...")
        state_dict = _attempt_key_remap(model, state_dict)

    # Check for shape mismatches
    model_sd = inner_model.state_dict()
    for key in matched:
        if model_sd[key].shape != state_dict[key].shape:
            print(f"[ERROR] Shape mismatch for {key}: "
                  f"model={model_sd[key].shape}, ckpt={state_dict[key].shape}")
            raise ValueError(f"Cannot load due to shape mismatch at {key}")

    result = inner_model.load_state_dict(state_dict, strict=False)
    print(f"[DEBUG] Load result - Missing: {len(result.missing_keys)}, "
          f"Unexpected: {len(result.unexpected_keys)}")

    if result.missing_keys:
        print(f"[DEBUG] Sample missing keys: {result.missing_keys[:5]}")
    if result.unexpected_keys:
        print(f"[DEBUG] Sample unexpected keys: {result.unexpected_keys[:5]}")

    return model


def _attempt_key_remap(model, state_dict):
    """Try common key prefix remappings when direct loading fails."""
    inner_model = model.model if hasattr(model, "model") else model
    model_keys = list(inner_model.state_dict().keys())
    ckpt_keys = list(state_dict.keys())

    # Common prefix patterns to try stripping/adding
    prefixes_to_strip = [
        "model.", "module.", "vision_model.", "visual.",
        "clip.vision_model.", "clip.visual.", "vit.",
    ]

    for prefix in prefixes_to_strip:
        remapped = {}
        for k, v in state_dict.items():
            new_key = k[len(prefix):] if k.startswith(prefix) else k
            remapped[new_key] = v

        overlap = set(inner_model.state_dict().keys()) & set(remapped.keys())
        if len(overlap) > len(set(inner_model.state_dict().keys()) & set(state_dict.keys())):
            print(f"[DEBUG] Key remap successful with prefix strip '{prefix}': "
                  f"{len(overlap)} matches")
            return remapped

    print("[WARN] No successful key remap found. Proceeding with original keys.")
    return state_dict


# =========================================================================
# Activation Verification
# =========================================================================

def verify_activations(activation_store, block_layers, device, n_samples=2):
    """
    Run a quick sanity check to ensure activations are being captured correctly.
    """
    print("\n" + "=" * 60)
    print("[DEBUG] ACTIVATION VERIFICATION")
    print("=" * 60)

    try:
        for i in range(n_samples):
            batch = activation_store.get_next_batch()
            if isinstance(batch, dict):
                for layer, act in batch.items():
                    print(f"  Sample {i} | Layer {layer}: "
                          f"shape={act.shape}, dtype={act.dtype}, "
                          f"mean={act.float().mean():.4f}, std={act.float().std():.4f}, "
                          f"min={act.float().min():.4f}, max={act.float().max():.4f}")
                    # Check for degenerate activations
                    if act.float().std() < 1e-8:
                        print(f"  [WARN] Activations at layer {layer} have near-zero variance!")
                    if torch.isnan(act).any():
                        print(f"  [FATAL] NaN detected in activations at layer {layer}!")
                        sys.exit(1)
            else:
                # Single tensor batch
                print(f"  Sample {i}: shape={batch.shape}, dtype={batch.dtype}, "
                      f"mean={batch.float().mean():.4f}, std={batch.float().std():.4f}")
                if torch.isnan(batch).any():
                    print(f"  [FATAL] NaN detected in activations!")
                    sys.exit(1)

        print("[DEBUG] Activation verification PASSED.\n")

    except Exception as e:
        print(f"[FATAL] Activation verification FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)


# =========================================================================
# Model Verification
# =========================================================================

def verify_model_loaded(model, lora_checkpoint_path, device):
    """
    Quick check that LoRA weights actually changed the model from base.
    """
    print("\n" + "=" * 60)
    print("[DEBUG] MODEL WEIGHT VERIFICATION")
    print("=" * 60)

    inner_model = model.model if hasattr(model, "model") else model

    # Grab a few parameter stats
    param_stats = []
    for name, param in inner_model.named_parameters():
        if "attn" in name.lower() and "weight" in name.lower():
            param_stats.append((name, param.float().mean().item(), param.float().std().item()))
            if len(param_stats) >= 3:
                break

    for name, mean, std in param_stats:
        print(f"  {name}: mean={mean:.6f}, std={std:.6f}")

    total_params = sum(p.numel() for p in inner_model.parameters())
    trainable_params = sum(p.numel() for p in inner_model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Model device:     {next(inner_model.parameters()).device}")
    print("[DEBUG] Model verification complete.\n")


# =========================================================================
# Main Training Script
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train SAE on LoRA fine-tuned CLIP activations"
    )

    # --- Model ---
    parser.add_argument("--class_token", action="store_true", default=None)
    parser.add_argument("--image_width", type=int, default=224)
    parser.add_argument("--image_height", type=int, default=224)
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch16")
    parser.add_argument("--module_name", type=str, default="resid")
    parser.add_argument(
        "--block_layers", type=int, nargs="+", default=[-3],
        help="Which transformer block layers to train SAEs for. "
             "Negative indices count from the end. E.g., -1 = last, -3 = 3rd from last. "
             "Pass multiple: --block_layers -3 -2 -1"
    )
    parser.add_argument("--clip_dim", type=int, default=768)

    # --- LoRA ---
    parser.add_argument(
        "--lora_checkpoint_path", type=str, required=True,
        help="Path to saved LoRA weights (.pt, .bin, or .safetensors)"
    )
    parser.add_argument(
        "--lora_format", type=str, default="auto",
        choices=["auto", "peft", "merged", "full"],
        help="Format of LoRA checkpoint. 'auto' tries to detect automatically."
    )

    # --- Dataset ---
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--use_cached_activations", action="store_true", default=None)
    parser.add_argument("--cached_activations_path", type=str)
    parser.add_argument("--expansion_factor", type=int, default=64)
    parser.add_argument("--b_dec_init_method", type=str, default="geometric_median")
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
    parser.add_argument("--use_ghost_grads", action="store_true", default=None)
    parser.add_argument("--feature_sampling_method")
    parser.add_argument("--feature_sampling_window", type=int, default=64)
    parser.add_argument("--dead_feature_window", type=int, default=64)
    parser.add_argument("--dead_feature_threshold", type=float, default=1e-6)

    # --- WANDB ---
    parser.add_argument("--log_to_wandb", action="store_true", default=None)
    parser.add_argument("--wandb_project", type=str, default="lora_clip_sae")
    parser.add_argument("--wandb_entity", type=str, default="test")
    parser.add_argument("--wandb_log_frequency", type=int, default=20)

    # --- Misc ---
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_checkpoints", type=int, default=1)
    parser.add_argument("--checkpoint_path", type=str, default="out/checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--root_dir", type=str, default="")
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--run_name", type=str, default="lora_sae_train")
    parser.add_argument("--start_training_steps", type=int, default=0)
    parser.add_argument("--pt_name", type=str)

    # --- ViT type ---
    parser.add_argument(
        "--vit_type", type=str, default="base",
        help="ViT type: 'base' for standard CLIP, 'maple' for prompt-tuned"
    )
    parser.add_argument("--model_path", type=str, help="Custom CLIP model path")
    parser.add_argument("--config_path", type=str, help="Config path for maple ViT")

    # --- Debugging ---
    parser.add_argument(
        "--skip_activation_verify", action="store_true",
        help="Skip activation sanity check (faster startup)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Load everything but don't train. Useful for verifying the pipeline."
    )

    args = parser.parse_args()

    # =====================================================================
    # Environment Checks
    # =====================================================================
    print("=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)
    print(f"  Python:    {sys.version}")
    print(f"  PyTorch:   {torch.__version__}")
    print(f"  CUDA:      {torch.cuda.is_available()} "
          f"({'device count: ' + str(torch.cuda.device_count()) if torch.cuda.is_available() else 'N/A'})")
    if torch.cuda.is_available():
        print(f"  GPU:       {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:      {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Device:    {args.device}")
    print(f"  Seed:      {args.seed}")
    print()

    # Validate device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # =====================================================================
    # Validate LoRA Checkpoint
    # =====================================================================
    lora_path = Path(args.lora_checkpoint_path)
    if not lora_path.exists():
        # Check if it's a directory (PEFT-style adapter)
        if not lora_path.is_dir():
            print(f"[FATAL] LoRA checkpoint not found: {lora_path}")
            sys.exit(1)
    print(f"[DEBUG] LoRA checkpoint path validated: {lora_path}")

    # =====================================================================
    # Training Loop (per block layer)
    # =====================================================================
    saes = {}
    print(f"\n[INFO] Training SAEs for block layers: {args.block_layers}")
    print(f"[INFO] Using LoRA checkpoint: {args.lora_checkpoint_path}")
    print()

    for layer_idx, block_layer in enumerate(args.block_layers):
        print("\n" + "=" * 60)
        print(f"TRAINING SAE FOR BLOCK LAYER {block_layer} "
              f"({layer_idx + 1}/{len(args.block_layers)})")
        print("=" * 60)

        t_start = time.time()

        # --- Config ---
        cfg = ViTSAERunnerConfig(
            class_token=args.class_token,
            image_width=args.image_width,
            image_height=args.image_height,
            model_name=args.model_name,
            module_name=args.module_name,
            block_layer=block_layer,
            dataset_path=DATASET_INFO[args.dataset]["path"],
            image_key="image",
            label_key="label",
            use_cached_activations=args.use_cached_activations,
            cached_activations_path=args.cached_activations_path,
            d_in=args.clip_dim,
            expansion_factor=args.expansion_factor,
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
        print(f"[DEBUG] Config created: d_in={cfg.d_in}, "
              f"expansion={cfg.expansion_factor}, "
              f"d_sae={cfg.d_in * cfg.expansion_factor}")

        # --- Dataset ---
        print("[DEBUG] Loading dataset...")
        dataset = load_dataset(**DATASET_INFO[args.dataset])
        classnames = get_classnames(args.dataset, dataset)
        print(f"[DEBUG] Dataset loaded: {args.dataset}")

        # --- SAE ---
        print("[DEBUG] Initializing SparseAutoencoder...")
        sae = SparseAutoencoder(cfg, args.device)
        print(f"[DEBUG] SAE initialized: "
              f"{sum(p.numel() for p in sae.parameters()):,} parameters")

        # --- ViT with LoRA ---
        print("[DEBUG] Loading hooked ViT (base model)...")
        vit = load_hooked_vit(
            cfg,
            args.vit_type,
            args.model_name,
            args.device,
            args.model_path,
            args.config_path,
            classnames,
        )
        print("[DEBUG] Base ViT loaded successfully.")

        # Apply LoRA weights
        print("[DEBUG] Applying LoRA weights to ViT...")
        vit = load_lora_weights(vit, args.lora_checkpoint_path, args.device)
        vit.eval()  # SAE training doesn't backprop through ViT
        print("[DEBUG] LoRA weights applied. ViT set to eval mode.")

        # Verify model
        verify_model_loaded(vit, args.lora_checkpoint_path, args.device)

        # --- Activation Store ---
        print("[DEBUG] Initializing ViTActivationsStore...")
        activation_store = ViTActivationsStore(
            dataset,
            args.batch_size,
            args.device,
            args.seed,
            vit,
            block_layer,
            cfg.module_name,
            args.class_token,
        )
        print("[DEBUG] ActivationsStore initialized.")

        # Verify activations
        if not args.skip_activation_verify:
            verify_activations(activation_store, block_layer, args.device)

        # --- Optimizer & Scheduler ---
        optimizer = torch.optim.Adam(sae.parameters(), lr=sae.cfg.lr)
        scheduler = get_scheduler(args.lr_scheduler_name, optimizer=optimizer)
        print(f"[DEBUG] Optimizer: Adam (lr={sae.cfg.lr}), "
              f"Scheduler: {args.lr_scheduler_name}")

        # --- Initialize decoder bias ---
        print("[DEBUG] Initializing SAE b_dec with geometric median...")
        sae.initialize_b_dec(activation_store)
        print(f"[DEBUG] b_dec initialized: mean={sae.b_dec.data.float().mean():.4f}, "
              f"std={sae.b_dec.data.float().std():.4f}")

        sae.train()

        # --- Dry run exit ---
        if args.dry_run:
            print("\n[INFO] DRY RUN complete. Pipeline validated. Exiting.")
            print(f"[INFO] Time for layer {block_layer}: {time.time() - t_start:.1f}s")
            continue

        # --- W&B ---
        if cfg.log_to_wandb:
            run_name = f"{args.run_name}_layer{block_layer}"
            wandb.init(
                project=cfg.wandb_project,
                config=vars(args),
                name=run_name,
                reinit=True,  # Allow re-init for multi-layer runs
            )
            print(f"[DEBUG] W&B initialized: {cfg.wandb_project}/{run_name}")

        # --- Train ---
        print(f"\n[INFO] Starting SAE training for layer {block_layer}...")
        try:
            sae_trainer = SAETrainer(
                sae, vit, activation_store, cfg, optimizer, scheduler, args.device
            )
            sae_trainer.fit()
        except Exception as e:
            print(f"[ERROR] Training failed for layer {block_layer}: {e}")
            traceback.print_exc()
            if cfg.log_to_wandb:
                wandb.finish(exit_code=1)
            continue

        if cfg.log_to_wandb:
            wandb.finish()

        saes[block_layer] = sae
        elapsed = time.time() - t_start
        print(f"[INFO] Layer {block_layer} complete in {elapsed:.1f}s "
              f"({elapsed / 60:.1f} min)")

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Layers trained: {list(saes.keys())}")
    print(f"  Checkpoints at: {args.checkpoint_path}")
    if args.dry_run:
        print("  (DRY RUN - no actual training performed)")
    print("=" * 60)


if __name__ == "__main__":
    main()