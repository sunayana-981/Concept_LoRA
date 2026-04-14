"""
Train Sparse Autoencoder on activations from a DINO/DINOv2 backbone on ImageNet.

This script:
1. Loads a DINOv2 ViT model from HuggingFace (or DINO from torch.hub)
2. Optionally applies PEFT LoRA weights from a saved checkpoint
3. Hooks into specified transformer block layers to extract activations
4. Trains an SAE on those activations

Usage:
    # Base DINOv2 (no LoRA):
    python train_sae_dino_imagenet.py \
        --model_name facebook/dinov2-base \
        --block_layers -3 -2 -1 \
        --dino_dim 768 \
        --expansion_factor 64 \
        --l1_coefficient 0.00008 \
        --batch_size 16 \
        --log_to_wandb

    # With PEFT LoRA weights:
    python train_sae_dino_imagenet.py \
        --model_name facebook/dinov2-base \
        --lora_checkpoint_path /path/to/lora_adapter/ \
        --block_layers -1 \
        --log_to_wandb

Supported model names:
    facebook/dinov2-small  (dim=384,  12 layers)
    facebook/dinov2-base   (dim=768,  12 layers)
    facebook/dinov2-large  (dim=1024, 24 layers)
    facebook/dinov2-giant  (dim=1536, 40 layers)

    torch.hub DINO (legacy): dino_vits16, dino_vitb16, dino_vitb8, dino_vits8
"""

import argparse
import sys
import traceback
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Tuple

# Ensure project root is on sys.path so "src" and "tasks" packages resolve
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset

print("[DEBUG] Importing project modules...")
try:
    from src.sae_training.config import ViTSAERunnerConfig
    from src.sae_training.sae_trainer import SAETrainer
    from src.sae_training.sparse_autoencoder import SparseAutoencoder
    from src.sae_training.utils import get_scheduler
    from tasks.utils import DATASET_INFO, get_classnames
    print("[DEBUG] All project imports successful.")
except ImportError as e:
    print(f"[FATAL] Import failed: {e}")
    traceback.print_exc()
    sys.exit(1)


# =========================================================================
# DINO-specific Hook & HookedVisionTransformer
# =========================================================================

class DINOHook:
    """
    Forward hook for DINO/DINOv2 transformer layers.

    Supports:
      - HuggingFace DINOv2: encoder.layer[{block_layer}]
      - torch.hub DINO:     blocks[{block_layer}]
    """

    def __init__(
        self,
        block_layer: int,
        module_name: str,
        hook_fn: Callable,
        hub_style: bool = False,
        return_module_output: bool = True,
    ):
        assert module_name == "resid", (
            f"Only 'resid' module supported for DINO, got '{module_name}'"
        )
        self.return_module_output = return_module_output
        self.function = self._make_hook(hook_fn)
        self.attr_path = self._get_attr_path(block_layer, hub_style)

    def _make_hook(self, hook_fn: Callable):
        def full_hook_fn(module, module_input, module_output):
            # HuggingFace DINOv2 layers output a tuple: (hidden_states, [attn_weights])
            # torch.hub DINO blocks output a tensor directly
            if isinstance(module_output, tuple):
                hidden = module_output[0]
            else:
                hidden = module_output

            hook_fn(hidden)

            if self.return_module_output:
                return module_output
            else:
                return module_output  # always pass through unchanged

        return full_hook_fn

    def _get_attr_path(self, block_layer: int, hub_style: bool) -> str:
        if hub_style:
            # torch.hub DINO: model.blocks[i]
            return f"blocks[{block_layer}]"
        else:
            # HuggingFace DINOv2: model.encoder.layer[i]
            return f"encoder.layer[{block_layer}]"

    def get_module(self, model):
        return self._get_nested_attr(model, self.attr_path)

    def _get_nested_attr(self, obj, path: str):
        for part in path.split("."):
            if "[" in part:
                name, idx = part[:-1].split("[")
                obj = getattr(obj, name)[int(idx)]
            else:
                obj = getattr(obj, part)
        return obj


class HookedDinoVisionTransformer:
    """
    Thin wrapper around a DINO/DINOv2 model that supports forward-hook caching,
    matching the interface expected by ViTActivationsStore / SAETrainer.

    Supports:
      - HuggingFace Dinov2Model  (model_source="hf")
      - torch.hub DINO ViT       (model_source="hub")
    """

    def __init__(self, model, processor, device="cuda", model_source="hf"):
        self.model = model.to(device)
        self.processor = processor
        self.model_source = model_source  # "hf" or "hub"
        self._hub_style = (model_source == "hub")

    # ------------------------------------------------------------------
    # Public interface (mirrors HookedVisionTransformer)
    # ------------------------------------------------------------------

    def run_with_cache(
        self,
        list_of_hook_locations: List[Tuple[int, str]],
        pixel_values: torch.Tensor,
        **kwargs,
    ):
        cache_dict = {}

        def save_activations(name, activations):
            cache_dict[name] = activations.detach()

        hooks = []
        for block_layer, module_name in list_of_hook_locations:
            fn = partial(save_activations, (block_layer, module_name))
            h = DINOHook(block_layer, module_name, fn, hub_style=self._hub_style)
            hooks.append(h)

        with self._register_hooks(hooks):
            with torch.no_grad():
                if self.model_source == "hf":
                    output = self.model(pixel_values=pixel_values)
                else:
                    output = self.model(pixel_values)

        return output, cache_dict

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def to(self, device):
        self.model = self.model.to(device)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, pixel_values: torch.Tensor, **kwargs):
        if self.model_source == "hf":
            return self.model(pixel_values=pixel_values)
        else:
            return self.model(pixel_values)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @contextmanager
    def _register_hooks(self, hooks: List[DINOHook]):
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


# =========================================================================
# DINO-specific Activations Store
# =========================================================================

class DinoActivationsStore:
    """
    Streams ImageNet images through a DINO model and stores patch/CLS activations
    for SAE training.

    Mirrors the ViTActivationsStore API so it's a drop-in replacement.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        device: str,
        seed: int,
        model: HookedDinoVisionTransformer,
        block_layer: int,
        module_name: str,
        class_token: bool,
    ):
        self.device = device
        self.model = model
        self.batch_size = batch_size
        self.block_layer = block_layer
        self.module_name = module_name
        self.class_token = class_token

        self.dataset = dataset.shuffle(seed=seed)
        self.dataset_iter = iter(self.dataset)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_next_batch(self) -> torch.Tensor:
        return self.get_batch_activations()

    def get_batch_activations(self) -> torch.Tensor:
        pixel_values = self._get_batch_pixel_values()
        return self._extract_activations(pixel_values)

    def initialize_b_dec(self, sae):
        """Delegate to the SAE's own method (uses get_next_batch internally)."""
        sae.initialize_b_dec(self)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_batch_pixel_values(self) -> torch.Tensor:
        images = []
        for _ in range(self.batch_size):
            try:
                item = next(self.dataset_iter)
            except StopIteration:
                self.dataset_iter = iter(self.dataset)
                item = next(self.dataset_iter)
            img = item["image"]
            # Convert grayscale to RGB
            if hasattr(img, "mode") and img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)

        # processor returns {"pixel_values": tensor}
        inputs = self.model.processor(images=images, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)

    def _extract_activations(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hook_location = (self.block_layer, self.module_name)
        _, cache = self.model.run_with_cache([hook_location], pixel_values=pixel_values)
        activations = cache[hook_location]  # (batch, seq_len, d_model)

        if self.class_token:
            # CLS token is always index 0 in DINO/DINOv2
            activations = activations[:, 0, :]   # (batch, d_model)
        else:
            # All patch tokens (drop CLS)
            activations = activations[:, 1:, :]  # (batch, num_patches, d_model)
            # Flatten batch and patch dims so SAE sees (batch*patches, d_model)
            B, P, D = activations.shape
            activations = activations.reshape(B * P, D)

        return activations


# =========================================================================
# Model Loading Utilities
# =========================================================================

def load_dino_model(
    model_name: str,
    device: str,
    model_source: str = "hf",
) -> HookedDinoVisionTransformer:
    """
    Load a DINO or DINOv2 model and wrap it in HookedDinoVisionTransformer.

    Args:
        model_name:   HuggingFace model id (e.g. "facebook/dinov2-base")
                      or torch.hub model name (e.g. "dino_vitb16")
        device:       "cuda" or "cpu"
        model_source: "hf" for HuggingFace, "hub" for torch.hub

    Returns:
        HookedDinoVisionTransformer
    """
    if model_source == "hf":
        from transformers import AutoImageProcessor, AutoModel

        print(f"[DEBUG] Loading DINOv2 from HuggingFace: {model_name}")
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        print(f"[DEBUG] DINOv2 loaded. "
              f"Params: {sum(p.numel() for p in model.parameters()):,}")

    elif model_source == "hub":
        from torchvision import transforms

        print(f"[DEBUG] Loading DINO from torch.hub: {model_name}")
        model = torch.hub.load("facebookresearch/dino:main", model_name, pretrained=True)

        # Build a standard ImageNet processor using torchvision transforms
        # wrapped to match the AutoImageProcessor.process() interface
        _transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        class TorchHubProcessor:
            """Minimal processor shim matching AutoImageProcessor's call signature."""
            def __call__(self, images, return_tensors="pt"):
                tensors = torch.stack([_transform(img) for img in images])
                return {"pixel_values": tensors}

        processor = TorchHubProcessor()
        print(f"[DEBUG] torch.hub DINO loaded. "
              f"Params: {sum(p.numel() for p in model.parameters()):,}")

    else:
        raise ValueError(f"Unknown model_source: {model_source!r}. Use 'hf' or 'hub'.")

    return HookedDinoVisionTransformer(model, processor, device=device, model_source=model_source)


def load_lora_weights_peft(model: HookedDinoVisionTransformer, lora_path: str, device: str):
    """
    Load PEFT-style LoRA adapter into a HookedDinoVisionTransformer.

    Supports:
      - PEFT adapter directory (adapter_config.json + adapter_model.bin)
      - Raw state_dict with 'lora_A' / 'lora_B' keys
    """
    ckpt = Path(lora_path)
    print(f"[DEBUG] Loading LoRA from: {ckpt}")

    # --- Try PEFT directory first ---
    if ckpt.is_dir() and (ckpt / "adapter_config.json").exists():
        try:
            from peft import PeftModel
            print("[DEBUG] PEFT adapter directory detected. Using PeftModel.from_pretrained...")
            inner = PeftModel.from_pretrained(model.model, str(ckpt))
            # Merge LoRA into base weights so inference is clean
            inner = inner.merge_and_unload()
            model.model = inner.to(device)
            print("[DEBUG] PEFT LoRA merged successfully.")
            return model
        except ImportError:
            print("[WARN] peft package not installed. Falling back to manual merge.")
        except Exception as e:
            print(f"[WARN] PeftModel.from_pretrained failed ({e}). Falling back.")

    # --- Manual merge from raw state_dict ---
    suffix = ckpt.suffix.lower()
    if suffix == ".safetensors":
        from safetensors.torch import load_file
        state_dict = load_file(str(ckpt), device=device)
    else:
        state_dict = torch.load(str(ckpt), map_location=device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

    is_peft = any("lora_A" in k or "lora_B" in k for k in state_dict)
    if not is_peft:
        print("[DEBUG] No lora_A/lora_B keys found — assuming merged weights, loading directly.")
        model.model.load_state_dict(state_dict, strict=False)
        return model

    # Group lora_A / lora_B pairs and merge into base weights
    inner = model.model
    base_sd = {k: v.clone() for k, v in inner.state_dict().items()}

    pairs = {}
    for k in state_dict:
        if "lora_A" in k:
            base_key = k.replace(".lora_A.weight", "").replace(".lora_A.default.weight", "")
            base_key = base_key.replace("base_model.model.", "")
            pairs.setdefault(base_key, {})["A"] = state_dict[k]
        elif "lora_B" in k:
            base_key = k.replace(".lora_B.weight", "").replace(".lora_B.default.weight", "")
            base_key = base_key.replace("base_model.model.", "")
            pairs.setdefault(base_key, {})["B"] = state_dict[k]

    print(f"[DEBUG] Found {len(pairs)} LoRA adapter pairs.")
    merged = 0
    for base_key, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            continue
        target = base_key + ".weight"
        if target not in base_sd:
            # Try matching without common prefixes
            for candidate in base_sd:
                if candidate.endswith(base_key.split(".")[-1] + ".weight"):
                    target = candidate
                    break
        if target not in base_sd:
            continue
        lA = ab["A"].to(device).float()
        lB = ab["B"].to(device).float()
        delta = lB @ lA
        if delta.shape != base_sd[target].shape:
            print(f"[WARN] Shape mismatch at {target}: "
                  f"delta={delta.shape}, base={base_sd[target].shape}, skipping.")
            continue
        base_sd[target] = base_sd[target].float() + delta
        merged += 1

    print(f"[DEBUG] Merged {merged}/{len(pairs)} LoRA pairs into DINO weights.")
    inner.load_state_dict(base_sd, strict=True)
    return model


# =========================================================================
# Verification Helpers
# =========================================================================

def verify_activations(store: DinoActivationsStore, block_layer: int, device: str, n_samples: int = 2):
    print("\n" + "=" * 60)
    print("[DEBUG] ACTIVATION VERIFICATION")
    print("=" * 60)

    for i in range(n_samples):
        batch = store.get_next_batch()
        print(f"  Sample {i}: shape={batch.shape}, dtype={batch.dtype}, "
              f"mean={batch.float().mean():.4f}, std={batch.float().std():.4f}, "
              f"min={batch.float().min():.4f}, max={batch.float().max():.4f}")
        if batch.float().std() < 1e-8:
            print(f"  [WARN] Near-zero variance in activations at layer {block_layer}!")
        if torch.isnan(batch).any():
            print(f"  [FATAL] NaN in activations at layer {block_layer}!")
            sys.exit(1)

    print("[DEBUG] Activation verification PASSED.\n")


def verify_model_loaded(model: HookedDinoVisionTransformer):
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

    total = sum(p.numel() for p in inner.parameters())
    trainable = sum(p.numel() for p in inner.parameters() if p.requires_grad)
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}")
    print(f"  Model device:     {next(inner.parameters()).device}")
    print("[DEBUG] Model verification complete.\n")


# =========================================================================
# Main Training Script
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train SAE on DINO/DINOv2 activations from ImageNet"
    )

    # --- Model ---
    parser.add_argument(
        "--model_name", type=str, default="facebook/dinov2-base",
        help="HuggingFace model id (e.g. facebook/dinov2-base) or torch.hub name (dino_vitb16)"
    )
    parser.add_argument(
        "--model_source", type=str, default="hf", choices=["hf", "hub"],
        help="'hf' for HuggingFace DINOv2, 'hub' for torch.hub DINO"
    )
    parser.add_argument(
        "--dino_dim", type=int, default=768,
        help="Hidden dimension of the DINO model. "
             "dinov2-small=384, dinov2-base=768, dinov2-large=1024, dinov2-giant=1536"
    )
    parser.add_argument(
        "--block_layers", type=int, nargs="+", default=[-3],
        help="Transformer block layers to train SAEs for. "
             "Negative indices count from the end. E.g., -1=last. "
             "Pass multiple: --block_layers -3 -2 -1"
    )
    parser.add_argument(
        "--class_token", action="store_true", default=True,
        help="If set, use only the CLS token activation (default). "
             "Otherwise use all patch tokens (flattened)."
    )
    parser.add_argument(
        "--no_class_token", dest="class_token", action="store_false",
        help="Use patch tokens instead of CLS token."
    )
    parser.add_argument("--module_name", type=str, default="resid")
    parser.add_argument("--image_width", type=int, default=224)
    parser.add_argument("--image_height", type=int, default=224)

    # --- LoRA (optional) ---
    parser.add_argument(
        "--lora_checkpoint_path", type=str, default=None,
        help="Optional: path to PEFT LoRA adapter dir or .pt/.safetensors file"
    )

    # --- Dataset ---
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--use_cached_activations", action="store_true", default=None)
    parser.add_argument("--cached_activations_path", type=str)

    # --- SAE ---
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

    # --- W&B ---
    parser.add_argument("--log_to_wandb", action="store_true", default=None)
    parser.add_argument("--wandb_project", type=str, default="dino_imagenet_sae")
    parser.add_argument("--wandb_entity", type=str, default="test")
    parser.add_argument("--wandb_log_frequency", type=int, default=20)

    # --- Misc ---
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_checkpoints", type=int, default=1)
    parser.add_argument("--checkpoint_path", type=str, default="out/checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run_name", type=str, default="dino_sae_train")
    parser.add_argument(
        "--skip_activation_verify", action="store_true",
        help="Skip activation sanity check (faster startup)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Load everything but skip training. Useful for pipeline validation."
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
    cuda_avail = torch.cuda.is_available()
    print(f"  CUDA:      {cuda_avail} "
          f"({'device count: ' + str(torch.cuda.device_count()) if cuda_avail else 'N/A'})")
    if cuda_avail:
        print(f"  GPU:       {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:      {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Device:    {args.device}")
    print(f"  Seed:      {args.seed}")
    print(f"  Model:     {args.model_name} (source={args.model_source})")
    print(f"  Dim:       {args.dino_dim}")
    print(f"  CLS token: {args.class_token}")
    print()

    if args.device == "cuda" and not cuda_avail:
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    torch.manual_seed(args.seed)
    if cuda_avail:
        torch.cuda.manual_seed_all(args.seed)

    # =====================================================================
    # Training Loop (per block layer)
    # =====================================================================
    saes = {}
    print(f"[INFO] Training SAEs for block layers: {args.block_layers}")
    if args.lora_checkpoint_path:
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
            d_in=args.dino_dim,
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
        print(f"[DEBUG] Config: d_in={cfg.d_in}, "
              f"expansion={cfg.expansion_factor}, "
              f"d_sae={cfg.d_in * cfg.expansion_factor}")

        # --- Dataset ---
        print("[DEBUG] Loading dataset...")
        dataset = load_dataset(**DATASET_INFO[args.dataset])
        # Use the train split if a DatasetDict is returned
        if hasattr(dataset, "keys"):
            split = DATASET_INFO[args.dataset].get("split", "train")
            dataset = dataset[split]
        print(f"[DEBUG] Dataset loaded: {args.dataset} ({len(dataset)} samples)")

        # --- SAE ---
        print("[DEBUG] Initializing SparseAutoencoder...")
        sae = SparseAutoencoder(cfg, args.device)
        print(f"[DEBUG] SAE params: {sum(p.numel() for p in sae.parameters()):,}")

        # --- DINO Model ---
        print("[DEBUG] Loading DINO model...")
        dino = load_dino_model(args.model_name, args.device, args.model_source)

        # Apply LoRA if requested
        if args.lora_checkpoint_path:
            print("[DEBUG] Applying LoRA weights...")
            dino = load_lora_weights_peft(dino, args.lora_checkpoint_path, args.device)

        dino.eval()
        verify_model_loaded(dino)

        # --- Activation Store ---
        print("[DEBUG] Initializing DinoActivationsStore...")
        activation_store = DinoActivationsStore(
            dataset=dataset,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
            model=dino,
            block_layer=block_layer,
            module_name=args.module_name,
            class_token=args.class_token,
        )
        print("[DEBUG] DinoActivationsStore initialized.")

        # Verify activations
        if not args.skip_activation_verify:
            verify_activations(activation_store, block_layer, args.device)

        # --- Optimizer & Scheduler ---
        optimizer = torch.optim.Adam(sae.parameters(), lr=sae.cfg.lr)
        scheduler = get_scheduler(args.lr_scheduler_name, optimizer=optimizer)
        print(f"[DEBUG] Optimizer: Adam (lr={sae.cfg.lr}), "
              f"Scheduler: {args.lr_scheduler_name}")

        # --- Initialize decoder bias ---
        print("[DEBUG] Initializing SAE b_dec...")
        sae.initialize_b_dec(activation_store)
        print(f"[DEBUG] b_dec: mean={sae.b_dec.data.float().mean():.4f}, "
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
                reinit=True,
            )
            print(f"[DEBUG] W&B: {cfg.wandb_project}/{run_name}")

        # --- Train ---
        print(f"\n[INFO] Starting SAE training for layer {block_layer}...")
        try:
            sae_trainer = SAETrainer(
                sae, dino, activation_store, cfg, optimizer, scheduler, args.device
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
        print(f"[INFO] Layer {block_layer} done in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Layers trained: {list(saes.keys())}")
    print(f"  Checkpoints at: {args.checkpoint_path}")
    if args.dry_run:
        print("  (DRY RUN — no actual training performed)")
    print("=" * 60)


if __name__ == "__main__":
    main()
