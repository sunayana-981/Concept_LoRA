#!/usr/bin/env python3
"""
SAE Inference & Top-Activating Neuron Visualization

Loads trained SAEs from checkpoints and generates top-activating neuron plots
using images from the respective datasets (EuroSAT, Caltech101, MedMNIST).

Usage:
    python run_sae_inference.py --dataset eurosat
    python run_sae_inference.py --dataset caltech101
    python run_sae_inference.py --dataset medmnist
    python run_sae_inference.py --all
"""

import argparse
import os
import sys
import glob
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# Add project root to path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- Project imports (adjust if your module structure differs) ---
SAE = None
try:
    from sae_lens.sae import SAE
except ImportError:
    try:
        from sae import SAE
    except ImportError:
        print("[WARN] Could not import SAE from sae_lens or sae. Will try loading from checkpoint directly.")

try:
    from lora_clip_model import LoRACLIPModel  # adjust module name as needed
except ImportError:
    pass

try:
    from dams_metric import compute_dams, compare_dams, DAMSResult
    _DAMS_AVAILABLE = True
except ImportError:
    _DAMS_AVAILABLE = False
    print("[WARN] dams_metric.py not found — DAMS scoring will be skipped.")

import open_clip
from torchvision import transforms, datasets


# =============================================================================
# Dataset Loaders — returns (dataset, class_names)
# =============================================================================

def get_clip_preprocess(model_name: str = "openai/clip-vit-base-patch16", image_size: int = 224):
    """Standard CLIP preprocessing."""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


class EuroSATDataset(Dataset):
    """EuroSAT dataset loader."""
    
    CLASS_NAMES = [
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
        "Industrial", "Pasture", "PermanentCrop", "Residential",
        "River", "SeaLake"
    ]
    
    def __init__(self, root: str = None, transform=None, split: str = "test"):
        if root is None:
            # Images are at: data/eurosat/2750/{ClassName}/*.jpg
            root = "/home/sunayana/Documents/Concept_LoRA/data/eurosat"
        
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"EuroSAT data not found at {root}"
            )
        
        self._use_tv = False
        self.root = root
        self.transform = transform
        self.class_names = self.CLASS_NAMES
        
        self.samples = []
        self.labels = []
        
        # Auto-discover: handle nested dirs like root/EuroSAT/ClassName/ or root/2750/ClassName/
        # First, find where the class subdirs actually are
        image_root = self._find_image_root(root)
        
        # Build class-to-index mapping from what's actually on disk
        subdirs = sorted([
            d for d in os.listdir(image_root)
            if os.path.isdir(os.path.join(image_root, d))
        ])
        
        if not subdirs:
            raise FileNotFoundError(
                f"No class subdirectories found in {image_root}. "
                f"Contents: {os.listdir(image_root)[:20]}"
            )
        
        # Try to match to known class names, otherwise use discovered names
        discovered_to_idx = {}
        for subdir in subdirs:
            # Try exact match first
            if subdir in self.CLASS_NAMES:
                discovered_to_idx[subdir] = self.CLASS_NAMES.index(subdir)
            else:
                # Try case-insensitive
                for i, cn in enumerate(self.CLASS_NAMES):
                    if subdir.lower() == cn.lower():
                        discovered_to_idx[subdir] = i
                        break
                else:
                    # Unknown class, assign new index
                    discovered_to_idx[subdir] = len(discovered_to_idx)
        
        if not any(s in self.CLASS_NAMES or s.lower() in [c.lower() for c in self.CLASS_NAMES] for s in subdirs):
            # Class names don't match predefined ones; use discovered
            self.class_names = subdirs
            discovered_to_idx = {s: i for i, s in enumerate(subdirs)}
        
        for subdir in subdirs:
            cls_idx = discovered_to_idx[subdir]
            cls_dir = os.path.join(image_root, subdir)
            for img_name in sorted(os.listdir(cls_dir)):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    self.samples.append(os.path.join(cls_dir, img_name))
                    self.labels.append(cls_idx)
        
        print(f"[EuroSAT] Loaded {len(self.samples)} images from {image_root} "
              f"({len(subdirs)} classes: {subdirs[:5]}...)")
    
    def _find_image_root(self, root):
        """Recursively find the directory containing class subdirectories with images."""
        # Check if root itself has class subdirs with images
        subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        
        # Check if any subdir contains image files
        for sd in subdirs:
            sd_path = os.path.join(root, sd)
            files = os.listdir(sd_path)
            has_images = any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')) for f in files)
            if has_images:
                return root
        
        # Otherwise recurse one level down
        for sd in subdirs:
            sd_path = os.path.join(root, sd)
            try:
                return self._find_image_root(sd_path)
            except FileNotFoundError:
                continue
        
        raise FileNotFoundError(f"No image class directories found under {root}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
    
    def get_raw_image(self, idx) -> Image.Image:
        return Image.open(self.samples[idx]).convert("RGB")


class Caltech101Dataset(Dataset):
    """Caltech-101 dataset loader."""
    
    def __init__(self, root: str = None, transform=None, split: str = "test"):
        if root is None:
            # Images are at: data/caltech-101/Caltech101/101_ObjectCategories/{ClassName}/*.jpg
            root = "/home/sunayana/Documents/Concept_LoRA/data/caltech-101"
        
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"Caltech101 data not found at {root}"
            )
        
        self._use_tv = False
        self.root = root
        self.transform = transform
        self.samples = []
        self.labels = []
        self.class_names = []
        
        # Auto-discover image root (handle nesting like root/101_ObjectCategories/ClassName/)
        image_root = self._find_image_root(root)
        
        subdirs = sorted([
            d for d in os.listdir(image_root)
            if os.path.isdir(os.path.join(image_root, d)) and d != "BACKGROUND_Google"
        ])
        
        if not subdirs:
            raise FileNotFoundError(
                f"No class subdirectories found in {image_root}. "
                f"Contents: {os.listdir(image_root)[:20]}"
            )
        
        self.class_names = subdirs
        
        for cls_idx, cls_name in enumerate(self.class_names):
            cls_dir = os.path.join(image_root, cls_name)
            for img_name in sorted(os.listdir(cls_dir)):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append(os.path.join(cls_dir, img_name))
                    self.labels.append(cls_idx)
        
        print(f"[Caltech101] Loaded {len(self.samples)} images from {image_root} "
              f"({len(self.class_names)} classes)")
    
    def _find_image_root(self, root):
        """Recursively find the directory containing class subdirectories with images."""
        subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        
        for sd in subdirs:
            sd_path = os.path.join(root, sd)
            files = os.listdir(sd_path)
            has_images = any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files)
            if has_images:
                return root
        
        # Check if we have many subdirs (likely class dirs even without checking images)
        if len(subdirs) > 10:
            # Probably the right level, check one subdir deeper
            for sd in subdirs[:3]:
                sd_path = os.path.join(root, sd)
                files = os.listdir(sd_path)
                has_images = any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files)
                if has_images:
                    return root
        
        for sd in subdirs:
            sd_path = os.path.join(root, sd)
            try:
                return self._find_image_root(sd_path)
            except (FileNotFoundError, PermissionError):
                continue
        
        raise FileNotFoundError(f"No image class directories found under {root}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
    
    def get_raw_image(self, idx) -> Image.Image:
        return Image.open(self.samples[idx]).convert("RGB")


class MedMNISTDataset(Dataset):
    """MedMNIST dataset loader. Uses pathmnist by default."""
    
    def __init__(self, root: str = None, transform=None, split: str = "test", 
                 medmnist_flag: str = "pathmnist"):
        try:
            import medmnist
            from medmnist import INFO
            
            info = INFO[medmnist_flag]
            self.class_names = list(info['label'].values())
            DataClass = getattr(medmnist, info['python_class'])
            
            if root is None:
                root = "/home/sunayana/Documents/Concept_LoRA/data/medmnist"
            
            os.makedirs(root, exist_ok=True)
            
            # pathmnist.npz is 28x28; the CLIP transform (Resize→224, CenterCrop, etc.) handles upscaling
            self.dataset = DataClass(
                split=split,
                download=False,
                root=root,
                transform=transform,
            )
            
            self._use_medmnist = True
            self._root = root
            print(f"[MedMNIST/{medmnist_flag}] Loaded {len(self.dataset)} images, "
                  f"{len(self.class_names)} classes")
            
        except ImportError:
            raise ImportError(
                "MedMNIST not installed. Install with: pip install medmnist"
            )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # MedMNIST labels are arrays
        if isinstance(label, np.ndarray):
            label = int(label.flatten()[0])
        elif isinstance(label, torch.Tensor):
            label = label.item()
        return img, label
    
    def get_raw_image(self, idx) -> Image.Image:
        """Get raw image without CLIP normalization for display."""
        # Create a version of dataset without the normalize transform
        # Access the underlying numpy data directly
        if hasattr(self.dataset, 'imgs'):
            img_array = self.dataset.imgs[idx]
            # img_array is (H, W, C) uint8 or (H, W) for grayscale
            img = Image.fromarray(img_array)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img.resize((224, 224), Image.BICUBIC)
        
        # Fallback
        img, _ = self.dataset[idx]
        if isinstance(img, torch.Tensor):
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
            img = img * std + mean
            img = img.clamp(0, 1)
            return transforms.ToPILImage()(img)
        return img


_DATA_ROOT = "/home/sunayana/Documents/Concept_LoRA/data"

# Default data directories for each dataset key
_DATASET_DATA_DIRS = {
    "eurosat":    f"{_DATA_ROOT}/eurosat/2750",
    "caltech101": f"{_DATA_ROOT}/caltech-101",
    "medmnist":   f"{_DATA_ROOT}/pathmnist_imagefolder",
    "dtd":        f"{_DATA_ROOT}/DTD/images",
    "cub2002011": f"{_DATA_ROOT}/cub2002011/test",
    "ucf101":     f"{_DATA_ROOT}/UCF101/UCF-101-midframes",
}

# Full mapping: checkpoint dir name → (canonical dataset key, lora subdir, lora seed)
ALL_SAE_CONFIGS = {
    "eurosat":               ("eurosat",   "eurosat",   "seed1"),
    "eurosat_maple":         ("eurosat",   "eurosat",   "seed1"),
    "caltech101":            ("caltech101","caltech101","seed1"),
    "caltech101_maple":      ("caltech101","caltech101","seed1"),
    "medmnist":              ("medmnist",  "medmnist",  "seed1"),
    "medmnist_maple":        ("medmnist",  "medmnist",  "seed1"),
    "masked_finetune":       ("medmnist",  "medmnist",  "seed1"),
    "masked_finetune_lora":  ("medmnist",  "medmnist",  "seed1"),
    "masked_finetune_maple": ("medmnist",  "medmnist",  "seed1"),
    "9g0pkku9":              ("medmnist",  None,        None),
    "owdr2cw0":              ("medmnist",  None,        None),
    "dtd":                   ("dtd",       "dtd",       "seed42"),
    "cub2002011":            ("cub2002011","cub2002011","seed1"),
    "ucf101":                ("ucf101",    "ucf101",    "seed1"),
}


def load_dataset(dataset_name: str, transform, data_root: str = None) -> Tuple[Dataset, List[str]]:
    """Load dataset by name. Returns (dataset, class_names)."""
    dataset_name = dataset_name.lower()

    if dataset_name == "eurosat":
        ds = EuroSATDataset(root=data_root, transform=transform)
        return ds, ds.class_names
    elif dataset_name == "caltech101":
        ds = Caltech101Dataset(root=data_root, transform=transform)
        return ds, ds.class_names
    elif dataset_name == "medmnist":
        ds = MedMNISTDataset(root=data_root, transform=transform)
        return ds, ds.class_names
    elif dataset_name in ("dtd", "cub2002011", "ucf101"):
        root = data_root or _DATASET_DATA_DIRS[dataset_name]
        if not os.path.isdir(root):
            raise FileNotFoundError(f"{dataset_name} data not found at {root}")
        # Walk into the directory until we find the ImageFolder-style root
        img_root = root
        for r, dirs, _ in os.walk(root):
            if dirs:
                sample = os.path.join(r, dirs[0])
                exts = {os.path.splitext(f)[1].lower()
                        for f in os.listdir(sample)
                        if os.path.isfile(os.path.join(sample, f))}
                if exts & {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                    img_root = r
                    break
        exclude = {"BACKGROUND_Google"} if dataset_name == "caltech101" else set()
        full_ds = datasets.ImageFolder(root=img_root, transform=transform)
        if exclude:
            keep = [i for i, (_, l) in enumerate(full_ds.samples)
                    if full_ds.classes[l] not in exclude]
            full_ds = torch.utils.data.Subset(full_ds, keep)
            class_names = [c for c in full_ds.dataset.classes if c not in exclude]
        else:
            class_names = full_ds.classes
        return full_ds, class_names
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported: {', '.join(_DATASET_DATA_DIRS)}"
        )


# =============================================================================
# Model Loading
# =============================================================================

def load_lora_clip_model(
    model_name: str,
    lora_checkpoint_path: str,
    device: str = "cuda",
):
    """Load CLIP model with LoRA weights applied."""
    from transformers import CLIPModel, CLIPProcessor
    
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    
    # Load and apply LoRA weights
    if lora_checkpoint_path and os.path.isfile(lora_checkpoint_path):
        print(f"Loading LoRA weights from {lora_checkpoint_path}")
        lora_state = torch.load(lora_checkpoint_path, map_location=device)
        
        # Try to apply LoRA weights — this depends on your LoRA implementation
        # Option 1: Direct state dict merge
        try:
            model_state = model.state_dict()
            applied = 0
            for k, v in lora_state.items():
                if k in model_state:
                    model_state[k] = model_state[k] + v
                    applied += 1
            if applied > 0:
                model.load_state_dict(model_state)
                print(f"  Applied {applied} LoRA parameters via additive merge")
        except Exception as e:
            print(f"  [WARN] Could not apply LoRA additively: {e}")
            # Option 2: Try loading via a custom function from the project
            try:
                from utils.lora_utils import apply_lora_weights
                apply_lora_weights(model, lora_state)
                print("  Applied LoRA via apply_lora_weights()")
            except ImportError:
                print("  [WARN] No apply_lora_weights found. Using base CLIP model.")
    
    model.eval()
    return model, processor


def find_latest_sae_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest SAE checkpoint in a directory."""
    patterns = ["*.pt", "*.bin", "**/*.pt", "**/*.bin"]
    all_files = []
    for p in patterns:
        all_files.extend(glob.glob(os.path.join(checkpoint_dir, p), recursive=True))
    
    if not all_files:
        return None
    
    # Sort by modification time, newest first
    all_files.sort(key=os.path.getmtime, reverse=True)
    
    # Prefer "final" checkpoints, but validate for NaN
    for f in all_files:
        fname = os.path.basename(f).lower()
        if "final" in fname:
            if _checkpoint_is_valid(f):
                return f
            else:
                print(f"  [WARN] Final checkpoint has NaN weights: {f}")
                break  # Fall through to find a valid earlier checkpoint
    
    # Try all checkpoints newest-first, pick first valid one
    for f in all_files:
        if _checkpoint_is_valid(f):
            print(f"  Using valid checkpoint: {f}")
            return f
    
    # No valid checkpoints at all
    print(f"  [ERROR] All checkpoints in {checkpoint_dir} have NaN weights!")
    return None


def _checkpoint_is_valid(path: str) -> bool:
    """Quick check that a checkpoint's W_enc doesn't contain NaN."""
    try:
        state = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(state, dict) and 'state_dict' in state:
            w = state['state_dict'].get('W_enc', None)
        elif isinstance(state, dict):
            w = state.get('W_enc', None)
        else:
            return True  # Can't check, assume valid
        if w is not None and torch.isnan(w).any():
            return False
        return True
    except Exception:
        return True  # Can't load to check, let main loader handle it


def load_sae_from_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Load SAE from checkpoint."""
    print(f"Loading SAE from {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    
    # Try various loading strategies based on what's in the checkpoint
    # Support both 'cfg' (sae_lens) and 'config' (base SAE) keys
    cfg_obj = None
    if isinstance(state, dict):
        cfg_obj = state.get("cfg", state.get("config", None))
    
    if cfg_obj is not None and SAE is not None:
        try:
            sae = SAE(cfg_obj)
            sae.load_state_dict(state["state_dict"] if "state_dict" in state else state)
            sae.to(device)
            sae.eval()
            return sae, cfg_obj
        except Exception as e:
            print(f"  [WARN] SAE(cfg) failed: {e}")
    
    if isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    elif isinstance(state, dict) and any(k.startswith("W_enc") or k.startswith("encoder") for k in state.keys()):
        state_dict = state
    else:
        state_dict = state
    
    # Infer dimensions from weights
    d_in, d_sae = None, None
    for key in state_dict:
        if "W_enc" in key or "encoder" in key:
            w = state_dict[key]
            if isinstance(w, torch.Tensor) and w.ndim == 2:
                d_in, d_sae = w.shape
                print(f"  Inferred d_in={d_in}, d_sae={d_sae} from {key}")
                break
    
    if d_in is None:
        raise ValueError(f"Cannot infer SAE dimensions from {checkpoint_path}")
    
    # Manual SAE construction
    sae = SimpleSAE(d_in=d_in, d_sae=d_sae)
    sae.load_state_dict(state_dict, strict=False)
    
    sae.to(device)
    sae.eval()
    return sae, cfg_obj or {}


def infer_sae_training_layer(sae_cfg, checkpoint_path: Optional[str] = None) -> Optional[int]:
    """Infer the CLIP block layer an SAE was trained on.

    Priority:
      1) explicit layer fields in SAE config
      2) hook point strings in config
      3) checkpoint filename pattern: *_<layer>_resid_*.pt
    """
    if sae_cfg is not None:
        if hasattr(sae_cfg, "block_layer"):
            value = getattr(sae_cfg, "block_layer")
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass

        if isinstance(sae_cfg, dict):
            for key in ("block_layer", "hook_layer", "layer"):
                value = sae_cfg.get(key)
                if value is not None:
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        continue

            hook_point = sae_cfg.get("hook_point")
            if isinstance(hook_point, str):
                matches = re.findall(r"-?\d+", hook_point)
                if matches:
                    try:
                        return int(matches[-1])
                    except (TypeError, ValueError):
                        pass

    if checkpoint_path:
        checkpoint_name = os.path.basename(checkpoint_path)
        match = re.search(r"_(-?\d+)_resid", checkpoint_name)
        if match:
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                pass

    return None


class SimpleSAE(torch.nn.Module):
    """Minimal SAE for inference if sae_lens is unavailable.

    Follows the sae_lens convention:
      encode:  z = ReLU((x − b_dec) @ W_enc + b_enc)
      decode:  x̂ = z @ W_dec + b_dec

    b_dec serves dual purpose: decoder bias AND input centering constant.
    Without this centering the encoder operates on an offset input, producing
    corrupted latents and catastrophic reconstruction MSE.
    """

    def __init__(self, d_in: int, d_sae: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.W_enc = torch.nn.Parameter(torch.zeros(d_in, d_sae))
        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
        self.W_dec = torch.nn.Parameter(torch.zeros(d_sae, d_in))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_in))

    def encode(self, x):
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# =============================================================================
# Feature Extraction
# =============================================================================

@torch.no_grad()
def extract_clip_features(
    model,
    dataloader: DataLoader,
    block_layer: int = -2,
    device: str = "cuda",
    max_batches: Optional[int] = None,
) -> Tuple[torch.Tensor, List[int], List[int]]:
    """
    Extract intermediate features from CLIP vision encoder at a given block layer.
    
    Returns:
        features: (N, seq_len, d_model) or (N, d_model) tensor
        labels: list of integer labels
        indices: list of dataset indices
    """
    all_features = []
    all_labels = []
    all_indices = []
    idx_offset = 0
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Extracting features")):
        if max_batches and batch_idx >= max_batches:
            break
        
        images = images.to(device)
        
        # Extract intermediate features using hooks
        features = []
        
        def hook_fn(module, input, output):
            # output shape: (batch, seq_len, d_model) for transformer blocks
            if isinstance(output, tuple):
                features.append(output[0].detach().cpu())
            else:
                features.append(output.detach().cpu())
        
        # Register hook on the target layer
        # For HuggingFace CLIP:
        if hasattr(model, 'vision_model'):
            target_layers = model.vision_model.encoder.layers
        elif hasattr(model, 'visual'):
            target_layers = model.visual.transformer.resblocks
        else:
            raise ValueError("Could not find vision encoder layers in model")
        
        layer_idx = block_layer if block_layer >= 0 else len(target_layers) + block_layer
        handle = target_layers[layer_idx].register_forward_hook(hook_fn)
        
        try:
            if hasattr(model, 'get_image_features'):
                _ = model.get_image_features(pixel_values=images)
            elif hasattr(model, 'encode_image'):
                _ = model.encode_image(images)
            else:
                _ = model(images)
        except Exception:
            # Try with pixel_values kwarg
            _ = model.vision_model(pixel_values=images)
        
        handle.remove()
        
        if features:
            batch_features = features[0]  # (B, seq_len, d_model)
            all_features.append(batch_features)
            
            if isinstance(labels, torch.Tensor):
                all_labels.extend(labels.cpu().tolist())
            elif isinstance(labels, np.ndarray):
                all_labels.extend(labels.flatten().tolist())
            else:
                all_labels.extend([int(l) for l in labels])
            
            all_indices.extend(range(idx_offset, idx_offset + len(images)))
            idx_offset += len(images)
    
    all_features = torch.cat(all_features, dim=0)
    return all_features, all_labels, all_indices


# =============================================================================
# SAE Activation Analysis
# =============================================================================

@torch.no_grad()
def compute_sae_activations(
    sae,
    features: torch.Tensor,
    device: str = "cuda",
    batch_size: int = 256,
) -> torch.Tensor:
    """
    Run features through SAE encoder to get *pooled* neuron activations.
    Only stores (N, d_sae) — NOT the full spatial tensor to avoid OOM.
    
    Args:
        sae: trained SAE model
        features: (N, seq_len, d_model) or (N, d_model)
        
    Returns:
        max_activations: (N, d_sae) — max-pooled over sequence length
    """
    has_seq = features.ndim == 3
    N = features.shape[0]
    
    if has_seq:
        S = features.shape[1]
        seq_len = S - 1  # excluding CLS token
        D = features.shape[2]
    else:
        seq_len = 1
        D = features.shape[1]
    
    # Process one image at a time, pooling incrementally to avoid materialising
    # the full (B*seq_len, d_sae) tensor which causes OOM with large d_sae.
    pooled_list = []
    sae_batch = 512  # patches per SAE forward pass

    for img_idx in tqdm(range(N), desc="SAE activations"):
        if has_seq:
            patches = features[img_idx, 1:, :].to(device)  # (seq_len, D), skip CLS
        else:
            patches = features[img_idx].unsqueeze(0).to(device)  # (1, D)

        # Incremental max-pool: never keep more than one sub-batch on GPU
        running_max: Optional[torch.Tensor] = None
        for j in range(0, patches.shape[0], sae_batch):
            chunk = patches[j : j + sae_batch]
            if hasattr(sae, 'encode'):
                z = sae.encode(chunk)
            else:
                _, z = sae(chunk)
            chunk_max = z.max(dim=0).values  # (d_sae,)
            if running_max is None:
                running_max = chunk_max 
            else:
                running_max = torch.maximum(running_max, chunk_max)
            del z, chunk_max

        pooled_list.append(running_max.cpu())
        del patches, running_max
        if img_idx % 100 == 0:
            torch.cuda.empty_cache()

    return torch.stack(pooled_list, dim=0)  # (N, d_sae)


@torch.no_grad()
def compute_spatial_activations_for_images(  
    sae,
    features: torch.Tensor,
    image_indices: List[int],
    device: str = "cuda",
) -> torch.Tensor:
    """
    Recompute per-patch SAE activations for a small set of images (for heatmaps).
    
    Args:
        sae: trained SAE model
        features: (N, seq_len, d_model) — full feature tensor
        image_indices: list of image indices to compute spatial activations for
        
    Returns:
        spatial_acts: (len(image_indices), S-1, d_sae)
    """
    has_seq = features.ndim == 3
    if not has_seq:
        # No spatial info
        subset = features[image_indices].to(device)
        if hasattr(sae, 'encode'):
            z = sae.encode(subset)
        else:
            _, z = sae(subset)
        return z.cpu().unsqueeze(1)  # (K, 1, d_sae)
    
    results = []
    for idx in image_indices:
        patches = features[idx, 1:, :].to(device)  # (S-1, D), skip CLS
        if hasattr(sae, 'encode'):
            z = sae.encode(patches)
        else:
            _, z = sae(patches)
        results.append(z.cpu())
    
    return torch.stack(results, dim=0)  # (K, S-1, d_sae)


def find_top_activating_neurons(
    activations: torch.Tensor,
    top_k: int = 20,
    min_activation_freq: float = 0.01,
) -> List[int]:
    """
    Find the most interesting SAE neurons based on activation magnitude and frequency.
    
    Args:
        activations: (N, d_sae)
        top_k: number of neurons to return
        min_activation_freq: minimum fraction of images that must activate
    """
    N, d_sae = activations.shape
    
    # Compute statistics per neuron
    mean_act = activations.mean(dim=0)          # Mean activation
    max_act = activations.max(dim=0).values     # Max activation
    freq = (activations > 0).float().mean(dim=0)  # Firing frequency
    
    # Score = mean activation * log(frequency + eps) to prefer neurons that
    # fire at a reasonable rate (not too sparse, not too dense)
    eps = 1e-8
    # Penalize neurons that fire too rarely or on everything
    freq_score = -((freq - 0.1).abs())  # Peak at 10% firing rate
    score = mean_act * (freq > min_activation_freq).float() + freq_score * 0.1
    
    # Also consider max activation as a tiebreaker
    score = score + 0.01 * max_act
    
    # Get top-k neuron indices
    top_indices = score.argsort(descending=True)[:top_k].tolist()
    
    # Print stats
    print(f"\nTop {top_k} neurons (out of {d_sae}):")
    print(f"{'Neuron':>8} {'MeanAct':>10} {'MaxAct':>10} {'Freq':>8} {'Score':>10}")
    print("-" * 50)
    for idx in top_indices:
        print(f"{idx:>8d} {mean_act[idx]:>10.4f} {max_act[idx]:>10.4f} "
              f"{freq[idx]:>8.3f} {score[idx]:>10.4f}")
    
    return top_indices


# =============================================================================
# Visualization
# =============================================================================

def get_raw_image_for_display(dataset, idx: int) -> Image.Image:
    """Get a raw (unnormalized) image from dataset for display."""
    if isinstance(dataset, torch.utils.data.Subset):
        return get_raw_image_for_display(dataset.dataset, dataset.indices[idx])

    if hasattr(dataset, 'get_raw_image'):
        return dataset.get_raw_image(idx)
    
    # Fallback: try to get image and undo normalization
    img, _ = dataset[idx]
    if isinstance(img, torch.Tensor):
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        img = img * std + mean
        img = img.clamp(0, 1)
        return transforms.ToPILImage()(img)
    return img


def plot_top_activating_images_for_neuron(
    neuron_idx: int,
    activations_pooled: torch.Tensor,   # (N, d_sae)
    sae,                                 # SAE model for on-demand spatial computation
    features: torch.Tensor,              # (N, seq_len, d_model) for spatial heatmaps
    dataset: Dataset,
    class_names: List[str],
    labels: List[int],
    indices: List[int],
    device: str = "cuda",
    top_k_images: int = 8,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Plot top-k images that most strongly activate a given SAE neuron,
    with patch-level activation heatmaps overlaid.
    Spatial activations are recomputed on-demand for only the top-k images.
    """
    # Get per-image activation for this neuron
    neuron_acts = activations_pooled[:, neuron_idx]  # (N,)
    
    # Top-k image indices
    topk_vals, topk_img_indices = neuron_acts.topk(min(top_k_images, len(neuron_acts)))
    topk_img_list = topk_img_indices.tolist()
    
    # Recompute spatial activations only for the top-k images
    has_spatial = features.ndim == 3
    if has_spatial and sae is not None:
        spatial_acts = compute_spatial_activations_for_images(
            sae, features, topk_img_list, device=device,
        )  # (top_k, S-1, d_sae)
    else:
        spatial_acts = None
    
    n_cols = min(top_k_images, 4)
    n_rows_per_img = 2 if spatial_acts is not None else 1
    n_img_rows = math.ceil(top_k_images / n_cols)
    n_rows = n_img_rows * n_rows_per_img
    
    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows + 1))
    fig.suptitle(
        f"SAE Neuron {neuron_idx} — Top {top_k_images} Activating Images",
        fontsize=14, fontweight='bold', y=0.98
    )
    
    for rank, (val, img_idx) in enumerate(zip(topk_vals, topk_img_indices)):
        img_idx = img_idx.item()
        dataset_idx = indices[img_idx]
        label = labels[img_idx]
        label_name = class_names[label] if label < len(class_names) else f"class_{label}"
        
        # Get raw image
        raw_img = get_raw_image_for_display(dataset, dataset_idx)
        
        row = (rank // n_cols) * n_rows_per_img
        col = rank % n_cols
        
        # Plot original image
        ax_img = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
        ax_img.imshow(raw_img)
        ax_img.set_title(f"#{rank+1} | {label_name}\nact={val:.3f}", fontsize=9)
        ax_img.axis("off")
        
        # Plot heatmap if we have spatial activations
        if spatial_acts is not None:
            spatial_act = spatial_acts[rank, :, neuron_idx]  # (S-1,)
            n_patches = spatial_act.shape[0]
            grid_size = int(math.sqrt(n_patches))
            
            if grid_size * grid_size == n_patches:
                heatmap = spatial_act.reshape(grid_size, grid_size).numpy()
            else:
                grid_size = int(math.ceil(math.sqrt(n_patches)))
                padded = torch.zeros(grid_size * grid_size)
                padded[:n_patches] = spatial_act
                heatmap = padded.reshape(grid_size, grid_size).numpy()
            
            ax_heat = fig.add_subplot(n_rows, n_cols, (row + 1) * n_cols + col + 1)
            ax_heat.imshow(raw_img)
            # Resize heatmap to image size
            from scipy.ndimage import zoom
            h, w = np.array(raw_img).shape[:2]
            heatmap_resized = zoom(heatmap, (h / grid_size, w / grid_size), order=1)
            ax_heat.imshow(heatmap_resized, alpha=0.5, cmap='hot', 
                          vmin=0, vmax=heatmap.max() + 1e-8)
            ax_heat.set_title(f"Patch activations", fontsize=8)
            ax_heat.axis("off")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_neuron_activation_histogram(
    neuron_idx: int,
    activations: torch.Tensor,
    labels: List[int],
    class_names: List[str],
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Plot histogram of activations for a neuron, colored by class."""
    acts = activations[:, neuron_idx].numpy()
    labels_arr = np.array(labels)
    
    # Skip if all NaN
    if np.all(np.isnan(acts)):
        print(f"  [WARN] Neuron {neuron_idx} has all-NaN activations, skipping histogram")
        return
    
    # Filter out NaN for plotting
    valid_mask = ~np.isnan(acts)
    acts_clean = acts[valid_mask]
    labels_arr = labels_arr[valid_mask]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall histogram
    ax1.hist(acts_clean, bins=50, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    ax1.set_xlabel("Activation")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Neuron {neuron_idx} — Activation Distribution")
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # Per-class mean activation
    unique_labels = np.unique(labels_arr)
    class_means = []
    class_label_names = []
    for lbl in unique_labels:
        mask = labels_arr == lbl
        mean_act = acts_clean[mask].mean()
        class_means.append(mean_act)
        name = class_names[lbl] if lbl < len(class_names) else f"class_{lbl}"
        class_label_names.append(name)
    
    # Sort by mean activation
    sorted_idx = np.argsort(class_means)[::-1]
    # Show top 20 classes max
    max_classes = min(20, len(sorted_idx))
    sorted_idx = sorted_idx[:max_classes]
    
    bars = ax2.barh(
        range(max_classes),
        [class_means[i] for i in sorted_idx],
        color='steelblue', edgecolor='black', linewidth=0.5
    )
    ax2.set_yticks(range(max_classes))
    ax2.set_yticklabels([class_label_names[i] for i in sorted_idx], fontsize=8)
    ax2.set_xlabel("Mean Activation")
    ax2.set_title(f"Neuron {neuron_idx} — Per-Class Mean Activation")
    ax2.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def create_neuron_dashboard(
    neuron_idx: int,
    activations_pooled: torch.Tensor,
    sae,
    features: torch.Tensor,
    dataset: Dataset,
    class_names: List[str],
    labels: List[int],
    indices: List[int],
    save_dir: str,
    device: str = "cuda",
    top_k_images: int = 8,
    show: bool = False,
):
    """Create a full dashboard for a single neuron."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Top activating images (spatial activations computed on-demand inside)
    plot_top_activating_images_for_neuron(
        neuron_idx=neuron_idx,
        activations_pooled=activations_pooled,
        sae=sae,
        features=features,
        dataset=dataset,
        class_names=class_names,
        labels=labels,
        indices=indices,
        device=device,
        top_k_images=top_k_images,
        save_path=os.path.join(save_dir, f"neuron_{neuron_idx:05d}_top_images.png"),
        show=show,
    )
    
    # Activation histogram
    plot_neuron_activation_histogram(
        neuron_idx=neuron_idx,
        activations=activations_pooled,
        labels=labels,
        class_names=class_names,
        save_path=os.path.join(save_dir, f"neuron_{neuron_idx:05d}_histogram.png"),
        show=show,
    )


# =============================================================================
# Comparison Visualizations (LoRA SAE vs Base SAE)
# =============================================================================

def plot_comparison_top_images(
    neuron_idx: int,
    lora_activations: torch.Tensor,
    base_activations: torch.Tensor,
    dataset: Dataset,
    class_names: List[str],
    labels: List[int],
    indices: List[int],
    top_k_images: int = 8,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Side-by-side top activating images for the same neuron in LoRA vs Base SAE.
    """
    n_cols = min(top_k_images, 4)
    n_img_rows = math.ceil(top_k_images / n_cols)
    n_rows = n_img_rows * 2  # top row: LoRA, bottom row: base

    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows + 1.5))
    fig.suptitle(
        f"Neuron {neuron_idx} — Top Activating Images\n"
        f"(Top: LoRA SAE, Bottom: Base SAE)",
        fontsize=13, fontweight='bold', y=0.99,
    )

    for sae_idx, (acts_tensor, sae_label) in enumerate([
        (lora_activations, "LoRA"),
        (base_activations, "Base"),
    ]):
        neuron_acts = acts_tensor[:, neuron_idx]
        # Handle all-NaN
        if torch.isnan(neuron_acts).all():
            continue
        topk_vals, topk_img = neuron_acts.topk(min(top_k_images, len(neuron_acts)))

        for rank, (val, img_idx) in enumerate(zip(topk_vals, topk_img)):
            img_idx_val = img_idx.item()
            dataset_idx = indices[img_idx_val]
            label = labels[img_idx_val]
            label_name = class_names[label] if label < len(class_names) else f"class_{label}"

            raw_img = get_raw_image_for_display(dataset, dataset_idx)

            row = sae_idx * n_img_rows + (rank // n_cols)
            col = rank % n_cols
            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
            ax.imshow(raw_img)
            ax.set_title(f"[{sae_label}] #{rank+1}\n{label_name}\nact={val:.3f}", fontsize=8)
            ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_comparison_histogram(
    neuron_idx: int,
    lora_activations: torch.Tensor,
    base_activations: torch.Tensor,
    labels: List[int],
    class_names: List[str],
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Overlaid activation histograms and per-class bar charts for LoRA vs Base SAE.
    """
    lora_acts = lora_activations[:, neuron_idx].numpy()
    base_acts = base_activations[:, neuron_idx].numpy()
    labels_arr = np.array(labels)

    # Clean NaN
    lora_valid = ~np.isnan(lora_acts)
    base_valid = ~np.isnan(base_acts)
    lora_clean = lora_acts[lora_valid]
    base_clean = base_acts[base_valid]

    if len(lora_clean) == 0 and len(base_clean) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # --- Overlaid histogram ---
    bins = np.linspace(
        min(lora_clean.min() if len(lora_clean) else 0, base_clean.min() if len(base_clean) else 0),
        max(lora_clean.max() if len(lora_clean) else 1, base_clean.max() if len(base_clean) else 1),
        50,
    )
    if len(lora_clean) > 0:
        ax1.hist(lora_clean, bins=bins, alpha=0.6, color='tab:blue', label='LoRA SAE', edgecolor='black', linewidth=0.3)
    if len(base_clean) > 0:
        ax1.hist(base_clean, bins=bins, alpha=0.6, color='tab:orange', label='Base SAE', edgecolor='black', linewidth=0.3)
    ax1.set_xlabel("Activation")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Neuron {neuron_idx} — Activation Distribution")
    ax1.legend()
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.4)

    # --- Per-class mean activation comparison ---
    unique_labels = np.unique(labels_arr)
    lora_means, base_means, names_list = [], [], []
    for lbl in unique_labels:
        mask = labels_arr == lbl
        name = class_names[lbl] if lbl < len(class_names) else f"class_{lbl}"
        lm = lora_acts[mask & lora_valid]
        bm = base_acts[mask & base_valid]
        lora_means.append(np.nanmean(lm) if len(lm) > 0 else 0.0)
        base_means.append(np.nanmean(bm) if len(bm) > 0 else 0.0)
        names_list.append(name)

    # Sort by LoRA mean activation descending, show top 20
    sorted_idx = np.argsort(lora_means)[::-1]
    max_cls = min(20, len(sorted_idx))
    sorted_idx = sorted_idx[:max_cls]

    y_pos = np.arange(max_cls)
    bar_h = 0.35
    ax2.barh(y_pos - bar_h/2, [lora_means[i] for i in sorted_idx],
             height=bar_h, color='tab:blue', label='LoRA SAE', edgecolor='black', linewidth=0.3)
    ax2.barh(y_pos + bar_h/2, [base_means[i] for i in sorted_idx],
             height=bar_h, color='tab:orange', label='Base SAE', edgecolor='black', linewidth=0.3)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([names_list[i] for i in sorted_idx], fontsize=7)
    ax2.set_xlabel("Mean Activation")
    ax2.set_title(f"Neuron {neuron_idx} — Per-Class Mean (LoRA vs Base)")
    ax2.legend(fontsize=8)
    ax2.invert_yaxis()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_global_comparison(
    lora_activations: torch.Tensor,
    base_activations: torch.Tensor,
    dataset_name: str,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Global comparison between LoRA SAE and Base SAE:
    - Sparsity (L0: mean number of active features per image)
    - Dead neuron fraction
    - Firing rate distribution
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{dataset_name.upper()} — LoRA SAE vs Base SAE (Global Stats)",
                 fontsize=14, fontweight='bold')

    for sae_idx, (acts, label, color) in enumerate([
        (lora_activations, "LoRA SAE", "tab:blue"),
        (base_activations, "Base SAE", "tab:orange"),
    ]):
        # Clean NaN for stats
        if torch.isnan(acts).all():
            continue

        # L0 per image
        l0 = (acts > 0).float().sum(dim=1)  # (N,)
        axes[0].hist(l0.numpy(), bins=50, alpha=0.6, color=color, label=f"{label} (mean={l0.mean():.0f})",
                     edgecolor='black', linewidth=0.3)

        # Firing rate per neuron
        firing_rate = (acts > 0).float().mean(dim=0)  # (d_sae,)
        dead_frac = (firing_rate == 0).float().mean().item()
        # Only plot non-zero firing rates on log scale
        active_fr = firing_rate[firing_rate > 0].numpy()
        if len(active_fr) > 0:
            axes[1].hist(np.log10(active_fr + 1e-10), bins=50, alpha=0.6, color=color,
                         label=f"{label} (dead={dead_frac:.1%})",
                         edgecolor='black', linewidth=0.3)

        # Mean activation per neuron
        mean_act = acts.mean(dim=0)
        valid = ~torch.isnan(mean_act)
        if valid.any():
            axes[2].hist(mean_act[valid].numpy(), bins=50, alpha=0.6, color=color,
                         label=label, edgecolor='black', linewidth=0.3)

    axes[0].set_xlabel("L0 (active features per image)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Sparsity (L0)")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("log10(firing rate)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Neuron Firing Rate Distribution")
    axes[1].legend(fontsize=8)

    axes[2].set_xlabel("Mean Activation")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Mean Activation per Neuron")
    axes[2].legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# Main Inference Pipeline
# =============================================================================

BASE_SAE_PATH = "/home/sunayana/Documents/Concept_LoRA/patchsae/data/sae_weight/base/out.pt"


def run_inference_for_dataset(
    dataset_name: str,
    checkpoint_root: str = "/home/sunayana/Documents/Concept_LoRA/patchsae/out/checkpoints",
    lora_weights_root: str = "/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16",
    output_root: str = "/home/sunayana/Documents/Concept_LoRA/patchsae/dams_results",
    summary_root: Optional[str] = None,
    model_name: str = "openai/clip-vit-base-patch16",
    base_sae_path: str = BASE_SAE_PATH,
    block_layer: Optional[int] = None,
    device: str = "cuda",
    top_k_neurons: int = 20,
    top_k_images: int = 8,
    max_images: int = 5000,
    subset_seed: int = 0,
    batch_size: int = 32,
    data_root: str = None,
    show_plots: bool = False,
    # --- DAMS parameters ---
    compute_dams_score: bool = True,
    dams_alpha: float = 0.5,
    dams_beta: float = 0.5,
    dams_css_saturation: float = 0.5,
    dams_top_features_fss: int = 0,
    dams_fss_min_support: Optional[int] = None,
    dams_only: bool = False,
    # --- SAE / LoRA overrides (for maple / masked-finetune variants) ---
    sae_subdir: Optional[str] = None,
    lora_subdir: Optional[str] = None,
    lora_seed: str = "seed1",
):
    """Full inference pipeline for a single dataset with LoRA-vs-Base comparison.

    dataset_name   — canonical dataset key (eurosat, caltech101, medmnist, dtd, …)
    sae_subdir     — checkpoint directory name under checkpoint_root (defaults to dataset_name)
    lora_subdir    — LoRA weights directory name under lora_weights_root (defaults to dataset_name)
    lora_seed      — seed subdirectory inside lora_subdir/16shots/ (default "seed1")
    """
    _sae_subdir  = sae_subdir  or dataset_name
    _lora_subdir = lora_subdir or dataset_name

    print(f"\n{'='*70}")
    print(f"  SAE INFERENCE: {_sae_subdir.upper()}  (dataset: {dataset_name})")
    print(f"{'='*70}\n")

    # --- Find SAE checkpoint ---
    ckpt_dir = os.path.join(checkpoint_root, _sae_subdir)
    sae_path = find_latest_sae_checkpoint(ckpt_dir)
    if sae_path is None:
        print(f"[ERROR] No SAE checkpoint found in {ckpt_dir}")
        return
    print(f"LoRA SAE checkpoint: {sae_path}")
    print(f"Base SAE checkpoint: {base_sae_path}")
    
    # --- Find LoRA weights ---
    lora_path = os.path.join(lora_weights_root, _lora_subdir, "16shots", lora_seed, "lora_weights.pt")
    if not os.path.isfile(lora_path):
        print(f"[WARN] LoRA weights not found: {lora_path}. Using base CLIP.")
        lora_path = None
    else:
        print(f"LoRA weights: {lora_path}")
    
    # --- Load LoRA CLIP model ---
    print("\nLoading LoRA CLIP model...")
    lora_model, processor = load_lora_clip_model(model_name, lora_path, device=device)
    
    # --- Load both SAEs ---
    print("Loading LoRA SAE...")
    lora_sae, lora_sae_cfg = load_sae_from_checkpoint(sae_path, device=device)
    
    has_base_sae = base_sae_path and os.path.isfile(base_sae_path)
    if has_base_sae:
        print("Loading Base SAE...")
        base_sae, base_sae_cfg = load_sae_from_checkpoint(base_sae_path, device=device)
    else:
        print(f"[WARN] Base SAE not found at {base_sae_path}, skipping comparison.")
        base_sae = None

    # --- Choose feature layers ---
    if block_layer is None:
        lora_block_layer = infer_sae_training_layer(lora_sae_cfg, sae_path)
        if lora_block_layer is None:
            lora_block_layer = -2
            print("[WARN] Could not infer LoRA SAE layer, defaulting to -2.")

        if base_sae is not None:
            base_block_layer = infer_sae_training_layer(base_sae_cfg, base_sae_path)
            if base_block_layer is None:
                base_block_layer = lora_block_layer
                print(f"[WARN] Could not infer Base SAE layer, using {base_block_layer}.")
        else:
            base_block_layer = None

        print(
            f"Using auto-selected feature layers: LoRA={lora_block_layer}"
            + (f", Base={base_block_layer}" if base_block_layer is not None else "")
        )
    else:
        lora_block_layer = block_layer
        base_block_layer = block_layer if base_sae is not None else None
        print(f"Using user-selected feature layer: {block_layer}")

    if base_block_layer is not None and base_block_layer != lora_block_layer:
        print(
            f"[INFO] Different feature layers in comparison: "
            f"LoRA={lora_block_layer}, Base={base_block_layer}"
        )
    
    # --- Load dataset ---
    print(f"\nLoading {dataset_name} dataset...")
    transform = get_clip_preprocess()
    dataset, class_names = load_dataset(dataset_name, transform=transform, data_root=data_root)
    
    # Limit dataset size for tractability
    if len(dataset) > max_images:
        total_images = len(dataset)
        print(
            f"  Using {max_images}/{total_images} images for analysis "
            f"(random subset, seed={subset_seed})"
        )
        rng = np.random.default_rng(subset_seed)
        subset_indices = rng.choice(total_images, size=max_images, replace=False).tolist()
        dataset = torch.utils.data.Subset(dataset, subset_indices)
    
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False,
    )
    
    # --- Extract features from LoRA CLIP ---
    print(f"\nExtracting LoRA CLIP features from layer {lora_block_layer}...")
    lora_features, labels, indices = extract_clip_features(
        lora_model, dataloader, block_layer=lora_block_layer, device=device,
    )
    print(f"  LoRA features shape: {lora_features.shape}")
    
    # --- Extract features from Base CLIP (no LoRA) ---
    # Needed for:
    #   1. Base SAE evaluation (base CLIP at base_block_layer)
    #   2. Domain SAE DAMS evaluation (base CLIP at lora_block_layer) — the domain SAEs
    #      were trained on base CLIP features (custom_clip_ckpt_path=None), NOT LoRA CLIP
    #      features. Using LoRA CLIP features at eval time introduces a spurious distribution
    #      shift that artificially lowers the domain SAE's EC and DAMS score.
    base_features = None
    base_features_domain_layer = None  # base CLIP at lora_block_layer, for domain SAE DAMS
    if base_sae is not None:
        print("\nLoading base CLIP model (no LoRA) for base SAE...")
        base_model, _ = load_lora_clip_model(model_name, lora_checkpoint_path=None, device=device)
        print(f"Extracting base CLIP features from layer {base_block_layer}...")
        base_features, _, _ = extract_clip_features(
            base_model, dataloader, block_layer=base_block_layer, device=device,
        )
        print(f"  Base features shape: {base_features.shape}")
        # Also extract at domain SAE's layer if it differs from base SAE's layer
        if lora_block_layer != base_block_layer:
            print(f"Extracting base CLIP features from layer {lora_block_layer} (for domain SAE DAMS)...")
            base_features_domain_layer, _, _ = extract_clip_features(
                base_model, dataloader, block_layer=lora_block_layer, device=device,
            )
            print(f"  Base features (domain layer) shape: {base_features_domain_layer.shape}")
        else:
            base_features_domain_layer = base_features
        del base_model
        torch.cuda.empty_cache()

    # --- Compute LoRA SAE activations (on LoRA CLIP features) — for visualization ---
    print("\nComputing LoRA SAE activations (pooled)...")
    lora_acts_pooled = compute_sae_activations(lora_sae, lora_features, device=device)
    print(f"  LoRA pooled activations: {lora_acts_pooled.shape}")

    # --- Compute LoRA SAE activations on base CLIP features — for DAMS ---
    # Domain SAEs were trained on base CLIP features; evaluate them on the same distribution.
    lora_acts_for_dams = lora_acts_pooled  # fallback if no base model was loaded
    if base_features_domain_layer is not None:
        print("Computing domain SAE activations on base CLIP features (for DAMS)...")
        lora_acts_for_dams = compute_sae_activations(lora_sae, base_features_domain_layer, device=device)
        print(f"  Domain SAE activations (base CLIP): {lora_acts_for_dams.shape}")

    # --- Compute Base SAE activations ---
    base_acts_pooled = None
    if base_sae is not None and base_features is not None:
        print("Computing Base SAE activations (pooled)...")
        base_acts_pooled = compute_sae_activations(base_sae, base_features, device=device)
        print(f"  Base pooled activations: {base_acts_pooled.shape}")
    
    # Free CLIP model from GPU — we still need SAEs for spatial heatmaps
    del lora_model
    torch.cuda.empty_cache()

    # --- Output directory (always needed for summary.json) ---
    viz_dir = os.path.join(output_root, dataset_name, f"layer{lora_block_layer}")
    os.makedirs(viz_dir, exist_ok=True)

    if not dams_only:
        # --- Find interesting neurons (from LoRA SAE) ---
        print("\nFinding top-activating neurons (LoRA SAE)...")
        top_neurons = find_top_activating_neurons(
            lora_acts_pooled, top_k=top_k_neurons,
        )

        lora_dir = os.path.join(viz_dir, "lora")
        compare_dir = os.path.join(viz_dir, "comparison")
        os.makedirs(lora_dir, exist_ok=True)
        if base_acts_pooled is not None:
            os.makedirs(compare_dir, exist_ok=True)

        viz_dataset = dataset

        # --- Generate per-neuron visualizations ---
        print(f"\nGenerating visualizations → {viz_dir}")

        for rank, neuron_idx in enumerate(top_neurons):
            print(f"\n  [{rank+1}/{len(top_neurons)}] Neuron {neuron_idx}")

            create_neuron_dashboard(
                neuron_idx=neuron_idx,
                activations_pooled=lora_acts_pooled,
                sae=lora_sae,
                features=lora_features,
                dataset=viz_dataset,
                class_names=class_names,
                labels=labels,
                indices=indices,
                save_dir=lora_dir,
                device=device,
                top_k_images=top_k_images,
                show=show_plots,
            )

            if base_acts_pooled is not None:
                plot_comparison_top_images(
                    neuron_idx=neuron_idx,
                    lora_activations=lora_acts_pooled,
                    base_activations=base_acts_pooled,
                    dataset=viz_dataset,
                    class_names=class_names,
                    labels=labels,
                    indices=indices,
                    top_k_images=top_k_images,
                    save_path=os.path.join(compare_dir, f"neuron_{neuron_idx:05d}_compare_images.png"),
                    show=show_plots,
                )
                plot_comparison_histogram(
                    neuron_idx=neuron_idx,
                    lora_activations=lora_acts_pooled,
                    base_activations=base_acts_pooled,
                    labels=labels,
                    class_names=class_names,
                    save_path=os.path.join(compare_dir, f"neuron_{neuron_idx:05d}_compare_hist.png"),
                    show=show_plots,
                )

        # --- Global comparison plot ---
        if base_acts_pooled is not None:
            print("\nGenerating global LoRA vs Base comparison...")
            plot_global_comparison(
                lora_activations=lora_acts_pooled,
                base_activations=base_acts_pooled,
                dataset_name=dataset_name,
                save_path=os.path.join(compare_dir, "global_comparison.png"),
                show=show_plots,
            )
    else:
        top_neurons = []
        print("\n[dams_only] Skipping neuron visualization.")
    
    # --- DAMS scoring ---
    dams_results: Dict[str, dict] = {}
    if compute_dams_score and _DAMS_AVAILABLE:
        print("\nComputing DAMS scores...")
        dams_kwargs = dict(
            labels=labels,
            num_classes=len(class_names),
            device=device,
            alpha=dams_alpha,
            beta=dams_beta,
            css_saturation=dams_css_saturation,
            top_features_fss=dams_top_features_fss,
            fss_min_support=dams_fss_min_support,
        )
        if base_acts_pooled is not None:
            # Use base CLIP features for domain SAE (matching its training distribution).
            # base_features_domain_layer is base CLIP at lora_block_layer; if that wasn't
            # extracted (e.g. layers are the same), fall back to lora_features.
            dams_features_a = base_features_domain_layer if base_features_domain_layer is not None else lora_features
            lora_result, base_result = compare_dams(
                sae_a=lora_sae,
                sae_b=base_sae,
                features_a=dams_features_a,
                features_b=base_features,
                label_a="Domain SAE",
                label_b="Base SAE",
                sae_cfg_a=lora_sae_cfg,
                sae_cfg_b=base_sae_cfg,
                feature_layer_a=lora_block_layer,
                feature_layer_b=base_block_layer,
                precomputed_a=lora_acts_for_dams,
                precomputed_b=base_acts_pooled,
                **dams_kwargs,
            )
            dams_results["lora"] = lora_result.to_dict()
            dams_results["base"] = base_result.to_dict()
        else:
            # No base SAE — base_features_domain_layer may not exist
            dams_feats = base_features_domain_layer if base_features_domain_layer is not None else lora_features
            lora_result = compute_dams(
                sae=lora_sae,
                features=dams_feats,
                precomputed_activations=lora_acts_for_dams,
                sae_cfg=lora_sae_cfg,
                feature_layer=lora_block_layer,
                **dams_kwargs,
            )
            dams_results["lora"] = lora_result.to_dict()
        print(f"\n  DAMS (LoRA SAE) = {dams_results['lora']['dams']:.4f}")
        if "base" in dams_results:
            print(f"  DAMS (Base SAE) = {dams_results['base']['dams']:.4f}")
    elif compute_dams_score and not _DAMS_AVAILABLE:
        print("\n[WARN] DAMS requested but dams_metric.py unavailable — skipping.")

    # --- Save summary ---
    summary = {
        "dataset": dataset_name,
        "lora_sae_checkpoint": sae_path,
        "base_sae_checkpoint": base_sae_path if has_base_sae else None,
        "lora_checkpoint": lora_path,
        "block_layer": lora_block_layer,
        "requested_block_layer": block_layer,
        "base_block_layer": base_block_layer,
        "subset_seed": subset_seed,
        "n_images": len(labels),
        "n_classes": len(class_names),
        "class_names": class_names,
        "top_neurons": top_neurons,
        "neuron_stats": {},
        "dams": dams_results,
    }
    
    # Per-neuron stats
    for neuron_idx in top_neurons:
        lora_acts_n = lora_acts_pooled[:, neuron_idx]
        entry = {
            "lora_mean_activation": float(torch.nanmean(lora_acts_n).item()),
            "lora_max_activation": float(lora_acts_n.max().item()),
            "lora_firing_frequency": float((lora_acts_n > 0).float().mean().item()),
        }
        if base_acts_pooled is not None:
            base_acts_n = base_acts_pooled[:, neuron_idx]
            entry["base_mean_activation"] = float(torch.nanmean(base_acts_n).item())
            entry["base_max_activation"] = float(base_acts_n.max().item())
            entry["base_firing_frequency"] = float((base_acts_n > 0).float().mean().item())
        summary["neuron_stats"][str(neuron_idx)] = entry
    
    # Global stats
    lora_l0 = (lora_acts_pooled > 0).float().sum(dim=1).mean().item()
    lora_dead = ((lora_acts_pooled > 0).float().mean(dim=0) == 0).float().mean().item()
    summary["global_stats"] = {
        "lora_mean_l0": lora_l0,
        "lora_dead_neuron_frac": lora_dead,
    }
    if base_acts_pooled is not None:
        base_l0 = (base_acts_pooled > 0).float().sum(dim=1).mean().item()
        base_dead = ((base_acts_pooled > 0).float().mean(dim=0) == 0).float().mean().item()
        summary["global_stats"]["base_mean_l0"] = base_l0
        summary["global_stats"]["base_dead_neuron_frac"] = base_dead
    
    summary_dir = viz_dir
    if summary_root is not None:
        summary_dir = os.path.join(summary_root, dataset_name, f"layer{lora_block_layer}")
        os.makedirs(summary_dir, exist_ok=True)

    summary_path = os.path.join(summary_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")
    
    # Free memory
    del lora_sae, base_sae, lora_features, base_features, lora_acts_pooled, base_acts_pooled
    torch.cuda.empty_cache()
    
    print(f"\n✓ {dataset_name} inference complete. Plots in: {viz_dir}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SAE Inference & Top-Activating Neuron Visualization"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=list(_DATASET_DATA_DIRS),
        help="Dataset to run inference on"
    )
    parser.add_argument("--all", action="store_true",
                        help="Run on all three original datasets (eurosat, caltech101, medmnist)")
    parser.add_argument("--all_saes", action="store_true",
                        help="Run DAMS for every SAE checkpoint in checkpoint_root "
                             "using the ALL_SAE_CONFIGS mapping")
    parser.add_argument(
        "--checkpoint_root", type=str,
        default="/home/sunayana/Documents/Concept_LoRA/patchsae/out/checkpoints",
    )
    parser.add_argument(
        "--lora_weights_root", type=str,
        default="/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16",
    )
    parser.add_argument(
        "--output_root", type=str,
        default="/home/sunayana/Documents/Concept_LoRA/patchsae/out/visualizations",
    )
    parser.add_argument(
        "--summary_root", type=str, default=None,
        help="Optional directory root for summary.json files. "
             "If omitted, summaries are saved under output_root.",
    )
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch16")
    parser.add_argument(
        "--base_sae_path", type=str,
        default="/home/sunayana/Documents/Concept_LoRA/patchsae/data/sae_weight/base/out.pt",
        help="Path to base SAE checkpoint (ImageNet) for comparison",
    )
    parser.add_argument("--no_base_comparison", action="store_true",
                        help="Skip base SAE comparison")
    parser.add_argument(
        "--block_layer", type=int, default=None,
        help="ViT layer to extract CLIP features from. Default: auto-detect from each SAE checkpoint.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top_k_neurons", type=int, default=20)
    parser.add_argument("--top_k_images", type=int, default=8)
    parser.add_argument("--max_images", type=int, default=5000)
    parser.add_argument("--subset_seed", type=int, default=0,
                        help="Random seed used when sampling max_images from large datasets")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    # DAMS arguments
    parser.add_argument("--dams_only", action="store_true",
                        help="Skip neuron visualizations — only compute and save DAMS scores")
    parser.add_argument("--no_dams", action="store_true",
                        help="Skip DAMS metric computation")
    parser.add_argument("--dams_alpha", type=float, default=0.5,
                        help="DAMS weight for CSS component (α, default 0.5)")
    parser.add_argument("--dams_beta", type=float, default=0.5,
                        help="DAMS weight for FSS component (β, default 0.5)")
    parser.add_argument("--dams_css_saturation", type=float, default=0.5,
                        help="CSS half-saturation constant κ (default 0.5)")
    parser.add_argument("--dams_top_features_fss", type=int, default=0,
                        help="Restrict FSS to top-k features by mean activation (0=all)")
    parser.add_argument("--dams_fss_min_support", type=int, default=None,
                        help="Minimum active-image support for FSS feature candidates. "
                             "Default: adaptive max(2, ceil(0.5% of dataset)).")
    
    args = parser.parse_args()
    
    if args.all_saes:
        # Build job list from ALL_SAE_CONFIGS, filtered to SAEs that exist on disk
        ckpt_root = args.checkpoint_root
        jobs = []  # list of (sae_subdir, dataset_name, lora_subdir, lora_seed)
        for sae_dir, (ds_name, lora_sub, lora_s) in ALL_SAE_CONFIGS.items():
            full_ckpt_dir = os.path.join(ckpt_root, sae_dir)
            if not os.path.isdir(full_ckpt_dir):
                print(f"[SKIP] checkpoint dir not found: {full_ckpt_dir}")
                continue
            jobs.append((sae_dir, ds_name, lora_sub, lora_s))
        if not jobs:
            print("[ERROR] No SAE checkpoints found under", ckpt_root)
            sys.exit(1)
        print(f"\nFound {len(jobs)} SAE(s) to evaluate:")
        for sae_dir, ds_name, lora_sub, lora_s in jobs:
            lora_info = f"{lora_sub}/{lora_s}" if lora_sub else "base-CLIP (no LoRA)"
            print(f"  {sae_dir:30s}  dataset={ds_name}  lora={lora_info}")
        print()
    elif args.all:
        jobs = [(ds, ds, ds, "seed1") for ds in ["eurosat", "caltech101", "medmnist"]]
    elif args.dataset:
        cfg = ALL_SAE_CONFIGS.get(args.dataset)
        if cfg:
            jobs = [(args.dataset, cfg[0], cfg[1], cfg[2] or "seed1")]
        else:
            jobs = [(args.dataset, args.dataset, args.dataset, "seed1")]
    else:
        parser.error("Specify --dataset, --all, or --all_saes")

    base_sae = None if args.no_base_comparison else args.base_sae_path

    for sae_dir, ds_name, lora_sub, lora_s in jobs:
        try:
            run_inference_for_dataset(
                dataset_name=ds_name,
                checkpoint_root=args.checkpoint_root,
                lora_weights_root=args.lora_weights_root,
                output_root=args.output_root,
                summary_root=args.summary_root,
                model_name=args.model_name,
                base_sae_path=base_sae,
                block_layer=args.block_layer,
                device=args.device,
                top_k_neurons=args.top_k_neurons,
                top_k_images=args.top_k_images,
                max_images=args.max_images,
                subset_seed=args.subset_seed,
                batch_size=args.batch_size,
                data_root=args.data_root,
                show_plots=args.show,
                compute_dams_score=not args.no_dams,
                dams_alpha=args.dams_alpha,
                dams_beta=args.dams_beta,
                dams_css_saturation=args.dams_css_saturation,
                dams_top_features_fss=args.dams_top_features_fss,
                dams_fss_min_support=args.dams_fss_min_support,
                dams_only=args.dams_only,
                sae_subdir=sae_dir,
                lora_subdir=lora_sub,
                lora_seed=lora_s or "seed1",
            )
        except Exception as e:
            print(f"\n[ERROR] {sae_dir} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 70)
    print("  ALL DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()