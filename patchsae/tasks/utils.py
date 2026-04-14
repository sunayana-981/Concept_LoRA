import json
import os
from collections import defaultdict
from typing import Dict, Tuple

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm

from src.models.utils import get_adapted_clip, get_base_clip
from src.sae_training.config import Config
from src.sae_training.hooked_vit import HookedVisionTransformer
from src.sae_training.sparse_autoencoder import SparseAutoencoder

# ----------------------------------------------------
# DATASET METADATA
# ----------------------------------------------------
DATASET_INFO = {
    "imagenet": {
        "path": "evanarlian/imagenet_1k_resized_256",
        "split": "train",
        "trust_remote_code": True,
    },
    "imagenet-sketch": {
        "path": "clip-benchmark/wds_imagenet_sketch",
        "split": "train",
        "trust_remote_code": True,
    },
    "oxford_flowers": {
        "path": "nelorth/oxford-flowers",
        "split": "train",
        "trust_remote_code": True,
    },
    "caltech101": {
        "path": "HuggingFaceM4/Caltech-101",
        "split": "train",
        "name": "with_background_category",
        "trust_remote_code": True,
    },
    "eurosat": {
        "path": "imagefolder",
        "data_dir": "/home/sunayana/Documents/Concept_LoRA/data/eurosat/2750",
        "split": "train",
        "trust_remote_code": True,
    },
    "medmnist": {
        "path": "imagefolder",
        "data_dir": "/home/sunayana/Documents/Concept_LoRA/data/pathmnist_imagefolder",
        "split": "train",
        "trust_remote_code": True,
    },
    "dtd": {
        "path": "imagefolder",
        "data_dir": "/home/sunayana/Documents/Concept_LoRA/data/DTD/images",
        "split": "train",
        "trust_remote_code": True,
    },
    "ucf101": {
        "path": "imagefolder",
        "data_dir": "/home/sunayana/Documents/Concept_LoRA/data/UCF101/UCF-101-midframes",
        "split": "train",
        "trust_remote_code": True,
    },
    "cub2002011": {
        "path": "imagefolder",
        "data_dir": "/home/sunayana/Documents/Concept_LoRA/data/cub2002011/train",
        "split": "train",
        "trust_remote_code": True,
    },
    "oxford_pets": {
        "path": "imagefolder",
        "data_dir": "/home/sunayana/Documents/Concept_LoRA/data/oxford_pets_imagefolder",
        "split": "train",
        "trust_remote_code": True,
    },
    "fgvc": {
        "path": "imagefolder",
        "data_dir": "/home/sunayana/Documents/Concept_LoRA/data/fgvc_imagefolder",
        "split": "train",
        "trust_remote_code": True,
    },
}

SAE_DIM = 49152


# ----------------------------------------------------
# SAE LOADING
# ----------------------------------------------------
def load_sae(sae_path: str, device: str) -> tuple[SparseAutoencoder, Config]:
    checkpoint = torch.load(sae_path, map_location="cpu")

    cfg = Config(checkpoint.get("cfg", checkpoint.get("config")))
    sae = SparseAutoencoder(cfg, device)
    sae.load_state_dict(checkpoint["state_dict"])
    sae.eval().to(device)

    return sae, cfg


# ----------------------------------------------------
# VIT LOADING
# ----------------------------------------------------
def load_hooked_vit(
    cfg: Config,
    vit_type: str,
    backbone: str,
    device: str,
    model_path: str = None,
    config_path: str = None,
    classnames: list[str] = None,
) -> HookedVisionTransformer:

    if vit_type == "base":
        model, processor = get_base_clip(backbone)
    else:
        model, processor = get_adapted_clip(
            cfg, vit_type, model_path, config_path, backbone, classnames
        )

    return HookedVisionTransformer(model, processor, device=device)


def get_sae_and_vit(
    sae_path: str,
    vit_type: str,
    device: str,
    backbone: str,
    model_path: str = None,
    config_path: str = None,
    classnames: list[str] = None,
) -> tuple[SparseAutoencoder, HookedVisionTransformer, Config]:

    sae, cfg = load_sae(sae_path, device)
    vit = load_hooked_vit(
        cfg, vit_type, backbone, device, model_path, config_path, classnames
    )
    return sae, vit, cfg


# ----------------------------------------------------
# LABEL + CLASSNAME HANDLING (FULLY PATCHED)
# ----------------------------------------------------
LABEL_KEYS = [
    "label",
    "fine_label",
    "coarse_label",
    "labels",
    "class_label",
    "class_id",
    "target",
    "y",
]


def detect_label_key(dataset: Dataset):
    """Find label column automatically."""
    # 1. First search for common names
    for key in LABEL_KEYS:
        if key in dataset.features:
            return key

    # 2. Fallback: search for ClassLabel feature
    for key, feat in dataset.features.items():
        if hasattr(feat, "names"):  # HuggingFace ClassLabel
            return key

    raise ValueError(
        f"No valid label column found. Available keys: {list(dataset.features.keys())}"
    )


def load_and_organize_dataset(dataset_name: str) -> Tuple[list, Dict]:
    dataset = load_dataset(**DATASET_INFO[dataset_name])

    # Ensure we always use the same split access (train/test)
    if isinstance(dataset, dict):
        dataset = dataset["train"]

    label_key = detect_label_key(dataset)
    classnames = get_classnames(dataset_name, dataset)

    data_by_class = defaultdict(list)
    for item in tqdm(dataset):
        label = item[label_key]
        classname = classnames[label]
        data_by_class[classname].append(item)

    return classnames, data_by_class


def get_classnames(
    dataset_name: str,
    dataset: Dataset = None,
    data_root: str = "./configs/classnames",
) -> list[str]:

    # If dataset has ClassLabel feature → use it directly
    if dataset is not None:
        for key in dataset.features:
            feat = dataset.features[key]
            if hasattr(feat, "names"):
                return feat.names  # perfect case

    # Try loading external files
    filename = os.path.join(data_root, f"{dataset_name}_classnames")
    txt_path, json_path = filename + ".txt", filename + ".json"

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)

    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            return [line.strip() for line in f.readlines()]

    raise ValueError(
        f"Could not determine classnames for {dataset_name}. "
        f"No ClassLabel feature and no {txt_path} or {json_path}"
    )


# ----------------------------------------------------
# ACTIVATIONS + PROCESSING
# ----------------------------------------------------
def setup_save_directory(
    root_dir: str, save_name: str, sae_path: str, vit_type: str, dataset_name: str
) -> str:
    sae_run_name = sae_path.split("/")[-2]
    save_directory = (
        f"{root_dir}/{save_name}/sae_{sae_run_name}/{vit_type}/{dataset_name}"
    )
    os.makedirs(save_directory, exist_ok=True)
    return save_directory


def get_sae_activations(model_activations: torch.Tensor, sae: SparseAutoencoder):
    _, cache = sae.run_with_cache(model_activations)
    acts = cache["hook_hidden_post"]
    if acts.ndim > 2:
        acts = acts.mean(dim=1)
    return acts


def get_sae_latents(sae: SparseAutoencoder, activations: torch.Tensor) -> torch.Tensor:
    """Extract sparse SAE latents from activations [B, seq, d_model]."""
    # Preferred API on newer SAE implementations.
    if hasattr(sae, "encode"):
        return sae.encode(activations)

    # Common forward signature: (reconstruction, latents, ...)
    out = sae(activations)
    if isinstance(out, tuple) and len(out) >= 2:
        candidate = out[1]
        if candidate.shape[-1] >= activations.shape[-1]:
            return candidate

    # Fallback for checkpoints exposing encoder weights directly.
    if hasattr(sae, "W_enc") and hasattr(sae, "b_enc"):
        x = activations
        if hasattr(sae, "b_dec"):
            x = x - sae.b_dec
        z = x @ sae.W_enc + sae.b_enc
        return torch.nn.functional.relu(z)

    raise RuntimeError(
        "Cannot extract SAE latents: no encode(), no forward latents, and no W_enc/b_enc."
    )


def process_batch(vit, batch_data, device):
    images = [d["image"] for d in batch_data]
    inputs = vit.processor(
        images=images, text="", return_tensors="pt", padding=True
    ).to(device)
    return inputs


# ----------------------------------------------------
# MAX ACTIVATIONS LOADING
# ----------------------------------------------------
def get_max_acts_and_images(
    datasets: dict, feat_data_root: str, sae_runname: str, vit_name: str
):
    max_act_imgs, mean_acts = {}, {}

    for name in datasets:
        base = os.path.join(feat_data_root, f"{sae_runname}/{vit_name}/{name}")

        max_act_imgs[name] = torch.load(
            os.path.join(base, "max_activating_image_indices.pt"), map_location="cpu"
        ).to(torch.int32)

        mean_acts[name] = torch.load(
            os.path.join(base, "sae_mean_acts.pt"), map_location="cpu"
        ).numpy()

    return max_act_imgs, mean_acts


# ----------------------------------------------------
# MULTI-DATASET LOADING
# ----------------------------------------------------
def load_datasets(include_imagenet: bool = False, seed: int = 1):
    out = {}

    if include_imagenet:
        out["imagenet"] = load_dataset(
            "evanarlian/imagenet_1k_resized_256", split="train"
        ).shuffle(seed=seed)

    out.update(
        {
            "imagenet-sketch": load_dataset(
                "clip-benchmark/wds_imagenet_sketch", split="test"
            ).shuffle(seed=seed),
            "caltech101": load_dataset(
                "HuggingFaceM4/Caltech-101",
                "with_background_category",
                split="train",
            ).shuffle(seed=seed)
        }
    )
    return out


def get_all_classnames(datasets, data_root):
    names = {}
    for name, ds in datasets.items():
        names[name] = get_classnames(name, ds, data_root)

    return names
    
