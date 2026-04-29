#!/usr/bin/env python3
"""
DnCBM-style SAE checkpoint evaluation on ViT-B/16.

For each dataset, this script:
  1. Discovers SAE checkpoints under out/checkpoints/{dataset}/.../final*/
  2. Prints discovered checkpoint names
  3. Evaluates four conditions per checkpoint:
       - ZS: zero-shot CLIP (no SAE)
       - LoRA-FT: LoRA CLIP (no SAE)
       - ZS + BaseSAE-1: zero-shot CLIP + discovered SAE
       - LoRA-FT + BaseSAE-1: LoRA CLIP + discovered SAE

Notes:
  - Backbone is fixed to OpenAI CLIP ViT-B/16.
  - "BaseSAE-1" refers to discovered SAE checkpoints.

Usage:
  python eval_dncbm_sae_lora_accuracy.py
  python eval_dncbm_sae_lora_accuracy.py --datasets eurosat medmnist cub2002011
  python eval_dncbm_sae_lora_accuracy.py --max_checkpoints 2 --save_json out/dncbm_sae_lora_acc.json
"""

import argparse
import gc
import glob
import json
import math
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from tqdm import tqdm

try:
    from src.sae_training.loaders import load_sae
    from src.models.templates.openai_imagenet_templates import openai_imagenet_template
except ImportError as e:
    print(f"[FATAL] {e}\nRun from the sae_vlm/ directory.")
    sys.exit(1)

try:
    import clip as openai_clip
except ImportError:
    print("[FATAL] OpenAI CLIP not found. Install with:\n"
          "  pip install git+https://github.com/openai/CLIP.git")
    sys.exit(1)


BACKBONE = "ViT-B/16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_ROOT = "out/checkpoints"
LORA_ROOT = "/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16"
DATA_ROOT = "/home/sunayana/Documents/Concept_LoRA/data"


MEDMNIST_CLASSES = [
    "adipose", "background", "debris", "lymphocytes", "mucus",
    "smooth muscle", "normal colon mucosa",
    "cancer-associated stroma", "colorectal adenocarcinoma epithelium",
]


DATASET_CFG = {
    "eurosat": {
        "data_dir": f"{DATA_ROOT}/eurosat/2750",
        "lora_path": f"{LORA_ROOT}/eurosat/16shots/seed1/lora_weights.pt",
        "class_names": None,
        "use_npz": False,
        "split": "all",
    },
    "caltech101": {
        "data_dir": f"{DATA_ROOT}/caltech-101",
        "lora_path": f"{LORA_ROOT}/caltech101/16shots/seed1/lora_weights.pt",
        "class_names": None,
        "use_npz": False,
        "split": "all",
        "exclude_classes": {"BACKGROUND_Google"},
    },
    "medmnist": {
        "data_dir": f"{DATA_ROOT}/pathmnist_imagefolder",
        "npz_path": f"{DATA_ROOT}/pathmnist.npz",
        "lora_path": f"{LORA_ROOT}/medmnist/16shots/seed1/lora_weights.pt",
        "class_names": MEDMNIST_CLASSES,
        "use_npz": True,
        "split": "test",
    },
    "cub2002011": {
        "data_dir": f"{DATA_ROOT}/cub2002011/test",
        "lora_path": f"{LORA_ROOT}/cub2002011/16shots/seed1/lora_weights.pt",
        "class_names": None,
        "use_npz": False,
        "split": "all",
    },
    "dtd": {
        "data_dir": f"{DATA_ROOT}/DTD/images",
        "lora_path": f"{LORA_ROOT}/dtd/16shots/seed1/lora_weights.pt",
        "class_names": None,
        "use_npz": False,
        "split": "all",
    },
    "ucf101": {
        "data_dir": f"{DATA_ROOT}/UCF101/UCF-101-midframes",
        "lora_path": f"{LORA_ROOT}/ucf101/16shots/seed1/lora_weights.pt",
        "class_names": None,
        "use_npz": False,
        "split": "all",
    },
    "fgvc": {
        "data_dir": f"{DATA_ROOT}/fgvc_imagefolder",
        "lora_path": f"{LORA_ROOT}/fgvc/16shots/seed1/lora_weights.pt",
        "class_names": None,
        "use_npz": False,
        "split": "all",
    },
    "oxford_pets": {
        "data_dir": f"{DATA_ROOT}/oxford_pets_imagefolder",
        "lora_path": f"{LORA_ROOT}/oxford_pets/16shots/seed1/lora_weights.pt",
        "class_names": None,
        "use_npz": False,
        "split": "all",
    },
}


def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def _lora_AB(layer_dict, proj_name):
    d = layer_dict
    if proj_name in d and isinstance(d[proj_name], dict):
        try:
            return d[proj_name]["w_lora_A"], d[proj_name]["w_lora_B"]
        except (KeyError, TypeError):
            pass
    try:
        return d[f"{proj_name}.w_lora_A"], d[f"{proj_name}.w_lora_B"]
    except KeyError:
        return None, None


def build_base_clip(device: str):
    model, preprocess = openai_clip.load(BACKBONE, device=device)
    model.eval()
    return model, preprocess


def build_lora_clip(lora_path: str, device: str):
    model, preprocess = openai_clip.load(BACKBONE, device=device)
    model.eval()

    lora_state = torch.load(lora_path, map_location=device)
    if "weights" not in lora_state:
        print("  [WARN] No 'weights' key in LoRA checkpoint; using base CLIP.")
        return model, preprocess

    layers_dict = lora_state["weights"]
    meta = lora_state["metadata"]
    scale = meta["alpha"] / meta["r"]
    print(f"  LoRA: rank={meta['r']}, alpha={meta['alpha']}, scale={scale:.6f}")

    with torch.no_grad():
        for i in range(12):
            ld = layers_dict.get(f"layer_{i}")
            if ld is None:
                continue
            w = model.transformer.resblocks[i].attn.in_proj_weight.data
            d = w.shape[1]
            for proj, off in [("q_proj", 0), ("k_proj", d), ("v_proj", 2 * d)]:
                A, B = _lora_AB(ld, proj)
                if A is None:
                    continue
                delta = scale * (B.float().to(device) @ A.float().to(device))
                w[off:off + d] += delta.to(w.dtype)

        for i in range(12, 24):
            ld = layers_dict.get(f"layer_{i}")
            if ld is None:
                continue
            w = model.visual.transformer.resblocks[i - 12].attn.in_proj_weight.data
            d = w.shape[1]
            for proj, off in [("q_proj", 0), ("k_proj", d), ("v_proj", 2 * d)]:
                A, B = _lora_AB(ld, proj)
                if A is None:
                    continue
                delta = scale * (B.float().to(device) @ A.float().to(device))
                w[off:off + d] += delta.to(w.dtype)

    print("  LoRA weights merged.")
    return model, preprocess


def _is_cuda_oom(err: Exception) -> bool:
    msg = str(err).lower()
    return "cuda" in msg and "out of memory" in msg


def load_models_with_fallback(lora_path: str, preferred_device: str):
    """
    Try loading base+LoRA CLIP on preferred_device.
    If CUDA OOM occurs, automatically retry on CPU.
    """
    run_device = preferred_device
    try:
        base_model, preprocess = build_base_clip(run_device)
        lora_model, _ = build_lora_clip(lora_path, run_device)
        return base_model, lora_model, preprocess, run_device
    except RuntimeError as e:
        if preferred_device == "cuda" and _is_cuda_oom(e):
            print("  [WARN] CUDA OOM while loading CLIP models; falling back to CPU.")
            flush()
            run_device = "cpu"
            base_model, preprocess = build_base_clip(run_device)
            lora_model, _ = build_lora_clip(lora_path, run_device)
            return base_model, lora_model, preprocess, run_device
        raise


def _find_imagefolder_root(root: str) -> str:
    subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not subdirs:
        raise FileNotFoundError(f"No sub-directories in {root}")

    for sd in subdirs[:3]:
        sd_path = os.path.join(root, sd)
        if any(f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
               for f in os.listdir(sd_path)):
            return root

    for sd in sorted(subdirs):
        sd_path = os.path.join(root, sd)
        inner = [d for d in os.listdir(sd_path) if os.path.isdir(os.path.join(sd_path, d))]
        if inner:
            try:
                return _find_imagefolder_root(sd_path)
            except (FileNotFoundError, PermissionError):
                continue

    raise FileNotFoundError(f"No image class directories found under {root}")


class FilteredImageFolder(Dataset):
    def __init__(self, root: str, transform, exclude_classes=None):
        img_root = _find_imagefolder_root(root)
        full_ds = datasets.ImageFolder(root=img_root, transform=transform)
        exclude = exclude_classes or set()

        keep_indices = [
            i for i, (_, lbl) in enumerate(full_ds.samples)
            if full_ds.classes[lbl] not in exclude
        ]
        kept_cls_names = sorted({full_ds.classes[full_ds.targets[i]] for i in keep_indices})
        old_to_new = {full_ds.class_to_idx[c]: new_i for new_i, c in enumerate(kept_cls_names)}

        self._dataset = full_ds
        self._indices = keep_indices
        self._label_map = old_to_new
        self.class_names = kept_cls_names

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        img, old_lbl = self._dataset[self._indices[idx]]
        return img, self._label_map[old_lbl]


class MedMNISTNpzDataset(Dataset):
    def __init__(self, npz_path: str, imagefolder_root: str, preprocess, split: str = "test"):
        data = np.load(npz_path)
        imgs_key = f"{split}_images"
        lbls_key = f"{split}_labels"
        if imgs_key not in data:
            raise KeyError(f"'{imgs_key}' not found in {npz_path}. Keys: {list(data.keys())}")

        self.imgs = data[imgs_key]
        self.labels = data[lbls_key].flatten().astype(int)
        self.preprocess = preprocess

        if_root = _find_imagefolder_root(imagefolder_root)
        if_ds = datasets.ImageFolder(root=if_root)
        if_map = {n.replace("_", " ").lower(): i for n, i in if_ds.class_to_idx.items()}

        self.label_map = {}
        for npz_idx, cls_name in enumerate(MEDMNIST_CLASSES):
            self.label_map[npz_idx] = if_map.get(cls_name.lower(), npz_idx)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.imgs[idx]).convert("RGB")
        img = self.preprocess(img)
        return img, self.label_map[int(self.labels[idx])]


def load_test_dataset(cfg_dict: dict, preprocess):
    if cfg_dict.get("use_npz", False):
        ds = MedMNISTNpzDataset(
            npz_path=cfg_dict["npz_path"],
            imagefolder_root=cfg_dict["data_dir"],
            preprocess=preprocess,
            split=cfg_dict.get("split", "test"),
        )
        if_root = _find_imagefolder_root(cfg_dict["data_dir"])
        if_ds = datasets.ImageFolder(root=if_root)
        class_names = [n.replace("_", " ") for n in if_ds.classes]
    else:
        ds = FilteredImageFolder(
            root=cfg_dict["data_dir"],
            transform=preprocess,
            exclude_classes=cfg_dict.get("exclude_classes"),
        )
        class_names = cfg_dict.get("class_names") or ds.class_names

    return ds, class_names


@torch.no_grad()
def get_text_features(model, class_names: list, device: str) -> torch.Tensor:
    n = len(class_names)
    mean_f = torch.zeros(n, 512, device=device)
    for tmpl in openai_imagenet_template:
        prompts = [tmpl(c) for c in class_names]
        tokens = openai_clip.tokenize(prompts, truncate=True).to(device)
        feats = model.encode_text(tokens).float()
        mean_f += l2_normalize(feats)
    mean_f /= len(openai_imagenet_template)
    return l2_normalize(mean_f)


class SAEForwardHook:
    def __init__(self, model, sae, sae_cfg):
        num_blocks = len(model.visual.transformer.resblocks)
        layer = sae_cfg.block_layer
        if layer < 0:
            layer = num_blocks + layer
        if not (0 <= layer < num_blocks):
            raise ValueError(
                f"block_layer={sae_cfg.block_layer} -> resblock[{layer}] out of range (n={num_blocks})"
            )

        self.sae = sae
        self.block = model.visual.transformer.resblocks[layer]
        self.handle = None
        self.layer = layer

    def _hook_fn(self, module, input, output):
        # output: [L, N, D] seq-first (OpenAI CLIP convention)
        # SAE was trained on CLS token only — only replace position 0
        orig_dtype = output.dtype
        cls = output[0].float()          # [N, D]
        sae_out, _, _ = self.sae(cls)    # [N, D]
        out = output.float().clone()
        out[0] = sae_out
        return out.to(orig_dtype)

    def register(self):
        self.handle = self.block.register_forward_hook(self._hook_fn)
        return self

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@torch.no_grad()
def compute_accuracy(model, text_features: torch.Tensor, loader: DataLoader, device: str) -> float:
    logit_scale = model.logit_scale.exp()
    tf = l2_normalize(text_features.float())

    correct = total = 0
    for imgs, labels in tqdm(loader, desc="    eval", leave=False):
        img_feat = l2_normalize(model.encode_image(imgs.to(device)).float())
        logits = logit_scale * (img_feat @ tf.t())
        preds = logits.argmax(dim=-1).cpu()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total


def discover_final_saes(sae_dir: str) -> list:
    pattern = os.path.join(sae_dir, "*/final*/*.pt")
    paths = sorted(glob.glob(pattern))

    results = []
    for p in paths:
        try:
            ckpt = torch.load(p, map_location="cpu")
            cfg = ckpt.get("cfg", ckpt.get("config"))
            layer = getattr(cfg, "block_layer", None) if not isinstance(cfg, dict) else cfg.get("block_layer")

            sd = ckpt.get("state_dict", ckpt)
            w = sd.get("W_enc") if isinstance(sd, dict) else None
            if w is not None and torch.isnan(w).any():
                print(f"  [SKIP] {p} -- NaN in W_enc")
                del ckpt
                gc.collect()
                continue

            run_id = p.split(os.sep)[-3]
            ckpt_name = os.path.basename(p)
            del ckpt
            gc.collect()
            results.append({
                "run_id": run_id,
                "layer": layer,
                "path": p,
                "ckpt_name": ckpt_name,
            })
        except Exception as e:
            print(f"  [SKIP] {p} -- {e}")

    return results


def evaluate_dataset(dataset_name: str, cfg_dict: dict, batch_size: int, num_workers: int,
                     max_checkpoints: int | None, device: str) -> dict:
    print(f"\n{'='*72}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'='*72}")

    out = {
        "dataset": dataset_name,
        "backbone": BACKBONE,
        "checkpoints": [],
    }

    lora_path = cfg_dict["lora_path"]
    if not os.path.isfile(lora_path):
        print(f"  [SKIP] LoRA checkpoint missing: {lora_path}")
        return out

    sae_dir = os.path.join(CKPT_ROOT, dataset_name)
    if not os.path.isdir(sae_dir):
        print(f"  [SKIP] SAE checkpoint directory not found: {sae_dir}")
        return out

    print("\n[1] Loading base + LoRA CLIP (ViT-B/16)")
    print(f"    LoRA weights: {lora_path}")
    base_model, lora_model, preprocess, run_device = load_models_with_fallback(lora_path, device)
    print(f"    Evaluation device: {run_device}")
    lora_model.eval()

    print("\n[3] Loading dataset")
    ds, class_names = load_test_dataset(cfg_dict, preprocess)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(run_device == "cuda"),
    )
    print(f"    {len(ds)} images | {len(class_names)} classes")

    print("\n[4] Building text features for base and LoRA models")
    base_text = get_text_features(base_model, class_names, run_device)
    lora_text = get_text_features(lora_model, class_names, run_device)

    print(f"\n[5] Discovering checkpoints in {sae_dir}/")
    ckpts = discover_final_saes(sae_dir)
    if max_checkpoints is not None:
        ckpts = ckpts[:max_checkpoints]

    if not ckpts:
        print("  None found.")
        del base_model, lora_model
        flush()
        return out

    print(f"  Found {len(ckpts)} checkpoint(s):")
    for i, c in enumerate(ckpts, start=1):
        print(f"    {i:02d}. run={c['run_id']:<12s} layer={c['layer']!s:<4s} {c['ckpt_name']}")

    print("\n[6] Evaluating each checkpoint")
    for i, c in enumerate(ckpts, start=1):
        print(f"\n  [{i}/{len(ckpts)}] {c['ckpt_name']} (run={c['run_id']}, layer={c['layer']})")
        try:
            # First compute accuracies WITHOUT SAE
            acc_zs = compute_accuracy(base_model, base_text, loader, run_device)
            acc_lora_ft = compute_accuracy(lora_model, lora_text, loader, run_device)

            # Then compute accuracies WITH SAE
            sae, sae_cfg = load_sae(c["path"], run_device)
            sae.eval()

            hook = SAEForwardHook(base_model, sae, sae_cfg).register()
            acc_zs_sae = compute_accuracy(base_model, base_text, loader, run_device)
            hook.remove()

            hook = SAEForwardHook(lora_model, sae, sae_cfg).register()
            acc_lora_ft_sae = compute_accuracy(lora_model, lora_text, loader, run_device)
            hook.remove()

            print(f"    ZS                : {acc_zs:.2f}%")
            print(f"    LoRA-FT           : {acc_lora_ft:.2f}%")
            print(f"    ZS + BaseSAE-1    : {acc_zs_sae:.2f}%")
            print(f"    LoRA-FT + BaseSAE-1 : {acc_lora_ft_sae:.2f}%")

            out["checkpoints"].append({
                "run_id": c["run_id"],
                "checkpoint_name": c["ckpt_name"],
                "checkpoint_path": c["path"],
                "sae_layer": c["layer"],
                "ZS": round(acc_zs, 4),
                "LoRA-FT": round(acc_lora_ft, 4),
                "ZS + BaseSAE-1": round(acc_zs_sae, 4),
                "LoRA-FT + BaseSAE-1": round(acc_lora_ft_sae, 4),
            })

            del sae
            flush()
        except Exception as e:
            print(f"    [ERROR] {e}")

    del base_model, lora_model
    flush()
    return out


def main():
    p = argparse.ArgumentParser(
        description="Discover SAE checkpoints and evaluate zero-shot+SAE and LoRA+SAE on ViT-B/16"
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_CFG.keys()),
        choices=list(DATASET_CFG.keys()),
        help="Datasets to evaluate",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--max_checkpoints",
        type=int,
        default=None,
        help="Limit number of discovered checkpoints per dataset",
    )
    p.add_argument(
        "--save_json",
        type=str,
        default="out/dncbm_sae_lora_accuracy.json",
        help="JSON output path",
    )
    args = p.parse_args()

    print(f"\n{'='*72}")
    print("DnCBM SAE Accuracy Evaluation")
    print(f"Device     : {DEVICE}")
    print(f"Backbone   : {BACKBONE}")
    print(f"Datasets   : {args.datasets}")
    print(f"CKPT root  : {CKPT_ROOT}")
    print(f"LoRA root  : {LORA_ROOT}")
    print(f"{'='*72}")

    all_results = []
    for ds_name in args.datasets:
        res = evaluate_dataset(
            dataset_name=ds_name,
            cfg_dict=DATASET_CFG[ds_name],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_checkpoints=args.max_checkpoints,
            device=DEVICE,
        )
        all_results.append(res)

    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    for res in all_results:
        print(f"\n{res['dataset'].upper()}")
        if not res["checkpoints"]:
            print("  (no successful checkpoint evals)")
            continue
        print(f"  {'Checkpoint':<45s} {'ZS':>7s} {'LoRA-FT':>9s} {'ZS+SAE':>9s} {'LoRA+SAE':>10s}")
        print(f"  {'-'*85}")
        for row in res["checkpoints"]:
            print(
                f"  {row['checkpoint_name']:<45s} "
                f"{row['ZS']:>6.2f}% "
                f"{row['LoRA-FT']:>8.2f}% "
                f"{row['ZS + BaseSAE-1']:>8.2f}% "
                f"{row['LoRA-FT + BaseSAE-1']:>9.2f}%"
            )

    out_path = args.save_json
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
