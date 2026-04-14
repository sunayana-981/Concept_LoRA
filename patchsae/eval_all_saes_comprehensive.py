#!/usr/bin/env python3
"""
Comprehensive accuracy evaluation for ALL SAEs across all depths.

Discovers every final-checkpoint SAE in out/checkpoints/, maps each to its
dataset and model type from the directory name, then runs zero-shot CLIP
classification accuracy with the SAE hooked into the vision encoder.

Output: out/all_sae_accuracy.json  (and printed table)

Run from the patchsae/ directory:
    python eval_all_saes_comprehensive.py
"""

import argparse, gc, glob, json, math, os, re, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

try:
    from tasks.utils import load_sae
    from src.models.templates.openai_imagenet_templates import openai_imagenet_template
except ImportError as e:
    print(f"[FATAL] {e}\nRun from the patchsae/ directory."); sys.exit(1)

try:
    import clip as openai_clip
except ImportError:
    print("[FATAL] pip install git+https://github.com/openai/CLIP.git"); sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
CKPT_ROOT       = "out/checkpoints"
BASE_SAE_PATH   = "data/sae_weight/base/out.pt"
LORA_ROOT       = "/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16"
DATA_ROOT       = "/home/sunayana/Documents/Concept_LoRA/data"
BACKBONE        = "ViT-B/16"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE      = 64

MEDMNIST_CLASSES = [
    "adipose", "background", "debris", "lymphocytes", "mucus",
    "smooth muscle", "normal colon mucosa",
    "cancer-associated stroma", "colorectal adenocarcinoma epithelium",
]

# ── Dataset configuration ─────────────────────────────────────────────────────
DATASET_CFG = {
    "eurosat": {
        "data_dir":  f"{DATA_ROOT}/eurosat/2750",
        "lora_path": f"{LORA_ROOT}/eurosat/16shots/seed1/lora_weights.pt",
        "use_npz":   False,
        "exclude_classes": set(),
    },
    "caltech101": {
        "data_dir":  f"{DATA_ROOT}/caltech-101",
        "lora_path": f"{LORA_ROOT}/caltech101/16shots/seed1/lora_weights.pt",
        "use_npz":   False,
        "exclude_classes": {"BACKGROUND_Google"},
    },
    "medmnist": {
        "data_dir":  f"{DATA_ROOT}/pathmnist_imagefolder",
        "npz_path":  f"{DATA_ROOT}/pathmnist.npz",
        "lora_path": f"{LORA_ROOT}/medmnist/16shots/seed1/lora_weights.pt",
        "use_npz":   True,
        "exclude_classes": set(),
    },
}

# Map checkpoint dir prefix → (dataset, model_type)
# model_type: "lora", "base_clip"
DIR_TO_DATASET = {
    "caltech101":            "caltech101",
    "caltech101_maple":      "caltech101",
    "eurosat":               "eurosat",
    "eurosat_maple":         "eurosat",
    "medmnist":              "medmnist",
    "medmnist_maple":        "medmnist",
    "masked_finetune":       "medmnist",
    "masked_finetune_lora":  "medmnist",
    "masked_finetune_maple": "medmnist",
    "9g0pkku9":              "medmnist",   # base-CLIP SAE, evaluate on medmnist
    "owdr2cw0":              "medmnist",
}

# Use base CLIP (no LoRA) for these dirs
USES_BASE_CLIP = {"9g0pkku9", "owdr2cw0"}


# ── Utilities ─────────────────────────────────────────────────────────────────

def flush():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def l2_norm(x):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)

def _find_imagefolder_root(path):
    """Walk into path until we find a directory that looks like ImageFolder root."""
    for root, dirs, files in os.walk(path):
        if dirs and all(os.path.isdir(os.path.join(root, d)) for d in dirs[:1]):
            # Check if subdirs contain images
            sample_dir = os.path.join(root, dirs[0])
            exts = {os.path.splitext(f)[1].lower()
                    for f in os.listdir(sample_dir)
                    if os.path.isfile(os.path.join(sample_dir, f))}
            if exts & {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                return root
    return path


# ── Dataset loaders ───────────────────────────────────────────────────────────

class FilteredImageFolder(Dataset):
    """ImageFolder with class exclusion and contiguous label remapping."""

    def __init__(self, root, transform, exclude_classes=None):
        img_root = _find_imagefolder_root(root)
        full_ds  = datasets.ImageFolder(root=img_root, transform=transform)
        exclude  = exclude_classes or set()

        keep_indices   = [i for i, (_, lbl) in enumerate(full_ds.samples)
                          if full_ds.classes[lbl] not in exclude]
        kept_cls_names = sorted({full_ds.classes[full_ds.targets[i]]
                                  for i in keep_indices})
        old_to_new     = {full_ds.class_to_idx[c]: new_i
                          for new_i, c in enumerate(kept_cls_names)}

        self._dataset   = full_ds
        self._indices   = keep_indices
        self._label_map = old_to_new
        self.class_names = [n.replace("_", " ") for n in kept_cls_names]

    def __len__(self):  return len(self._indices)

    def __getitem__(self, idx):
        img, old_lbl = self._dataset[self._indices[idx]]
        return img, self._label_map[old_lbl]


class MedMNISTNpzDataset(Dataset):
    """PathMNIST test split with label remapping to ImageFolder ordering."""

    def __init__(self, npz_path, imagefolder_root, preprocess, split="test"):
        data = np.load(npz_path)
        self.imgs   = data[f"{split}_images"]
        self.labels = data[f"{split}_labels"].flatten().astype(int)
        self.preprocess = preprocess

        if_root = _find_imagefolder_root(imagefolder_root)
        if_ds   = datasets.ImageFolder(root=if_root)
        if_map  = {n.replace("_", " ").lower(): i for n, i in if_ds.class_to_idx.items()}

        self.label_map  = {i: if_map.get(c.lower(), i)
                           for i, c in enumerate(MEDMNIST_CLASSES)}
        self.class_names = [n.replace("_", " ") for n in if_ds.classes]

    def __len__(self):  return len(self.labels)

    def __getitem__(self, i):
        img = self.preprocess(Image.fromarray(self.imgs[i]))
        return img, self.label_map[int(self.labels[i])]


def load_dataset(dataset_name, cfg, preprocess):
    if cfg["use_npz"]:
        ds = MedMNISTNpzDataset(
            cfg["npz_path"], cfg["data_dir"], preprocess, split="test")
        return ds, ds.class_names
    else:
        ds = FilteredImageFolder(
            cfg["data_dir"], preprocess, cfg.get("exclude_classes"))
        return ds, ds.class_names


# ── LoRA loading ──────────────────────────────────────────────────────────────

def _lora_AB(ld, proj):
    if proj in ld and isinstance(ld[proj], dict):
        try: return ld[proj]["w_lora_A"], ld[proj]["w_lora_B"]
        except: pass
    try: return ld[f"{proj}.w_lora_A"], ld[f"{proj}.w_lora_B"]
    except: return None, None

def build_lora_clip(lora_path, device):
    model, preprocess = openai_clip.load(BACKBONE, device=device)
    state = torch.load(lora_path, map_location=device, weights_only=False)
    if "weights" not in state:
        model.load_state_dict(state)
        return model, preprocess
    layers, meta = state["weights"], state["metadata"]
    scale = meta["alpha"] / math.sqrt(meta["r"])
    with torch.no_grad():
        for i in range(12):
            ld = layers.get(f"layer_{i+12}", {})
            if not ld: continue
            blk = model.visual.transformer.resblocks[i]
            w   = blk.attn.in_proj_weight.data
            d   = w.shape[1]
            for proj, off in [("q_proj", 0), ("k_proj", d), ("v_proj", 2*d)]:
                A, B = _lora_AB(ld, proj)
                if A is None: continue
                delta = scale * (B.float().to(device) @ A.float().to(device))
                w[off:off+d] += delta.to(w.dtype)
    return model, preprocess

def build_base_clip(device):
    return openai_clip.load(BACKBONE, device=device)


# ── Text features ─────────────────────────────────────────────────────────────

@torch.no_grad()
def get_text_features(model, class_names, device):
    n = len(class_names)
    mean_f = torch.zeros(n, 512, device=device)
    for tmpl in openai_imagenet_template:
        prompts = [tmpl(c) for c in class_names]
        tokens  = openai_clip.tokenize(prompts, truncate=True).to(device)
        feats   = model.encode_text(tokens).float()
        mean_f += l2_norm(feats)
    mean_f /= len(openai_imagenet_template)
    return l2_norm(mean_f)


# ── Accuracy computation ──────────────────────────────────────────────────────

@torch.no_grad()
def compute_accuracy(model, text_feat, loader, device):
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        vis    = l2_norm(model.encode_image(imgs).float())
        logits = 100.0 * vis @ text_feat.T
        pred   = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


# ── SAE hook ──────────────────────────────────────────────────────────────────

class SAEHook:
    """
    Inserts SAE into a visual residual block's output.
    OpenAI CLIP resblock: [seq, batch, d_model] → transpose → [batch, seq, d_model]
    """
    def __init__(self, model, sae, sae_cfg, device):
        self.sae    = sae.to(device)
        self.device = device
        n = len(model.visual.transformer.resblocks)
        layer = sae_cfg.block_layer if sae_cfg.block_layer >= 0 else n + sae_cfg.block_layer
        assert 0 <= layer < n, f"layer {layer} out of range {n}"
        self.block  = model.visual.transformer.resblocks[layer]
        self.handle = None
        print(f"  [SAEHook] resblock[{layer}] (block_layer={sae_cfg.block_layer})")

    def _hook_fn(self, module, input, output):
        orig_dtype = output.dtype
        # OpenAI CLIP: [seq, batch, d_model] → need [batch, seq, d_model]
        act_bsd = output.transpose(0, 1).float()
        with torch.no_grad():
            result = self.sae(act_bsd)
            # sae returns (sae_out, feature_acts, ...) or just sae_out
            sae_out = result[0] if isinstance(result, (tuple, list)) else result
        # Back to [seq, batch, d_model]
        return sae_out.transpose(0, 1).to(orig_dtype)

    def register(self):
        self.handle = self.block.register_forward_hook(self._hook_fn)
        return self

    def remove(self):
        if self.handle:
            self.handle.remove()


# ── SAE discovery ─────────────────────────────────────────────────────────────

def discover_all_saes(ckpt_root):
    """Return list of (dir_name, run_id, layer, sae_path) for final checkpoints."""
    results = []
    # depth-2: out/checkpoints/{dir}/{run}/final*/*.pt
    for p in sorted(glob.glob(os.path.join(ckpt_root, "*/*/final*/*.pt"))):
        rel   = p[len(ckpt_root)+1:]
        parts = rel.split("/")
        dir_name = parts[0]
        run_id   = parts[1]
        m = re.search(r"_(-\d+)_resid", parts[-1])
        layer = int(m.group(1)) if m else None
        results.append((dir_name, run_id, layer, p))

    # depth-1: out/checkpoints/{run}/final*/*.pt  (standalone run dirs)
    for p in sorted(glob.glob(os.path.join(ckpt_root, "*/final*/*.pt"))):
        rel   = p[len(ckpt_root)+1:]
        parts = rel.split("/")
        if len(parts) == 3:   # dir/subdir/file.pt
            dir_name = parts[0]
            m = re.search(r"_(-\d+)_resid", parts[-1])
            layer = int(m.group(1)) if m else None
            results.append((dir_name, dir_name, layer, p))

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_json",  default="out/all_sae_accuracy.json")
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    all_saes = discover_all_saes(CKPT_ROOT)
    print(f"\nFound {len(all_saes)} SAE checkpoints\n")

    # Group SAEs by (dataset, model_type) so we load the model once per group
    groups = {}
    for dir_name, run_id, layer, sae_path in all_saes:
        dataset = DIR_TO_DATASET.get(dir_name)
        if dataset is None:
            print(f"  [SKIP] Unknown dir: {dir_name}")
            continue
        model_type = "base_clip" if dir_name in USES_BASE_CLIP else "lora"
        key = (dataset, model_type)
        groups.setdefault(key, []).append((dir_name, run_id, layer, sae_path))

    all_results = []

    for (dataset, model_type), saes in sorted(groups.items()):
        print(f"\n{'═'*70}")
        print(f"  DATASET: {dataset.upper()}  |  MODEL: {model_type}")
        print(f"{'═'*70}")

        cfg = DATASET_CFG[dataset]

        # ── Load model ────────────────────────────────────────────────────────
        if model_type == "lora":
            lora_path = cfg["lora_path"]
            if not os.path.isfile(lora_path):
                print(f"  [SKIP] LoRA not found: {lora_path}"); continue
            model, preprocess = build_lora_clip(lora_path, DEVICE)
            print(f"  LoRA merged from: {lora_path}")
        else:
            model, preprocess = build_base_clip(DEVICE)
            print("  Base CLIP (no LoRA)")
        model.eval()

        # ── Load dataset ──────────────────────────────────────────────────────
        try:
            ds, class_names = load_dataset(dataset, cfg, preprocess)
        except Exception as e:
            print(f"  [ERROR] Dataset load failed: {e}"); del model; flush(); continue

        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=(DEVICE == "cuda"))
        print(f"  {len(ds)} images | {len(class_names)} classes")

        # ── Text features ─────────────────────────────────────────────────────
        text_feat = get_text_features(model, class_names, DEVICE)

        # ── Baseline (no SAE) ─────────────────────────────────────────────────
        acc_base = compute_accuracy(model, text_feat, loader, DEVICE)
        print(f"\n  Baseline ({model_type}, no SAE): {acc_base:.2f}%")

        entry = {
            "dataset": dataset, "model_type": model_type,
            "conditions": [{
                "name": f"{model_type} (no SAE)",
                "accuracy": round(acc_base, 4),
                "sae_path": None, "sae_layer": None,
                "run_id": None, "dir": None,
            }]
        }

        # ── ImageNet base SAE ─────────────────────────────────────────────────
        if os.path.isfile(BASE_SAE_PATH):
            try:
                sae, sae_cfg = load_sae(BASE_SAE_PATH, DEVICE)
                sae.eval()
                hook = SAEHook(model, sae, sae_cfg, DEVICE).register()
                acc  = compute_accuracy(model, text_feat, loader, DEVICE)
                hook.remove()
                del sae; flush()
                print(f"  ImageNet base SAE (layer={sae_cfg.block_layer}): {acc:.2f}%")
                entry["conditions"].append({
                    "name": "ImageNet base SAE",
                    "accuracy": round(acc, 4),
                    "sae_path": BASE_SAE_PATH,
                    "sae_layer": sae_cfg.block_layer,
                    "run_id": "imagenet_base", "dir": "base",
                })
            except Exception as e:
                print(f"  [ERROR] ImageNet base SAE: {e}")

        # ── Per-SAE evaluation ────────────────────────────────────────────────
        for dir_name, run_id, layer, sae_path in saes:
            label = f"{dir_name} run={run_id} layer={layer}"
            print(f"\n  ↳ {label}")
            try:
                sae, sae_cfg = load_sae(sae_path, DEVICE)
                sae.eval()
                hook = SAEHook(model, sae, sae_cfg, DEVICE).register()
                acc  = compute_accuracy(model, text_feat, loader, DEVICE)
                hook.remove()
                del sae; flush()
                print(f"    → {acc:.2f}%")
                entry["conditions"].append({
                    "name": label,
                    "accuracy": round(acc, 4),
                    "sae_path": sae_path,
                    "sae_layer": layer,
                    "run_id": run_id, "dir": dir_name,
                })
            except Exception as e:
                print(f"    [ERROR] {e}")
                import traceback; traceback.print_exc()

        all_results.append(entry)
        del model; flush()

    # ── Print summary table ────────────────────────────────────────────────────
    print(f"\n\n{'═'*80}")
    print("  SUMMARY — Zero-Shot Accuracy (%) with SAE Hooks")
    print(f"{'═'*80}\n")
    for res in all_results:
        print(f"\n  [{res['dataset'].upper()} / {res['model_type']}]")
        print(f"  {'─'*72}")
        for cond in res["conditions"]:
            ltag = f"  [L{cond['sae_layer']}]" if cond["sae_layer"] is not None else ""
            print(f"  {cond['name']:<58s}  {cond['accuracy']:6.2f}%{ltag}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
    with open(args.save_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved → {args.save_json}")


if __name__ == "__main__":
    main()
