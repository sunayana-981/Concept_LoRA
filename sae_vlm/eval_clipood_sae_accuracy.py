#!/usr/bin/env python3
"""
CLIPood CLIP Accuracy Evaluation on PACS (leave-one-out protocol).

For each PACS split (held-out domain):
  1. CLIPood model alone            — zero-shot top-1 accuracy on target domain
  2. CLIPood + base (ImageNet) SAE  — data/sae_weight/base/out.pt hooked in
  3. CLIPood + PACS SAE             — checkpoints in out/checkpoints/clipood_pacs_{split}/

Mirrors eval_lora_sae_accuracy.py and eval_maple_sae_accuracy.py structure.

Usage (run from sae_vlm/ directory):
    python eval_clipood_sae_accuracy.py
    python eval_clipood_sae_accuracy.py --splits no_photo no_art
    python eval_clipood_sae_accuracy.py --save_json out/clipood_sae_acc.json
"""

import argparse
import gc
import glob
import json
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast

try:
    from src.sae_training.loaders import load_sae
except ImportError as e:
    print(f"[FATAL] {e}\nRun from the sae_vlm/ directory.")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
BACKBONE      = "openai/clip-vit-base-patch16"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
BASE_SAE_PATH = "data/sae_weight/base/out.pt"
CKPT_ROOT     = "out/checkpoints"
CLIPOOD_ROOT  = "out/clipood_pacs"
PACS_ROOT     = "/home/sunayana/Documents/Concept_LoRA/data/PACS"

PACS_CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]

# Map: split_name → (held_out_domain, source_domains)
PACS_SPLITS = {
    "no_photo":       ("photo",       ["art_painting", "cartoon", "sketch"]),
    "no_art":         ("art_painting",["cartoon", "photo", "sketch"]),
    "no_cartoon":     ("cartoon",     ["art_painting", "photo", "sketch"]),
    "no_sketch":      ("sketch",      ["art_painting", "cartoon", "photo"]),
}

# CLIP normalization
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


# ── Utilities ─────────────────────────────────────────────────────────────────

def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)


# ── Model loading ─────────────────────────────────────────────────────────────

def build_clipood_model(clipood_ckpt_path: str, device: str):
    """Load HF CLIPModel and apply CLIPood fine-tuned state dict."""
    model     = CLIPModel.from_pretrained(BACKBONE).to(device).eval()
    processor = CLIPProcessor.from_pretrained(BACKBONE)
    tokenizer = CLIPTokenizerFast.from_pretrained(BACKBONE)

    if clipood_ckpt_path and os.path.isfile(clipood_ckpt_path):
        state = torch.load(clipood_ckpt_path, map_location=device, weights_only=False)
        state_dict = state.get("state_dict", state)
        model.load_state_dict(state_dict, strict=False)
        print(f"  CLIPood weights loaded from {os.path.basename(clipood_ckpt_path)}")
    else:
        print(f"  [WARN] No CLIPood checkpoint found at {clipood_ckpt_path} — using base CLIP")

    return model, processor, tokenizer


# ── Dataset loading ───────────────────────────────────────────────────────────

def build_pacs_target_loader(pacs_root: str, domain: str,
                             batch_size: int, num_workers: int):
    """Load a single PACS domain as the target-domain test set."""
    domain_dir = os.path.join(pacs_root, domain)
    if not os.path.isdir(domain_dir):
        raise FileNotFoundError(f"PACS domain not found: {domain_dir}")

    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
    ])
    ds = datasets.ImageFolder(domain_dir, transform=tfm)
    # ImageFolder sorts classes alphabetically → matches PACS_CLASSES order
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(DEVICE == "cuda"))
    return loader, len(ds), ds.classes


# ── Text features ─────────────────────────────────────────────────────────────

@torch.no_grad()
def get_text_features(model, tokenizer, class_names: list, device: str) -> torch.Tensor:
    """Single-template text features: 'a photo of a {class}'."""
    prompts = [f"a photo of a {c}" for c in class_names]
    inputs  = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    feats   = model.get_text_features(**inputs).float()
    return l2_normalize(feats)   # [C, d]


# ── SAE hook (HF CLIP) ────────────────────────────────────────────────────────

class HFCLIPSAEHook:
    """
    Registers a forward hook on an HF CLIPModel vision encoder layer that
    replaces the layer's output hidden states with the SAE reconstruction.

    HF CLIP encoder layer output: tuple where [0] is [batch, seq, d_model].
    """

    def __init__(self, model: CLIPModel, sae, sae_cfg, device: str):
        n_layers  = model.config.vision_config.num_hidden_layers  # 12
        layer_idx = sae_cfg.block_layer
        if layer_idx < 0:
            layer_idx = n_layers + layer_idx

        self.sae     = sae
        self.device  = device
        self.layer   = model.vision_model.encoder.layers[layer_idx]
        self.handle  = None
        self._logged = False
        print(f"  [HFCLIPSAEHook] encoder.layers[{layer_idx}] "
              f"(block_layer={sae_cfg.block_layer}, d_sae={sae.d_sae})")

    def _hook_fn(self, module, input, output):
        # output is a tuple; [0] is hidden states [B, T, d]
        hs        = output[0].float()
        orig_dtype = output[0].dtype

        sae_out, _, _ = self.sae(hs)

        if not self._logged:
            self._logged = True
            mse = (hs - sae_out).pow(2).mean().item()
            cos = F.cosine_similarity(
                hs.reshape(-1, hs.shape[-1]),
                sae_out.reshape(-1, sae_out.shape[-1]),
                dim=-1,
            ).mean().item()
            print(f"  [HOOK debug] MSE={mse:.6f}  cos={cos:.6f}")

        # Rebuild output tuple with reconstructed hidden states
        return (sae_out.to(orig_dtype),) + output[1:]

    def register(self):
        self.handle = self.layer.register_forward_hook(self._hook_fn)
        return self

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


# ── Accuracy ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_accuracy(model: CLIPModel, text_features: torch.Tensor,
                     loader: DataLoader, device: str) -> float:
    logit_scale = model.logit_scale.exp()
    tf = l2_normalize(text_features.float())

    correct = total = 0
    for imgs, labels in tqdm(loader, desc="    eval", leave=False):
        img_feat = l2_normalize(
            model.get_image_features(pixel_values=imgs.to(device)).float()
        )
        logits = logit_scale * (img_feat @ tf.t())
        preds  = logits.argmax(dim=-1).cpu()
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return correct / total * 100.0


# ── SAE discovery ─────────────────────────────────────────────────────────────

def discover_saes(sae_dir: str) -> list:
    """Return [(run_id, block_layer, path)] for all final SAE checkpoints."""
    paths   = sorted(glob.glob(os.path.join(sae_dir, "*/final*/*.pt")))
    results = []
    for p in paths:
        try:
            ckpt  = torch.load(p, map_location="cpu", weights_only=False)
            cfg   = ckpt.get("cfg", ckpt.get("config"))
            layer = (getattr(cfg, "block_layer", None)
                     if not isinstance(cfg, dict) else cfg.get("block_layer"))
            sd = ckpt.get("state_dict", ckpt)
            w  = sd.get("W_enc") if isinstance(sd, dict) else None
            if w is not None and torch.isnan(w).any():
                print(f"  [SKIP] {p} — NaN in W_enc")
                del ckpt; gc.collect(); continue
            run_id = p.split(os.sep)[-3]
            del ckpt; gc.collect()
            results.append((run_id, layer, p))
        except Exception as e:
            print(f"  [SKIP] {p} — {e}")
    return results


# ── Per-split evaluation ──────────────────────────────────────────────────────

def evaluate_split(split_name: str, held_out: str, sources: list,
                   batch_size: int, num_workers: int, device: str,
                   extra_sae_dirs: list = None) -> dict:
    print(f"\n{'═'*70}")
    print(f"  SPLIT: {split_name}  |  held-out (target): {held_out}")
    print(f"  sources: {sources}")
    print(f"{'═'*70}")

    results = {"split": split_name, "held_out": held_out,
               "sources": sources, "conditions": []}

    # ── Load CLIPood model ────────────────────────────────────────────────────
    clipood_ckpt = os.path.join(CLIPOOD_ROOT, f"clipood_pacs_held_{held_out}.pt")
    print(f"\n[1] Loading CLIPood model")
    model, processor, tokenizer = build_clipood_model(clipood_ckpt, device)

    # ── Load target-domain test set ───────────────────────────────────────────
    print(f"\n[2] Target domain: {held_out}")
    try:
        loader, n_imgs, class_names = build_pacs_target_loader(
            PACS_ROOT, held_out, batch_size, num_workers
        )
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        del model; flush(); return results

    print(f"  {n_imgs} images | classes: {class_names}")

    # ── Text features ─────────────────────────────────────────────────────────
    print("\n[3] Building text features...")
    text_feat = get_text_features(model, tokenizer, PACS_CLASSES, device)

    # ── A: CLIPood only ───────────────────────────────────────────────────────
    print("\n[A] CLIPood (no SAE)")
    acc = compute_accuracy(model, text_feat, loader, device)
    print(f"    → {acc:.2f}%")
    results["conditions"].append({
        "name": "CLIPood (no SAE)", "accuracy": round(acc, 4),
        "sae_path": None, "sae_layer": None, "run_id": None,
    })

    # ── B: CLIPood + ImageNet base SAE ────────────────────────────────────────
    print(f"\n[B] CLIPood + ImageNet (base) SAE")
    if not os.path.isfile(BASE_SAE_PATH):
        print(f"  [SKIP] {BASE_SAE_PATH} not found")
    else:
        try:
            sae, sae_cfg = load_sae(BASE_SAE_PATH, device)
            sae.eval()
            hook = HFCLIPSAEHook(model, sae, sae_cfg, device).register()
            acc  = compute_accuracy(model, text_feat, loader, device)
            hook.remove()
            del sae; flush()
            print(f"    → {acc:.2f}%")
            results["conditions"].append({
                "name": "CLIPood + ImageNet SAE", "accuracy": round(acc, 4),
                "sae_path": BASE_SAE_PATH,
                "sae_layer": sae_cfg.block_layer, "run_id": "imagenet_base",
            })
        except Exception as e:
            print(f"  [ERROR] {e}")

    # ── C+: CLIPood + PACS SAEs ───────────────────────────────────────────────
    sae_dir   = os.path.join(CKPT_ROOT, f"clipood_pacs_{split_name}")
    extra_dirs = [d for d in (extra_sae_dirs or []) if os.path.isdir(d)]
    search_dirs = [sae_dir] + extra_dirs
    print(f"\n[C] PACS CLIPood SAEs in: {search_dirs}")
    pacs_saes = []
    for sd in search_dirs:
        pacs_saes.extend(discover_saes(sd))
    if not pacs_saes:
        print("  None found.")
    else:
        print(f"  Found {len(pacs_saes)}:")
        for run_id, layer, path in pacs_saes:
            print(f"    run={run_id}  layer={layer}")

    for run_id, layer, sae_path in pacs_saes:
        label = f"CLIPood + PACS SAE (run={run_id}, layer={layer})"
        print(f"\n    {label}")
        try:
            sae, sae_cfg = load_sae(sae_path, device)
            sae.eval()
            hook = HFCLIPSAEHook(model, sae, sae_cfg, device).register()
            acc  = compute_accuracy(model, text_feat, loader, device)
            hook.remove()
            del sae; flush()
            print(f"    → {acc:.2f}%")
            results["conditions"].append({
                "name": label, "accuracy": round(acc, 4),
                "sae_path": sae_path,
                "sae_layer": layer, "run_id": run_id,
            })
        except Exception as e:
            print(f"    [ERROR] {e}")

    # ── D: Base CLIP alone + base SAE — reference conditions ─────────────────
    print(f"\n[D] Base CLIP (no adapter) — reference conditions")
    try:
        base_model = CLIPModel.from_pretrained(BACKBONE).to(device).eval()
        base_tf    = get_text_features(base_model, tokenizer, PACS_CLASSES, device)

        # D1: Base CLIP, no SAE
        acc = compute_accuracy(base_model, base_tf, loader, device)
        print(f"  [D1] Base CLIP (no SAE)         → {acc:.2f}%")
        results["conditions"].append({
            "name": "Base CLIP (no SAE)", "accuracy": round(acc, 4),
            "sae_path": None, "sae_layer": None, "run_id": "base",
        })

        # D2: Base CLIP + ImageNet (base) SAE
        if os.path.isfile(BASE_SAE_PATH):
            try:
                sae, sae_cfg = load_sae(BASE_SAE_PATH, device)
                sae.eval()
                hook = HFCLIPSAEHook(base_model, sae, sae_cfg, device).register()
                acc2 = compute_accuracy(base_model, base_tf, loader, device)
                hook.remove()
                del sae; flush()
                print(f"  [D2] Base CLIP + ImageNet SAE   → {acc2:.2f}%")
                results["conditions"].append({
                    "name": "Base CLIP + ImageNet SAE", "accuracy": round(acc2, 4),
                    "sae_path": BASE_SAE_PATH,
                    "sae_layer": sae_cfg.block_layer, "run_id": "imagenet_base",
                })
            except Exception as e:
                print(f"  [D2 ERROR] {e}")

        del base_model; flush()
    except Exception as e:
        print(f"  [ERROR] {e}")

    del model; flush()
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Zero-shot accuracy: CLIPood vs CLIPood+SAE on PACS target domains"
    )
    p.add_argument("--splits", nargs="+",
                   default=list(PACS_SPLITS.keys()),
                   choices=list(PACS_SPLITS.keys()))
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_json",      default="out/clipood_sae_accuracy.json")
    p.add_argument("--extra_sae_dirs", nargs="*", default=[],
                   help="Extra checkpoint directories to search for CLIPood SAEs")
    args = p.parse_args()

    print(f"\n{'═'*70}")
    print(f"  CLIPood SAE Accuracy Evaluation — PACS leave-one-out")
    print(f"  Device  : {DEVICE}")
    print(f"  Backbone: {BACKBONE}")
    print(f"  Splits  : {args.splits}")
    print(f"  Base SAE: {BASE_SAE_PATH}")
    print(f"{'═'*70}")

    all_results = []
    for split_name in args.splits:
        held_out, sources = PACS_SPLITS[split_name]
        res = evaluate_split(split_name, held_out, sources,
                             args.batch_size, args.num_workers, DEVICE,
                             extra_sae_dirs=args.extra_sae_dirs)
        all_results.append(res)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n\n{'═'*70}")
    print("  SUMMARY")
    print(f"{'═'*70}\n")

    for res in all_results:
        print(f"  {res['split'].upper()}  (target: {res['held_out']})")
        print(f"  {'─'*60}")
        for cond in res["conditions"]:
            layer_tag = (f"  [layer={cond['sae_layer']}]"
                         if cond["sae_layer"] is not None else "")
            print(f"    {cond['name']:<55s}  {cond['accuracy']:6.2f}%{layer_tag}")
        print()

    os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
    with open(args.save_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved → {args.save_json}")


if __name__ == "__main__":
    main()
