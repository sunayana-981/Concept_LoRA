#!/usr/bin/env python3
"""
Evaluate LoRA + base SAE accuracy for DTD and FGVC-Aircraft.

This script intentionally avoids importing tasks.utils (and yacs dependency)
and uses the same LoRA merge + SAE hook math as eval_lora_sae_accuracy.py.

Usage:
  conda activate patchsae
  cd /home/sunayana/Documents/Concept_LoRA/patchsae
  python eval_lora_base_sae_dtd_fgvc.py
"""

import gc
import json
import math
import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = "/home/sunayana/Documents/Concept_LoRA"
PATCHSAE = f"{ROOT}/patchsae"

sys.path.insert(0, ROOT)
sys.path.insert(0, PATCHSAE)

from CLIP_LoRA import clip as openai_clip
from CLIP_LoRA.datasets import build_dataset
from CLIP_LoRA.datasets.utils import build_data_loader
from src.models.templates.openai_imagenet_templates import openai_imagenet_template

os.environ["TOKENIZERS_PARALLELISM"] = "false"

BACKBONE = "ViT-B/16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = f"{ROOT}/data"
BASE_SAE_PATH = f"{PATCHSAE}/data/sae_weight/base/out.pt"

TARGETS = [
    {
        "dataset": "dtd",
        "shots": 16,
        "lora_path": f"{ROOT}/lora_weights/vitb16/dtd/16shots/seed42/lora_weights.pt",
    },
    {
        "dataset": "fgvc",
        "shots": 16,
        "lora_path": f"{ROOT}/lora_weights/vitb16/fgvc/16shots/seed1/lora_weights.pt",
    },
]


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


def build_lora_clip(lora_path: str, device: str):
    model, preprocess = openai_clip.load(BACKBONE, device=device)
    model.eval()

    lora_state = torch.load(lora_path, map_location=device)
    if "weights" not in lora_state:
        print("  [WARN] No 'weights' key in LoRA checkpoint — using base CLIP.")
        return model, preprocess

    layers_dict = lora_state["weights"]
    meta = lora_state["metadata"]
    scale = meta["alpha"] / math.sqrt(meta["r"])
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


class MinimalSAE:
    def __init__(self, ckpt_path: str, device: str):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt.get("cfg", ckpt.get("config", {}))
        if isinstance(cfg, dict):
            self.block_layer = cfg.get("block_layer", -2)
        else:
            self.block_layer = getattr(cfg, "block_layer", -2)

        sd = ckpt.get("state_dict", ckpt)
        self.W_enc = sd["W_enc"].to(device).float()
        self.b_enc = sd["b_enc"].to(device).float()
        self.W_dec = sd["W_dec"].to(device).float()
        self.b_dec = sd["b_dec"].to(device).float()

    def __call__(self, x_bsd: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu((x_bsd - self.b_dec) @ self.W_enc + self.b_enc)
        out = hidden @ self.W_dec + self.b_dec
        return out


class SAEForwardHook:
    def __init__(self, model, sae, block_layer):
        num_blocks = len(model.visual.transformer.resblocks)
        layer = block_layer
        if layer < 0:
            layer = num_blocks + layer
        if not (0 <= layer < num_blocks):
            raise ValueError(f"Invalid block layer {block_layer} -> {layer}")

        self.sae = sae
        self.block = model.visual.transformer.resblocks[layer]
        self.handle = None
        self._logged = False

        print(f"  [SAEHook] resblock[{layer}] (block_layer={block_layer})")

    def _hook_fn(self, module, input, output):
        orig_dtype = output.dtype
        act_bsd = output.transpose(0, 1).float()
        sae_out = self.sae(act_bsd)

        if not self._logged:
            self._logged = True
            mse = (act_bsd - sae_out).pow(2).mean().item()
            cos = F.cosine_similarity(
                act_bsd.reshape(-1, act_bsd.shape[-1]),
                sae_out.reshape(-1, sae_out.shape[-1]),
                dim=-1,
            ).mean().item()
            print(f"  [SAEHook debug] MSE={mse:.6f}  cos={cos:.6f}")

        return sae_out.transpose(0, 1).to(orig_dtype)

    def register(self):
        self.handle = self.block.register_forward_hook(self._hook_fn)
        return self

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@torch.no_grad()
def get_text_features(model, class_names, device):
    n = len(class_names)
    mean_f = torch.zeros(n, 512, device=device)

    for tmpl in openai_imagenet_template:
        prompts = [tmpl(c) for c in class_names]
        tokens = openai_clip.tokenize(prompts, truncate=True).to(device)
        feats = model.encode_text(tokens).float()
        mean_f += l2_normalize(feats)

    mean_f /= len(openai_imagenet_template)
    return l2_normalize(mean_f)


@torch.no_grad()
def compute_accuracy(model, text_features, loader, device):
    logit_scale = model.logit_scale.exp()
    tf = l2_normalize(text_features.float())

    correct = 0
    total = 0
    for imgs, labels in tqdm(loader, desc="eval", leave=False):
        img_feat = l2_normalize(model.encode_image(imgs.to(device)).float())
        logits = logit_scale * (img_feat @ tf.t())
        preds = logits.argmax(dim=-1).cpu()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total * 100.0


def main():
    print(f"Device: {DEVICE}")

    if not os.path.isfile(BASE_SAE_PATH):
        raise FileNotFoundError(f"Base SAE missing: {BASE_SAE_PATH}")

    base_sae = MinimalSAE(BASE_SAE_PATH, DEVICE)
    print(f"Base SAE loaded: {BASE_SAE_PATH} | block_layer={base_sae.block_layer}")

    all_results = []

    for cfg in TARGETS:
        ds_name = cfg["dataset"]
        lora_path = cfg["lora_path"]
        shots = cfg["shots"]

        print("\n" + "=" * 80)
        print(f"DATASET: {ds_name} | LORA: {lora_path}")
        print("=" * 80)

        if not os.path.isfile(lora_path):
            print(f"[SKIP] missing LoRA checkpoint: {lora_path}")
            continue

        model, preprocess = build_lora_clip(lora_path, DEVICE)

        dataset = build_dataset(ds_name, DATA_ROOT, shots, preprocess)
        loader = build_data_loader(
            data_source=dataset.test,
            batch_size=64,
            is_train=False,
            tfm=preprocess,
            shuffle=False,
            num_workers=4,
        )
        class_names = dataset.classnames

        print(f"Test samples: {len(dataset.test)} | classes: {len(class_names)}")
        text_feat = get_text_features(model, class_names, DEVICE)

        hook = SAEForwardHook(model, base_sae, base_sae.block_layer).register()
        acc = compute_accuracy(model, text_feat, loader, DEVICE)
        hook.remove()

        print(f"LoRA + base SAE accuracy ({ds_name}) = {acc:.4f}%")

        all_results.append(
            {
                "dataset": ds_name,
                "lora_path": lora_path,
                "lora_plus_base_sae_accuracy": round(acc, 6),
                "base_sae_path": BASE_SAE_PATH,
                "base_sae_layer": int(base_sae.block_layer),
            }
        )

        del model, dataset, loader, text_feat
        flush()

    out_path = f"{PATCHSAE}/out/lora_plus_base_sae_dtd_fgvc.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nRESULTS:")
    for r in all_results:
        print(f"  {r['dataset']}: {r['lora_plus_base_sae_accuracy']:.4f}%")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
