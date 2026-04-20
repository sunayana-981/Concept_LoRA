"""Compute per-feature statistics from a trained SAE over a dataset.

All tensors are kept on CPU to support large feature dictionaries.

Returns a dict (artifacts) with keys:
  max_activating_image_indices  Tensor[N_FEAT, N_TOP]  — global dataset indices
  max_activating_image_values   Tensor[N_FEAT, N_TOP]  — activation values
  sae_mean_acts                 Tensor[N_FEAT]          — conditioned mean
  sae_sparsity                  Tensor[N_FEAT]          — fraction that fired
  cls_sae_cnt                   ndarray[N_CLASS, N_FEAT]— per-class fire counts
  l0, recon_mse, explained_variance  — scalars
  n_images                      int
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ACT_THRESHOLD = 0.2   # binarisation threshold for cls_sae_cnt
N_TOP_IMAGES  = 10    # rolling top-K per feature


def _get_new_top_k(first_vals, first_idx, second_vals, second_idx, k):
    total_vals = torch.cat([first_vals, second_vals], dim=1)
    total_idx  = torch.cat([first_idx,  second_idx],  dim=1)
    new_vals, pos = torch.topk(total_vals, k=k, dim=1)
    new_idx = torch.gather(total_idx, 1, pos)
    return new_vals, new_idx


def make_collate_fn(processor, label_key: str, is_align: bool = False):
    """Build a DataLoader collate function for any backbone processor."""
    def collate(batch):
        imgs, labels = [], []
        for item in batch:
            img = item["image"]
            if hasattr(img, "mode") and img.mode != "RGB":
                img = img.convert("RGB")
            imgs.append(img)
            labels.append(item[label_key])
        if is_align:
            pv = processor(images=imgs, text="", return_tensors="pt")["pixel_values"]
        else:
            pv = processor(images=imgs, return_tensors="pt")["pixel_values"]
        return pv, torch.tensor(labels, dtype=torch.long)
    return collate


@torch.no_grad()
def compute_feature_stats(
    backbone,
    sae,
    dataset,
    label_key: str,
    n_classes: int,
    layer: int = -1,
    batch_size: int = 64,
    device: str = "cuda",
    max_images: Optional[int] = None,
    num_workers: int = 4,
) -> dict:
    """Run full dataset through backbone+SAE; return artifact dict.

    backbone  : BackboneBase subclass (already loaded)
    sae       : SparseAutoencoder (already loaded, on device)
    dataset   : HF Dataset
    label_key : column name for integer labels
    n_classes : number of classes
    """
    from ..models.base import BackboneBase

    if max_images is not None:
        dataset = dataset.select(range(min(max_images, len(dataset))))

    is_align = backbone.__class__.__name__ == "ALIGNBackbone"
    collate = make_collate_fn(backbone.processor, label_key, is_align=is_align)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda"),
        collate_fn=collate,
    )

    N_FEAT = sae.W_enc.shape[1]

    max_values  = torch.zeros(N_FEAT, N_TOP_IMAGES)
    max_indices = torch.zeros(N_FEAT, N_TOP_IMAGES, dtype=torch.long)
    total_acts  = torch.zeros(N_FEAT)
    sparsity    = torch.zeros(N_FEAT)
    cls_cnt     = torch.zeros(n_classes, N_FEAT)

    total_l0  = 0.0
    total_mse = 0.0
    total_ev  = 0.0
    total_n   = 0
    images_processed = 0

    # Register hook once
    layer_idx = backbone.resolve_layer(layer)
    captured = {}

    if is_align:
        def _hook(_m, _i, out):
            captured["act"] = out.detach().float().mean(dim=[-2, -1])  # GAP
        hook_mod = backbone.model.vision_model.encoder.blocks[layer_idx]
    elif hasattr(backbone.model, "encoder"):  # DINOv2
        def _hook(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured["act"] = hs.detach().float()[:, 0, :]  # CLS
        hook_mod = backbone.model.encoder.layer[layer_idx]
    else:  # CLIP
        def _hook(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured["act"] = hs.detach().float()[:, 0, :]  # CLS
        hook_mod = backbone.model.vision_model.encoder.layers[layer_idx]

    h = hook_mod.register_forward_hook(_hook)

    for pixel_values, labels in tqdm(loader, desc="  eval", leave=True):
        pixel_values = pixel_values.to(device)
        B = pixel_values.shape[0]

        # Backbone forward (hook fires)
        if is_align:
            backbone.model.vision_model(pixel_values=pixel_values)
        elif hasattr(backbone.model, "vision_model"):
            backbone.model.vision_model(pixel_values=pixel_values)
        else:
            backbone.model(pixel_values=pixel_values)

        cls_act = captured["act"]  # [B, d]

        # SAE forward
        sae_out, feat_acts, _ = sae(cls_act)
        feat_acts_cpu = feat_acts.detach().cpu()  # [B, N_FEAT]

        # Global metrics
        total_l0  += (feat_acts_cpu > 0).float().sum(-1).mean().item() * B
        mse_s = (sae_out.cpu() - cls_act.cpu()).pow(2).sum(-1)
        var_s = cls_act.cpu().pow(2).sum(-1).clamp(min=1e-8)
        total_mse += mse_s.mean().item() * B
        total_ev  += (1 - mse_s / var_s).mean().item() * B
        total_n   += B

        # Per-feature stats
        sae_acts_T = feat_acts_cpu.T  # [N_FEAT, B]
        total_acts += sae_acts_T.sum(dim=1)
        sparsity   += (sae_acts_T > 0).sum(dim=1).float()

        # Rolling top-K
        k_batch = min(N_TOP_IMAGES, B)
        batch_vals, batch_local = torch.topk(sae_acts_T, k=k_batch, dim=1)
        batch_global = batch_local + images_processed
        max_values, max_indices = _get_new_top_k(
            max_values, max_indices, batch_vals, batch_global, k=N_TOP_IMAGES
        )

        # Class-wise counts
        binary  = (feat_acts_cpu > ACT_THRESHOLD).float()
        one_hot = F.one_hot(labels, n_classes).float()
        cls_cnt += one_hot.T @ binary

        images_processed += B

    h.remove()

    # Finalise
    fired_mask = sparsity > 0
    sae_mean_acts = torch.zeros(N_FEAT)
    sae_mean_acts[fired_mask] = total_acts[fired_mask] / sparsity[fired_mask]
    sae_sparsity = sparsity / max(images_processed, 1)

    metrics = {
        "n_images":           int(images_processed),
        "l0":                 total_l0 / total_n,
        "recon_mse":          total_mse / total_n,
        "explained_variance": total_ev  / total_n,
    }

    artifacts = {
        "max_activating_image_indices": max_indices,
        "max_activating_image_values":  max_values,
        "sae_mean_acts":                sae_mean_acts,
        "sae_sparsity":                 sae_sparsity,
        "cls_sae_cnt":                  cls_cnt.numpy(),
        "metrics":                      metrics,
        "n_classes":                    n_classes,
    }
    return artifacts
