"""Embedding comparison and representation-space monosemanticity.

Consolidates compare_embeddings.py, compare_models.py, and integrate_mono.py
into two functions that share a single forward pass.

compare_representations()
    Compares two backbone instances (e.g. base CLIP vs LoRA-merged CLIP) on a
    dataset.  Per-sample and aggregate statistics:
      - SAE latent cosine similarity, L2 distance, sparsity
      - CLIP image embedding cosine similarity, L2 distance
    Single forward pass per backbone (not two separate passes).

embedding_monosemanticity()
    Weighted pairwise cosine monosemanticity (MS_h) from integrate_mono.py.
    For each SAE neuron h:
        MS_h = Σ_{i<j} z_i^h * z_j^h * cos(x_i, x_j)
               ─────────────────────────────────────
               Σ_{i<j} z_i^h * z_j^h
    Uses the key identity  z^T S z = ||X^T z||²  so the N×N similarity matrix
    is never materialised — O(N·D + N·H) memory instead of O(N²).
"""

from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


# ── Embedding comparison ──────────────────────────────────────────────────────

@dataclass
class EmbeddingStats:
    dataset_name: str
    n_samples: int
    sae_cosine_mean: float
    sae_cosine_std:  float
    sae_l2_mean:     float
    sae_l2_std:      float
    clip_cosine_mean: float
    clip_cosine_std:  float
    clip_l2_mean:     float
    clip_l2_std:      float
    sae_sparsity_a_mean: float
    sae_sparsity_b_mean: float


def _make_collate(processor, is_align: bool = False):
    def collate(batch):
        imgs = []
        for item in batch:
            img = item["image"] if isinstance(item, dict) else item
            if hasattr(img, "mode") and img.mode != "RGB":
                img = img.convert("RGB")
            imgs.append(img)
        kw = dict(images=imgs, return_tensors="pt")
        if is_align:
            kw["text"] = ""
        return processor(**kw)["pixel_values"]
    return collate


@torch.no_grad()
def compare_representations(
    backbone_a,         # BackboneBase — e.g. base CLIP
    backbone_b,         # BackboneBase — e.g. LoRA-merged CLIP (same architecture)
    sae_a,              # SparseAutoencoder for backbone_a
    sae_b,              # SparseAutoencoder for backbone_b (can be same object)
    dataset,
    dataset_name: str   = "dataset",
    layer: int          = -1,
    batch_size: int     = 64,
    device: str         = "cuda",
    num_workers: int    = 4,
) -> EmbeddingStats:
    """Compare SAE latents and backbone embeddings between two models.

    Both models are run in the same loop to avoid loading the dataset twice.
    Works with any BackboneBase subclass.
    """
    is_align = backbone_a.__class__.__name__ == "ALIGNBackbone"
    collate  = _make_collate(backbone_a.processor, is_align=is_align)
    loader   = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda"),
        collate_fn=collate,
    )

    layer_idx = backbone_a.resolve_layer(layer)

    def _make_hook(cap, name):
        def _h(_m, _i, out):
            hs = out if isinstance(out, torch.Tensor) else out[0]
            cap[name] = hs.detach().float()
        return _h

    cap_a, cap_b = {}, {}

    if is_align:
        ha = backbone_a.model.vision_model.encoder.blocks[layer_idx].register_forward_hook(
            _make_hook(cap_a, "act"))
        hb = backbone_b.model.vision_model.encoder.blocks[layer_idx].register_forward_hook(
            _make_hook(cap_b, "act"))
    elif hasattr(backbone_a.model, "vision_model"):  # CLIP
        ha = backbone_a.model.vision_model.encoder.layers[layer_idx].register_forward_hook(
            _make_hook(cap_a, "act"))
        hb = backbone_b.model.vision_model.encoder.layers[layer_idx].register_forward_hook(
            _make_hook(cap_b, "act"))
    else:  # DINOv2
        ha = backbone_a.model.encoder.layer[layer_idx].register_forward_hook(
            _make_hook(cap_a, "act"))
        hb = backbone_b.model.encoder.layer[layer_idx].register_forward_hook(
            _make_hook(cap_b, "act"))

    sae_cosines, sae_l2s  = [], []
    clip_cosines, clip_l2s = [], []
    sp_a_list, sp_b_list   = [], []

    for pixel_values in tqdm(loader, desc=f"  compare {dataset_name}", leave=True):
        pv = pixel_values.to(device)

        # Forward both models
        if is_align:
            backbone_a.model.vision_model(pixel_values=pv)
            backbone_b.model.vision_model(pixel_values=pv)
            act_a = cap_a["act"].mean(dim=[-2, -1])   # GAP → [B, C]
            act_b = cap_b["act"].mean(dim=[-2, -1])
        elif hasattr(backbone_a.model, "vision_model"):
            backbone_a.model.vision_model(pixel_values=pv)
            backbone_b.model.vision_model(pixel_values=pv)
            act_a = cap_a["act"][:, 0, :]   # CLS
            act_b = cap_b["act"][:, 0, :]
        else:
            backbone_a.model(pixel_values=pv)
            backbone_b.model(pixel_values=pv)
            act_a = cap_a["act"][:, 0, :]
            act_b = cap_b["act"][:, 0, :]

        # SAE latents
        _, lat_a, _ = sae_a(act_a)
        _, lat_b, _ = sae_b(act_b)

        # SAE metrics (flatten to 1D per sample for cosine)
        sae_cosines.append(
            F.cosine_similarity(lat_a, lat_b, dim=-1).cpu())
        sae_l2s.append(
            (lat_a - lat_b).norm(p=2, dim=-1).cpu())
        sp_a_list.append((lat_a.abs() < 1e-5).float().mean(-1).cpu())
        sp_b_list.append((lat_b.abs() < 1e-5).float().mean(-1).cpu())

        # Backbone embedding metrics
        clip_cosines.append(
            F.cosine_similarity(act_a, act_b, dim=-1).cpu())
        clip_l2s.append(
            (act_a - act_b).norm(p=2, dim=-1).cpu())

    ha.remove()
    hb.remove()

    def agg(lst): return torch.cat(lst).numpy()
    sc = agg(sae_cosines);   sl = agg(sae_l2s)
    cc = agg(clip_cosines);  cl = agg(clip_l2s)
    spa = agg(sp_a_list);    spb = agg(sp_b_list)

    return EmbeddingStats(
        dataset_name       = dataset_name,
        n_samples          = len(sc),
        sae_cosine_mean    = float(sc.mean()),
        sae_cosine_std     = float(sc.std()),
        sae_l2_mean        = float(sl.mean()),
        sae_l2_std         = float(sl.std()),
        clip_cosine_mean   = float(cc.mean()),
        clip_cosine_std    = float(cc.std()),
        clip_l2_mean       = float(cl.mean()),
        clip_l2_std        = float(cl.std()),
        sae_sparsity_a_mean= float(spa.mean()),
        sae_sparsity_b_mean= float(spb.mean()),
    )


# ── Embedding-space monosemanticity ───────────────────────────────────────────

@torch.no_grad()
def embedding_monosemanticity(
    image_embeddings: torch.Tensor,   # X: [N, D]  (e.g. CLIP CLS tokens)
    neuron_activations: torch.Tensor, # Z: [N, H]  (SAE feature activations)
    chunk_size: int = 4096,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Weighted pairwise cosine monosemanticity per SAE neuron.

    MS_h = Σ_{i<j} w_{ij}^h · cos(x_i, x_j) / Σ_{i<j} w_{ij}^h
    where w_{ij}^h = z_i^h · z_j^h  (product of normalised activations).

    Memory-efficient identity: z^T (X X^T) z = ||X^T z||²
    so we never materialise the N×N similarity matrix.

    Returns
    -------
    ms : Tensor[H] — per-neuron MS score (NaN where neuron never fires)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    N, D = image_embeddings.shape
    H    = neuron_activations.shape[1]

    X_norm = F.normalize(image_embeddings.float().to(device), dim=1)  # [N, D]

    # Normalise activations to [0,1] per neuron (on CPU)
    Z_cpu = neuron_activations.float()
    z_min = Z_cpu.min(dim=0, keepdim=True)[0]
    z_max = Z_cpu.max(dim=0, keepdim=True)[0]
    Z_cpu = (Z_cpu - z_min) / (z_max - z_min + 1e-6)

    z_sq_sum   = (Z_cpu ** 2).sum(dim=0)           # [H] on CPU
    z_sum      = Z_cpu.sum(dim=0)                  # [H]
    denominator = (0.5 * (z_sum ** 2 - z_sq_sum)).to(device)
    z_sq_dev    = z_sq_sum.to(device)

    numerator = torch.zeros(H, device=device)
    for h0 in tqdm(range(0, H, chunk_size), desc="  embed_mono", leave=False):
        h1 = min(h0 + chunk_size, H)
        Zc = Z_cpu[:, h0:h1].to(device)           # [N, chunk]
        V  = X_norm.T @ Zc                         # [D, chunk]
        numerator[h0:h1] = 0.5 * ((V ** 2).sum(0) - z_sq_dev[h0:h1])

    ms = torch.where(
        denominator > 0,
        numerator / denominator,
        torch.tensor(float("nan"), device=device),
    )
    return ms.cpu()


@torch.no_grad()
def collect_embeddings_and_activations(
    backbone,
    sae,
    dataset,
    layer: int      = -1,
    batch_size: int = 64,
    device: str     = "cuda",
    num_workers: int = 4,
    max_samples: Optional[int] = None,
):
    """Collect [N, D] CLIP embeddings and [N, H] SAE activations in one pass.

    Used as the data-collection step before calling embedding_monosemanticity().
    """
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    is_align = backbone.__class__.__name__ == "ALIGNBackbone"
    collate  = _make_collate(backbone.processor, is_align=is_align)
    loader   = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda"),
        collate_fn=collate,
    )

    layer_idx = backbone.resolve_layer(layer)
    captured  = {}

    if is_align:
        def _hook(_m, _i, out):
            captured["act"] = out.detach().float().mean(dim=[-2, -1])
        h = backbone.model.vision_model.encoder.blocks[layer_idx].register_forward_hook(_hook)
    elif hasattr(backbone.model, "vision_model"):
        def _hook(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured["act"] = hs.detach().float()[:, 0, :]
        h = backbone.model.vision_model.encoder.layers[layer_idx].register_forward_hook(_hook)
    else:
        def _hook(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured["act"] = hs.detach().float()[:, 0, :]
        h = backbone.model.encoder.layer[layer_idx].register_forward_hook(_hook)

    all_embeds = []
    all_acts   = []

    for pixel_values in tqdm(loader, desc="  collect", leave=True):
        pv = pixel_values.to(device)
        if is_align:
            backbone.model.vision_model(pixel_values=pv)
        elif hasattr(backbone.model, "vision_model"):
            backbone.model.vision_model(pixel_values=pv)
        else:
            backbone.model(pixel_values=pv)

        act_cls = captured["act"]                # [B, d]
        _, feat, _ = sae(act_cls)                # [B, H]

        all_embeds.append(act_cls.cpu())
        all_acts.append(feat.cpu())

    h.remove()
    return torch.cat(all_embeds, dim=0), torch.cat(all_acts, dim=0)
