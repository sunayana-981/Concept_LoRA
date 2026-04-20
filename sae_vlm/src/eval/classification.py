"""Classification evaluation with SAE-augmented CLIP.

Three complementary paradigms — all sharing a single backbone forward pass
per batch and using the same hook infrastructure.

1. zero_shot_accuracy / classification_suite
   Standard CLIP zero-shot classification with optional SAE reconstruction
   applied to the intermediate sequence before the final projection.

   SAE modes:
     none          Standard CLIP zero-shot (no SAE).
     preserve_cls  Reconstruct only patch tokens; keep CLS token unchanged.
                   Best accuracy preservation (recommended).
     original      Reconstruct all tokens including CLS.
     blend         alpha * SAE_recon + (1-alpha) * original.

2. linear_probe_accuracy
   Extract [N, d_sae] SAE sparse latents from a dataset, then fit a
   logistic regression (SGDClassifier) and evaluate test accuracy.
   Equivalent to eval_sae_classifier.py but backbone-agnostic.
   Useful for comparing: raw backbone features vs SAE latents.

3. feature_masking_accuracy
   For each top-K SAE features of a target class, clamp them ON or OFF
   and measure how classification accuracy changes.  This tests the
   *causal* role of each feature in driving class predictions.
   Equivalent to tasks/classification_with_top_k_masking.py.
   CLIP-family only (requires text encoder).

Usage
-----
    from src.eval.classification import (
        classification_suite,
        linear_probe_accuracy,
        feature_masking_accuracy,
    )
"""

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


# ── Shared helpers ────────────────────────────────────────────────────────────

def _require_clip(backbone):
    if backbone.__class__.__name__ != "CLIPBackbone":
        raise ValueError(
            f"This function requires a CLIPBackbone, got {backbone.__class__.__name__}. "
            "ALIGN and DINOv2 have no text encoder."
        )


def _get_text_features(backbone, classnames: List[str], device: str) -> torch.Tensor:
    """Encode classnames with CLIP text encoder → L2-normalised [N_CLASS, d_proj].

    Uses the backbone's own processor/tokenizer, so no hardcoded model IDs.
    """
    model     = backbone.model
    processor = backbone.processor
    prompts   = [f"a photo of a {c.replace('_', ' ')}" for c in classnames]

    inputs = processor(
        text=prompts, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    return F.normalize(feats.float(), dim=-1)   # [N_CLASS, d_proj]


def _make_collate(processor, label_key: str, is_align: bool = False):
    def collate(batch):
        imgs, labels = [], []
        for item in batch:
            img = item["image"]
            if hasattr(img, "mode") and img.mode != "RGB":
                img = img.convert("RGB")
            imgs.append(img)
            labels.append(item[label_key])
        kw = dict(images=imgs, return_tensors="pt")
        if is_align:
            kw["text"] = ""
        pv = processor(**kw)["pixel_values"]
        return pv, torch.tensor(labels, dtype=torch.long)
    return collate


def _register_cls_hook(backbone, layer_idx: int, captured: dict):
    """Register a forward hook that captures the CLS (or GAP) activation."""
    is_align = backbone.__class__.__name__ == "ALIGNBackbone"

    if is_align:
        def _hook(_m, _i, out):
            captured["seq"] = out.detach().float()          # [B, C, H, W]
            captured["cls"] = out.detach().float().mean(dim=[-2, -1])  # GAP
        h = backbone.model.vision_model.encoder.blocks[layer_idx].register_forward_hook(_hook)
    elif hasattr(backbone.model, "vision_model"):           # CLIP / SigLIP
        def _hook(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured["seq"] = hs.detach().float()            # [B, T, d]
            captured["cls"] = hs.detach().float()[:, 0, :]   # CLS
        h = backbone.model.vision_model.encoder.layers[layer_idx].register_forward_hook(_hook)
    else:                                                    # DINOv2
        def _hook(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured["seq"] = hs.detach().float()
            captured["cls"] = hs.detach().float()[:, 0, :]
        h = backbone.model.encoder.layer[layer_idx].register_forward_hook(_hook)
    return h


def _forward_backbone(backbone, pixel_values):
    """Run the vision encoder (trigger registered hooks)."""
    is_align = backbone.__class__.__name__ == "ALIGNBackbone"
    if is_align:
        backbone.model.vision_model(pixel_values=pixel_values)
    elif hasattr(backbone.model, "vision_model"):
        backbone.model.vision_model(pixel_values=pixel_values)
    else:
        backbone.model(pixel_values=pixel_values)


# ── 1. Zero-shot classification ───────────────────────────────────────────────

def _apply_sae_to_sequence(seq: torch.Tensor, sae, mode: str,
                            blend_alpha: float = 0.2) -> torch.Tensor:
    """Apply SAE reconstruction to an intermediate sequence [B, T, d]."""
    B, T, d = seq.shape
    if mode == "preserve_cls":
        patches = seq[:, 1:, :].reshape(-1, d)
        recon, _, _ = sae(patches)
        return torch.cat([seq[:, 0:1, :], recon.reshape(B, T - 1, d)], dim=1)
    elif mode == "original":
        recon, _, _ = sae(seq.reshape(-1, d))
        return recon.reshape(B, T, d)
    elif mode == "blend":
        flat = seq.reshape(-1, d)
        recon, _, _ = sae(flat)
        return (blend_alpha * recon + (1 - blend_alpha) * flat).reshape(B, T, d)
    raise ValueError(f"Unknown sae_mode '{mode}'. Use: preserve_cls, original, blend")


@torch.no_grad()
def zero_shot_accuracy(
    backbone,
    dataset,
    classnames: List[str],
    label_key:   str   = "label",
    sae                = None,
    sae_mode:    str   = "none",
    blend_alpha: float = 0.2,
    layer:       int   = -1,
    batch_size:  int   = 64,
    device:      str   = "cuda",
    num_workers: int   = 4,
) -> dict:
    """Zero-shot CLIP classification with optional SAE reconstruction.

    Returns
    -------
    dict: accuracy, n_correct, n_total, sae_mode,
          recon_mse (if sae used), cosine_sim_vs_baseline (if sae used)
    """
    _require_clip(backbone)
    model     = backbone.model
    processor = backbone.processor

    text_feats  = _get_text_features(backbone, classnames, device)
    collate     = _make_collate(processor, label_key)
    loader      = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=(device == "cuda"),
                             collate_fn=collate)

    layer_idx = backbone.resolve_layer(layer)
    captured  = {}
    h = _register_cls_hook(backbone, layer_idx, captured)

    correct = total = 0
    recon_mse_sum = cosine_sum = n_batches = 0

    for pixel_values, labels in tqdm(loader, desc="  zero-shot", leave=True):
        pixel_values = pixel_values.to(device)
        labels       = labels.to(device)

        if sae is not None and sae_mode != "none":
            _forward_backbone(backbone, pixel_values)
            seq      = captured["seq"]                              # [B, T, d]
            baseline = F.normalize(model.get_image_features(pixel_values).float(), dim=-1)

            seq_recon      = _apply_sae_to_sequence(seq, sae, sae_mode, blend_alpha)
            recon_mse_sum += (seq - seq_recon).pow(2).mean().item()

            cls_recon  = model.vision_model.post_layernorm(seq_recon[:, 0, :])
            img_feats  = F.normalize(model.visual_projection(cls_recon).float(), dim=-1)
            cosine_sum += F.cosine_similarity(img_feats, baseline, dim=-1).mean().item()
            n_batches  += 1
        else:
            img_feats = F.normalize(model.get_image_features(pixel_values).float(), dim=-1)

        pred     = (img_feats @ text_feats.T).argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total   += labels.size(0)

    h.remove()

    result = {"accuracy": 100.0 * correct / total,
              "n_correct": correct, "n_total": total, "sae_mode": sae_mode}
    if n_batches > 0:
        result["recon_mse"]              = recon_mse_sum / n_batches
        result["cosine_sim_vs_baseline"] = cosine_sum    / n_batches
    return result


def classification_suite(
    backbone,
    dataset,
    classnames:  List[str],
    label_key:   str            = "label",
    sae                         = None,
    sae_modes:   Optional[List[str]] = None,
    blend_alpha: float          = 0.2,
    layer:       int            = -1,
    batch_size:  int            = 64,
    device:      str            = "cuda",
    num_workers: int            = 4,
) -> dict:
    """Run zero-shot for baseline + all requested SAE modes.

    Returns ``{mode_name: accuracy_dict, ...}`` including ``"baseline"``.
    """
    if sae_modes is None:
        sae_modes = ["preserve_cls", "original"] if sae is not None else []

    results = {}
    results["baseline"] = zero_shot_accuracy(
        backbone, dataset, classnames, label_key=label_key,
        sae=None, sae_mode="none", layer=layer,
        batch_size=batch_size, device=device, num_workers=num_workers,
    )
    for mode in sae_modes:
        if sae is None:
            continue
        key = f"sae_{mode}"
        results[key] = zero_shot_accuracy(
            backbone, dataset, classnames, label_key=label_key,
            sae=sae, sae_mode=mode, blend_alpha=blend_alpha, layer=layer,
            batch_size=batch_size, device=device, num_workers=num_workers,
        )
        base = results["baseline"]["accuracy"]
        results[key]["accuracy_drop_pct"] = (
            100.0 * (base - results[key]["accuracy"]) / base if base > 0 else 0.0
        )
    return results


# ── 2. Linear probe on SAE latents ────────────────────────────────────────────

@torch.no_grad()
def _extract_sae_latents(backbone, sae, dataset, label_key, layer,
                          batch_size, device, num_workers, pool="cls"):
    """Extract (N, d_sae) SAE latents and (N,) labels from a dataset."""
    is_align = backbone.__class__.__name__ == "ALIGNBackbone"
    collate  = _make_collate(backbone.processor, label_key, is_align=is_align)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=(device == "cuda"),
                          collate_fn=collate)

    layer_idx = backbone.resolve_layer(layer)
    captured  = {}
    h = _register_cls_hook(backbone, layer_idx, captured)

    all_latents, all_labels = [], []

    for pixel_values, labels in tqdm(loader, desc="  extract latents", leave=False):
        pixel_values = pixel_values.to(device)
        _forward_backbone(backbone, pixel_values)

        if pool == "cls" or is_align:
            act = captured["cls"]                              # [B, d]
        else:
            seq = captured["seq"]                              # [B, T, d] or [B, C, H, W]
            if seq.ndim == 4:
                act = seq.mean(dim=[-2, -1])
            else:
                act = seq.mean(dim=1)                          # mean-pool tokens

        _, latents, _ = sae(act)                               # [B, d_sae]
        all_latents.append(latents.cpu().half().numpy())
        all_labels.append(labels.numpy())

    h.remove()
    return (np.concatenate(all_latents).astype(np.float32),
            np.concatenate(all_labels))


def linear_probe_accuracy(
    backbone,
    train_dataset,
    test_dataset,
    sae,
    label_key:    str   = "label",
    layer:        int   = -1,
    pool:         str   = "cls",
    batch_size:   int   = 64,
    device:       str   = "cuda",
    num_workers:  int   = 4,
    probe_max_iter: int = 1000,
    also_clip_features: bool = True,
) -> dict:
    """Fit a logistic regression on SAE latents; compare to raw backbone features.

    Equivalent to eval_sae_classifier.py but backbone-agnostic and using
    the unified hook infrastructure.

    Returns
    -------
    dict with "sae_latents" entry (and optionally "clip_features" baseline):
      {
        "sae_latents":   {"test_acc": float, "train_acc": float, "feature_dim": int, "sparsity": float},
        "clip_features": {"test_acc": float, "train_acc": float, "feature_dim": int},
      }
    """
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import MaxAbsScaler

    def _probe(X_tr, y_tr, X_te, y_te):
        scaler = MaxAbsScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        clf = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=probe_max_iter,
                            tol=1e-3, random_state=42, n_jobs=-1)
        clf.fit(X_tr_s, y_tr)
        return clf.score(X_te_s, y_te) * 100, clf.score(X_tr_s, y_tr) * 100

    results = {}

    # SAE latents
    print("  Extracting train SAE latents...")
    X_tr, y_tr = _extract_sae_latents(backbone, sae, train_dataset, label_key,
                                       layer, batch_size, device, num_workers, pool)
    print("  Extracting test SAE latents...")
    X_te, y_te = _extract_sae_latents(backbone, sae, test_dataset, label_key,
                                       layer, batch_size, device, num_workers, pool)

    test_acc, train_acc = _probe(X_tr, y_tr, X_te, y_te)
    sparsity = float((X_tr == 0).mean())
    results["sae_latents"] = {
        "test_acc":    round(test_acc,  4),
        "train_acc":   round(train_acc, 4),
        "feature_dim": X_tr.shape[1],
        "sparsity":    round(sparsity,  4),
    }
    print(f"  SAE latents: train={train_acc:.2f}%  test={test_acc:.2f}%  "
          f"d={X_tr.shape[1]}  sparsity={sparsity:.1%}")

    # Optional: raw backbone features for comparison (CLIP image projection)
    if also_clip_features and backbone.__class__.__name__ == "CLIPBackbone":
        model  = backbone.model
        is_align = False
        collate  = _make_collate(backbone.processor, label_key)

        @torch.no_grad()
        def _extract_clip(ds):
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers,
                                pin_memory=(device == "cuda"), collate_fn=collate)
            feats, lbls = [], []
            for pv, lbl in tqdm(loader, desc="  clip features", leave=False):
                f = F.normalize(model.get_image_features(pv.to(device)).float(), dim=-1)
                feats.append(f.cpu().numpy()); lbls.append(lbl.numpy())
            return np.concatenate(feats), np.concatenate(lbls)

        print("  Extracting CLIP features (baseline)...")
        X_tr_c, _ = _extract_clip(train_dataset)
        X_te_c, _ = _extract_clip(test_dataset)
        test_c, train_c = _probe(X_tr_c, y_tr, X_te_c, y_te)
        results["clip_features"] = {
            "test_acc":    round(test_c,  4),
            "train_acc":   round(train_c, 4),
            "feature_dim": X_tr_c.shape[1],
        }
        print(f"  CLIP features: train={train_c:.2f}%  test={test_c:.2f}%  d={X_tr_c.shape[1]}")

    return results


# ── 3. Feature masking / causal ablation ─────────────────────────────────────

@torch.no_grad()
def feature_masking_accuracy(
    backbone,
    sae,
    dataset,
    classnames:    List[str],
    cls_sae_cnt:   np.ndarray,   # [N_CLASS, N_FEAT] from compute_feature_stats
    label_key:     str   = "label",
    topk_list:     Optional[List[int]] = None,
    layer:         int   = -1,
    batch_size:    int   = 32,
    device:        str   = "cuda",
    num_workers:   int   = 4,
) -> dict:
    """Causal ablation: clamp top-K SAE features ON or OFF, measure accuracy change.

    For each class c and each top-K value k:
      - top-K features for class c = highest cls_sae_cnt[c] features
      - "on"  mode: force those K features to their mean active value, others to 0
      - "off" mode: zero those K features out, keep rest unchanged
      - measure classification accuracy on images of class c

    Equivalent to tasks/classification_with_top_k_masking.py but using the
    unified backbone/SAE hook infrastructure.

    Returns
    -------
    dict: {class_idx: {condition: accuracy, ...}, ...}
    where conditions are "baseline", "on_K", "off_K" for each K in topk_list.
    """
    _require_clip(backbone)

    if topk_list is None:
        topk_list = [1, 2, 5, 10, 50, 100, 500, 1000]

    model      = backbone.model
    processor  = backbone.processor
    text_feats = _get_text_features(backbone, classnames, device)

    layer_idx = backbone.resolve_layer(layer)
    n_feat    = sae.W_enc.shape[1]
    captured  = {}
    h = _register_cls_hook(backbone, layer_idx, captured)

    def _classify_with_clamped_sae(pixel_values: torch.Tensor,
                                   mask_on: torch.Tensor,
                                   mask_off: torch.Tensor) -> torch.Tensor:
        """
        mask_on  [d_sae]: bool — force these features to 1.0 (active)
        mask_off [d_sae]: bool — force these features to 0.0 (silent)
        Returns predicted class indices [B].
        """
        _forward_backbone(backbone, pixel_values)
        cls_act = captured["cls"]                               # [B, d]

        # SAE encode
        _, feat, _ = sae(cls_act)                              # [B, d_sae]

        # Apply clamping
        if mask_on.any():
            feat = feat.clone()
            feat[:, mask_on] = 1.0
        if mask_off.any():
            feat = feat.clone()
            feat[:, mask_off] = 0.0

        # Decode: W_dec is [d_sae, d_in], so feat @ W_dec gives [B, d_in]
        recon = feat @ sae.W_dec + sae.b_dec                  # [B, d_in]

        # Push through post-layernorm + projection
        cls_recon  = model.vision_model.post_layernorm(recon)
        img_feats  = F.normalize(model.visual_projection(cls_recon).float(), dim=-1)

        return (img_feats @ text_feats.T).argmax(dim=-1)

    # Group dataset by class
    from collections import defaultdict
    by_class = defaultdict(list)
    for i in range(len(dataset)):
        lbl = dataset[i][label_key] if isinstance(dataset[i], dict) else dataset[i][1]
        by_class[int(lbl)].append(i)

    results = {}

    for cls_idx, cls_name in enumerate(tqdm(classnames, desc="  masking")):
        indices = by_class.get(cls_idx, [])
        if not indices:
            continue

        # Build a small subset loader for this class
        from torch.utils.data import Subset
        subset  = Subset(dataset, indices)
        collate = _make_collate(processor, label_key)
        loader  = DataLoader(subset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate)

        # Feature rank for this class
        feat_rank = np.argsort(cls_sae_cnt[cls_idx])[::-1]   # highest first

        preds_per_cond = defaultdict(list)
        empty_on  = torch.zeros(n_feat, dtype=torch.bool, device=device)
        empty_off = torch.zeros(n_feat, dtype=torch.bool, device=device)

        for pv, _ in loader:
            pv = pv.to(device)

            # Baseline (no clamping)
            _forward_backbone(backbone, pv)
            img_feats = F.normalize(model.get_image_features(pv).float(), dim=-1)
            preds_per_cond["baseline"].extend(
                (img_feats @ text_feats.T).argmax(dim=-1).cpu().tolist())

            for k in topk_list:
                top_feat = feat_rank[:k]
                on_mask  = empty_on.clone()
                off_mask = empty_off.clone()
                on_mask[top_feat]  = True
                off_mask[top_feat] = True

                preds_per_cond[f"on_{k}"].extend(
                    _classify_with_clamped_sae(pv, on_mask, empty_off).cpu().tolist())
                preds_per_cond[f"off_{k}"].extend(
                    _classify_with_clamped_sae(pv, empty_on, off_mask).cpu().tolist())

        n = len(indices)
        results[cls_idx] = {
            cond: sum(1 for p in preds if p == cls_idx) / n * 100
            for cond, preds in preds_per_cond.items()
        }

    h.remove()
    return results
