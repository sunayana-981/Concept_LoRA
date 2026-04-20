"""Monosemanticity metrics from class-wise SAE activation counts.

All metrics are computed from `cls_sae_cnt` [N_CLASS, N_FEAT] — the per-class
firing count matrix produced by feature_stats.py.  No extra forward pass needed.

Two complementary monosemanticity measures:

  selectivity   = 1 - H_norm       (H_norm = entropy / log(N_CLASSES))
    → 1 = fires only for one class, 0 = fires uniformly across all classes.
    Equivalent to the repo's original compute_class_wise_sae_activation.py metric.

  label_entropy_norm = H / log(N_CLASSES)  (weighted mean across alive features)
    → 0 = monosemantic, 1 = polysemantic.
    Equivalent to compute_sae_eval_metrics.py's label_entropy_norm.
    Weights each feature by its total fire count so high-frequency features
    contribute more to the aggregate score.

Noisy filter: adaptive 95th-percentile of alive mean_acts
(avoids hardcoding a scale-dependent constant like CLIP's 0.1).
"""

import math

import numpy as np


def compute_monosemanticity(
    cls_sae_cnt: np.ndarray,    # [N_CLASS, N_FEAT]
    sae_mean_acts: np.ndarray,  # [N_FEAT]
    sae_sparsity: np.ndarray,   # [N_FEAT]  (fraction of images that fired)
) -> dict:
    """Return all monosemanticity metrics and masks for all features.

    Computes selectivity, top1_fraction, AND label_entropy in a single pass
    over cls_sae_cnt — no extra backbone/SAE forward passes required.
    """
    N_CLASS, N_FEAT = cls_sae_cnt.shape

    dead_mask  = sae_sparsity == 0
    alive_mask = ~dead_mask

    top1_fraction  = np.zeros(N_FEAT)
    selectivity    = np.zeros(N_FEAT)   # 1 - H_norm  (high = monosemantic)
    label_entropy  = np.zeros(N_FEAT)   # H in nats
    label_entropy_norm = np.zeros(N_FEAT)  # H / log(N_CLASS) ∈ [0,1]

    log_n = math.log(max(N_CLASS, 2))
    fire_counts = np.zeros(N_FEAT)      # total fires per feature (for weighting)

    for f in np.where(alive_mask)[0]:
        cnt = cls_sae_cnt[:, f]
        total = cnt.sum()
        if total == 0:
            continue
        p = cnt / total
        top1_fraction[f] = p.max()
        nonzero = p[p > 0]
        H = -(nonzero * np.log(nonzero)).sum()
        H_norm = H / log_n
        selectivity[f]       = 1.0 - H_norm
        label_entropy[f]     = H
        label_entropy_norm[f] = H_norm
        fire_counts[f]       = total

    # Adaptive noisy threshold: 95th percentile of alive mean_acts
    alive_mean = sae_mean_acts[alive_mask]
    adaptive_noisy_thr = float(np.percentile(alive_mean, 95)) if len(alive_mean) > 0 else 0.1

    noisy_mask  = (sae_mean_acts > adaptive_noisy_thr) & alive_mask
    usable_mask = alive_mask & ~noisy_mask

    n_dead, n_alive = int(dead_mask.sum()), int(alive_mask.sum())
    n_noisy, n_usable = int(noisy_mask.sum()), int(usable_mask.sum())

    alive_sel  = selectivity[alive_mask]
    alive_top1 = top1_fraction[alive_mask]
    alive_he   = label_entropy_norm[alive_mask]
    usable_sel = selectivity[usable_mask]
    alive_w    = fire_counts[alive_mask]

    # Weighted label entropy mean (weight by feature fire count)
    w_sum = alive_w.sum()
    label_entropy_weighted_norm = float((alive_he * alive_w).sum() / w_sum) if w_sum > 0 else 0.0
    label_entropy_unweighted_norm = float(alive_he.mean()) if n_alive > 0 else 0.0

    metrics = {
        "dead_pct":              n_dead  / N_FEAT * 100,
        "alive_pct":             n_alive / N_FEAT * 100,
        "noisy_pct":             n_noisy / N_FEAT * 100,
        "usable_pct":            n_usable / N_FEAT * 100,
        "n_dead":                n_dead,
        "n_alive":               n_alive,
        "n_noisy":               n_noisy,
        "n_usable":              n_usable,
        "adaptive_noisy_thr":    adaptive_noisy_thr,
        # selectivity-based (1 - H_norm): high = monosemantic
        "top1_fraction_mean":      float(alive_top1.mean()) if n_alive > 0 else 0.0,
        "selectivity_mean":        float(alive_sel.mean())  if n_alive > 0 else 0.0,
        "frac_sel_gt_0.5":         float((alive_sel > 0.5).mean())  if n_alive > 0 else 0.0,
        "frac_sel_gt_0.75":        float((alive_sel > 0.75).mean()) if n_alive > 0 else 0.0,
        "frac_sel_gt_0.9":         float((alive_sel > 0.90).mean()) if n_alive > 0 else 0.0,
        "usable_selectivity_mean": float(usable_sel.mean()) if n_usable > 0 else 0.0,
        "usable_frac_sel_gt_0.5":  float((usable_sel > 0.5).mean()) if n_usable > 0 else 0.0,
        # entropy-based: low = monosemantic (complementary view, same underlying data)
        "label_entropy_weighted_norm":   label_entropy_weighted_norm,
        "label_entropy_unweighted_norm": label_entropy_unweighted_norm,
    }

    masks = {
        "dead_mask":   dead_mask,
        "alive_mask":  alive_mask,
        "noisy_mask":  noisy_mask,
        "usable_mask": usable_mask,
    }

    arrays = {
        "top1_fraction":       top1_fraction,
        "selectivity":         selectivity,
        "label_entropy":       label_entropy,
        "label_entropy_norm":  label_entropy_norm,
    }

    return {**metrics, **masks, **arrays}
