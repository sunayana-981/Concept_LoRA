#!/usr/bin/env python3
"""
Distribution-Aware Monosemanticity Score (DAMS)  — v2 (bugfixed)

Fixes from v1:
  - EC now auto-detects SAE training layer from checkpoint config and
    validates layer match; also adds diagnostic for b_dec loading.
  - FSS uses bounded precision × recall:  specificity(f,c) = P(c|f) × P(f|c)
    which is in [0, 1] by construction.
  - EC computation uses per-dimension MSE / per-dimension Var (true R²).

Decomposes SAE quality under domain shift into three components:

  1. Effective Coverage (EC)  — R²-style reconstruction quality in feature space.
     Gates the entire score: a SAE that can't reconstruct the target domain
     cannot appear spuriously monosemantic.

  2. Concept Separability Score (CSS) — Fisher discriminant ratio in SAE
     activation space.  Measures whether the SAE geometry separates target-
     domain concepts, regardless of individual feature semantics.

  3. Feature Specificity Score (FSS) — probabilistic precision × recall per
     concept, max-aggregated over features.  Measures whether each concept
     has a dedicated, non-shared region of feature space.

  4. Domain Alignment Score (DAS) — class-balanced kernel target alignment
     between SAE concept activations and target-domain labels.  Measures
     whether concept geometry is statistically dependent on the domain class
     partition.

  5. SAE Utility Score (SUS) — cross-validated, chance-normalised balanced
     accuracy of a ridge readout on frozen SAE activations.  Measures whether
     the frozen concept representation is actually useful for this dataset.

Composite:
    DAMS = EC^rho × (α × CSS_norm + β × FSS + γ × DAS)

Usage (as library):
    from src.metrics.dams import compute_dams, DAMSResult

    result = compute_dams(
        sae=sae_model,
        features=clip_features,       # (N, seq_len, d_model)
        labels=labels,                # List[int], length N
        num_classes=10,
        device="cuda",
    )
    print(result)
"""

import argparse
import os
import sys
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════
# Result Container
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DAMSResult:
    """Stores all DAMS sub-metrics and the composite score.

    DAMS = EC^rho × (α × CSS_norm + β × FSS + γ × DAS)

    Component mapping (default new metrics):
        EC        — CKA(X_cls, A_pool) with sigmoid kernel in concept space
        CSS_norm  — Normalised inter-class MMD in SAE concept space
        FSS       — Entropy-based feature specificity (raw activation weights)
        DAS       — Class-balanced CKA(A_pool, Y), a normalized HSIC dependence
                    score between SAE concept geometry and target-domain labels
    """

    # --- Component scores ---
    ec: float            # EC ∈ [0, 1]  (kernel alignment / concept coverage)
    ec_metric: str       # 'cka' | 'r2'
    css_raw: float       # Raw CSS value (unbounded): MMD mean or Fisher ratio
    css_norm: float      # Normalised CSS ∈ [0, 1] via saturation: raw/(raw+κ)
    css_metric: str      # 'mmd' | 'fisher'
    fss: float           # FSS ∈ [0, 1]  (feature specificity)
    fss_method: str      # 'entropy' | 'threshold'
    das: float           # DAS ∈ [0, 1]  (domain/label dependence)
    das_metric: str
    utility: float       # SUS ∈ [0, 1]  (chance-normalised readout utility)
    utility_balanced_acc: float
    utility_chance: float
    utility_metric: str

    # --- Composite ---
    alpha: float
    beta: float
    gamma: float
    coverage_power: float
    coverage_gate: float
    dams: float          # EC^rho × (alpha × CSS_norm + beta × FSS + gamma × DAS)

    # --- Diagnostics ---
    recon_mse_per_dim: float
    baseline_var_per_dim: float
    n_samples: int
    n_classes: int
    n_features: int
    layer_match: bool
    b_dec_norm: float
    css_per_pair: Optional[Dict[str, float]] = field(default_factory=dict)
    fss_per_class: Optional[Dict[int, float]] = field(default_factory=dict)

    def __repr__(self):
        layer_warn = "" if self.layer_match else "  ⚠ LAYER MISMATCH — EC is unreliable\n"
        bdec_warn  = "" if self.b_dec_norm > 1e-3 else "  ⚠ b_dec ≈ 0 — decoder bias may not have loaded\n"
        lines = [
            "DAMSResult(",
            layer_warn + bdec_warn,
            f"  DAMS                    = {self.dams:.4f}",
            f"  EC  [{self.ec_metric:<5}]          = {self.ec:.4f}"
            f"  (MSE/dim={self.recon_mse_per_dim:.4f})",
            f"  CSS [{self.css_metric:<6}] raw/norm = {self.css_raw:.4f} / {self.css_norm:.4f}",
            f"  FSS [{self.fss_method:<9}]     = {self.fss:.4f}",
            f"  DAS [{self.das_metric:<9}]     = {self.das:.4f}",
            f"  SUS [{self.utility_metric:<9}]     = {self.utility:.4f}"
            f"  (bal_acc={self.utility_balanced_acc:.4f}, chance={self.utility_chance:.4f})",
            f"  α={self.alpha}, β={self.beta}, γ={self.gamma}, ρ={self.coverage_power}",
            f"  n={self.n_samples}, C={self.n_classes}, d_sae={self.n_features}",
            f"  layer_match={self.layer_match}, ||b_dec||={self.b_dec_norm:.4f}",
            ")",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, (np.floating, np.integer)):
                d[k] = float(v)
        return d


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostics: SAE health checks
# ═══════════════════════════════════════════════════════════════════════════

def check_sae_b_dec(sae) -> float:
    """Return ||b_dec|| to detect zero-initialised decoder bias."""
    if hasattr(sae, 'b_dec'):
        return float(sae.b_dec.data.norm().item())
    for attr in ['bias_dec', 'b_pre', 'decoder_bias']:
        if hasattr(sae, attr):
            return float(getattr(sae, attr).data.norm().item())
    return 0.0


def infer_sae_training_layer(sae, sae_cfg) -> Optional[int]:
    """Try to infer which ViT layer this SAE was trained on."""
    if hasattr(sae_cfg, 'block_layer'):
        return sae_cfg.block_layer
    if isinstance(sae_cfg, dict):
        return sae_cfg.get('block_layer', None)
    return None


# ═══════════════════════════════════════════════════════════════════════════
# 1. Effective Coverage (EC)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_effective_coverage(
    sae,
    features: torch.Tensor,
    device: str = "cuda",
    batch_size: int = 512,
) -> Tuple[float, float, float]:
    """
    EC = 1 − Σ_d MSE_d / Σ_d Var_d

    Standard multivariate R²: fraction of total variance explained.
    Computed per-dimension then aggregated.
    Patch tokens only (CLS excluded for ViT features).

    Returns:
        ec, avg_mse_per_dim, avg_var_per_dim
    """
    has_seq = features.ndim == 3

    if has_seq:
        patches = features[:, 1:, :]
        flat = patches.reshape(-1, patches.shape[-1])
    else:
        flat = features

    M, D = flat.shape

    sum_x = torch.zeros(D, dtype=torch.float64)
    sum_x2 = torch.zeros(D, dtype=torch.float64)
    sum_se = torch.zeros(D, dtype=torch.float64)
    count = 0

    for i in tqdm(range(0, M, batch_size), desc="EC: reconstruct", leave=False):
        chunk = flat[i : i + batch_size].to(device)

        if hasattr(sae, "decode") and hasattr(sae, "encode"):
            z = sae.encode(chunk)
            x_hat = sae.decode(z)
        else:
            out = sae(chunk)
            x_hat = out[0] if isinstance(out, tuple) else out

        se = (chunk - x_hat).pow(2)
        sum_se += se.sum(dim=0).cpu().double()

        chunk_cpu = chunk.cpu().double()
        sum_x += chunk_cpu.sum(dim=0)
        sum_x2 += (chunk_cpu ** 2).sum(dim=0)
        count += chunk.shape[0]

        del chunk, x_hat, se
        if device == "cuda":
            torch.cuda.empty_cache()

    mse_per_dim = sum_se / count
    mean_x = sum_x / count
    var_per_dim = sum_x2 / count - mean_x ** 2

    # Clamp negative variances from floating-point error
    var_per_dim = var_per_dim.clamp(min=0.0)

    total_mse = mse_per_dim.sum().item()
    total_var = var_per_dim.sum().item()

    avg_mse_per_dim = total_mse / D
    avg_var_per_dim = total_var / D

    ec = max(0.0, 1.0 - total_mse / total_var) if total_var > 1e-12 else 0.0

    # ── EC diagnostics ────────────────────────────────────────────────────
    print(f"    [EC debug] M={M} patches, D={D} dims, count={count}")
    print(f"    [EC debug] total_mse={total_mse:.6f}, total_var={total_var:.6f}")
    print(f"    [EC debug] ratio mse/var = {total_mse/total_var:.6f}" if total_var > 1e-12 else "    [EC debug] total_var ≈ 0 — features are constant!")
    n_neg_var = (var_per_dim < 0).sum().item()
    if n_neg_var > 0:
        print(f"    [EC debug] ⚠ {n_neg_var}/{D} dims had negative variance (clamped to 0)")
    zero_var_dims = (var_per_dim < 1e-10).sum().item()
    print(f"    [EC debug] {zero_var_dims}/{D} dims have near-zero variance")
    # Check x_hat shape by probing first batch
    print(f"    [EC debug] Features shape into EC: {features.shape}")

    return float(ec), float(avg_mse_per_dim), float(avg_var_per_dim)


# ═══════════════════════════════════════════════════════════════════════════
# 1b. Kernel Alignment Score — CKA (replaces R²-based EC)
# ═══════════════════════════════════════════════════════════════════════════

def _hsic_from_grams(Kc: torch.Tensor, Lc: torch.Tensor) -> float:
    """HSIC from two pre-centred gram matrices."""
    n = Kc.shape[0]
    return float((Kc * Lc).sum() / (n - 1) ** 2)


def _centre_gram(K: torch.Tensor) -> torch.Tensor:
    """Double-centre a gram matrix: Kc = HKH."""
    row = K.mean(dim=1, keepdim=True)
    col = K.mean(dim=0, keepdim=True)
    return K - row - col + K.mean()


def _hsic_linear(A: torch.Tensor, B: torch.Tensor) -> float:
    """
    Unbiased linear HSIC estimator.
    A, B: (N, D) feature matrices.  Builds gram matrices internally.
    """
    Kc = _centre_gram(A @ A.T)
    Lc = _centre_gram(B @ B.T)
    return _hsic_from_grams(Kc, Lc)


@torch.no_grad()
def compute_kernel_alignment(
    sae,
    features: torch.Tensor,            # (N, seq_len, d_model) or (N, d_model)
    device: str = "cuda",
    batch_size: int = 512,
    subsample: int = 2000,
    precomputed_acts: Optional[torch.Tensor] = None,  # (N, d_sae) — reuse if available
) -> Tuple[float, float, float]:
    """
    CKA between the CLIP feature space (K_X) and the SAE concept space (K_A).

    Why this captures domain adaptation:
    ─────────────────────────────────────
    • X_pool: per-image CLS-token CLIP features (N, d_model).
      K_X[i,j] = cos-similarity structure of CLIP on the target domain.
    • A_pool: per-image max-pooled SAE activations (N, d_sae).
      K_A[i,j] = sigmoid( a_i·a_j / d_sae )  ← sigmoid kernel in concept space.
      The sigmoid kernel avoids R²-like linear dominance by large-norm activations.

    CKA(K_X, K_A) asks: "does the neighbourhood structure of CLIP features
    map to the same neighbourhood structure in the SAE concept space?"

    • A domain-adapted SAE (trained on the target domain) organises its
      feature dictionary around domain-specific patterns → high CKA.
    • A generic base SAE (trained on ImageNet) uses concepts irrelevant to the
      target domain → lower CKA, because its activation geometry is misaligned.

    Old CKA(X, X̂) ≈ 0.999 for all SAEs because both reconstruct well.
    CKA(X_pool, A_pool) separates base from adapted because it measures
    structural alignment of the *concept space*, not reconstruction quality.

    Returns:
        cka_score ∈ [0, 1], avg_recon_mse_per_dim (diagnostic), avg_var_per_dim
    """
    has_seq = features.ndim == 3

    # ── Per-image CLS features ────────────────────────────────────────────
    # CLS token is index 0; it summarises the whole image in CLIP
    if has_seq:
        X_pool = features[:, 0, :].float()   # (N, d_model)
    else:
        X_pool = features.float()

    N, d_model = X_pool.shape

    # ── SAE activations (per-image, max-pooled over patches) ─────────────
    if precomputed_acts is not None:
        A_pool = precomputed_acts.float()     # (N, d_sae)
        print(f"    [CKA] Using precomputed activations: {A_pool.shape}")
    else:
        print(f"    [CKA] Computing SAE activations ({N} images)...")
        A_pool = _compute_pooled_activations(
            sae, features, device=device, batch_size=batch_size,
        ).float()                             # (N, d_sae)

    d_sae = A_pool.shape[1]

    # ── Reconstruction MSE diagnostic (on CLS token only) ────────────────
    mse_sum = 0.0
    var_sum = 0.0
    for i in range(0, N, batch_size):
        chunk = X_pool[i : i + batch_size].to(device)
        out = sae(chunk)
        x_hat = out[0] if isinstance(out, (tuple, list)) else out
        mse_sum += (chunk - x_hat.to(chunk.dtype)).pow(2).mean(dim=-1).sum().item()
        var_sum += chunk.var(dim=0).sum().item()
        del chunk, x_hat
        if device == "cuda":
            torch.cuda.empty_cache()
    avg_mse_per_dim = mse_sum / max(N, 1) / d_model
    avg_var_per_dim = var_sum / max(N, 1) / d_model

    # ── Subsample images for gram matrix (O(N²) cost) ────────────────────
    if N > subsample:
        idx = torch.randperm(N)[:subsample]
        X_sub = X_pool[idx]
        A_sub = A_pool[idx]
        print(f"    [CKA] Subsampling {N} → {subsample} images for gram matrices")
    else:
        X_sub = X_pool
        A_sub = A_pool

    # ── Gram matrices ─────────────────────────────────────────────────────
    # CLIP feature space: linear kernel (standard inner product).
    # Linear kernel preserves the cosine-similarity structure of CLIP features,
    # which is what CLIP was trained to produce.
    K_X = X_sub @ X_sub.T                                 # (n, n)

    # SAE concept space: sigmoid kernel
    #   K_A[i,j] = sigmoid( a_i·a_j / d_sae )
    #   Compresses scale so large-norm activations don't dominate.
    K_A = torch.sigmoid((A_sub @ A_sub.T) / d_sae)        # (n, n)

    Kc = _centre_gram(K_X)
    Lc = _centre_gram(K_A)

    hsic_xy = _hsic_from_grams(Kc, Lc)
    hsic_xx = _hsic_from_grams(Kc, Kc)
    hsic_yy = _hsic_from_grams(Lc, Lc)

    denom = math.sqrt(max(hsic_xx, 0.0) * max(hsic_yy, 0.0))
    cka = float(max(0.0, min(1.0, hsic_xy / denom if denom > 1e-12 else 0.0)))

    print(f"    [CKA] n={X_sub.shape[0]}, d_model={d_model}, d_sae={d_sae}")
    print(f"    [CKA] HSIC(K_X,K_A)={hsic_xy:.6f}, "
          f"HSIC(K_X,K_X)={hsic_xx:.6f}, HSIC(K_A,K_A)={hsic_yy:.6f}")
    print(f"    [CKA] CKA(X_cls [linear], A_pool [sigmoid]) = {cka:.4f}")

    return cka, avg_mse_per_dim, avg_var_per_dim


# ═══════════════════════════════════════════════════════════════════════════
# 1c. MMD Score — inter-class separation in SAE concept space
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_mmd_score(
    sae,
    features: torch.Tensor,            # (N, seq_len, d_model) or (N, d_model)
    labels: List[int],
    num_classes: int,
    device: str = "cuda",
    batch_size: int = 512,
    subsample_per_class: int = 200,
    precomputed_acts: Optional[torch.Tensor] = None,  # (N, d_sae)
) -> Tuple[float, float, float]:
    """
    Mean pairwise MMD² between class distributions in SAE concept space.

    Why this captures domain adaptation:
    ─────────────────────────────────────
    • For each class c, form the set of per-image SAE activation vectors A_c.
    • Compute MMD²(A_i, A_j) for every pair of classes using an RBF kernel
      with median-heuristic bandwidth estimated from A.
    • Score = mean_pairs MMD²  (higher = classes are more separated in concept
      space → the SAE has learned domain-specific, class-discriminative features).

    A domain-adapted SAE concentrates its active features around target-domain
    concepts, pushing class distributions further apart.
    A generic base SAE fires on ImageNet-generic features that are shared across
    domain classes, so class distributions overlap heavily → low MMD².

    Note: the old implementation measured MMD(X, X̂), which is reconstruction
    quality and is high for both SAEs equally.

    Returns:
        mmd_mean ∈ [0, ∞) — higher is better (more separated classes),
        avg_mse_per_dim (diagnostic, CLS-token MSE),
        avg_var_per_dim (diagnostic)
    """
    has_seq = features.ndim == 3
    N = features.shape[0]
    d_model = features.shape[-1]

    # ── Diagnostic: CLS-token reconstruction MSE ─────────────────────────
    X_cls = features[:, 0, :].float() if has_seq else features.float()
    mse_sum = 0.0
    var_sum = 0.0
    for i in range(0, N, batch_size):
        chunk = X_cls[i : i + batch_size].to(device)
        out = sae(chunk)
        x_hat = out[0] if isinstance(out, (tuple, list)) else out
        mse_sum += (chunk - x_hat.to(chunk.dtype)).pow(2).mean(dim=-1).sum().item()
        var_sum += chunk.var(dim=0).sum().item()
        del chunk, x_hat
        if device == "cuda":
            torch.cuda.empty_cache()
    avg_mse_per_dim = mse_sum / max(N, 1) / d_model
    avg_var_per_dim = var_sum / max(N, 1) / d_model

    # ── SAE activations (per-image, max-pooled) ───────────────────────────
    if precomputed_acts is not None:
        A_all = precomputed_acts.float()
        print(f"    [MMD] Using precomputed activations: {A_all.shape}")
    else:
        print(f"    [MMD] Computing SAE activations ({N} images)...")
        A_all = _compute_pooled_activations(
            sae, features, device=device, batch_size=batch_size,
        ).float()

    d_sae = A_all.shape[1]
    labels_t = torch.tensor(labels, dtype=torch.long)

    # Collect per-class activation sets (subsampled)
    class_acts: Dict[int, torch.Tensor] = {}
    for c in range(num_classes):
        idx_c = (labels_t == c).nonzero(as_tuple=True)[0]
        if idx_c.numel() == 0:
            continue
        if idx_c.numel() > subsample_per_class:
            perm = torch.randperm(idx_c.numel())[:subsample_per_class]
            idx_c = idx_c[perm]
        class_acts[c] = A_all[idx_c]  # (n_c, d_sae)

    valid = [c for c in class_acts if class_acts[c].shape[0] >= 2]
    if len(valid) < 2:
        print("    [MMD] ⚠ fewer than 2 classes with samples; returning 0")
        return 0.0, avg_mse_per_dim, avg_var_per_dim

    # Median-heuristic bandwidth from a pooled subsample
    pool = torch.cat([class_acts[c][:50] for c in valid], dim=0)
    dists2 = torch.cdist(pool, pool).pow(2)
    sigma2 = float(dists2[dists2 > 0].median().item() / 2.0)
    sigma2 = max(sigma2, 1e-6)

    def rbf(A, B):
        return torch.exp(-torch.cdist(A, B).pow(2) / (2 * sigma2))

    def mmd2_unbiased(X, Y):
        nx, ny = X.shape[0], Y.shape[0]
        Kxx = rbf(X, X); Kxy = rbf(X, Y); Kyy = rbf(Y, Y)
        return (
            (Kxx.sum() - Kxx.trace()) / max(nx * (nx - 1), 1)
            + (Kyy.sum() - Kyy.trace()) / max(ny * (ny - 1), 1)
            - 2 * Kxy.mean()
        ).item()

    total_mmd2 = 0.0
    n_pairs = 0
    for i, ci in enumerate(valid):
        for cj in valid[i + 1:]:
            total_mmd2 += max(0.0, mmd2_unbiased(class_acts[ci], class_acts[cj]))
            n_pairs += 1

    mmd_mean = total_mmd2 / max(n_pairs, 1)
    print(f"    [MMD] σ²={sigma2:.4f}, mean_pairwise_MMD²={mmd_mean:.6f} "
          f"over {n_pairs} pairs ({len(valid)} classes), d_sae={d_sae}")

    return mmd_mean, avg_mse_per_dim, avg_var_per_dim


# ═══════════════════════════════════════════════════════════════════════════
# 1d. Domain Alignment Score — class-balanced kernel target alignment
# ═══════════════════════════════════════════════════════════════════════════

def _class_balanced_indices(
    labels_t: torch.Tensor,
    max_samples: int,
    min_class_count: int = 2,
) -> torch.Tensor:
    """Deterministic class-balanced subsample for O(N^2) kernel metrics."""
    valid_classes = []
    for c in labels_t.unique(sorted=True).tolist():
        idx_c = (labels_t == int(c)).nonzero(as_tuple=True)[0]
        if idx_c.numel() >= min_class_count:
            valid_classes.append((int(c), idx_c))

    if not valid_classes:
        return torch.empty(0, dtype=torch.long)

    per_class = max(min_class_count, int(math.ceil(max_samples / len(valid_classes))))
    chunks = [idx_c[:per_class] for _, idx_c in valid_classes]
    idx = torch.cat(chunks, dim=0)
    if idx.numel() > max_samples:
        idx = idx[:max_samples]
    return idx.sort().values


def _balanced_label_kernel(labels_t: torch.Tensor) -> torch.Tensor:
    """
    Class-balanced label kernel.

    Y_ic = 1/sqrt(n_c) if sample i belongs to class c, else 0.
    K_Y = Y Y^T gives every class equal total mass, preventing common
    classes from dominating the alignment.
    """
    classes, inverse = torch.unique(labels_t, sorted=True, return_inverse=True)
    oh = F.one_hot(inverse, num_classes=len(classes)).float()
    counts = oh.sum(dim=0).clamp(min=1.0)
    Y = oh / counts.sqrt().unsqueeze(0)
    return Y @ Y.T


def _normalised_cka_from_grams(K: torch.Tensor, L: torch.Tensor) -> float:
    Kc = _centre_gram(K)
    Lc = _centre_gram(L)
    hsic_kl = _hsic_from_grams(Kc, Lc)
    hsic_kk = _hsic_from_grams(Kc, Kc)
    hsic_ll = _hsic_from_grams(Lc, Lc)
    denom = math.sqrt(max(hsic_kk, 0.0) * max(hsic_ll, 0.0))
    if denom <= 1e-12:
        return 0.0
    return float(max(0.0, min(1.0, hsic_kl / denom)))


@torch.no_grad()
def compute_domain_alignment_score(
    activations: torch.Tensor,
    labels: List[int],
    num_classes: int,
    subsample: int = 2000,
    min_class_count: int = 2,
) -> float:
    """
    Domain Alignment Score (DAS): class-balanced CKA between SAE concept
    activations and the target-domain label kernel.

    DAS = CKA(K_A, K_Y)

    K_A is a cosine kernel over log-compressed non-negative SAE activations.
    K_Y is a class-balanced label kernel: samples in the same class are
    similar, but each class contributes equal total mass. This is normalized
    HSIC, so it has a direct dependence-testing interpretation:

        DAS is high when the SAE concept geometry is statistically aligned
        with the target-domain class partition.

    Unlike EC, DAS does not reward merely preserving generic CLIP geometry.
    Unlike raw separability, it is normalized and class-balanced.
    """
    labels_t = torch.tensor(labels, dtype=torch.long)
    idx = _class_balanced_indices(labels_t, max_samples=subsample, min_class_count=min_class_count)
    if idx.numel() < 4:
        print("    [DAS] fewer than 4 balanced samples; returning 0")
        return 0.0

    A = torch.log1p(activations[idx].float().clamp(min=0.0))
    y = labels_t[idx]

    # Remove dead dimensions on the selected subset, then row-normalize.
    keep = A.var(dim=0) > 1e-12
    if keep.any():
        A = A[:, keep]
    A = F.normalize(A, p=2, dim=1, eps=1e-12)

    K_A = A @ A.T
    K_Y = _balanced_label_kernel(y)
    das = _normalised_cka_from_grams(K_A, K_Y)

    n_valid_classes = int(torch.unique(y).numel())
    print(f"    [DAS] n={idx.numel()}, classes={n_valid_classes}/{num_classes}, "
          f"d_eff={A.shape[1]}, CKA(K_A,K_Y)={das:.4f}")
    return das


# ═══════════════════════════════════════════════════════════════════════════
# 1e. SAE Utility Score — frozen readout utility
# ═══════════════════════════════════════════════════════════════════════════

def _balanced_accuracy_score(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
) -> Tuple[float, int]:
    recalls = []
    for c in range(num_classes):
        mask = y_true == c
        if mask.any():
            recalls.append(float((y_pred[mask] == c).float().mean().item()))
    return (float(np.mean(recalls)) if recalls else 0.0, len(recalls))


def _stratified_fold_indices(
    labels_t: torch.Tensor,
    n_splits: int,
    min_class_count: int = 2,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Deterministic stratified folds with rare classes kept when possible."""
    folds: List[List[int]] = [[] for _ in range(n_splits)]
    all_idx = torch.arange(labels_t.numel())

    for c in labels_t.unique(sorted=True).tolist():
        idx_c = (labels_t == int(c)).nonzero(as_tuple=True)[0]
        if idx_c.numel() < min_class_count:
            continue
        for j, idx in enumerate(idx_c.tolist()):
            folds[j % n_splits].append(int(idx))

    split_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    used = set()
    for fold in folds:
        if not fold:
            continue
        test_idx = torch.tensor(sorted(fold), dtype=torch.long)
        used.update(test_idx.tolist())
        test_mask = torch.zeros(labels_t.numel(), dtype=torch.bool)
        test_mask[test_idx] = True
        train_idx = all_idx[~test_mask]
        if train_idx.numel() > 0:
            split_pairs.append((train_idx, test_idx))

    # If very rare classes were excluded from folds, they remain in training
    # only. That keeps test balanced accuracy well-defined.
    _ = used
    return split_pairs


def _prepare_readout_features(
    activations: torch.Tensor,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    top_features: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    weights = torch.log1p(activations.float().clamp(min=0.0))
    X_train = weights[train_idx]
    X_test = weights[test_idx]

    if top_features > 0 and top_features < X_train.shape[1]:
        # Unsupervised feature selection: high-variance SAE features carry
        # more signal, while avoiding label leakage from the held-out fold.
        var = X_train.var(dim=0)
        keep = var.argsort(descending=True)[:top_features]
        X_train = X_train[:, keep]
        X_test = X_test[:, keep]

    mu = X_train.mean(dim=0, keepdim=True)
    sigma = X_train.std(dim=0, keepdim=True).clamp(min=1e-6)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
    return X_train, X_test


@torch.no_grad()
def compute_sae_utility_score(
    activations: torch.Tensor,
    labels: List[int],
    num_classes: int,
    n_splits: int = 3,
    ridge: float = 1.0,
    top_features: int = 4096,
    min_class_count: int = 2,
) -> Tuple[float, float, float]:
    """
    SAE Utility Score (SUS): can a simple frozen readout use this SAE?

    For each deterministic stratified fold, train a closed-form ridge readout
    on log-compressed SAE activations and evaluate balanced accuracy on held-out
    examples. The reported score is chance-normalised:

        SUS = clip((balanced_acc - chance) / (1 - chance), 0, 1)

    where chance = 1 / (# classes represented in held-out folds).

    This is intentionally not a monosemanticity score. It is a decision score:
    if SUS is high for the base SAE, the frozen base representation is already
    useful on the dataset; if adapted SUS improves materially, adaptation buys
    practical value.
    """
    labels_t = torch.tensor(labels, dtype=torch.long)
    split_pairs = _stratified_fold_indices(labels_t, n_splits=n_splits, min_class_count=min_class_count)
    if not split_pairs:
        print("    [SUS] no valid stratified folds; returning 0")
        return 0.0, 0.0, 0.0

    fold_bacc = []
    fold_chance = []
    eye_cache: Dict[int, torch.Tensor] = {}

    for train_idx, test_idx in split_pairs:
        X_train, X_test = _prepare_readout_features(
            activations,
            train_idx=train_idx,
            test_idx=test_idx,
            top_features=top_features,
        )
        y_train = labels_t[train_idx]
        y_test = labels_t[test_idx]

        Y = F.one_hot(y_train, num_classes=num_classes).float()
        K = X_train @ X_train.T
        n_train = K.shape[0]
        if n_train not in eye_cache:
            eye_cache[n_train] = torch.eye(n_train, dtype=K.dtype)
        dual = torch.linalg.solve(K + ridge * eye_cache[n_train], Y)
        W = X_train.T @ dual
        logits = X_test @ W
        y_pred = logits.argmax(dim=1)

        bacc, n_eval_classes = _balanced_accuracy_score(y_test, y_pred, num_classes)
        chance = 1.0 / max(n_eval_classes, 1)
        fold_bacc.append(bacc)
        fold_chance.append(chance)

    balanced_acc = float(np.mean(fold_bacc)) if fold_bacc else 0.0
    chance = float(np.mean(fold_chance)) if fold_chance else 0.0
    utility = (balanced_acc - chance) / max(1.0 - chance, 1e-12)
    utility = float(max(0.0, min(1.0, utility)))

    print(
        f"    [SUS] folds={len(split_pairs)}, top_features={top_features}, ridge={ridge}, "
        f"balanced_acc={balanced_acc:.4f}, chance={chance:.4f}, utility={utility:.4f}"
    )
    return utility, balanced_acc, chance


# ═══════════════════════════════════════════════════════════════════════════
# 1f. Feature-level domain metrics — DAMS v4 components
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_feature_purity_score(
    activations: torch.Tensor,
    labels: List[int],
    num_classes: int,
    active_threshold: float = 0.0,
    min_fire_count: int = 5,
    min_fire_frac: float = 0.005,
    chunk_size: int = 4096,
) -> Tuple[float, Dict[str, float]]:
    """
    Feature Purity (FP): are active SAE features class-specific?

    For each feature f with enough active samples, compute

        purity(f) = 1 - H(Y | f active) / log(C_eff)

    where C_eff is the number of represented labels in the evaluated subset.
    The final score is the fire-count-weighted mean purity over supported
    features. This is deliberately feature-level: generic features that fire
    across many classes are penalised even if a downstream classifier can still
    recover the label by combining many features.
    """
    labels_t = torch.tensor(labels, dtype=torch.long)
    present_classes = labels_t.unique(sorted=True).tolist()
    c_eff = len(present_classes)
    if c_eff <= 1 or activations.numel() == 0:
        return 0.0, {
            "fp_supported_features": 0,
            "fp_total_features": int(activations.shape[1]) if activations.ndim == 2 else 0,
            "fp_effective_classes": c_eff,
            "fp_mean_unweighted": 0.0,
        }

    class_masks = [(labels_t == int(c)).float() for c in present_classes]
    weights = activations.float().clamp(min=0.0)
    n_samples, n_features = weights.shape
    min_fires = max(int(min_fire_count), int(math.ceil(min_fire_frac * n_samples)))
    log_c = math.log(c_eff)
    eps = 1e-10

    weighted_purity_sum = 0.0
    fire_weight_sum = 0.0
    purity_sum = 0.0
    supported_total = 0

    for start in range(0, n_features, chunk_size):
        end = min(start + chunk_size, n_features)
        chunk = weights[:, start:end]
        active = chunk > active_threshold
        fire_count = active.float().sum(dim=0)
        support = fire_count >= min_fires
        if not support.any():
            continue

        active_f = active.float()
        class_counts = torch.stack(
            [(mask.to(active_f.device).unsqueeze(0) @ active_f).squeeze(0) for mask in class_masks],
            dim=0,
        )
        p_y_given_f = class_counts / (fire_count.unsqueeze(0) + eps)
        entropy = -(p_y_given_f * (p_y_given_f + eps).log()).sum(dim=0)
        purity = (1.0 - entropy / log_c).clamp(0.0, 1.0)
        purity = purity[support]
        fires = fire_count[support]

        weighted_purity_sum += float((purity * fires).sum().item())
        fire_weight_sum += float(fires.sum().item())
        purity_sum += float(purity.sum().item())
        supported_total += int(support.sum().item())

    fp = weighted_purity_sum / max(fire_weight_sum, eps)
    stats = {
        "fp_supported_features": supported_total,
        "fp_total_features": int(n_features),
        "fp_effective_classes": int(c_eff),
        "fp_min_fires": int(min_fires),
        "fp_fire_weight_sum": float(fire_weight_sum),
        "fp_mean_unweighted": float(purity_sum / max(supported_total, 1)),
    }
    print(
        f"    [FP] supported={supported_total}/{n_features}, min_fires={min_fires}, "
        f"weighted_purity={fp:.4f}, unweighted={stats['fp_mean_unweighted']:.4f}"
    )
    return float(fp), stats


@torch.no_grad()
def compute_topk_feature_discriminability(
    activations: torch.Tensor,
    labels: List[int],
    num_classes: int,
    top_k: int = 200,
    active_threshold: float = 0.0,
    min_fire_count: int = 5,
    min_fire_frac: float = 0.005,
    min_pos: int = 2,
    chunk_size: int = 2048,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Top-k Feature Discriminability (TFD): how good are the best feature detectors?

    For every SAE feature, compute the best one-vs-rest rank AUC over represented
    classes, then average the top-k features. AUC is computed with average ranks,
    so ties from zero activations are handled correctly. Returned values are:

        tfd_norm = clip((mean_topk_auc - 0.5) / 0.5, 0, 1)
        mean_topk_auc
        diagnostics

    `tfd_norm` is the [0, 1] chance-normalised value used by composites, while
    `mean_topk_auc` is easier to read as the raw detector quality.
    """
    try:
        from scipy.stats import rankdata
    except Exception as exc:  # pragma: no cover - dependency is available in env
        raise ImportError("compute_topk_feature_discriminability requires scipy") from exc

    labels_np = np.asarray(labels, dtype=np.int64)
    present_classes = np.unique(labels_np)
    n_samples, n_features = activations.shape
    if n_samples < 3 or n_features == 0 or present_classes.size <= 1:
        return 0.0, 0.5, {
            "tfd_top_k": int(top_k),
            "tfd_effective_classes": int(present_classes.size),
            "tfd_supported_features": 0,
            "tfd_features_auc_gt_0_8": 0,
            "tfd_features_auc_gt_0_9": 0,
        }

    class_masks = []
    for c in present_classes.tolist():
        pos = labels_np == int(c)
        n_pos = int(pos.sum())
        n_neg = int(n_samples - n_pos)
        if n_pos >= min_pos and n_neg >= min_pos:
            class_masks.append((int(c), pos, n_pos, n_neg))

    if not class_masks:
        return 0.0, 0.5, {
            "tfd_top_k": int(top_k),
            "tfd_effective_classes": int(present_classes.size),
            "tfd_supported_features": 0,
            "tfd_features_auc_gt_0_8": 0,
            "tfd_features_auc_gt_0_9": 0,
        }

    weights = activations.float().clamp(min=0.0).cpu()
    min_fires = max(int(min_fire_count), int(math.ceil(min_fire_frac * n_samples)))
    all_best_auc = []
    supported_total = 0

    for start in range(0, n_features, chunk_size):
        end = min(start + chunk_size, n_features)
        chunk = weights[:, start:end].numpy()
        support = (chunk > active_threshold).sum(axis=0) >= min_fires
        supported_total += int(support.sum())

        # rankdata(method="average") gives exact Mann-Whitney AUC under ties.
        ranks = rankdata(chunk, axis=0, method="average")
        best_auc = np.full(chunk.shape[1], 0.5, dtype=np.float64)

        for _class_id, pos_mask, n_pos, n_neg in class_masks:
            rank_sum = ranks[pos_mask].sum(axis=0)
            auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
            best_auc = np.maximum(best_auc, auc)

        best_auc[~support] = 0.5
        all_best_auc.append(best_auc.astype(np.float32))

    all_best_auc_np = np.concatenate(all_best_auc, axis=0)
    k = min(max(int(top_k), 1), all_best_auc_np.size)
    top_auc = np.partition(all_best_auc_np, -k)[-k:]
    mean_topk_auc = float(top_auc.mean())
    tfd_norm = float(np.clip((mean_topk_auc - 0.5) / 0.5, 0.0, 1.0))

    stats = {
        "tfd_top_k": int(k),
        "tfd_effective_classes": int(present_classes.size),
        "tfd_valid_ovr_classes": int(len(class_masks)),
        "tfd_supported_features": int(supported_total),
        "tfd_total_features": int(n_features),
        "tfd_min_fires": int(min_fires),
        "tfd_features_auc_gt_0_8": int((all_best_auc_np >= 0.8).sum()),
        "tfd_features_auc_gt_0_9": int((all_best_auc_np >= 0.9).sum()),
        "tfd_max_auc": float(all_best_auc_np.max()),
    }
    print(
        f"    [TFD] top{k}_auc={mean_topk_auc:.4f}, norm={tfd_norm:.4f}, "
        f"auc>=0.8={stats['tfd_features_auc_gt_0_8']}, supported={supported_total}/{n_features}"
    )
    return tfd_norm, mean_topk_auc, stats


def _best_ovr_auc_for_indices(
    weights: torch.Tensor,
    labels_np: np.ndarray,
    sample_idx: np.ndarray,
    active_threshold: float,
    min_fires: int,
    min_pos: int,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return per-feature best one-vs-rest AUC and its selected class."""
    from scipy.stats import rankdata

    y = labels_np[sample_idx]
    present_classes = np.unique(y)
    class_masks = []
    n = int(sample_idx.size)
    for c in present_classes.tolist():
        pos = y == int(c)
        n_pos = int(pos.sum())
        n_neg = n - n_pos
        if n_pos >= min_pos and n_neg >= min_pos:
            class_masks.append((int(c), pos, n_pos, n_neg))

    n_features = int(weights.shape[1])
    all_auc = []
    all_class = []
    supported_total = 0

    for start in range(0, n_features, chunk_size):
        end = min(start + chunk_size, n_features)
        chunk = weights[sample_idx, start:end].numpy()
        support = (chunk > active_threshold).sum(axis=0) >= min_fires
        supported_total += int(support.sum())
        ranks = rankdata(chunk, axis=0, method="average")

        best_auc = np.full(chunk.shape[1], 0.5, dtype=np.float64)
        best_class = np.full(chunk.shape[1], -1, dtype=np.int64)
        for class_id, pos_mask, n_pos, n_neg in class_masks:
            rank_sum = ranks[pos_mask].sum(axis=0)
            auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
            update = auc > best_auc
            best_auc[update] = auc[update]
            best_class[update] = class_id

        best_auc[~support] = 0.5
        best_class[~support] = -1
        all_auc.append(best_auc.astype(np.float32))
        all_class.append(best_class)

    return np.concatenate(all_auc), np.concatenate(all_class), supported_total


@torch.no_grad()
def compute_cv_topk_feature_discriminability(
    activations: torch.Tensor,
    labels: List[int],
    num_classes: int,
    top_k: int = 200,
    n_splits: int = 3,
    active_threshold: float = 0.0,
    min_fire_count: int = 5,
    min_fire_frac: float = 0.005,
    min_pos: int = 2,
    chunk_size: int = 2048,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Cross-validated Top-k Feature Discriminability.

    Raw top-k AUC over 49k features can overfit badly: with enough features, both
    base and adapted SAEs can produce apparently perfect top detectors on the
    same samples used for selection. This held-out version selects feature/class
    detectors by train-fold AUC, then evaluates those detectors on the held-out
    fold. It measures generalisable concept detectors rather than lucky spikes.
    """
    try:
        from scipy.stats import rankdata
    except Exception as exc:  # pragma: no cover
        raise ImportError("compute_cv_topk_feature_discriminability requires scipy") from exc

    labels_t = torch.tensor(labels, dtype=torch.long)
    split_pairs = _stratified_fold_indices(labels_t, n_splits=n_splits, min_class_count=max(2, min_pos))
    if not split_pairs:
        return 0.0, 0.5, {
            "tfd_top_k": int(top_k),
            "tfd_cv_folds": 0,
            "tfd_features_auc_gt_0_8": 0,
            "tfd_features_auc_gt_0_9": 0,
            "tfd_supported_features": 0,
        }

    weights = activations.float().clamp(min=0.0).cpu()
    labels_np = np.asarray(labels, dtype=np.int64)
    n_samples, n_features = weights.shape
    min_fires = max(int(min_fire_count), int(math.ceil(min_fire_frac * n_samples)))

    fold_auc_means = []
    fold_auc_ge_08 = []
    fold_auc_ge_09 = []
    supported_counts = []
    valid_detector_counts = []

    for train_idx_t, test_idx_t in split_pairs:
        train_idx = train_idx_t.cpu().numpy()
        test_idx = test_idx_t.cpu().numpy()
        train_auc, train_class, supported = _best_ovr_auc_for_indices(
            weights=weights,
            labels_np=labels_np,
            sample_idx=train_idx,
            active_threshold=active_threshold,
            min_fires=max(2, min(min_fires, int(train_idx.size))),
            min_pos=min_pos,
            chunk_size=chunk_size,
        )
        supported_counts.append(supported)

        k = min(max(int(top_k), 1), train_auc.size)
        selected = np.argpartition(train_auc, -k)[-k:]
        selected = selected[np.argsort(train_auc[selected])[::-1]]
        selected_class = train_class[selected]

        valid_selection = selected_class >= 0
        selected = selected[valid_selection]
        selected_class = selected_class[valid_selection]
        if selected.size == 0:
            continue

        test_scores = weights[test_idx][:, selected].numpy()
        y_test = labels_np[test_idx]
        ranks = rankdata(test_scores, axis=0, method="average")
        test_auc = np.full(selected.size, np.nan, dtype=np.float64)

        for class_id in np.unique(selected_class).tolist():
            cols = selected_class == int(class_id)
            pos = y_test == int(class_id)
            n_pos = int(pos.sum())
            n_neg = int(y_test.size - n_pos)
            if n_pos < 1 or n_neg < 1:
                continue
            rank_sum = ranks[pos][:, cols].sum(axis=0)
            auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
            test_auc[cols] = auc

        valid_auc = test_auc[np.isfinite(test_auc)]
        if valid_auc.size == 0:
            continue
        fold_auc_means.append(float(valid_auc.mean()))
        fold_auc_ge_08.append(int((valid_auc >= 0.8).sum()))
        fold_auc_ge_09.append(int((valid_auc >= 0.9).sum()))
        valid_detector_counts.append(int(valid_auc.size))

    if not fold_auc_means:
        return 0.0, 0.5, {
            "tfd_top_k": int(top_k),
            "tfd_cv_folds": 0,
            "tfd_features_auc_gt_0_8": 0,
            "tfd_features_auc_gt_0_9": 0,
            "tfd_supported_features": int(np.mean(supported_counts)) if supported_counts else 0,
        }

    mean_topk_auc = float(np.mean(fold_auc_means))
    tfd_norm = float(np.clip((mean_topk_auc - 0.5) / 0.5, 0.0, 1.0))
    stats = {
        "tfd_top_k": int(top_k),
        "tfd_cv_folds": int(len(fold_auc_means)),
        "tfd_supported_features": int(np.mean(supported_counts)) if supported_counts else 0,
        "tfd_total_features": int(n_features),
        "tfd_min_fires": int(min_fires),
        "tfd_valid_detectors": float(np.mean(valid_detector_counts)) if valid_detector_counts else 0.0,
        "tfd_features_auc_gt_0_8": float(np.mean(fold_auc_ge_08)) if fold_auc_ge_08 else 0.0,
        "tfd_features_auc_gt_0_9": float(np.mean(fold_auc_ge_09)) if fold_auc_ge_09 else 0.0,
        "tfd_fold_auc_std": float(np.std(fold_auc_means)) if len(fold_auc_means) > 1 else 0.0,
    }
    print(
        f"    [TFD-cv] top{top_k}_heldout_auc={mean_topk_auc:.4f}, norm={tfd_norm:.4f}, "
        f"auc>=0.8/fold={stats['tfd_features_auc_gt_0_8']:.1f}, folds={len(fold_auc_means)}"
    )
    return tfd_norm, mean_topk_auc, stats


@torch.no_grad()
def compute_activation_hoyer_sparsity(
    activations: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[float, Dict[str, float]]:
    """
    Mean Hoyer sparsity of SAE activations per sample.

    Hoyer(x) = (sqrt(d) - ||x||_1 / ||x||_2) / (sqrt(d) - 1)

    It is 0 for dense/uniform activations and approaches 1 for one-sparse
    activations. A matched SAE should often produce a cleaner, sparser code on
    the same target-model features.
    """
    x = activations.float().clamp(min=0.0)
    if x.ndim != 2 or x.numel() == 0:
        return 0.0, {"hoyer_sparsity_std": 0.0, "active_features_per_sample": 0.0}

    d = x.shape[1]
    l1 = x.sum(dim=1)
    l2 = x.norm(p=2, dim=1).clamp(min=eps)
    sparsity = ((math.sqrt(d) - (l1 / l2)) / max(math.sqrt(d) - 1.0, eps)).clamp(0.0, 1.0)
    active_per_sample = (x > 0).float().sum(dim=1)
    stats = {
        "hoyer_sparsity_std": float(sparsity.std(unbiased=False).item()),
        "active_features_per_sample": float(active_per_sample.mean().item()),
        "active_features_per_sample_std": float(active_per_sample.std(unbiased=False).item()),
    }
    score = float(sparsity.mean().item())
    print(
        f"    [Hoyer] sparsity={score:.4f}, active/sample={stats['active_features_per_sample']:.1f}"
    )
    return score, stats


def _image_level_features_from_tokens(
    features: torch.Tensor,
    token_mode: str,
) -> torch.Tensor:
    if features.ndim != 3:
        return features.float().cpu()
    if token_mode == "cls":
        return features[:, 0, :].float().cpu()
    if token_mode == "mean_patch":
        return features[:, 1:, :].mean(dim=1).float().cpu()
    if token_mode == "mean_all":
        return features.mean(dim=1).float().cpu()
    raise ValueError(f"Unknown token_mode={token_mode!r}; expected cls|mean_patch|mean_all")


@torch.no_grad()
def _reconstruct_image_level_features(
    sae,
    features: torch.Tensor,
    device: str,
    batch_size: int,
    token_mode: str,
) -> torch.Tensor:
    if features.ndim != 3:
        recon_batches = []
        for start in tqdm(range(0, features.shape[0], batch_size), desc="TSF reconstruct", leave=False):
            chunk = features[start : start + batch_size].to(device)
            if hasattr(sae, "decode") and hasattr(sae, "encode"):
                recon = sae.decode(sae.encode(chunk)).float()
            else:
                out = sae(chunk)
                recon = out[0] if isinstance(out, (tuple, list)) else out
            recon_batches.append(recon.cpu())
        return torch.cat(recon_batches, dim=0)

    if token_mode == "cls":
        tokens = features[:, 0, :].contiguous()
        recon_batches = []
        for start in tqdm(range(0, tokens.shape[0], batch_size), desc="TSF reconstruct", leave=False):
            chunk = tokens[start : start + batch_size].to(device)
            if hasattr(sae, "decode") and hasattr(sae, "encode"):
                recon = sae.decode(sae.encode(chunk)).float()
            else:
                out = sae(chunk)
                recon = out[0] if isinstance(out, (tuple, list)) else out
            recon_batches.append(recon.cpu())
        return torch.cat(recon_batches, dim=0)

    if token_mode == "mean_patch":
        token_tensor = features[:, 1:, :].contiguous()
    elif token_mode == "mean_all":
        token_tensor = features.contiguous()
    else:
        raise ValueError(f"Unknown token_mode={token_mode!r}; expected cls|mean_patch|mean_all")

    n_images, tokens_per_image, d_model = token_tensor.shape
    flat = token_tensor.view(n_images * tokens_per_image, d_model)
    image_ids = torch.arange(n_images, device=device).repeat_interleave(tokens_per_image)
    recon_sum = torch.zeros(n_images, d_model, dtype=torch.float32, device=device)

    for start in tqdm(range(0, flat.shape[0], batch_size), desc="TSF reconstruct", leave=False):
        end = min(start + batch_size, flat.shape[0])
        chunk = flat[start:end].to(device)
        if hasattr(sae, "decode") and hasattr(sae, "encode"):
            recon = sae.decode(sae.encode(chunk)).float()
        else:
            out = sae(chunk)
            recon = out[0] if isinstance(out, (tuple, list)) else out
        recon_sum.index_add_(0, image_ids[start:end], recon)
        del chunk, recon

    return (recon_sum / float(tokens_per_image)).cpu()


@torch.no_grad()
def compute_task_subspace_fidelity(
    sae,
    features: torch.Tensor,
    labels: List[int],
    num_classes: int,
    device: str = "cuda",
    batch_size: int = 2048,
    token_mode: str = "cls",
    max_components: int = 0,
    eig_eps: float = 1e-9,
) -> Tuple[float, Dict[str, float]]:
    """
    Task Subspace Fidelity (TSF).

    TSF is reconstruction R² restricted to the between-class discriminant
    subspace of the *original adapted-model features*:

        S_B = sum_c n_c (mu_c - mu_bar)(mu_c - mu_bar)^T
        P   = top eigenvectors of S_B
        TSF = 1 - ||(X - X_hat)P||_F² / ||(X - mu_bar)P||_F²

    This focuses the reconstruction test on class-separating directions rather
    than the full hidden space, so small SAE errors in task-relevant LoRA
    directions are not washed out by generic high-variance directions.
    """
    x = _image_level_features_from_tokens(features, token_mode=token_mode).float()
    x_hat = _reconstruct_image_level_features(
        sae,
        features,
        device=device,
        batch_size=batch_size,
        token_mode=token_mode,
    ).float()
    labels_t = torch.tensor(labels, dtype=torch.long)
    present = [int(c) for c in labels_t.unique(sorted=True).tolist() if (labels_t == int(c)).sum() >= 1]
    n, d = x.shape
    if len(present) <= 1 or n <= 1:
        return 0.0, {
            "tsf_token_mode": token_mode,
            "tsf_rank": 0,
            "tsf_error_energy": 0.0,
            "tsf_total_energy": 0.0,
        }

    mu_bar = x.mean(dim=0)
    sb = torch.zeros(d, d, dtype=torch.float64)
    for c in present:
        mask = labels_t == c
        x_c = x[mask]
        if x_c.numel() == 0:
            continue
        diff = (x_c.mean(dim=0) - mu_bar).double().unsqueeze(1)
        sb += float(mask.sum().item()) * (diff @ diff.T)

    evals, evecs = torch.linalg.eigh(sb)
    order = evals.argsort(descending=True)
    evals = evals[order]
    evecs = evecs[:, order].float()
    rank = int((evals > eig_eps * max(float(evals[0].item()), 1.0)).sum().item()) if evals.numel() else 0
    rank = min(rank, len(present) - 1, d)
    if max_components and max_components > 0:
        rank = min(rank, int(max_components))
    if rank <= 0:
        return 0.0, {
            "tsf_token_mode": token_mode,
            "tsf_rank": 0,
            "tsf_error_energy": 0.0,
            "tsf_total_energy": 0.0,
        }

    p = evecs[:, :rank]
    z = (x - mu_bar) @ p
    z_hat = (x_hat - mu_bar) @ p
    error_energy = float((z - z_hat).pow(2).sum().item())
    total_energy = float(z.pow(2).sum().item())
    tsf = 1.0 - error_energy / max(total_energy, 1e-12)
    tsf = float(np.clip(tsf, 0.0, 1.0))
    stats = {
        "tsf_token_mode": token_mode,
        "tsf_rank": int(rank),
        "tsf_error_energy": error_energy,
        "tsf_total_energy": total_energy,
        "tsf_top_eigenvalue": float(evals[0].item()) if evals.numel() else 0.0,
        "tsf_kept_eigen_energy": float(evals[:rank].sum().item()) if rank > 0 else 0.0,
    }
    print(
        f"    [TSF] mode={token_mode}, rank={rank}, error/energy="
        f"{error_energy / max(total_energy, 1e-12):.4f}, tsf={tsf:.4f}"
    )
    return tsf, stats


def _prepare_dense_readout_features(
    features: torch.Tensor,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    top_features: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    X_train = features.float()[train_idx]
    X_test = features.float()[test_idx]

    if top_features > 0 and top_features < X_train.shape[1]:
        var = X_train.var(dim=0)
        keep = var.argsort(descending=True)[:top_features]
        X_train = X_train[:, keep]
        X_test = X_test[:, keep]

    mu = X_train.mean(dim=0, keepdim=True)
    sigma = X_train.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (X_train - mu) / sigma, (X_test - mu) / sigma


@torch.no_grad()
def compute_dense_readout_score(
    features: torch.Tensor,
    labels: List[int],
    num_classes: int,
    n_splits: int = 3,
    ridge: float = 1.0,
    top_features: int = 0,
    min_class_count: int = 2,
) -> Tuple[float, float, float]:
    """Chance-normalised held-out ridge balanced accuracy for dense features."""
    labels_t = torch.tensor(labels, dtype=torch.long)
    split_pairs = _stratified_fold_indices(labels_t, n_splits=n_splits, min_class_count=min_class_count)
    if not split_pairs:
        return 0.0, 0.0, 0.0

    fold_bacc = []
    fold_chance = []
    eye_cache: Dict[int, torch.Tensor] = {}
    for train_idx, test_idx in split_pairs:
        X_train, X_test = _prepare_dense_readout_features(features, train_idx, test_idx, top_features)
        y_train = labels_t[train_idx]
        y_test = labels_t[test_idx]

        Y = F.one_hot(y_train, num_classes=num_classes).float()
        K = X_train @ X_train.T
        n_train = K.shape[0]
        if n_train not in eye_cache:
            eye_cache[n_train] = torch.eye(n_train, dtype=K.dtype)
        dual = torch.linalg.solve(K + ridge * eye_cache[n_train], Y)
        W = X_train.T @ dual
        y_pred = (X_test @ W).argmax(dim=1)

        bacc, n_eval_classes = _balanced_accuracy_score(y_test, y_pred, num_classes)
        fold_bacc.append(bacc)
        fold_chance.append(1.0 / max(n_eval_classes, 1))

    balanced_acc = float(np.mean(fold_bacc))
    chance = float(np.mean(fold_chance))
    utility = (balanced_acc - chance) / max(1.0 - chance, 1e-12)
    utility = float(max(0.0, min(1.0, utility)))
    return utility, balanced_acc, chance


@torch.no_grad()
def compute_task_reconstruction_retention(
    sae,
    features: torch.Tensor,
    labels: List[int],
    num_classes: int,
    device: str = "cuda",
    batch_size: int = 2048,
    n_splits: int = 3,
    ridge: float = 1.0,
    top_features: int = 0,
) -> Tuple[float, float, float, float]:
    """
    Task Reconstruction Retention (TRR).

    Instead of asking whether the SAE reconstructs variance, ask whether a
    simple task readout still works after decode(encode(x)). We represent each
    image by its mean patch hidden state before and after SAE reconstruction and
    compute:

        TRR = balanced_acc(reconstructed) / balanced_acc(original), clipped to [0, 1].

    Returns (trr, recon_balanced_acc, original_balanced_acc, chance).
    """
    if features.ndim == 3:
        patches = features[:, 1:, :].contiguous()
        orig_image_features = patches.mean(dim=1).cpu()
        n_images, patches_per_image, d_model = patches.shape
        flat = patches.view(n_images * patches_per_image, d_model)
        image_ids = torch.arange(n_images, device=device).repeat_interleave(patches_per_image)
        recon_sum = torch.zeros(n_images, d_model, dtype=torch.float32, device=device)

        for start in tqdm(range(0, flat.shape[0], batch_size), desc="TRR reconstruct", leave=False):
            end = min(start + batch_size, flat.shape[0])
            chunk = flat[start:end].to(device)
            if hasattr(sae, "decode") and hasattr(sae, "encode"):
                recon = sae.decode(sae.encode(chunk)).float()
            else:
                out = sae(chunk)
                recon = out[0] if isinstance(out, (tuple, list)) else out
            recon_sum.index_add_(0, image_ids[start:end], recon)
            del chunk, recon

        recon_image_features = (recon_sum / float(patches_per_image)).cpu()
    else:
        orig_image_features = features.float().cpu()
        recon_batches = []
        for start in tqdm(range(0, features.shape[0], batch_size), desc="TRR reconstruct", leave=False):
            chunk = features[start : start + batch_size].to(device)
            if hasattr(sae, "decode") and hasattr(sae, "encode"):
                recon = sae.decode(sae.encode(chunk)).float()
            else:
                out = sae(chunk)
                recon = out[0] if isinstance(out, (tuple, list)) else out
            recon_batches.append(recon.cpu())
        recon_image_features = torch.cat(recon_batches, dim=0)

    _, orig_bacc, chance = compute_dense_readout_score(
        orig_image_features,
        labels,
        num_classes,
        n_splits=n_splits,
        ridge=ridge,
        top_features=top_features,
    )
    _, recon_bacc, _ = compute_dense_readout_score(
        recon_image_features,
        labels,
        num_classes,
        n_splits=n_splits,
        ridge=ridge,
        top_features=top_features,
    )
    trr = float(np.clip(recon_bacc / max(orig_bacc, 1e-12), 0.0, 1.0))
    print(
        f"    [TRR] recon_bacc={recon_bacc:.4f}, orig_bacc={orig_bacc:.4f}, "
        f"chance={chance:.4f}, trr={trr:.4f}"
    )
    return trr, float(recon_bacc), float(orig_bacc), float(chance)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Concept Separability Score (CSS) — Fisher Discriminant Ratio
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_concept_separability(
    activations: torch.Tensor,
    labels: List[int],
    num_classes: int,
    css_saturation: float = 0.5,
    return_per_pair: bool = False,
) -> Tuple[float, float, Dict[str, float]]:
    """
    CSS = (2 / C(C-1)) Σ_{i<j} ||μ_i − μ_j||² / (σ²_i + σ²_j + ε)

    Normalised to [0, 1] via:  CSS_norm = CSS / (CSS + κ)
    """
    labels_t = torch.tensor(labels, dtype=torch.long)
    N, D = activations.shape

    class_means = torch.zeros(num_classes, D)
    class_vars = torch.zeros(num_classes)
    class_counts = torch.zeros(num_classes, dtype=torch.long)

    for c in range(num_classes):
        mask = labels_t == c
        if mask.sum() < 2:
            continue
        x_c = activations[mask]
        mu_c = x_c.mean(dim=0)
        var_c = (x_c - mu_c).pow(2).sum(dim=1).mean()
        class_means[c] = mu_c
        class_vars[c] = var_c
        class_counts[c] = mask.sum()

    eps = 1e-8
    total_ratio = 0.0
    n_pairs = 0
    per_pair = {}

    for i in range(num_classes):
        if class_counts[i] < 2:
            continue
        for j in range(i + 1, num_classes):
            if class_counts[j] < 2:
                continue
            between = (class_means[i] - class_means[j]).pow(2).sum().item()
            within = class_vars[i].item() + class_vars[j].item() + eps
            ratio = between / within
            total_ratio += ratio
            n_pairs += 1
            if return_per_pair:
                per_pair[f"{i}_vs_{j}"] = ratio

    css_raw = total_ratio / max(n_pairs, 1)
    css_norm = css_raw / (css_raw + css_saturation)

    # ── CSS diagnostics ───────────────────────────────────────────────────
    valid_classes = [c for c in range(num_classes) if class_counts[c] >= 2]
    print(f"    [CSS debug] {len(valid_classes)}/{num_classes} classes have ≥2 samples: {valid_classes[:10]}{'...' if len(valid_classes) > 10 else ''}")
    print(f"    [CSS debug] n_pairs={n_pairs}, css_raw={css_raw:.4f}, css_norm={css_norm:.4f} (κ={css_saturation})")
    if n_pairs > 0:
        mean_within = sum(class_vars[c].item() for c in valid_classes) / len(valid_classes)
        print(f"    [CSS debug] avg within-class scatter (Tr Cov): {mean_within:.4f}")
        mean_norms = sum(class_means[c].norm().item() for c in valid_classes) / len(valid_classes)
        print(f"    [CSS debug] avg class-mean ||μ||: {mean_norms:.4f}")
        if per_pair:
            ratios = list(per_pair.values())
            print(f"    [CSS debug] per-pair ratio stats: min={min(ratios):.4f} mean={sum(ratios)/len(ratios):.4f} max={max(ratios):.4f}")
    if css_raw < 0.01:
        print(f"    [CSS debug] ⚠ CSS_raw is very small — class means nearly overlap in SAE space")
    if css_raw > css_saturation * 10:
        print(f"    [CSS debug] ⚠ CSS_raw >> κ — consider increasing css_saturation (currently {css_saturation})")

    return float(css_norm), float(css_raw), per_pair


# ═══════════════════════════════════════════════════════════════════════════
# 3. Feature Specificity Score (FSS) — bounded precision × recall
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_feature_specificity(
    activations: torch.Tensor,
    labels: List[int],
    num_classes: int,
    activation_threshold: float = 0.0,
    top_features: int = 0,
    min_support: Optional[int] = None,
) -> Tuple[float, Dict[int, float]]:
    """
    For each concept c and feature f:

        precision(f, c) = P(c | f active)
        recall(f, c)    = P(f active | c)
        specificity(f, c) = precision × recall  ∈ [0, 1]

    FSS(c) = max_f specificity(f, c)
    FSS    = mean_c FSS(c)
    """
    labels_t = torch.tensor(labels, dtype=torch.long)
    N, D = activations.shape
    eps = 1e-10

    active = (activations > activation_threshold).float()  # (N, D)

    if top_features > 0 and top_features < D:
        mean_act = activations.mean(dim=0)
        topk_idx = mean_act.argsort(descending=True)[:top_features]
        active = active[:, topk_idx]

    n_f_active = active.sum(dim=0)  # (D_eff,)
    effective_min_support = max(2, int(math.ceil(0.005 * N))) if min_support is None else max(1, int(min_support))
    global_support_mask = n_f_active >= effective_min_support

    per_class_fss = {}
    fss_values = []

    for c in range(num_classes):
        mask_c = labels_t == c
        n_c = mask_c.sum().item()
        if n_c == 0:
            per_class_fss[c] = 0.0
            fss_values.append(0.0)
            continue

        recall = active[mask_c].mean(dim=0)            # P(f active | c)
        n_cf = active[mask_c].sum(dim=0)
        precision = n_cf / (n_f_active + eps)           # P(c | f active)

        specificity = precision * recall                # ∈ [0, 1]
        specificity = specificity * global_support_mask.float()
        best_spec = specificity.max().item()
        per_class_fss[c] = best_spec
        fss_values.append(best_spec)

    fss = float(np.mean(fss_values)) if fss_values else 0.0

    # ── FSS diagnostics ───────────────────────────────────────────────────
    N_act, D_act = activations.shape
    mean_active_per_sample = active.sum(dim=1).mean().item()
    mean_active_rate = active.mean().item()
    dead_features = int((n_f_active == 0).sum().item())
    unsupported_features = int((~global_support_mask).sum().item())
    print(f"    [FSS debug] activations shape: ({N_act}, {D_act}), threshold={activation_threshold}")
    print(f"    [FSS debug] mean active features/sample: {mean_active_per_sample:.1f}/{D_act} ({mean_active_rate*100:.1f}%)")
    print(f"    [FSS debug] dead features (never fire): {dead_features}/{D_act}")
    print(f"    [FSS debug] min_support={effective_min_support} => filtered features: {unsupported_features}/{D_act}")
    print(f"    [FSS debug] FSS={fss:.4f}")
    if mean_active_rate > 0.5:
        print(f"    [FSS debug] ⚠ >50% features active — threshold too low, inflate n_f_active → low precision → low FSS")
        positive_acts = activations[activations > 0]
        if positive_acts.numel() > 0:
            print(f"    [FSS debug]   Try: activation_threshold = {positive_acts.median().item():.4f} (median of positive acts)")
    if fss_values:
        print(f"    [FSS debug] per-class FSS: min={min(fss_values):.4f} mean={fss:.4f} max={max(fss_values):.4f}")
        best_class = max(per_class_fss, key=per_class_fss.get)
        print(f"    [FSS debug] best class: {best_class} (FSS={per_class_fss[best_class]:.4f})")

    return fss, per_class_fss


# ═══════════════════════════════════════════════════════════════════════════
# 3b. Entropy-based FSS — sigmoid gating + class-conditional entropy
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_feature_specificity_entropy(
    activations: torch.Tensor,
    labels: List[int],
    num_classes: int,
    top_features: int = 0,
    min_fire_frac: float = 0.005,
    entropy_sharpening: float = 2.0,
) -> Tuple[float, Dict[int, float]]:
    """
    Entropy-based FSS using raw ReLU activation values as soft weights.

    Why raw weights, not sigmoid(a/T):
    ────────────────────────────────────
    SAE activations are already ReLU-gated: a_{i,f} >= 0.
    • a_{i,f} = 0  → feature f did not fire for image i  → contributes zero
    • a_{i,f} > 0  → feature f fired; weight proportional to strength

    sigmoid(0) = 0.5, so the previous implementation gave every non-firing
    image equal weight to a firing one, collapsing the class distribution
    to near-uniform for all features in both base and adapted SAEs.

    With raw weights, features that mostly fire for one class produce a
    peaky p(c|f) and hence low entropy, which is what we want to measure.

    Algorithm:
      1. Weight:  w_{i,f} = a_{i,f}   (raw activation, >= 0)
      2. Weighted class distribution:
             p(c | f) = Σ_{i: label=c} w_{i,f} / Σ_i w_{i,f}
      3. Normalised entropy:
             h(f) = H(p(·|f)) / log(C)  ∈ [0,1]
             0 = all activation weight falls on one class (perfectly specific)
             1 = uniform over all classes
      4. Specificity:
             spec(f, c) = p(c|f) × (1 − h(f))
             High ↔ feature f fires strongly and almost exclusively for class c.

    FSS(c) = max_f  spec(f, c)
    FSS    = mean_c FSS(c)

    Args:
        activations:   (N, D) SAE activations, ReLU-gated so values ≥ 0.
        labels:        list of int class labels, length N.
        num_classes:   C.
        top_features:  if > 0, restrict to top-k features by mean activation
                       (speeds up computation on very wide SAEs).
        min_fire_frac: minimum fraction of images a feature must fire on
                       to be included; filters completely dead features.
        entropy_sharpening: exponent applied to (1 − h_norm) before computing
                       specificity.  Values > 1 sharpen the penalty for high-
                       entropy (generic) features and amplify the reward for
                       low-entropy (specific) ones.  Default 2.0 (squared).

    Returns:
        fss: float ∈ [0, 1]
        per_class_fss: dict {class_idx: fss_value}
    """
    labels_t = torch.tensor(labels, dtype=torch.long)
    N, D = activations.shape
    eps = 1e-10

    # Raw non-negative weights (ReLU acts — already >= 0 from SAE)
    weights = activations.clamp(min=0.0)   # (N, D)

    if top_features > 0 and top_features < D:
        mean_act = weights.mean(dim=0)
        topk_idx = mean_act.argsort(descending=True)[:top_features]
        weights = weights[:, topk_idx]

    D_eff = weights.shape[1]

    # Support filter: feature must fire on at least min_fire_frac of images
    fire_count = (weights > 0).float().sum(dim=0)   # (D_eff,)
    min_fires = max(2, int(math.ceil(min_fire_frac * N)))
    support_mask = fire_count >= min_fires           # (D_eff,)

    # Total activation weight per feature
    total_w = weights.sum(dim=0)                    # (D_eff,)

    # Weighted class distribution: p(c|f)
    oh = F.one_hot(labels_t, num_classes=num_classes).float()   # (N, C)
    class_w_sum = oh.T @ weights                                 # (C, D_eff)
    p_c_given_f = class_w_sum / (total_w.unsqueeze(0) + eps)    # (C, D_eff)

    # Normalised entropy per feature
    log_C = math.log(num_classes) if num_classes > 1 else 1.0
    H_per_f = -(p_c_given_f * (p_c_given_f + eps).log()).sum(dim=0)   # (D_eff,)
    h_norm = (H_per_f / log_C).clamp(0.0, 1.0)                        # ∈ [0,1]

    # Specificity, masked to supported features only.
    # entropy_sharpening > 1 amplifies the gap between specific (h_norm≈0)
    # and generic (h_norm≈1) features — (0.9)^2=0.81 vs (0.3)^2=0.09.
    specificity = p_c_given_f * (1.0 - h_norm).pow(entropy_sharpening).unsqueeze(0)  # (C, D_eff)
    specificity = specificity * support_mask.float().unsqueeze(0)

    per_class_fss: Dict[int, float] = {}
    fss_values: List[float] = []
    for c in range(num_classes):
        best = float(specificity[c].max().item())
        per_class_fss[c] = best
        fss_values.append(best)

    fss = float(np.mean(fss_values)) if fss_values else 0.0

    # ── Diagnostics ───────────────────────────────────────────────────────
    dead = int((fire_count == 0).sum().item())
    unsupported = int((~support_mask).sum().item())
    mean_h = float(h_norm[support_mask].mean().item()) if support_mask.any() else float("nan")
    mean_act_val = float(weights[weights > 0].mean().item()) if (weights > 0).any() else 0.0
    print(f"    [FSS-entropy] shape=({N},{D_eff}), min_fires={min_fires}")
    print(f"    [FSS-entropy] dead={dead}/{D_eff}, unsupported={unsupported}/{D_eff}, "
          f"mean_act(>0)={mean_act_val:.4f}")
    print(f"    [FSS-entropy] mean h_norm={mean_h:.4f}  (0=specific, 1=uniform)")
    if fss_values:
        print(f"    [FSS-entropy] FSS: min={min(fss_values):.4f} "
              f"mean={fss:.4f} max={max(fss_values):.4f}")
        best_c = max(per_class_fss, key=per_class_fss.get)
        print(f"    [FSS-entropy] best class: {best_c} (FSS={per_class_fss[best_c]:.4f})")

    return fss, per_class_fss


def fss_from_components(
    p_c_given_f: torch.Tensor,   # (C, D)
    h_norm: torch.Tensor,         # (D,)
    support_mask: torch.Tensor,   # (D,) bool
    num_classes: int,
    entropy_sharpening: float = 2.0,
) -> float:
    """Recompute FSS from pre-extracted components at any sharpening power.
    Used by the hyperparameter sweep to avoid re-running the SAE forward pass."""
    specificity = p_c_given_f * (1.0 - h_norm).pow(entropy_sharpening).unsqueeze(0)
    specificity = specificity * support_mask.float().unsqueeze(0)
    fss_vals = [float(specificity[c].max().item()) for c in range(num_classes)]
    return float(np.mean(fss_vals)) if fss_vals else 0.0


@torch.no_grad()
def extract_fss_components(
    activations: torch.Tensor,
    labels: List[int],
    num_classes: int,
    top_features: int = 0,
    min_fire_frac: float = 0.005,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract raw FSS components without applying a sharpening exponent.
    Returns (p_c_given_f, h_norm, support_mask) — pass to fss_from_components()
    to compute FSS at any entropy_sharpening value without re-running the SAE.
    """
    labels_t = torch.tensor(labels, dtype=torch.long)
    N, D = activations.shape
    eps = 1e-10

    weights = activations.clamp(min=0.0)
    if top_features > 0 and top_features < D:
        mean_act = weights.mean(dim=0)
        topk_idx = mean_act.argsort(descending=True)[:top_features]
        weights = weights[:, topk_idx]

    fire_count = (weights > 0).float().sum(dim=0)
    min_fires = max(2, int(math.ceil(min_fire_frac * N)))
    support_mask = fire_count >= min_fires

    total_w = weights.sum(dim=0)
    oh = F.one_hot(labels_t, num_classes=num_classes).float()
    class_w_sum = oh.T @ weights
    p_c_given_f = class_w_sum / (total_w.unsqueeze(0) + eps)

    log_C = math.log(num_classes) if num_classes > 1 else 1.0
    H_per_f = -(p_c_given_f * (p_c_given_f + eps).log()).sum(dim=0)
    h_norm = (H_per_f / log_C).clamp(0.0, 1.0)

    return p_c_given_f, h_norm, support_mask


# ═══════════════════════════════════════════════════════════════════════════
# 4. Composite DAMS
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_dams(
    sae,
    features: torch.Tensor,
    labels: List[int],
    num_classes: int,
    device: str = "cuda",
    alpha: float = 0.05,
    beta: float = 0.95,
    gamma: float = 0.0,
    coverage_power: float = 1.0,
    css_saturation: float = 0.4,
    # EC: 'cka' (default — kernel alignment concept space) | 'r2' (legacy)
    ec_metric: str = "cka",
    ec_batch_size: int = 512,
    ec_subsample: int = 2000,
    das_subsample: int = 2000,
    compute_utility_score: bool = True,
    utility_top_features: int = 4096,
    utility_splits: int = 3,
    utility_ridge: float = 1.0,
    # CSS: 'mmd' (default — inter-class MMD in concept space) | 'fisher' (legacy)
    css_metric: str = "mmd",
    # FSS: 'entropy' (default — raw activation weights) | 'threshold' (legacy)
    fss_method: str = "entropy",
    fss_min_support_weight: float = 0.005,
    fss_entropy_sharpening: float = 2.5,
    # threshold-FSS legacy params
    activation_threshold: Optional[float] = None,
    top_features_fss: int = 0,
    fss_min_support: Optional[int] = None,
    sae_batch_size: int = 512,
    precomputed_activations: Optional[torch.Tensor] = None,
    sae_cfg: Optional[object] = None,
    feature_layer: Optional[int] = None,
) -> DAMSResult:
    """
    Full DAMS computation.

    DAMS = EC^rho × (α × CSS_norm + β × FSS + γ × DAS)

    Default metric configuration (all three replace R²/hard-threshold approaches):
        EC  = CKA(X_cls, A_pool) with sigmoid kernel in SAE concept space  ∈ [0,1]
              Measures whether the domain's CLIP feature neighbourhood structure
              is preserved in the SAE's activation geometry.

        CSS = Mean pairwise inter-class MMD in SAE concept space, normalised
              via saturation: CSS_norm = MMD_raw / (MMD_raw + κ)  ∈ [0,1]
              Measures how well the SAE separates target-domain classes.

        FSS = Entropy-based feature specificity with raw activation weights.
              p(c|f) = activation-weighted class distribution per feature;
              spec(f,c) = p(c|f) × (1 − H_norm(f))  ∈ [0,1]
              Measures whether individual SAE features are class-dedicated.

        DAS = Class-balanced kernel target alignment CKA(A_pool, Y).
              Measures statistical dependence between SAE concept geometry and
              target-domain labels without rewarding generic CLIP geometry alone.

        SUS = Chance-normalised balanced accuracy of a simple ridge readout on
              frozen SAE activations. Measures whether the SAE is useful on the
              dataset, independent of whether it is base or adapted.

    Args:
        ec_metric:  'cka' (default) | 'r2' (legacy R²)
        css_metric: 'mmd' (default) | 'fisher' (legacy Fisher discriminant)
        fss_method: 'entropy' (default) | 'threshold' (legacy)
        css_saturation: κ for half-saturation normalisation of CSS raw value.
        precomputed_activations: (N, d_sae) — pass to avoid recomputing activations.
        sae_cfg, feature_layer: for layer-match diagnostics.
    """
    weight_sum = alpha + beta + gamma
    assert abs(weight_sum - 1.0) < 1e-6, f"alpha + beta + gamma must equal 1, got {weight_sum}"
    assert coverage_power >= 0.0, f"coverage_power must be >= 0, got {coverage_power}"
    assert ec_metric in ("r2", "cka"), f"ec_metric must be 'r2' or 'cka'"
    assert css_metric in ("mmd", "fisher"), f"css_metric must be 'mmd' or 'fisher'"
    assert fss_method in ("entropy", "threshold"), f"fss_method must be 'entropy' or 'threshold'"

    N = features.shape[0]

    # ── Diagnostics ───────────────────────────────────────────────────────
    b_dec_norm = check_sae_b_dec(sae)
    sae_layer  = infer_sae_training_layer(sae, sae_cfg) if sae_cfg is not None else None
    layer_match = True
    if sae_layer is not None and feature_layer is not None:
        layer_match = (sae_layer == feature_layer)

    print(f"\n{'─' * 60}")
    print(f"  DAMS = EC^rho × (alpha·CSS + beta·FSS + gamma·DAS)")
    print(f"  EC={ec_metric}  CSS={css_metric}  FSS={fss_method}  "
          f"alpha={alpha} beta={beta} gamma={gamma} rho={coverage_power} kappa={css_saturation}")
    print(f"  N={N}, C={num_classes}, SAE layer={sae_layer}, feature layer={feature_layer}")
    print(f"  ||b_dec|| = {b_dec_norm:.4f}")
    if not layer_match:
        print(f"  ⚠ LAYER MISMATCH: SAE layer {sae_layer} ≠ feature layer {feature_layer}")
    if b_dec_norm < 1e-3:
        print(f"  ⚠ b_dec ≈ 0: decoder bias may not have loaded.")
    print(f"{'─' * 60}")

    # ── SAE activations — computed once, shared by all sub-metrics ────────
    if precomputed_activations is not None:
        acts = precomputed_activations
        print(f"\n  Precomputed activations: {acts.shape}")
    else:
        print("\n  Computing SAE activations (max-pooled over patches)...")
        acts = _compute_pooled_activations(
            sae, features, device=device, batch_size=sae_batch_size,
        )
        print(f"  Activations: {acts.shape}")

    d_sae = acts.shape[1]

    # ── EC: Kernel Alignment or R² ────────────────────────────────────────
    if ec_metric == "cka":
        print("\n[1/3] EC — Kernel Alignment CKA(X_cls, A_pool, sigmoid kernel)...")
        ec, mse_per_dim, var_per_dim = compute_kernel_alignment(
            sae, features, device=device,
            batch_size=ec_batch_size, subsample=ec_subsample,
            precomputed_acts=acts,
        )
    else:  # r2
        print("\n[1/3] EC — R² (legacy)...")
        ec, mse_per_dim, var_per_dim = compute_effective_coverage(
            sae, features, device=device, batch_size=ec_batch_size,
        )
    print(f"  EC [{ec_metric}] = {ec:.4f}")

    # ── CSS: Inter-class MMD or Fisher discriminant ───────────────────────
    if css_metric == "mmd":
        print("\n[2/3] CSS — Inter-class MMD in SAE concept space...")
        css_raw, _, _ = compute_mmd_score(
            sae, features, labels, num_classes,
            device=device, batch_size=sae_batch_size,
            precomputed_acts=acts,
        )
        css_norm = css_raw / (css_raw + css_saturation)
        css_pairs: Dict[str, float] = {}
    else:  # fisher
        print("\n[2/3] CSS — Fisher discriminant ratio (legacy)...")
        css_norm, css_raw, css_pairs = compute_concept_separability(
            acts, labels, num_classes,
            css_saturation=css_saturation,
            return_per_pair=True,
        )
    print(f"  CSS [{css_metric}] raw={css_raw:.4f}  norm={css_norm:.4f}")

    # ── 4. Feature Specificity Score ─────────────────────────────────────
    print("\n[3/4] Computing Feature Specificity (FSS)...")
    if fss_method == "entropy":
        fss, fss_per_class = compute_feature_specificity_entropy(
            acts, labels, num_classes,
            top_features=top_features_fss,
            min_fire_frac=fss_min_support_weight,
            entropy_sharpening=fss_entropy_sharpening,
        )
    else:  # threshold
        if activation_threshold is None:
            positive_acts = acts[acts > 0]
            if positive_acts.numel() > 0:
                fss_threshold = float(positive_acts.median().item())
                print(f"  Auto-selected threshold (median positive act): {fss_threshold:.6f}")
            else:
                fss_threshold = 0.0
                print("  [WARN] No positive activations; using activation_threshold=0.0")
        else:
            fss_threshold = float(activation_threshold)
            print(f"  Using user activation threshold: {fss_threshold:.6f}")
        fss, fss_per_class = compute_feature_specificity(
            acts, labels, num_classes,
            activation_threshold=fss_threshold,
            top_features=top_features_fss,
            min_support=fss_min_support,
        )
    print(f"  FSS = {fss:.4f}")

    # ── DAS: label dependence in concept space ───────────────────────────
    print("\n[4/4] Computing Domain Alignment Score (DAS)...")
    das = compute_domain_alignment_score(
        acts, labels, num_classes,
        subsample=das_subsample,
    )
    print(f"  DAS = {das:.4f}")

    # ── SUS: frozen representation utility ───────────────────────────────
    if compute_utility_score:
        print("\n[utility] Computing SAE Utility Score (SUS)...")
        utility, utility_balanced_acc, utility_chance = compute_sae_utility_score(
            acts,
            labels,
            num_classes,
            n_splits=utility_splits,
            ridge=utility_ridge,
            top_features=utility_top_features,
        )
    else:
        utility, utility_balanced_acc, utility_chance = 0.0, 0.0, 0.0
    print(f"  SUS = {utility:.4f}")

    # ── DAMS = EC^rho × (α·CSS_norm + β·FSS + γ·DAS) ────────────────────
    coverage_gate = float(ec ** coverage_power) if coverage_power > 0 else 1.0
    dams = coverage_gate * (alpha * css_norm + beta * fss + gamma * das)
    print(
        f"\n  DAMS = {coverage_gate:.4f} × "
        f"({alpha}×{css_norm:.4f} + {beta}×{fss:.4f} + {gamma}×{das:.4f}) = {dams:.4f}"
    )
    print(
        f"  EC[{ec_metric}]={ec:.4f}  gate={coverage_gate:.4f}  "
        f"CSS[{css_metric}]_norm={css_norm:.4f}  FSS[{fss_method}]={fss:.4f}  DAS={das:.4f}"
    )

    return DAMSResult(
        ec=ec,
        ec_metric=ec_metric,
        css_raw=css_raw,
        css_norm=css_norm,
        css_metric=css_metric,
        fss=fss,
        fss_method=fss_method,
        das=das,
        das_metric="label_cka",
        utility=utility,
        utility_balanced_acc=utility_balanced_acc,
        utility_chance=utility_chance,
        utility_metric="ridge_readout",
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        coverage_power=coverage_power,
        coverage_gate=coverage_gate,
        dams=dams,
        recon_mse_per_dim=mse_per_dim,
        baseline_var_per_dim=var_per_dim,
        n_samples=N,
        n_classes=num_classes,
        n_features=d_sae,
        layer_match=layer_match,
        b_dec_norm=b_dec_norm,
        css_per_pair=css_pairs,
        fss_per_class=fss_per_class,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Internal: pooled activation helper (OOM-safe, incremental max-pool)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _compute_pooled_activations(
    sae,
    features: torch.Tensor,
    device: str = "cuda",
    batch_size: int = 512,
) -> torch.Tensor:
    """
    Max-pool SAE activations over sequence length. Returns (N, d_sae) on CPU.

    Processes one image at a time with incremental max-pooling to avoid
    materialising the full (B × seq_len × d_sae) tensor on GPU.
    """
    has_seq = features.ndim == 3
    N = features.shape[0]
    pooled = []

    for img_idx in tqdm(range(N), desc="SAE encode", leave=False):
        if has_seq:
            patches = features[img_idx, 1:, :].to(device)   # (seq_len, D), skip CLS
        else:
            patches = features[img_idx].unsqueeze(0).to(device)

        running_max: Optional[torch.Tensor] = None
        for j in range(0, patches.shape[0], batch_size):
            chunk = patches[j : j + batch_size]
            if hasattr(sae, "encode"):
                z = sae.encode(chunk)
            else:
                out = sae(chunk)
                # sae returns (sae_out, feature_acts, ...) — activations are index 1
                z = out[1] if isinstance(out, (tuple, list)) else out
            chunk_max = z.max(dim=0).values
            running_max = chunk_max if running_max is None else torch.maximum(running_max, chunk_max)
            del z, chunk_max

        pooled.append(running_max.cpu())
        del patches, running_max
        if img_idx % 100 == 0:
            torch.cuda.empty_cache()

    return torch.stack(pooled, dim=0)   # (N, d_sae)


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: compare two SAEs side-by-side
# ═══════════════════════════════════════════════════════════════════════════

def compare_dams(
    sae_a,
    sae_b,
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    labels: List[int],
    num_classes: int,
    label_a: str = "LoRA SAE",
    label_b: str = "Base SAE",
    device: str = "cuda",
    sae_cfg_a: Optional[object] = None,
    sae_cfg_b: Optional[object] = None,
    feature_layer_a: Optional[int] = None,
    feature_layer_b: Optional[int] = None,
    feature_layer: Optional[int] = None,
    precomputed_a: Optional[torch.Tensor] = None,
    precomputed_b: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[DAMSResult, DAMSResult]:
    """
    Compute DAMS for two SAEs side-by-side (e.g., domain-finetuned vs base).

    Pass precomputed_a / precomputed_b to skip redundant SAE encode steps
    when pooled activations were already computed by run_sae_inference.py.
    """
    print(f"\n{'═' * 70}")
    print(f"  DAMS COMPARISON: {label_a} vs {label_b}")
    print(f"{'═' * 70}")

    local_kwargs = dict(kwargs)
    fss_method = local_kwargs.get("fss_method", "entropy")

    # For legacy threshold-FSS only: compute a shared activation threshold
    # from SAE-A activations so both SAEs are compared at the same cut-off.
    # With entropy-FSS this is not needed (sigmoid gating has no hard threshold).
    if fss_method == "threshold":
        threshold_source_acts = precomputed_a
        if threshold_source_acts is None:
            print("\n[threshold] Computing domain SAE activations for shared FSS threshold...")
            threshold_source_acts = _compute_pooled_activations(
                sae_a,
                features_a,
                device=device,
                batch_size=local_kwargs.get("sae_batch_size", 512),
            )
        if local_kwargs.get("activation_threshold", None) is None:
            positive_acts = threshold_source_acts[threshold_source_acts > 0]
            if positive_acts.numel() > 0:
                shared_threshold = float(positive_acts.median().item())
                print(f"\n[threshold] Shared threshold from {label_a}: {shared_threshold:.6f}")
            else:
                shared_threshold = 0.0
                print(f"\n[threshold] [WARN] {label_a} has no positive acts; using 0.0")
        else:
            shared_threshold = float(local_kwargs["activation_threshold"])
            print(f"\n[threshold] Using provided threshold: {shared_threshold:.6f}")
        local_kwargs["activation_threshold"] = shared_threshold
        precomputed_a_pass = threshold_source_acts
    else:
        precomputed_a_pass = precomputed_a

    print(f"\n▶ {label_a}")
    result_a = compute_dams(
        sae_a, features_a, labels, num_classes, device=device,
        sae_cfg=sae_cfg_a,
        feature_layer=feature_layer_a if feature_layer_a is not None else feature_layer,
        precomputed_activations=precomputed_a_pass,
        **local_kwargs,
    )

    print(f"\n▶ {label_b}")
    result_b = compute_dams(
        sae_b, features_b, labels, num_classes, device=device,
        sae_cfg=sae_cfg_b,
        feature_layer=feature_layer_b if feature_layer_b is not None else feature_layer,
        precomputed_activations=precomputed_b,
        **local_kwargs,
    )

    ec_label  = f"EC  [{result_a.ec_metric}]"
    css_label = f"CSS [{result_a.css_metric}] norm"
    fss_label = f"FSS [{result_a.fss_method}]"
    print(f"\n{'─' * 64}")
    print(f"{'Metric':<22s} {label_a:>16s} {label_b:>16s} {'Δ':>8s}")
    print(f"{'─' * 64}")
    for name, va, vb in [
        (ec_label,         result_a.ec,       result_b.ec),
        (f"CSS raw",       result_a.css_raw,  result_b.css_raw),
        (css_label,        result_a.css_norm, result_b.css_norm),
        (fss_label,        result_a.fss,      result_b.fss),
        ("DAS [label_cka]", result_a.das,      result_b.das),
        ("SUS [readout]",  result_a.utility,   result_b.utility),
        ("DAMS",           result_a.dams,     result_b.dams),
    ]:
        print(f"{name:<22s} {va:>16.4f} {vb:>16.4f} {va - vb:>+8.4f}")
    print(f"{'─' * 56}")

    if not result_a.layer_match:
        print(f"  ⚠ {label_a}: layer mismatch (SAE layer ≠ feature layer)")
    if not result_b.layer_match:
        print(f"  ⚠ {label_b}: layer mismatch (SAE layer ≠ feature layer)")
    if result_a.b_dec_norm < 1e-3:
        print(f"  ⚠ {label_a}: ||b_dec|| ≈ 0")
    if result_b.b_dec_norm < 1e-3:
        print(f"  ⚠ {label_b}: ||b_dec|| ≈ 0")

    return result_a, result_b


# ═══════════════════════════════════════════════════════════════════════════
# CLI: standalone evaluation
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Compute DAMS for a trained SAE")
    parser.add_argument("--sae_checkpoint", type=str, required=True)
    parser.add_argument("--base_sae_checkpoint", type=str, default=None,
                        help="Optional base SAE for comparison")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["eurosat", "caltech101", "medmnist"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch16")
    parser.add_argument("--lora_weights", type=str, default=None)
    parser.add_argument("--block_layer", type=int, default=None,
                        help="ViT layer to extract features from. "
                             "If None, auto-detected from SAE config.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_images", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--coverage_power", type=float, default=1.0)
    parser.add_argument("--css_saturation", type=float, default=0.5)
    parser.add_argument(
        "--ec_metric", type=str, default="cka", choices=["r2", "cka"],
        help="EC metric: 'cka' (kernel alignment, default) or 'r2' (legacy).",
    )
    parser.add_argument("--ec_subsample", type=int, default=2000)
    parser.add_argument("--das_subsample", type=int, default=2000)
    parser.add_argument("--no_utility_score", action="store_true")
    parser.add_argument("--utility_top_features", type=int, default=4096)
    parser.add_argument("--utility_splits", type=int, default=3)
    parser.add_argument("--utility_ridge", type=float, default=1.0)
    parser.add_argument(
        "--css_metric", type=str, default="mmd", choices=["mmd", "fisher"],
        help="CSS metric: 'mmd' (inter-class MMD, default) or 'fisher' (legacy).",
    )
    parser.add_argument(
        "--fss_method", type=str, default="entropy",
        choices=["entropy", "threshold"],
        help="FSS method: 'entropy' (sigmoid gating + class-conditional entropy) "
             "or 'threshold' (original hard-threshold precision×recall).",
    )
    parser.add_argument("--fss_min_support_weight", type=float, default=0.01,
                        help="Min gate weight fraction of N for a feature to count in entropy-FSS.")
    # Legacy threshold-FSS args
    parser.add_argument(
        "--activation_threshold", type=float, default=None,
        help="(threshold-FSS only) Hard activation threshold.",
    )
    parser.add_argument(
        "--fss_min_support", type=int, default=None,
        help="(threshold-FSS only) Minimum active images per feature.",
    )
    parser.add_argument("--top_features_fss", type=int, default=0)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    try:
        from run_sae_inference import (
            load_dataset, get_clip_preprocess, load_lora_clip_model,
            load_sae_from_checkpoint, extract_clip_features,
        )
    except ImportError:
        print("[FATAL] Could not import from run_sae_inference.py.")
        sys.exit(1)

    device = args.device
    transform = get_clip_preprocess()

    # Load SAE first to auto-detect training layer
    print("Loading SAE...")
    sae, sae_cfg = load_sae_from_checkpoint(args.sae_checkpoint, device=device)
    sae_layer = infer_sae_training_layer(sae, sae_cfg)

    if args.block_layer is not None:
        feature_layer = args.block_layer
    elif sae_layer is not None:
        feature_layer = sae_layer
        print(f"Auto-detected SAE training layer: {sae_layer}")
    else:
        feature_layer = -2
        print(f"Could not detect SAE layer, defaulting to {feature_layer}")

    # Load dataset
    print(f"Loading {args.dataset}...")
    dataset, class_names = load_dataset(args.dataset, transform=transform, data_root=args.data_root)
    num_classes = len(class_names)
    if len(dataset) > args.max_images:
        dataset = torch.utils.data.Subset(dataset, range(args.max_images))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Load CLIP + LoRA and extract from the correct layer
    print("Loading CLIP model...")
    model, _ = load_lora_clip_model(args.model_name, args.lora_weights, device=device)

    print(f"Extracting CLIP features from layer {feature_layer}...")
    features, labels, _ = extract_clip_features(
        model, dataloader, block_layer=feature_layer, device=device,
    )
    del model
    torch.cuda.empty_cache()

    # Compute DAMS
    result = compute_dams(
        sae=sae, features=features, labels=labels,
        num_classes=num_classes, device=device,
        alpha=args.alpha, beta=args.beta,
        gamma=args.gamma,
        coverage_power=args.coverage_power,
        css_saturation=args.css_saturation,
        ec_metric=args.ec_metric,
        ec_subsample=args.ec_subsample,
        das_subsample=args.das_subsample,
        compute_utility_score=not args.no_utility_score,
        utility_top_features=args.utility_top_features,
        utility_splits=args.utility_splits,
        utility_ridge=args.utility_ridge,
        css_metric=args.css_metric,
        fss_method=args.fss_method,
        fss_min_support_weight=args.fss_min_support_weight,
        activation_threshold=args.activation_threshold,
        top_features_fss=args.top_features_fss,
        fss_min_support=args.fss_min_support,
        sae_cfg=sae_cfg, feature_layer=feature_layer,
    )
    print(f"\n{result}")

    # Optional base SAE comparison
    result_base = None
    if args.base_sae_checkpoint:
        base_sae, base_cfg = load_sae_from_checkpoint(args.base_sae_checkpoint, device=device)
        base_layer = infer_sae_training_layer(base_sae, base_cfg)
        base_feat_layer = base_layer if base_layer is not None else feature_layer

        print(f"\nLoading base CLIP (no LoRA) for base SAE (layer {base_feat_layer})...")
        base_model, _ = load_lora_clip_model(args.model_name, None, device=device)
        base_features, _, _ = extract_clip_features(
            base_model, dataloader, block_layer=base_feat_layer, device=device,
        )
        del base_model
        torch.cuda.empty_cache()

        _, result_base = compare_dams(
            sae_a=sae, sae_b=base_sae,
            features_a=features, features_b=base_features,
            labels=labels, num_classes=num_classes,
            label_a="Domain SAE", label_b="Base SAE",
            device=device,
            sae_cfg_a=sae_cfg, sae_cfg_b=base_cfg,
            feature_layer_a=feature_layer,
            feature_layer_b=base_feat_layer,
            alpha=args.alpha, beta=args.beta,
            css_saturation=args.css_saturation,
            activation_threshold=args.activation_threshold,
            top_features_fss=args.top_features_fss,
            fss_min_support=args.fss_min_support,
        )

    if args.output_json:
        out = {"domain_sae": result.to_dict()}
        if result_base is not None:
            out["base_sae"] = result_base.to_dict()
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()
