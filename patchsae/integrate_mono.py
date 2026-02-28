import torch
import torch.nn.functional as F
import tqdm


@torch.no_grad()
def per_class_monosemanticity(X, Z, y, num_classes, k=10, min_images=20):
    """
    Returns:
        per_class_topk: dict[class_id] -> list of (neuron_idx, MS score)
    """
    per_class_topk = {}

    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=True)[0]

        # Skip tiny classes (important!)
        if idx.numel() < min_images:
            continue

        Xc = X[idx]
        Zc = Z[idx]

        ms_c = weighted_pairwise_cosine(Xc, Zc)

        valid = ~torch.isnan(ms_c)
        scores = ms_c.clone()
        scores[~valid] = -1e9

        topk_vals, topk_idx = torch.topk(scores, k)

        per_class_topk[c] = [
            (int(i), float(s))
            for i, s in zip(topk_idx, topk_vals)
        ]

    return per_class_topk


def topk_neurons_overall(ms_scores, k=10):
    """
    ms_scores: [H]
    Returns: indices of top-k neurons
    """
    valid = ~torch.isnan(ms_scores)
    scores = ms_scores.clone()
    scores[~valid] = -1e9  # push NaNs to bottom

    topk_vals, topk_idx = torch.topk(scores, k)
    return topk_idx, topk_vals


@torch.no_grad()
def dataset_monosemanticity(X, Z):
    """
    Returns:
        ms: [H] per-neuron MS
        avg_ms: scalar dataset-level average
    """
    ms = weighted_pairwise_cosine(X, Z)
    avg_ms = torch.nanmean(ms).item()
    return ms, avg_ms


@torch.no_grad()
def weighted_pairwise_cosine(
    image_embeddings,        # X: [N, D]  CLIP embeddings
    neuron_activations,      # Z: [N, H]  SAE activations
    neuron_chunk_size=4096,  # process neurons in chunks to limit memory
    device=None,             # target device (default: cuda if available)
):
    """
    Computes weighted pairwise cosine monosemanticity per neuron.

    For each neuron h, the score is:
        MS_h = Σ_{i<j} w_{ij}^h * cos(x_i, x_j) / Σ_{i<j} w_{ij}^h
    where w_{ij}^h = z_i^h * z_j^h  (product of normalized activations).

    Key identity (avoids materializing N×N similarity matrix):
        z^T S z = z^T (X X^T) z = ||X^T z||²
    where X^T z is only a [D]-dim vector (D=512), so this is O(N*D + N*H)
    memory instead of O(N²). Enables GPU computation even for N=90k.

    Similarly for the denominator:
        Σ_{i<j} z_i * z_j = 0.5 * ((Σ z_i)² - Σ z_i²)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    N, D = image_embeddings.shape
    H = neuron_activations.shape[1]

    print(f"    [mono] N={N}, D={D}, H={H}, chunk={neuron_chunk_size}, "
          f"device={device}")

    # Move to target device and normalize
    X_norm = F.normalize(image_embeddings.float().to(device), dim=1)  # [N, D]

    # Normalize activations to [0,1] per neuron — do this on CPU if needed
    # to avoid holding full [N,H] on GPU at once
    Z_cpu = neuron_activations.float()
    a_min = Z_cpu.min(dim=0, keepdim=True)[0]
    a_max = Z_cpu.max(dim=0, keepdim=True)[0]
    Z_cpu = (Z_cpu - a_min) / (a_max - a_min + 1e-6)  # [N, H] on CPU

    # Precompute per-neuron statistics on CPU (cheap, avoids moving full Z to GPU)
    z_sq_sum = (Z_cpu ** 2).sum(dim=0)                # [H]
    z_sum = Z_cpu.sum(dim=0)                           # [H]
    denominator = (0.5 * (z_sum ** 2 - z_sq_sum)).to(device)  # [H]
    z_sq_sum_dev = z_sq_sum.to(device)

    # Compute numerator in chunks over neurons
    # For each chunk: V = X_norm^T @ Z_chunk → [D, chunk], zTSz = ||V||² per col
    # Peak GPU memory per chunk: [N, chunk] + [D, chunk] ≈ N*chunk*4 bytes
    numerator = torch.zeros(H, device=device)
    n_chunks = (H + neuron_chunk_size - 1) // neuron_chunk_size

    for h_start in tqdm.tqdm(range(0, H, neuron_chunk_size),
                             desc="    mono", total=n_chunks):
        h_end = min(h_start + neuron_chunk_size, H)
        Z_chunk = Z_cpu[:, h_start:h_end].to(device)  # [N, chunk]

        # z^T S z = z^T (X X^T) z = ||X^T z||²
        V = X_norm.T @ Z_chunk                         # [D, chunk]
        zTSz = (V ** 2).sum(dim=0)                     # [chunk]
        numerator[h_start:h_end] = 0.5 * (zTSz - z_sq_sum_dev[h_start:h_end])

    # MS = numerator / denominator, NaN where denominator is 0
    scores = torch.where(
        denominator > 0,
        numerator / denominator,
        torch.tensor(float('nan'), device=device),
    )

    n_valid = (~torch.isnan(scores)).sum().item()
    print(f"    [mono] Done. Valid neurons: {n_valid}/{H}")

    return scores.cpu()