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


def weighted_pairwise_cosine(
    image_embeddings,        # X: [num_images, embedding_dim]
    neuron_activations,      # Z: [num_images, num_neurons]
    pair_batch_size=100
):
    """
    Computes weighted pairwise cosine monosemanticity per neuron.

    image_embeddings: CLIP embeddings (N X D)
    neuron_activations: SAE activations (N X H)
    """

    num_images, embedding_dim = image_embeddings.shape
    num_neurons = neuron_activations.shape[1]

    # Normalize embeddings so dot product = cosine similarity
    normalized_embeddings = F.normalize(image_embeddings, dim=1)

    # Normalize activations to [0,1] per neuron
    activation_min = neuron_activations.min(dim=0, keepdim=True)[0]
    activation_max = neuron_activations.max(dim=0, keepdim=True)[0]
    normalized_activations = (
        neuron_activations - activation_min
    ) / (activation_max - activation_min + 1e-6)

    # Accumulators per neuron
    weighted_similarity_sum = torch.zeros(num_neurons)
    activation_weight_sum = torch.zeros(num_neurons)

    for anchor_idx in tqdm.tqdm(range(num_images)):
        anchor_embedding = normalized_embeddings[anchor_idx]
        anchor_activation = normalized_activations[anchor_idx]

        for other_start in range(anchor_idx + 1, num_images, pair_batch_size):
            other_end = min(other_start + pair_batch_size, num_images)

            other_embeddings = normalized_embeddings[other_start:other_end]
            other_activations = normalized_activations[other_start:other_end]

            # Cosine similarity between anchor and batch
            pairwise_cosine = other_embeddings @ anchor_embedding

            # Activation product weights (batch × neurons)
            activation_weights = anchor_activation.unsqueeze(0) * other_activations

            # Accumulate numerator
            weighted_similarity_sum += (
                activation_weights * pairwise_cosine.unsqueeze(1)
            ).sum(dim=0)

            # Accumulate denominator
            activation_weight_sum += activation_weights.sum(dim=0)

    monosemanticity_scores = torch.where(
        activation_weight_sum > 0,
        weighted_similarity_sum / activation_weight_sum,
        torch.nan
    )

    return monosemanticity_scores
