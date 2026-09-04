from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from tasks.eval_causal_steering import (
    error_preserving_latent_edit,
    extract_targets,
    match_random_controls,
    select_class_balanced_features,
)


class _ToySAE:
    def __init__(self):
        self.W_enc = torch.tensor(
            [
                [1.0, 0.0, 0.5],
                [0.0, 1.0, 0.5],
            ]
        )
        self.b_enc = torch.tensor([0.0, 0.0, 0.0])
        # Deliberately incomplete decoder so reconstruction error is nonzero.
        self.W_dec = torch.tensor(
            [
                [0.8, 0.0],
                [0.0, 0.7],
                [0.1, 0.2],
            ]
        )
        self.b_dec = torch.tensor([0.2, -0.1])

    def __call__(self, x):
        z = torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        reconstruction = z @ self.W_dec + self.b_dec
        return reconstruction, z, {}


class _TargetsDataset(Dataset):
    def __init__(self, targets):
        self.targets = list(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return index, self.targets[index]


def test_error_preserving_ablation_changes_only_selected_decoder_component():
    sae = _ToySAE()
    x = torch.tensor([[2.0, 3.0], [1.5, 0.5]])
    _, z, _ = sae(x)
    edited = error_preserving_latent_edit(
        sae, x, feature_id=1, intervention="ablate"
    )
    expected = x - z[:, 1, None] * sae.W_dec[1]
    assert torch.allclose(edited, expected)


def test_error_preserving_noop_retains_original_not_sae_reconstruction():
    sae = _ToySAE()
    x = torch.tensor([[2.0, 3.0], [1.5, 0.5]])
    reconstruction, _, _ = sae(x)
    edited = error_preserving_latent_edit(
        sae,
        x,
        feature_id=0,
        intervention="amplify",
        amplify_mode="multiply",
        amplify_value=1.0,
    )
    assert torch.allclose(edited, x)
    assert not torch.allclose(edited, reconstruction)


def test_quantile_amplification_uses_train_calibrated_floor():
    sae = _ToySAE()
    x = torch.tensor([[0.4, 0.3], [2.0, 1.0]])
    _, z, _ = sae(x)
    floor = 1.5
    edited = error_preserving_latent_edit(
        sae,
        x,
        feature_id=0,
        intervention="amplify",
        amplify_mode="quantile",
        amplify_value=floor,
    )
    delta = (torch.maximum(z[:, 0], torch.tensor(floor)) - z[:, 0])[:, None]
    assert torch.allclose(edited, x + delta * sae.W_dec[0])


def test_class_balanced_selection_is_unique_and_covers_classes():
    score = np.asarray(
        [
            [9, 8, 1, 0, 0, 0],
            [0, 1, 9, 8, 0, 0],
            [0, 0, 0, 1, 9, 8],
        ],
        dtype=float,
    )
    eligible = score > 0
    selected = select_class_balanced_features(
        score, eligible, num_latents=5, seed=7
    )
    classes = [item[0] for item in selected]
    features = [item[1] for item in selected]
    assert set(classes) == {0, 1, 2}
    assert len(features) == len(set(features)) == 5
    assert max(classes.count(c) for c in set(classes)) <= 2


def test_matched_controls_are_deterministic_unique_and_not_selected():
    selected = [(0, 0, 3.0), (1, 1, 2.5)]
    d_sae = 8
    metrics = {
        "global_active_fraction": np.linspace(0.1, 0.8, d_sae),
        "global_mean": np.linspace(0.2, 1.0, d_sae),
        "decoder_norm": np.linspace(0.8, 1.2, d_sae),
        "means": np.vstack(
            [np.linspace(0.2, 1.0, d_sae), np.linspace(1.0, 0.2, d_sae)]
        ),
        "class_active_fraction": np.vstack(
            [np.linspace(0.1, 0.8, d_sae), np.linspace(0.8, 0.1, d_sae)]
        ),
    }
    args = SimpleNamespace(
        min_active_fraction=0.01,
        max_active_fraction=0.95,
        control_pool_size=3,
    )
    first = match_random_controls(selected, metrics, args, seed=123)
    second = match_random_controls(selected, metrics, args, seed=123)
    control_ids = [item[0] for item in first]
    assert first == second
    assert len(control_ids) == len(set(control_ids))
    assert set(control_ids).isdisjoint({0, 1})


def test_extract_targets_preserves_nested_subset_order():
    dataset = _TargetsDataset([2, 0, 1, 2, 1])
    nested = Subset(Subset(dataset, [4, 0, 3, 1]), [2, 0])
    assert extract_targets(nested).tolist() == [2, 1]
