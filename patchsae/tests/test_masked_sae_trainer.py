from types import SimpleNamespace

import pytest
import torch

from src.sae_training.masked_sae_trainer import (
    MaskedSAETrainer,
    compute_protected_mask,
)


class _ActivitySAE:
    d_sae = 4
    device = "cpu"

    def eval(self):
        return self

    def forward(self, x):
        acts = x
        return x, acts, {}


class _Store:
    def get_batch_activations(self):
        return torch.tensor([[1.0, 4.0, 3.0, 2.0]])


@pytest.mark.parametrize("frac,count", [(0.0, 0), (0.5, 2), (1.0, 4)])
def test_protected_mask_endpoints_and_count(frac, count):
    mask = compute_protected_mask(_ActivitySAE(), _Store(), frac, 1, "cpu")
    assert mask.dtype == torch.bool
    assert mask.sum().item() == count


def test_protected_mask_validates_fraction():
    with pytest.raises(ValueError):
        compute_protected_mask(_ActivitySAE(), _Store(), 1.1, 1, "cpu")


def test_restore_reverts_optimizer_and_renorm_changes():
    sae = SimpleNamespace(
        W_enc=torch.nn.Parameter(torch.arange(12.0).reshape(3, 4)),
        b_enc=torch.nn.Parameter(torch.arange(4.0)),
        W_dec=torch.nn.Parameter(torch.arange(12.0).reshape(4, 3)),
        b_dec=torch.nn.Parameter(torch.arange(3.0)),
    )
    trainer = MaskedSAETrainer.__new__(MaskedSAETrainer)
    trainer.sae = sae
    trainer.protected_mask = torch.tensor([True, False, True, False])
    trainer._protected_parameter_values = {
        "W_enc": sae.W_enc.detach()[:, trainer.protected_mask].clone(),
        "b_enc": sae.b_enc.detach()[trainer.protected_mask].clone(),
        "W_dec": sae.W_dec.detach()[trainer.protected_mask].clone(),
    }
    expected = {k: v.clone() for k, v in trainer._protected_parameter_values.items()}
    with torch.no_grad():
        sae.W_enc.add_(100)
        sae.b_enc.add_(100)
        sae.W_dec.add_(100)
    trainer._restore_protected_parameters()
    assert torch.equal(sae.W_enc[:, trainer.protected_mask], expected["W_enc"])
    assert torch.equal(sae.b_enc[trainer.protected_mask], expected["b_enc"])
    assert torch.equal(sae.W_dec[trainer.protected_mask], expected["W_dec"])
    # Free units must retain the simulated update.
    assert torch.all(sae.b_enc[~trainer.protected_mask] >= 100)
