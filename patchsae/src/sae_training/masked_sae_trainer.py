"""
Masked SAE Trainer: fine-tunes an existing SAE while protecting high-activity units.

Strategy:
  - Step A: Estimate per-unit activity on a reference distribution (e.g. ImageNet or
    MedMNIST activations through base CLIP). The top `protect_frac` most-active units
    are marked as "protected".
  - Step B (fast path, default): split W_enc/b_enc/W_dec into protected/free column
    or row slices and run the protected slice's forward under torch.no_grad(). Only
    the free slice ever enters the autograd graph, so backward() never computes a
    gradient for protected parameters at all -- there is nothing to zero afterward.
    This is what actually turns protect_frac into a training-time saving: encoder/
    decoder backward cost scales with the free-unit count, not d_sae. Reconstruction
    (sae_out) is numerically the sum of both slices' contributions, so it is
    unaffected -- protected units still fire on new data and still contribute to
    the reconstruction, only their *weights* stop moving.
  - Step B (legacy path, `fast_forward=False`): run the standard dense forward over
    all d_sae units, then zero out gradients for protected units' encoder columns,
    encoder biases, decoder rows (and gated params if applicable) after backward().
    Kept only for correctness comparison against the fast path -- it recomputes
    everything the fast path skips and is proportionally slower at high protect_frac.

Both paths also restore protected parameters to their exact pre-step values after
every optimizer step, as a defensive check independent of whichever gradient
mechanism was used (see `_restore_protected_parameters`).

Usage:
    trainer = MaskedSAETrainer(sae, model, activation_store, cfg, optimizer,
                               scheduler, device, protected_mask)
    trainer.fit()
"""

from typing import Any, Optional

import torch
import wandb
from tqdm import tqdm

from src.sae_training.config import Config
from src.sae_training.hooked_vit import Hook, HookedVisionTransformer
from src.sae_training.provenance import build_training_metadata
from src.sae_training.sparse_autoencoder import SparseAutoencoder
from src.sae_training.vit_activations_store import ViTActivationsStore


# =========================================================================
# Activity estimation (Step A)
# =========================================================================

@torch.no_grad()
def estimate_unit_activity(
    sae: SparseAutoencoder,
    activation_store: ViTActivationsStore,
    n_batches: int = 50,
) -> torch.Tensor:
    """
    Compute mean |feature_activation| per SAE unit over `n_batches` batches.

    Returns:
        activity: Tensor of shape [d_sae] with mean absolute activation per unit.
    """
    sae.eval()
    acc = torch.zeros(sae.d_sae, device=sae.device, dtype=torch.float32)

    for i in range(n_batches):
        x = activation_store.get_batch_activations()  # [..., d_in]
        # Use forward() (not forward_standard) so gated SAEs are scored on their
        # actual gate*magnitude feature_acts, not a standard-SAE reading that
        # ignores r_mag/b_mag entirely.
        _, acts, _ = sae.forward(x)                    # acts: [..., d_sae]
        # Flatten all leading dims, keep d_sae last
        acc += acts.abs().reshape(-1, sae.d_sae).mean(dim=0)

    acc /= n_batches
    return acc


def compute_protected_mask(
    sae: SparseAutoencoder,
    activation_store: ViTActivationsStore,
    protect_frac: float = 0.2,
    n_batches: int = 50,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Identify units whose activity is in the top `protect_frac` fraction,
    and return a boolean mask of shape [d_sae].

    Args:
        sae: Pre-trained SparseAutoencoder.
        activation_store: Store that yields activation batches.
        protect_frac: Fraction of units to protect (0.0 – 1.0).
        n_batches: Number of batches to estimate activity over.
        device: Target device.

    Returns:
        protected_mask: bool Tensor [d_sae]. True = protected (frozen).
    """
    print(f"[MaskedTrainer] Estimating unit activity over {n_batches} batches...")
    activity = estimate_unit_activity(sae, activation_store, n_batches)

    if not 0.0 <= protect_frac <= 1.0:
        raise ValueError(f"protect_frac must be in [0, 1], got {protect_frac}")

    # round() makes requested fractions map to the closest realizable number of
    # units. Handle endpoints explicitly so the ablation includes true full
    # fine-tuning (0%) and a no-op SAE control (100%).
    k = round(protect_frac * sae.d_sae)
    protected_idx = (
        torch.topk(activity, k=k, largest=True, sorted=False).indices
        if k > 0 else torch.empty(0, device=activity.device, dtype=torch.long)
    )
    protected_mask = torch.zeros(sae.d_sae, device=device, dtype=torch.bool)
    protected_mask[protected_idx] = True

    n_protected = protected_mask.sum().item()
    n_free = sae.d_sae - n_protected

    print(f"[MaskedTrainer] Activity stats: "
          f"mean={activity.mean():.6f}, max={activity.max():.6f}, "
          f"min={activity.min():.6f}")
    print(f"[MaskedTrainer] Protected {n_protected}/{sae.d_sae} units "
          f"({100 * n_protected / sae.d_sae:.1f}%), "
          f"{n_free} units free to adapt.")

    return protected_mask


# =========================================================================
# Masked SAE Trainer (Step B)
# =========================================================================

class MaskedSAETrainer:
    """
    Fine-tunes an SAE with gradient masking to protect high-activity units.

    Identical to SAETrainer except:
      1. After loss.backward(), gradients on protected units are zeroed.
      2. After optimizer.step(), decoder norms are re-normalized to unit norm.
    """

    def __init__(
        self,
        sae: SparseAutoencoder,
        model: HookedVisionTransformer,
        activation_store: ViTActivationsStore,
        cfg: Config,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        protected_mask: torch.Tensor,
        # --- Early stopping ---
        early_stop: bool = False,
        early_stop_patience: int = 50,
        early_stop_min_delta: float = 1e-5,
        early_stop_warmup: int = 100,
        fast_forward: bool = True,
    ):
        self.sae = sae
        self.model = model
        self.activation_store = activation_store
        self.cfg = cfg
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.protected_mask = protected_mask  # bool [d_sae], True = frozen
        self.free_idx = torch.nonzero(~protected_mask, as_tuple=True)[0]
        self.protected_idx = torch.nonzero(protected_mask, as_tuple=True)[0]

        # The fast split-forward below only implements the standard (non-gated)
        # forward_standard math. Gated/Top-K/JumpReLU SAEs fall back to the
        # legacy dense-forward + gradient-zeroing path instead. For gated and
        # jumprelu this is "just" a missed optimization (their math could in
        # principle be split too). For Top-K it is a correctness requirement,
        # not an optimization choice: top-k selection is *global* across all
        # d_sae units for a given token, so splitting into independent
        # free/protected forward passes and taking top-k *within each subset
        # separately* would select a different set of active units than the
        # dense forward -- a different algorithm, not just a slower one.
        sae_variant = (
            "gated" if sae.cfg.gated_sae else
            "topk" if getattr(sae.cfg, "topk_sae", False) else
            "jumprelu" if getattr(sae.cfg, "jumprelu_sae", False) else
            "standard"
        )
        self.sae_variant = sae_variant
        self.fast_forward = fast_forward and sae_variant == "standard"
        if fast_forward and sae_variant != "standard":
            print(f"[MaskedSAETrainer] sae_variant={sae_variant}: fast split-forward is not "
                  "implemented for this variant, falling back to the legacy "
                  "dense-forward + gradient-zeroing path.")
        print(f"[MaskedSAETrainer] fast_forward={self.fast_forward}")

        # Gradient masking alone is insufficient: decoder normalization occurs
        # outside the optimizer and can move protected rows. Keep immutable
        # copies and restore them after every update, making "protected" exact.
        mask = self.protected_mask
        self._protected_parameter_values = {
            "W_enc": sae.W_enc.detach()[:, mask].clone(),
            "b_enc": sae.b_enc.detach()[mask].clone(),
            "W_dec": sae.W_dec.detach()[mask, :].clone(),
        }
        if bool(mask.all()) and hasattr(sae, "b_dec"):
            self._protected_parameter_values["b_dec"] = sae.b_dec.detach().clone()
        if sae.cfg.gated_sae:
            if hasattr(sae, "r_mag"):
                self._protected_parameter_values["r_mag"] = sae.r_mag.detach()[mask].clone()
            if hasattr(sae, "b_mag"):
                self._protected_parameter_values["b_mag"] = sae.b_mag.detach()[mask].clone()
        if self.sae_variant == "jumprelu" and hasattr(sae, "log_threshold"):
            self._protected_parameter_values["log_threshold"] = sae.log_threshold.detach()[mask].clone()

        self.act_freq_scores = torch.zeros(sae.cfg.d_sae, device=device)
        self.n_forward_passes_since_fired = torch.zeros(sae.cfg.d_sae, device=device)
        self.n_frac_active_tokens = 0
        self.n_training_tokens = 0
        self.ghost_grad_neuron_mask = None
        self.n_training_steps = 0

        # Early stopping state
        self.early_stop = early_stop
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.early_stop_warmup = early_stop_warmup
        self._best_loss = float('inf')
        self._steps_without_improvement = 0
        self._ema_loss = None
        self._ema_alpha = 0.05  # smoothing factor for exponential moving avg

        self.checkpoint_thresholds = list(
            range(
                0,
                cfg.total_training_tokens,
                cfg.total_training_tokens // max(self.cfg.n_checkpoints, 1),
            )
        )[1:]

        n_prot = protected_mask.sum().item()
        n_free = sae.d_sae - n_prot
        print(f"[MaskedSAETrainer] Initialized: "
              f"{n_prot} protected, {n_free} trainable units.")
        if self.early_stop:
            print(f"[MaskedSAETrainer] Early stopping ON: "
                  f"patience={early_stop_patience}, min_delta={early_stop_min_delta}, "
                  f"warmup={early_stop_warmup} steps")

    # -----------------------------------------------------------------
    # Fast split-forward (Step B, default): protected units never enter the
    # autograd graph, so backward() only pays for the free-unit slice of the
    # encoder/decoder. Mirrors SparseAutoencoder.forward_standard exactly
    # (same math, same loss terms) except: (a) computation is split by
    # free/protected index instead of dense over all d_sae units, and
    # (b) ghost-grad revival is restricted to units that are both "dead" and
    # free -- protected units were selected as the *highest*-activity units
    # on the reference distribution, so in practice they are never flagged
    # dead; this restriction changes nothing observable and keeps ghost-grad
    # backward cost bounded by the free-unit count too.
    # -----------------------------------------------------------------
    def _fast_masked_forward(self, x: torch.Tensor, dead_neuron_mask: Optional[torch.Tensor]):
        sae = self.sae
        free_idx = self.free_idx
        prot_idx = self.protected_idx

        x = x.to(sae.dtype)
        sae_in = sae.hook_sae_in(x - sae.b_dec)

        # Free branch: normal autograd, matmuls sized [d_in, n_free] / [n_free, d_in].
        hidden_pre_free = sae_in @ sae.W_enc[:, free_idx] + sae.b_enc[free_idx]
        feature_acts_free = torch.relu(hidden_pre_free)
        free_term = feature_acts_free @ sae.W_dec[free_idx, :]

        # Protected branch: entirely inside no_grad, so no backward node is ever
        # created for W_enc/b_enc/W_dec's protected slice -- this is the actual
        # compute saving, not just a post-hoc gradient zeroing.
        with torch.no_grad():
            hidden_pre_prot = sae_in @ sae.W_enc[:, prot_idx] + sae.b_enc[prot_idx]
            feature_acts_prot = torch.relu(hidden_pre_prot)
            protected_term = feature_acts_prot @ sae.W_dec[prot_idx, :]

        sae_out = sae.hook_sae_out(free_term + protected_term + sae.b_dec)

        # Reassemble full-width [*, d_sae] tensors for logging / L1 / dead-unit
        # tracking. index_copy correctly routes gradient back to hidden_pre_free
        # only; the protected slice is a constant here (from the no_grad block).
        hidden_pre = torch.zeros(*x.shape[:-1], sae.d_sae, dtype=x.dtype, device=x.device)
        hidden_pre = hidden_pre.index_copy(-1, free_idx, hidden_pre_free)
        hidden_pre = hidden_pre.index_copy(-1, prot_idx, hidden_pre_prot)
        feature_acts = torch.zeros_like(hidden_pre)
        feature_acts = feature_acts.index_copy(-1, free_idx, feature_acts_free)
        feature_acts = feature_acts.index_copy(-1, prot_idx, feature_acts_prot)

        # --- everything below mirrors forward_standard exactly ---
        mse_loss = torch.pow((sae_out - x.float()), 2) / (x**2).sum(dim=-1, keepdim=True).sqrt()

        mse_loss_ghost_resid = torch.tensor(0.0, dtype=sae.dtype, device=sae.device)
        if sae.cfg.use_ghost_grads and sae.training and dead_neuron_mask is not None:
            dead_free_mask = dead_neuron_mask & (~self.protected_mask)
            if dead_free_mask.sum() > 0:
                residual = x - sae_out
                l2_norm_residual = torch.norm(residual, dim=-1)
                if len(hidden_pre.size()) == 3:
                    feature_acts_dead_neurons_only = torch.exp(hidden_pre[:, :, dead_free_mask])
                else:
                    feature_acts_dead_neurons_only = torch.exp(hidden_pre[:, dead_free_mask])
                ghost_out = feature_acts_dead_neurons_only @ sae.W_dec[dead_free_mask, :]
                l2_norm_ghost_out = torch.norm(ghost_out, dim=-1)
                norm_scaling_factor = l2_norm_residual / (1e-6 + l2_norm_ghost_out * 2)
                if len(hidden_pre.size()) == 3:
                    ghost_out = ghost_out * norm_scaling_factor[:, :, None].detach()
                else:
                    ghost_out = ghost_out * norm_scaling_factor[:, None].detach()
                mse_loss_ghost_resid = (
                    torch.pow((ghost_out - residual.detach().float()), 2)
                    / (residual.detach() ** 2).sum(dim=-1, keepdim=True).sqrt()
                )
                mse_rescaling_factor = (mse_loss / (mse_loss_ghost_resid + 1e-6)).detach()
                mse_loss_ghost_resid = mse_rescaling_factor * mse_loss_ghost_resid

        mse_loss_ghost_resid = mse_loss_ghost_resid.mean()

        if len(mse_loss.size()) == 3 and sae.training:
            mse_loss[:, 0, :] = mse_loss[:, 0, :] * sae.cfg.mse_cls_coefficient

        mse_loss = mse_loss.mean()
        sparsity = torch.abs(feature_acts).sum(dim=-1).mean(dim=(0,))
        l1_loss = sae.l1_coefficient * sparsity
        loss = mse_loss + l1_loss + mse_loss_ghost_resid

        loss_dict = {
            "mse_loss": mse_loss,
            "l1_loss": l1_loss.mean(),
            "mse_loss_ghost_resid": mse_loss_ghost_resid,
            "loss": loss.mean(),
        }
        return sae_out, feature_acts, loss_dict

    # -----------------------------------------------------------------
    # Gradient masking
    # -----------------------------------------------------------------
    def _apply_gradient_mask(self):
        """Zero out gradients for all protected SAE units after backward()."""
        mask = self.protected_mask  # [d_sae]

        # W_enc: [d_in, d_sae] — freeze columns for protected units
        if self.sae.W_enc.grad is not None:
            self.sae.W_enc.grad[:, mask] = 0.0

        # b_enc: [d_sae]
        if self.sae.b_enc.grad is not None:
            self.sae.b_enc.grad[mask] = 0.0

        # W_dec: [d_sae, d_in] — freeze rows for protected units
        if self.sae.W_dec.grad is not None:
            self.sae.W_dec.grad[mask, :] = 0.0

        # Gated SAE extra params
        if self.sae.cfg.gated_sae:
            if hasattr(self.sae, 'r_mag') and self.sae.r_mag.grad is not None:
                self.sae.r_mag.grad[mask] = 0.0
            if hasattr(self.sae, 'b_mag') and self.sae.b_mag.grad is not None:
                self.sae.b_mag.grad[mask] = 0.0

        # JumpReLU SAE extra param (Top-K has none beyond W_enc/b_enc/W_dec)
        if self.sae_variant == "jumprelu" and hasattr(self.sae, 'log_threshold') and self.sae.log_threshold.grad is not None:
            self.sae.log_threshold.grad[mask] = 0.0

    @torch.no_grad()
    def _restore_protected_parameters(self):
        """Restore protected unit parameters exactly after optimizer/renorm steps."""
        mask = self.protected_mask
        values = self._protected_parameter_values
        # Boolean advanced indexing returns a copy; use indexed assignment so
        # restoration writes into the underlying Parameters.
        self.sae.W_enc[:, mask] = values["W_enc"]
        self.sae.b_enc[mask] = values["b_enc"]
        self.sae.W_dec[mask, :] = values["W_dec"]
        if "r_mag" in values:
            self.sae.r_mag[mask] = values["r_mag"]
        if "b_mag" in values:
            self.sae.b_mag[mask] = values["b_mag"]
        if "log_threshold" in values:
            self.sae.log_threshold[mask] = values["log_threshold"]
        if "b_dec" in values:
            self.sae.b_dec.copy_(values["b_dec"])

    # -----------------------------------------------------------------
    # Training step
    # -----------------------------------------------------------------
    def _train_step(self, sae_in: torch.Tensor):
        self.optimizer.zero_grad()

        self.sae.train()
        self.sae.set_decoder_norm_to_unit_norm()

        # Log and reset sparsity stats periodically
        if (self.n_training_steps + 1) % self.cfg.feature_sampling_window == 0:
            if self.cfg.log_to_wandb:
                sparsity_log_dict = self._build_sparsity_log_dict()
                wandb.log(sparsity_log_dict, step=self.n_training_steps)
            self._reset_running_sparsity_stats()

        ghost_grad_neuron_mask = (
            self.n_forward_passes_since_fired > self.cfg.dead_feature_window
        ).bool()
        if self.fast_forward:
            sae_out, feature_acts, loss_dict = self._fast_masked_forward(sae_in, ghost_grad_neuron_mask)
        else:
            sae_out, feature_acts, loss_dict = self.sae(sae_in, ghost_grad_neuron_mask)

        with torch.no_grad():
            if self.cfg.class_token:
                did_fire = (feature_acts > 0).float().sum(-2) > 0
                self.act_freq_scores += (feature_acts.abs() > 0).float().sum(0)
            else:
                did_fire = (((feature_acts > 0).float().sum(-2) > 0).sum(-2)) > 0
                self.act_freq_scores += (feature_acts.abs() > 0).float().sum(0).sum(0)

            self.n_forward_passes_since_fired += 1
            self.n_forward_passes_since_fired[did_fire] = 0
            self.n_frac_active_tokens += sae_out.size(0)

        self.ghost_grad_neuron_mask = ghost_grad_neuron_mask

        # --- Backward ---
        loss_dict["loss"].backward()

        # --- Gradient masking (Step B) ---
        self._apply_gradient_mask()

        # Remove gradient component parallel to decoder directions (Anthropic trick)
        self.sae.remove_gradient_parallel_to_decoder_directions()

        # --- Optimizer step ---
        self.optimizer.step()
        self.scheduler.step()

        # --- Re-normalize decoder to unit norm after step ---
        with torch.no_grad():
            self.sae.set_decoder_norm_to_unit_norm()
            self._restore_protected_parameters()

        return sae_out, feature_acts, loss_dict

    # -----------------------------------------------------------------
    # Sparsity logging
    # -----------------------------------------------------------------
    def _build_sparsity_log_dict(self) -> dict[str, Any]:
        feature_freq = self.act_freq_scores / max(self.n_frac_active_tokens, 1)
        log_feature_freq = torch.log10(feature_freq + 1e-10).detach().cpu()
        return {
            "plots/feature_density_line_chart": wandb.Histogram(
                log_feature_freq.numpy()
            ),
            "metrics/mean_log10_feature_sparsity": log_feature_freq.mean().item(),
        }

    @torch.no_grad()
    def _reset_running_sparsity_stats(self) -> None:
        self.act_freq_scores = torch.zeros(self.cfg.d_sae, device=self.device)
        self.n_frac_active_tokens = 0

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------
    def _calculate_sparsity_metrics(self) -> dict:
        feature_freq = self.act_freq_scores / max(self.n_frac_active_tokens, 1)
        return {
            "sparsity/mean_passes_since_fired": self.n_forward_passes_since_fired.mean().item(),
            "sparsity/n_passes_since_fired_over_threshold": (
                self.ghost_grad_neuron_mask.sum().item()
                if self.ghost_grad_neuron_mask is not None else 0
            ),
            "sparsity/below_1e-5": (feature_freq < 1e-5).float().mean().item(),
            "sparsity/below_1e-6": (feature_freq < 1e-6).float().mean().item(),
            "sparsity/dead_features": (
                feature_freq < self.cfg.dead_feature_threshold
            ).float().mean().item(),
        }

    @torch.no_grad()
    def _calculate_metrics(
        self, feature_acts: torch.Tensor, sae_out: torch.Tensor, sae_in: torch.Tensor
    ) -> dict:
        if self.cfg.class_token:
            l0 = (feature_acts > 0).float().sum(-1).mean()
        else:
            l0 = (feature_acts > 0).float().sum(-1).mean(-1).mean()
        per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).mean().squeeze()
        total_variance = sae_in.pow(2).sum(-1).mean()
        explained_variance = 1 - per_token_l2_loss / total_variance

        # Extra: track activity in protected vs free units
        mask = self.protected_mask
        prot_l0 = (feature_acts[..., mask] > 0).float().sum(-1).mean().item()
        free_l0 = (feature_acts[..., ~mask] > 0).float().sum(-1).mean().item()

        return {
            "metrics/explained_variance": explained_variance.mean().item(),
            "metrics/explained_variance_std": explained_variance.std().item(),
            "metrics/l0": l0.item(),
            "metrics/l0_protected": prot_l0,
            "metrics/l0_free": free_l0,
        }

    @torch.no_grad()
    def _log_train_step(
        self,
        feature_acts: torch.Tensor,
        loss_dict: dict[str, torch.Tensor],
        sae_out: torch.Tensor,
        sae_in: torch.Tensor,
    ):
        metrics = self._calculate_metrics(feature_acts, sae_out, sae_in)
        sparsity_metrics = self._calculate_sparsity_metrics()

        log_dict = {
            "losses/overall_loss": loss_dict["loss"].item(),
            "losses/mse_loss": loss_dict["mse_loss"].item(),
            "losses/l1_loss": loss_dict["l1_loss"].item(),
            "losses/ghost_grad_loss": loss_dict["mse_loss_ghost_resid"].item(),
            **metrics,
            **sparsity_metrics,
            "details/n_training_tokens": self.n_training_tokens,
            "details/current_learning_rate": self.optimizer.param_groups[0]["lr"],
        }
        wandb.log(log_dict, step=self.n_training_steps)

    @torch.no_grad()
    def _update_pbar(self, loss_dict, pbar, batch_size):
        pbar.set_description(
            f"{self.n_training_steps}| MSE {loss_dict['mse_loss'].item():.4f} | "
            f"L1 {loss_dict['l1_loss'].item():.4f}"
        )
        pbar.update(batch_size)

    # -----------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------
    @torch.no_grad()
    def _checkpoint_if_needed(self):
        if (
            self.checkpoint_thresholds
            and self.n_training_tokens > self.checkpoint_thresholds[0]
        ):
            self.save_checkpoint()
            self.run_evals()
            self.checkpoint_thresholds.pop(0)

    def save_checkpoint(self, is_final=False):
        if is_final:
            path = f"{self.cfg.checkpoint_path}/final_{self.sae.get_name()}.pt"
        else:
            path = f"{self.cfg.checkpoint_path}/{self.n_training_tokens}_{self.sae.get_name()}.pt"

        # Save protected_mask alongside model so downstream knows which units were frozen
        import os
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)

        vectors_per_example = getattr(
            self.cfg, "activation_vectors_per_example", None
        ) or 1
        requested_examples = getattr(
            self.cfg, "training_examples", None
        ) or self.cfg.total_training_tokens
        training_metadata = build_training_metadata(
            examples_seen=self.n_training_tokens,
            training_steps=self.n_training_steps,
            requested_examples=requested_examples,
            activation_vectors_per_example=vectors_per_example,
        )
        training_metadata.update({
            "n_protected": int(self.protected_mask.sum().item()),
            "n_free": int((~self.protected_mask).sum().item()),
            "exact_protected_restore": True,
        })
        state = {
            "cfg": self.sae.cfg,
            "state_dict": self.sae.state_dict(),
            "protected_mask": self.protected_mask.cpu(),
            "training_metadata": training_metadata,
        }
        experiment_metadata = getattr(self.cfg, "experiment_metadata", None)
        if experiment_metadata is not None:
            state["experiment_metadata"] = experiment_metadata
        torch.save(state, path)
        print(f"[MaskedSAETrainer] Saved checkpoint to {path}")

    # -----------------------------------------------------------------
    # Early stopping check
    # -----------------------------------------------------------------
    def _check_early_stop(self, loss_dict) -> bool:
        """
        Check if training should stop early based on EMA of total loss.
        Returns True if training should stop.
        """
        if not self.early_stop:
            return False

        current_loss = loss_dict["loss"].item()

        # Update EMA
        if self._ema_loss is None:
            self._ema_loss = current_loss
        else:
            self._ema_loss = (self._ema_alpha * current_loss
                             + (1 - self._ema_alpha) * self._ema_loss)

        # Don't check during warmup
        if self.n_training_steps < self.early_stop_warmup:
            return False

        # Check improvement
        if self._ema_loss < self._best_loss - self.early_stop_min_delta:
            self._best_loss = self._ema_loss
            self._steps_without_improvement = 0
        else:
            self._steps_without_improvement += 1

        if self._steps_without_improvement >= self.early_stop_patience:
            print(f"\n[MaskedSAETrainer] EARLY STOPPING at step {self.n_training_steps}: "
                  f"EMA loss {self._ema_loss:.6f} has not improved by {self.early_stop_min_delta} "
                  f"for {self.early_stop_patience} steps (best={self._best_loss:.6f}).")
            return True

        return False

    # -----------------------------------------------------------------
    # Main training loop
    # -----------------------------------------------------------------
    def fit(self) -> SparseAutoencoder:
        pbar = tqdm(
            total=self.cfg.total_training_tokens,
            desc="Masked SAE Fine-tune (examples)",
        )
        stopped_early = False

        try:
            while self.n_training_tokens < self.cfg.total_training_tokens:
                sae_acts = self.activation_store.get_batch_activations()
                self.n_training_tokens += sae_acts.size(0)

                sae_out, feature_acts, loss_dict = self._train_step(sae_in=sae_acts)

                if (
                    self.cfg.log_to_wandb
                    and (self.n_training_steps + 1) % self.cfg.wandb_log_frequency == 0
                ):
                    self._log_train_step(
                        feature_acts=feature_acts,
                        loss_dict=loss_dict,
                        sae_out=sae_out,
                        sae_in=sae_acts,
                    )

                self._checkpoint_if_needed()
                self.n_training_steps += 1
                self._update_pbar(loss_dict, pbar, sae_out.size(0))

                # --- Early stopping ---
                if self._check_early_stop(loss_dict):
                    stopped_early = True
                    break
        finally:
            print("[MaskedSAETrainer] Saving final checkpoint...")
            self.save_checkpoint(is_final=True)
            self.run_evals()

        pbar.close()
        if stopped_early:
            print(f"[MaskedSAETrainer] Finished early at {self.n_training_tokens:,} / "
                  f"{self.cfg.total_training_tokens:,} examples "
                  f"({self.n_training_steps} steps).")
        return self.sae

    # -----------------------------------------------------------------
    # Evals (mirrors SAETrainer.run_evals)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def run_evals(self):
        self.sae.eval()

        # Some imagefolder datasets (e.g. cityscapes, whose "train" class dir
        # collides with HF's split-name autodetection) end up with no label
        # column at all. The contrastive-loss eval below needs real per-sample
        # class names ("A photo of a {label}"), so it's meaningless -- and
        # crashes on the empty label list -- without one. Skip it gracefully
        # rather than losing an otherwise-complete training run.
        if "label" not in self.activation_store.dataset.features:
            print("[MaskedSAETrainer] No label column on this dataset -- "
                  "skipping label-dependent contrastive-loss eval.")
            self.sae.train()
            return

        backbone_spec = getattr(self.model, "backbone_spec", None)
        backbone_name = getattr(self.model, "backbone", "clip")

        # DINOv2 has no text tower, so the contrastive-loss eval below (which
        # compares original vs. SAE-reconstructed vs. zero-ablated contrastive
        # loss) is meaningless for it -- there is no per-backbone equivalent
        # eval implemented yet, so skip rather than compute a fake number.
        if backbone_spec is not None and not backbone_spec.has_text_tower:
            print(f"[MaskedSAETrainer] Backbone '{backbone_name}' has no text "
                  "tower -- skipping contrastive-loss eval (no vision-only "
                  "equivalent is implemented).")
            self.sae.train()
            return

        # ALIGN's conv blocks have no lossless way to splice a modified
        # pooled/flattened activation back into [B, C, H, W] for downstream
        # blocks (see Hook.get_full_hook_fn's ALIGN branch), so the
        # reconstruction/zero-ablation hooks below (which rely on
        # return_module_output=False to feed a modified activation forward)
        # aren't supported for it. Skip rather than crash.
        if backbone_spec is not None and not backbone_spec.is_transformer_block:
            print(f"[MaskedSAETrainer] Backbone '{backbone_name}' is not a "
                  "transformer-block backbone -- skipping contrastive-loss "
                  "eval (reconstruction/ablation hooks are ViT-specific).")
            self.sae.train()
            return

        def _create_hook(hook_fn):
            return Hook(
                self.sae.cfg.block_layer,
                self.sae.cfg.module_name,
                hook_fn,
                backbone=backbone_name,
                return_module_output=False,
            )

        def _zero_ablation_hook(activations):
            activations[:, 0, :] = torch.zeros_like(activations[:, 0, :]).to(
                activations.device
            )
            return (activations,)

        def _sae_reconstruction_hook(activations):
            activations[:, 0, :] = self.sae(activations[:, 0, :])[0]
            return (activations,)

        model_inputs = self.activation_store.get_batch_model_inputs(process_labels=True)
        original_loss = self.model(return_type="loss", **model_inputs).item()

        sae_hooks = [_create_hook(_sae_reconstruction_hook)]
        reconstruction_loss = self.model.run_with_hooks(
            sae_hooks, return_type="loss", **model_inputs
        ).item()

        zero_hooks = [_create_hook(_zero_ablation_hook)]
        zero_ablation_loss = self.model.run_with_hooks(
            zero_hooks, return_type="loss", **model_inputs
        ).item()

        denominator = zero_ablation_loss - original_loss
        if abs(denominator) < 1e-8:
            reconstruction_score = (
                1.0 if abs(reconstruction_loss - original_loss) < 1e-8 else 0.0
            )
        else:
            reconstruction_score = (reconstruction_loss - original_loss) / denominator

        if self.cfg.log_to_wandb:
            wandb.log(
                {
                    "metrics/contrastive_loss_score": reconstruction_score,
                    "metrics/original_contrastive_loss": original_loss,
                    "metrics/contrastive_loss_with_sae": reconstruction_loss,
                    "metrics/contrastive_loss_with_ablation": zero_ablation_loss,
                },
                step=self.n_training_steps,
            )

        del model_inputs
        torch.cuda.empty_cache()
        self.sae.train()
