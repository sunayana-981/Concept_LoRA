"""
Masked SAE Trainer: fine-tunes an existing SAE while protecting high-activity units.

Strategy:
  - Step A: Estimate per-unit activity on a reference distribution (e.g. ImageNet or
    MedMNIST activations through base CLIP). The top `protect_frac` most-active units
    are marked as "protected".
  - Step B: After each loss.backward(), zero out gradients for all protected units'
    encoder columns, encoder biases, decoder rows (and gated params if applicable).
    This lets only the unused / low-activity capacity adapt to new concepts.

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
        _, acts, _ = sae.forward_standard(x)           # acts: [..., d_sae]
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

    k = int(protect_frac * sae.d_sae)
    protected_idx = torch.topk(activity, k=k).indices
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
    ):
        self.sae = sae
        self.model = model
        self.activation_store = activation_store
        self.cfg = cfg
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.protected_mask = protected_mask  # bool [d_sae], True = frozen

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

        state = {
            "cfg": self.sae.cfg,
            "state_dict": self.sae.state_dict(),
            "protected_mask": self.protected_mask.cpu(),
        }
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
        pbar = tqdm(total=self.cfg.total_training_tokens, desc="Masked SAE Fine-tune")
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
                  f"{self.cfg.total_training_tokens:,} tokens "
                  f"({self.n_training_steps} steps).")
        return self.sae

    # -----------------------------------------------------------------
    # Evals (mirrors SAETrainer.run_evals)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def run_evals(self):
        self.sae.eval()

        inner_model = self.model.model if hasattr(self.model, 'model') else self.model
        is_maple = (
            hasattr(inner_model, 'image_encoder')
            and not hasattr(inner_model, 'vision_model')
        )

        def _create_hook(hook_fn):
            return Hook(
                self.sae.cfg.block_layer,
                self.sae.cfg.module_name,
                hook_fn,
                is_custom=is_maple,
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
