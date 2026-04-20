"""
# This code is a modified version of Arthur Conmy's code:
    https://github.com/ArthurConmy/sae/blob/main/sae/model.py
# Portions of this file are based on code from the "HugoFry/mats_sae_training_for_ViTs" repository (MIT-licensed):
    https://github.com/HugoFry/mats_sae_training_for_ViTs/blob/main/sae_training/hooked_vit.py
"""

import gzip
import os
import pickle

import einops
import torch
import torch.nn.functional as F
from geom_median.torch import compute_geometric_median
from jaxtyping import Float
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from tqdm import tqdm
from transformer_lens.hook_points import HookedRootModule, HookPoint

from src.sae_training.config import ViTSAERunnerConfig


class _JumpReLUFunction(torch.autograd.Function):
    """
    JumpReLU activation with rectangular-kernel STE for the threshold gradient.

    Forward:  f(z) = z * Θ(z - θ),  θ = exp(log_threshold)
    Backward (∂/∂log_θ): uses kernel K(u) = 1[|u| < 0.5] scaled by θ/bandwidth
                           so gradients flow when z ≈ θ.
    """

    @staticmethod
    def forward(ctx, hidden_pre, log_threshold, bandwidth):
        threshold = log_threshold.exp()
        mask = (hidden_pre > threshold)
        ctx.save_for_backward(hidden_pre, log_threshold, mask)
        ctx.bandwidth = bandwidth
        return hidden_pre * mask.to(hidden_pre.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        hidden_pre, log_threshold, mask = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        threshold = log_threshold.exp()

        # Gradient w.r.t. hidden_pre: straight-through where active
        grad_hidden = grad_output * mask.to(grad_output.dtype)

        # Gradient w.r.t. log_threshold via rectangular kernel
        # u = (z - θ) / ε;  |u| < 0.5  →  within one bandwidth of threshold
        u = (hidden_pre - threshold) / bandwidth
        kernel = (u.abs() < 0.5).to(grad_output.dtype)
        # d/d(log_θ) = d/dθ * θ  (since θ = exp(log_θ))
        # d/dθ [z * Θ(z-θ)] ≈ -z * K(u)/ε  →  times θ for log-space
        grad_log_threshold = -(grad_output * hidden_pre * kernel / bandwidth * threshold).sum(
            tuple(range(grad_output.dim() - 1))
        )

        return grad_hidden, grad_log_threshold, None


class SparseAutoencoder(HookedRootModule):
    """ """

    def __init__(
        self,
        cfg,
        device,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = cfg.d_in
        if not isinstance(self.d_in, int):
            raise ValueError(
                f"d_in must be an int but was {self.d_in=}; {type(self.d_in)=}"
            )
        self.d_sae = cfg.d_sae
        self.l1_coefficient = cfg.l1_coefficient
        self.dtype = cfg.dtype
        self.device = device

        # NOTE: if using resampling neurons method, you must ensure that we initialise the weights in the order W_enc, b_enc, W_dec, b_dec
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_in, self.d_sae, dtype=self.dtype, device=self.device)
            )
        )
        self.b_enc = nn.Parameter(
            torch.zeros(self.d_sae, dtype=self.dtype, device=self.device)
        )

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_sae, self.d_in, dtype=self.dtype, device=self.device)
            )
        )

        if self.cfg.gated_sae:
            self.r_mag = nn.Parameter(
                torch.zeros(self.d_sae, dtype=self.dtype, device=self.device)
            )

            self.b_mag = nn.Parameter(
                torch.zeros(self.d_sae, dtype=self.dtype, device=self.device)
            )

        with torch.no_grad():
            # Anthropic normalize this to have unit columns
            self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

        self.b_dec = nn.Parameter(
            torch.zeros(self.d_in, dtype=self.dtype, device=self.device)
        )

        self.hook_sae_in = HookPoint()
        self.hook_hidden_pre = HookPoint()
        self.hook_hidden_post = HookPoint()
        self.hook_sae_out = HookPoint()

        # JumpReLU: per-feature log-threshold (stored in log-space for positivity)
        if getattr(cfg, "sae_type", "standard") == "jumprelu":
            self.log_threshold = nn.Parameter(
                torch.zeros(self.d_sae, dtype=self.dtype, device=self.device)
            )

        self.setup()  # Required for `HookedRootModule`s

    def forward(self, x, dead_neuron_mask=None):
        sae_type = getattr(self.cfg, "sae_type", "standard")
        if self.cfg.gated_sae or sae_type == "standard":
            if self.cfg.gated_sae:
                return self.forward_gated(x, dead_neuron_mask)
            return self.forward_standard(x, dead_neuron_mask)
        elif sae_type == "jumprelu":
            return self.forward_jumprelu(x, dead_neuron_mask)
        elif sae_type == "topk":
            return self.forward_topk(x, dead_neuron_mask)
        else:
            raise ValueError(f"Unknown sae_type '{sae_type}'. Use 'standard', 'jumprelu', or 'topk'.")

    def forward_standard(self, x, dead_neuron_mask=None):
        x = x.to(self.dtype)

        sae_in = self.hook_sae_in(
            x - self.b_dec
        )  # Remove encoder bias as per Anthropic

        hidden_pre = self.hook_hidden_pre(
            einops.einsum(
                sae_in,
                self.W_enc,
                "... d_in, d_in d_sae -> ... d_sae",
            )
            + self.b_enc
        )
        feature_acts = self.hook_hidden_post(torch.nn.functional.relu(hidden_pre))

        sae_out = self.hook_sae_out(
            einops.einsum(
                feature_acts,
                self.W_dec,
                "... d_sae, d_sae d_in -> ... d_in",
            )
            + self.b_dec
        )

        # add config for whether l2 is normalized:
        mse_loss = (
            torch.pow((sae_out - x.float()), 2)
            / (x**2).sum(dim=-1, keepdim=True).sqrt()
        )

        mse_loss_ghost_resid = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        # gate on config and training so evals is not slowed down.
        if self.cfg.use_ghost_grads and self.training and dead_neuron_mask.sum() > 0:
            assert dead_neuron_mask is not None

            # ghost protocol

            # 1.
            residual = x - sae_out
            l2_norm_residual = torch.norm(residual, dim=-1)

            # 2.
            if len(hidden_pre.size()) == 3:
                feature_acts_dead_neurons_only = torch.exp(
                    hidden_pre[:, :, dead_neuron_mask]
                )
            else:
                feature_acts_dead_neurons_only = torch.exp(
                    hidden_pre[:, dead_neuron_mask]
                )
            ghost_out = feature_acts_dead_neurons_only @ self.W_dec[dead_neuron_mask, :]
            l2_norm_ghost_out = torch.norm(ghost_out, dim=-1)
            norm_scaling_factor = l2_norm_residual / (1e-6 + l2_norm_ghost_out * 2)
            if len(hidden_pre.size()) == 3:
                ghost_out = ghost_out * norm_scaling_factor[:, :, None].detach()
            else:
                ghost_out = ghost_out * norm_scaling_factor[:, None].detach()

            # 3.
            mse_loss_ghost_resid = (
                torch.pow((ghost_out - residual.detach().float()), 2)
                / (residual.detach() ** 2).sum(dim=-1, keepdim=True).sqrt()
            )
            mse_rescaling_factor = (mse_loss / (mse_loss_ghost_resid + 1e-6)).detach()
            mse_loss_ghost_resid = mse_rescaling_factor * mse_loss_ghost_resid

        mse_loss_ghost_resid = mse_loss_ghost_resid.mean()

        # mse_loss shape is (batch_size, token_length, sae_dim), then multiply mse_cls_coeff to [:, 0, :]
        if len(mse_loss.size()) == 3 and self.training:
            mse_loss[:, 0, :] = mse_loss[:, 0, :] * self.cfg.mse_cls_coefficient

        mse_loss = mse_loss.mean()
        sparsity = torch.abs(feature_acts).sum(dim=-1).mean(dim=(0,))
        l1_loss = self.l1_coefficient * sparsity
        loss = mse_loss + l1_loss + mse_loss_ghost_resid

        loss_dict = {
            "mse_loss": mse_loss,
            "l1_loss": l1_loss.mean(),
            "mse_loss_ghost_resid": mse_loss_ghost_resid,
            "loss": loss.mean(),
        }

        return sae_out, feature_acts, loss_dict

    def forward_clamp(
        self, x, dead_neuron_mask=None, clamp_feat_dim=None, clamp_value=10
    ):
        # move x to correct dtype
        x = x.to(self.dtype)
        sae_in = self.hook_sae_in(
            x - self.b_dec
        )  # Remove encoder bias as per Anthropic

        hidden_pre = self.hook_hidden_pre(
            einops.einsum(
                sae_in,
                self.W_enc,
                "... d_in, d_in d_sae -> ... d_sae",
            )
            + self.b_enc
        )
        feature_acts = self.hook_hidden_post(torch.nn.functional.relu(hidden_pre))

        # TODO: make it compatible with args.steer_option
        # for option 1)
        # feature_acts[:, clamp_feat_dim] = feature_acts[:, clamp_feat_dim] * clamp_value

        # for option 3)
        feature_acts = (
            feature_acts[:, :, clamp_feat_dim] * clamp_value
        )  # TODO: check if this is compatabile for both cls and non-cls SAE

        sae_out = self.hook_sae_out(
            einops.einsum(
                feature_acts,
                self.W_dec,
                "... d_sae, d_sae d_in -> ... d_in",
            )
            + self.b_dec
        )

        # add config for whether l2 is normalized:
        mse_loss = (
            torch.pow((sae_out - x.float()), 2)
            / (x**2).sum(dim=-1, keepdim=True).sqrt()
        )

        mse_loss_ghost_resid = torch.tensor(0.0, dtype=self.dtype, device=self.device)

        # # gate on config and training so evals is not slowed down.
        # if self.cfg.use_ghost_grads and self.training and dead_neuron_mask.sum() > 0:
        #     assert dead_neuron_mask is not None

        #     # ghost protocol

        #     # 1.
        #     residual = x - sae_out
        #     l2_norm_residual = torch.norm(residual, dim=-1)

        #     # 2.
        #     feature_acts_dead_neurons_only = torch.exp(hidden_pre[:, dead_neuron_mask])
        #     ghost_out =  feature_acts_dead_neurons_only @ self.W_dec[dead_neuron_mask,:]
        #     l2_norm_ghost_out = torch.norm(ghost_out, dim = -1)
        #     norm_scaling_factor = l2_norm_residual / (1e-6 + l2_norm_ghost_out* 2)
        #     ghost_out = ghost_out*norm_scaling_factor[:, None].detach()

        #     # 3.
        #     mse_loss_ghost_resid = (
        #         torch.pow((ghost_out - residual.detach().float()), 2) / (residual.detach()**2).sum(dim=-1, keepdim=True).sqrt()
        #     )
        #     mse_rescaling_factor = (mse_loss / (mse_loss_ghost_resid + 1e-6)).detach()
        #     mse_loss_ghost_resid = mse_rescaling_factor * mse_loss_ghost_resid

        mse_loss_ghost_resid = mse_loss_ghost_resid.mean()
        mse_loss = mse_loss.mean()
        sparsity = torch.abs(feature_acts).sum(dim=1).mean(dim=(0,))
        l1_loss = self.l1_coefficient * sparsity
        loss = mse_loss + l1_loss + mse_loss_ghost_resid

        return sae_out, feature_acts, loss, mse_loss, l1_loss, mse_loss_ghost_resid

    def forward_gated(self, x, dead_neuron_mask=None):
        """
        Gated SAE (Rajamanoharan et al. 2024, DeepMind).

        Two parallel streams share the encoder weight matrix W_enc:

          Gate stream:      π_gate = (x - b_dec) @ W_enc + b_enc
          Magnitude stream: π_mag  = (x - b_dec) @ (W_enc * exp(r_mag)) + b_mag

          feature_acts = ReLU(π_gate > 0) * ReLU(π_mag)   [element-wise gate × magnitude]

        Losses:
          mse_loss : normalized MSE of main reconstruction
          L1 loss  : l1_coefficient * ||ReLU(π_gate)||_1   (sparsity on gate logits)
          aux_loss : normalized MSE of auxiliary path that reconstructs using
                     ReLU(π_gate) as activations with a detached decoder
                     (trains gating stream to reconstruct without collapsing)
        """
        x = x.to(self.dtype)

        sae_in = self.hook_sae_in(x - self.b_dec)

        # Gate pre-activation
        pi_gate = self.hook_hidden_pre(
            einops.einsum(sae_in, self.W_enc, "... d_in, d_in d_sae -> ... d_sae") + self.b_enc
        )

        # Magnitude pre-activation (shared W_enc scaled by exp(r_mag))
        W_enc_mag = self.W_enc * self.r_mag.exp()
        pi_mag = einops.einsum(sae_in, W_enc_mag, "... d_in, d_in d_sae -> ... d_sae") + self.b_mag

        # Feature activations: gate (Heaviside) × magnitude (ReLU)
        gate = (pi_gate > 0).to(self.dtype)
        feature_acts = self.hook_hidden_post(gate * F.relu(pi_mag))

        sae_out = self.hook_sae_out(
            einops.einsum(feature_acts, self.W_dec, "... d_sae, d_sae d_in -> ... d_in") + self.b_dec
        )

        # Main MSE loss
        mse_loss = (
            torch.pow((sae_out - x.float()), 2)
            / (x ** 2).sum(dim=-1, keepdim=True).sqrt()
        )

        # Auxiliary reconstruction: use ReLU(π_gate) with detached decoder
        # to train the gating stream to reconstruct independently
        pi_gate_relu = F.relu(pi_gate)
        aux_out = (
            einops.einsum(pi_gate_relu, self.W_dec.detach(), "... d_sae, d_sae d_in -> ... d_in")
            + self.b_dec.detach()
        )
        aux_loss = (
            torch.pow((aux_out - x.float()), 2)
            / (x ** 2).sum(dim=-1, keepdim=True).sqrt()
        ).mean()

        if len(mse_loss.size()) == 3 and self.training:
            mse_loss[:, 0, :] = mse_loss[:, 0, :] * self.cfg.mse_cls_coefficient

        mse_loss = mse_loss.mean()
        l1_loss  = self.l1_coefficient * pi_gate_relu.sum(dim=-1).mean()
        loss     = mse_loss + l1_loss + aux_loss

        loss_dict = {
            "mse_loss":           mse_loss,
            "l1_loss":            l1_loss,
            "mse_loss_ghost_resid": aux_loss,   # reuse key so trainer logs it
            "loss":               loss,
        }
        return sae_out, feature_acts, loss_dict

    def forward_jumprelu(self, x, dead_neuron_mask=None):
        """
        JumpReLU SAE (Rajamanoharan et al. 2024).

        Activation: f(z) = z * Θ(z - θ)  where Θ is the Heaviside step function.
        θ = exp(log_threshold) enforces positivity of the threshold.
        Gradients through the non-differentiable step are computed via a
        rectangular-kernel straight-through estimator (STE) with bandwidth ε.

        Loss: MSE + l1_coefficient * L0   (L0 = mean number of active features)
        """
        x = x.to(self.dtype)
        bandwidth = getattr(self.cfg, "jumprelu_bandwidth", 0.001)

        sae_in = self.hook_sae_in(x - self.b_dec)
        hidden_pre = self.hook_hidden_pre(
            einops.einsum(sae_in, self.W_enc, "... d_in, d_in d_sae -> ... d_sae") + self.b_enc
        )

        feature_acts = self.hook_hidden_post(
            _JumpReLUFunction.apply(hidden_pre, self.log_threshold, bandwidth)
        )

        sae_out = self.hook_sae_out(
            einops.einsum(feature_acts, self.W_dec, "... d_sae, d_sae d_in -> ... d_in") + self.b_dec
        )

        mse_loss = (
            torch.pow((sae_out - x.float()), 2)
            / (x ** 2).sum(dim=-1, keepdim=True).sqrt()
        )

        # Ghost residual for dead neurons (same as standard)
        mse_loss_ghost_resid = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        if self.cfg.use_ghost_grads and self.training and dead_neuron_mask is not None and dead_neuron_mask.sum() > 0:
            residual = x - sae_out
            l2_norm_residual = torch.norm(residual, dim=-1)
            if len(hidden_pre.size()) == 3:
                dead_acts = torch.exp(hidden_pre[:, :, dead_neuron_mask])
            else:
                dead_acts = torch.exp(hidden_pre[:, dead_neuron_mask])
            ghost_out = dead_acts @ self.W_dec[dead_neuron_mask, :]
            l2_norm_ghost = torch.norm(ghost_out, dim=-1)
            scale = l2_norm_residual / (1e-6 + l2_norm_ghost * 2)
            if len(hidden_pre.size()) == 3:
                ghost_out = ghost_out * scale[:, :, None].detach()
            else:
                ghost_out = ghost_out * scale[:, None].detach()
            mse_loss_ghost_resid = (
                torch.pow((ghost_out - residual.detach().float()), 2)
                / (residual.detach() ** 2).sum(dim=-1, keepdim=True).sqrt()
            )
            rescale = (mse_loss / (mse_loss_ghost_resid + 1e-6)).detach()
            mse_loss_ghost_resid = rescale * mse_loss_ghost_resid

        mse_loss_ghost_resid = mse_loss_ghost_resid.mean()

        if len(mse_loss.size()) == 3 and self.training:
            mse_loss[:, 0, :] = mse_loss[:, 0, :] * self.cfg.mse_cls_coefficient

        mse_loss = mse_loss.mean()

        # L0 penalty: smooth proxy using same rectangular kernel so gradients flow to log_threshold
        threshold = self.log_threshold.exp()
        u = (hidden_pre - threshold) / bandwidth
        l0_proxy = torch.clamp(0.5 + u, 0.0, 1.0)  # rectangular integral ≈ Heaviside
        l0_loss = self.l1_coefficient * l0_proxy.sum(dim=-1).mean()

        loss = mse_loss + l0_loss + mse_loss_ghost_resid

        loss_dict = {
            "mse_loss": mse_loss,
            "l1_loss": l0_loss,            # logged under same key for trainer compatibility
            "mse_loss_ghost_resid": mse_loss_ghost_resid,
            "loss": loss,
        }
        return sae_out, feature_acts, loss_dict

    def forward_topk(self, x, dead_neuron_mask=None):
        """
        Top-K SAE (Gao et al. 2024, OpenAI).

        Keeps exactly k largest post-ReLU activations per sample; all others zeroed.
        No sparsity penalty — k itself controls L0.  An auxiliary loss on the k
        largest dead-neuron pre-activations prevents neuron collapse.
        """
        x = x.to(self.dtype)
        k = getattr(self.cfg, "topk_k", 64)

        sae_in = self.hook_sae_in(x - self.b_dec)
        hidden_pre = self.hook_hidden_pre(
            einops.einsum(sae_in, self.W_enc, "... d_in, d_in d_sae -> ... d_sae") + self.b_enc
        )

        # Top-k selection (over the last dimension, supporting batch × seq or batch)
        pre_relu = F.relu(hidden_pre)
        topk_vals, topk_idx = torch.topk(pre_relu, k=min(k, self.d_sae), dim=-1)
        feature_acts = self.hook_hidden_post(
            torch.zeros_like(hidden_pre).scatter_(-1, topk_idx, topk_vals)
        )

        sae_out = self.hook_sae_out(
            einops.einsum(feature_acts, self.W_dec, "... d_sae, d_sae d_in -> ... d_in") + self.b_dec
        )

        mse_loss = (
            torch.pow((sae_out - x.float()), 2)
            / (x ** 2).sum(dim=-1, keepdim=True).sqrt()
        )

        # Auxiliary loss: push dead neurons' pre-activations toward firing
        # Uses the top-k of *dead* neuron pre-activations (aux-k from Gao et al.)
        aux_loss = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        if self.training and dead_neuron_mask is not None and dead_neuron_mask.sum() > 0:
            if len(hidden_pre.size()) == 3:
                dead_pre = hidden_pre[:, :, dead_neuron_mask]
            else:
                dead_pre = hidden_pre[:, dead_neuron_mask]
            n_dead = dead_pre.shape[-1]
            aux_k = max(1, min(k, n_dead))
            aux_topk_vals, aux_topk_idx = torch.topk(F.relu(dead_pre), k=aux_k, dim=-1)
            aux_acts = torch.zeros_like(dead_pre).scatter_(-1, aux_topk_idx, aux_topk_vals)
            # Reconstruct from dead features only and penalize residual
            aux_out = einops.einsum(
                aux_acts, self.W_dec[dead_neuron_mask, :],
                "... d_dead, d_dead d_in -> ... d_in"
            )
            residual = (x - sae_out).detach()
            aux_loss = (
                torch.pow((aux_out - residual.float()), 2)
                / (residual ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-8).sqrt()
            ).mean() * self.l1_coefficient

        if len(mse_loss.size()) == 3 and self.training:
            mse_loss[:, 0, :] = mse_loss[:, 0, :] * self.cfg.mse_cls_coefficient

        mse_loss = mse_loss.mean()
        loss = mse_loss + aux_loss

        loss_dict = {
            "mse_loss": mse_loss,
            "l1_loss": aux_loss,           # logged under same key for trainer compatibility
            "mse_loss_ghost_resid": torch.tensor(0.0, dtype=self.dtype, device=self.device),
            "loss": loss,
        }
        return sae_out, feature_acts, loss_dict

    @torch.no_grad()
    def initialize_b_dec(self, activation_store):
        if self.cfg.b_dec_init_method == "geometric_median":
            self.initialize_b_dec_with_geometric_median(activation_store)
        elif self.cfg.b_dec_init_method == "mean":
            self.initialize_b_dec_with_mean(activation_store)
        elif self.cfg.b_dec_init_method == "zeros":
            pass
        else:
            raise ValueError(
                f"Unexpected b_dec_init_method: {self.cfg.b_dec_init_method}"
            )

    @torch.no_grad()
    def initialize_b_dec_with_geometric_median(self, activation_store, maxiter=100):
        previous_b_dec = self.b_dec.clone().cpu()
        all_activations = activation_store.get_batch_activations().detach().cpu()
        out = compute_geometric_median(
            all_activations, skip_typechecks=True, maxiter=maxiter, per_component=False
        ).median

        if len(out.shape) == 2:
            out = out.mean(dim=0)
            # out = out.view(-1)

        previous_distances = torch.norm(all_activations - previous_b_dec, dim=-1)
        distances = torch.norm(all_activations - out, dim=-1)

        print("Reinitializing b_dec with geometric median of activations")
        print(
            f"Previous distances: {previous_distances.median(0).values.mean().item()}"
        )
        print(f"New distances: {distances.median(0).values.mean().item()}")

        out = torch.tensor(out, dtype=self.dtype, device=self.device)

        # print('out.shape', out.shape)
        # print('self.b_dec.shape', self.b_dec.shape)

        self.b_dec.data = out

    @torch.no_grad()
    def initialize_b_dec_with_mean(self, activation_store):
        previous_b_dec = self.b_dec.clone().cpu()
        all_activations = activation_store.get_batch_activations().detach().cpu()
        out = all_activations.mean(dim=0)

        previous_distances = torch.norm(all_activations - previous_b_dec, dim=-1)
        distances = torch.norm(all_activations - out, dim=-1)

        print("Reinitializing b_dec with mean of activations")
        print(
            f"Previous distances: {previous_distances.median(0).values.mean().item()}"
        )
        print(f"New distances: {distances.median(0).values.mean().item()}")

        self.b_dec.data = out.to(self.dtype).to(self.device)

    @torch.no_grad()
    def resample_neurons_l2(
        self,
        x: Float[Tensor, "batch_size n_hidden"],  # noqa: F722
        feature_sparsity: Float[Tensor, "n_hidden_ae"],  # noqa: F821
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """
        Resamples neurons that have been dead for `dead_neuron_window` steps, according to `frac_active`.

        I'll probably break this now and fix it later!
        """

        feature_reinit_scale = self.cfg.feature_reinit_scale

        sae_out, _, _, _, _ = self.forward(x)
        per_token_l2_loss = (sae_out - x).pow(2).sum(dim=-1).squeeze()

        # Find the dead neurons in this instance. If all neurons are alive, continue
        is_dead = feature_sparsity < self.cfg.dead_feature_threshold
        dead_neurons = torch.nonzero(is_dead).squeeze(-1)
        alive_neurons = torch.nonzero(~is_dead).squeeze(-1)
        n_dead = dead_neurons.numel()

        if n_dead == 0:
            return 0  # If there are no dead neurons, we don't need to resample neurons

        # Compute L2 loss for each element in the batch
        # TODO: Check whether we need to go through more batches as features get sparse to find high l2 loss examples.
        if per_token_l2_loss.max() < 1e-6:
            return 0  # If we have zero reconstruction loss, we don't need to resample neurons

        # Draw `n_hidden_ae` samples from [0, 1, ..., batch_size-1], with probabilities proportional to l2_loss squared
        per_token_l2_loss = per_token_l2_loss.to(
            torch.float32
        )  # wont' work with bfloat16
        distn = Categorical(
            probs=per_token_l2_loss.pow(2) / (per_token_l2_loss.pow(2).sum())
        )
        replacement_indices = distn.sample((n_dead,))  # shape [n_dead]

        # Index into the batch of hidden activations to get our replacement values
        replacement_values = (x - self.b_dec)[
            replacement_indices
        ]  # shape [n_dead n_input_ae]

        # unit norm
        replacement_values = replacement_values / (
            replacement_values.norm(dim=1, keepdim=True) + 1e-8
        )

        # St new decoder weights
        self.W_dec.data[is_dead, :] = replacement_values

        # Get the norm of alive neurons (or 1.0 if there are no alive neurons)
        W_enc_norm_alive_mean = (
            1.0
            if len(alive_neurons) == 0
            else self.W_enc[:, alive_neurons].norm(dim=0).mean().item()
        )

        # Lastly, set the new weights & biases
        self.W_enc.data[:, is_dead] = (
            replacement_values * W_enc_norm_alive_mean * feature_reinit_scale
        ).T
        self.b_enc.data[is_dead] = 0.0

        # reset the Adam Optimiser for every modified weight and bias term
        # Reset all the Adam parameters
        for dict_idx, (k, v) in enumerate(optimizer.state.items()):
            for v_key in ["exp_avg", "exp_avg_sq"]:
                if dict_idx == 0:
                    assert k.data.shape == (self.d_in, self.d_sae)
                    v[v_key][:, is_dead] = 0.0
                elif dict_idx == 1:
                    assert k.data.shape == (self.d_sae,)
                    v[v_key][is_dead] = 0.0
                elif dict_idx == 2:
                    assert k.data.shape == (self.d_sae, self.d_in)
                    v[v_key][is_dead, :] = 0.0
                elif dict_idx == 3:
                    assert k.data.shape == (self.d_in,)
                else:
                    raise ValueError(f"Unexpected dict_idx {dict_idx}")

        # Check that the opt is really updated
        for dict_idx, (k, v) in enumerate(optimizer.state.items()):
            for v_key in ["exp_avg", "exp_avg_sq"]:
                if dict_idx == 0:
                    if k.data.shape != (self.d_in, self.d_sae):
                        print(
                            "Warning: it does not seem as if resetting the Adam parameters worked, there are shapes mismatches"
                        )
                    if v[v_key][:, is_dead].abs().max().item() > 1e-6:
                        print(
                            "Warning: it does not seem as if resetting the Adam parameters worked"
                        )

        return n_dead

    @torch.no_grad()
    def resample_neurons_anthropic(
        self, dead_neuron_indices, model, optimizer, activation_store
    ):
        """
        Arthur's version of Anthropic's feature resampling
        procedure.
        """
        # collect global loss increases, and input activations
        global_loss_increases, global_input_activations = (
            self.collect_anthropic_resampling_losses(model, activation_store)
        )

        # sample according to losses
        probs = global_loss_increases / global_loss_increases.sum()
        sample_indices = torch.multinomial(
            probs,
            min(len(dead_neuron_indices), probs.shape[0]),
            replacement=False,
        )
        # if we don't have enough samples for for all the dead neurons, take the first n
        if sample_indices.shape[0] < len(dead_neuron_indices):
            dead_neuron_indices = dead_neuron_indices[: sample_indices.shape[0]]

        # Replace W_dec with normalized differences in activations
        self.W_dec.data[dead_neuron_indices, :] = (
            (
                global_input_activations[sample_indices]
                / torch.norm(
                    global_input_activations[sample_indices], dim=1, keepdim=True
                )
            )
            .to(self.dtype)
            .to(self.device)
        )

        # Lastly, set the new weights & biases
        self.W_enc.data[:, dead_neuron_indices] = self.W_dec.data[
            dead_neuron_indices, :
        ].T
        self.b_enc.data[dead_neuron_indices] = 0.0

        # Reset the Encoder Weights
        if dead_neuron_indices.shape[0] < self.d_sae:
            sum_of_all_norms = torch.norm(self.W_enc.data, dim=0).sum()
            sum_of_all_norms -= len(dead_neuron_indices)
            average_norm = sum_of_all_norms / (self.d_sae - len(dead_neuron_indices))
            self.W_enc.data[:, dead_neuron_indices] *= (
                self.cfg.feature_reinit_scale * average_norm
            )

            # Set biases to resampled value
            relevant_biases = self.b_enc.data[dead_neuron_indices].mean()
            self.b_enc.data[dead_neuron_indices] = (
                relevant_biases * 0
            )  # bias resample factor (put in config?)

        else:
            self.W_enc.data[:, dead_neuron_indices] *= self.cfg.feature_reinit_scale
            self.b_enc.data[dead_neuron_indices] = -5.0

        # TODO: Refactor this resetting to be outside of resampling.
        # reset the Adam Optimiser for every modified weight and bias term
        # Reset all the Adam parameters
        for dict_idx, (k, v) in enumerate(optimizer.state.items()):
            for v_key in ["exp_avg", "exp_avg_sq"]:
                if dict_idx == 0:
                    assert k.data.shape == (self.d_in, self.d_sae)
                    v[v_key][:, dead_neuron_indices] = 0.0
                elif dict_idx == 1:
                    assert k.data.shape == (self.d_sae,)
                    v[v_key][dead_neuron_indices] = 0.0
                elif dict_idx == 2:
                    assert k.data.shape == (self.d_sae, self.d_in)
                    v[v_key][dead_neuron_indices, :] = 0.0
                elif dict_idx == 3:
                    assert k.data.shape == (self.d_in,)
                else:
                    raise ValueError(f"Unexpected dict_idx {dict_idx}")

        # Check that the opt is really updated
        for dict_idx, (k, v) in enumerate(optimizer.state.items()):
            for v_key in ["exp_avg", "exp_avg_sq"]:
                if dict_idx == 0:
                    if k.data.shape != (self.d_in, self.d_sae):
                        print(
                            "Warning: it does not seem as if resetting the Adam parameters worked, there are shapes mismatches"
                        )
                    if v[v_key][:, dead_neuron_indices].abs().max().item() > 1e-6:
                        print(
                            "Warning: it does not seem as if resetting the Adam parameters worked"
                        )

        return

    @torch.no_grad()
    def collect_anthropic_resampling_losses(self, model, activation_store):
        """
        Collects the losses for resampling neurons (anthropic)
        """
        if isinstance(self.cfg, ViTSAERunnerConfig):
            raise Exception(
                "Currently, resampling is not supported for training on ViTs."
            )

        batch_size = self.cfg.store_batch_size

        # we're going to collect this many forward passes
        number_final_activations = self.cfg.resample_batches * batch_size
        # but have seq len number of tokens in each
        anthropic_iterator = range(0, number_final_activations, batch_size)
        anthropic_iterator = tqdm(
            anthropic_iterator, desc="Collecting losses for resampling..."
        )

        global_loss_increases = torch.zeros(
            (number_final_activations,), dtype=self.dtype, device=self.device
        )
        global_input_activations = torch.zeros(
            (number_final_activations, self.d_in), dtype=self.dtype, device=self.device
        )

        for refill_idx in anthropic_iterator:
            # get a batch, calculate loss with/without using SAE reconstruction.
            batch_tokens = activation_store.get_batch_tokens()
            ce_loss_with_recons = self.get_test_loss(batch_tokens, model)
            ce_loss_without_recons, normal_activations_cache = model.run_with_cache(
                batch_tokens,
                names_filter=self.cfg.hook_point,
                return_type="loss",
                loss_per_token=True,
            )
            # ce_loss_without_recons = model.loss_fn(normal_logits, batch_tokens, True)
            # del normal_logits

            normal_activations = normal_activations_cache[self.cfg.hook_point]
            if self.cfg.hook_point_head_index is not None:
                normal_activations = normal_activations[
                    :, :, self.cfg.hook_point_head_index
                ]

            # calculate the difference in loss
            changes_in_loss = ce_loss_with_recons - ce_loss_without_recons
            changes_in_loss = changes_in_loss.cpu()

            # sample from the loss differences
            probs = F.relu(changes_in_loss) / F.relu(changes_in_loss).sum(
                dim=1, keepdim=True
            )
            changes_in_loss_dist = Categorical(probs)
            samples = changes_in_loss_dist.sample()

            assert samples.shape == (batch_size,), (
                f"{samples.shape=}; {self.cfg.store_batch_size=}"
            )

            end_idx = refill_idx + batch_size
            global_loss_increases[refill_idx:end_idx] = changes_in_loss[
                torch.arange(batch_size), samples
            ]
            global_input_activations[refill_idx:end_idx] = normal_activations[
                torch.arange(batch_size), samples
            ]

        return global_loss_increases, global_input_activations

    @torch.no_grad()
    def get_test_loss(self, batch_tokens, model):
        """
        A method for running the model with the SAE activations in order to return the loss.
        returns per token loss when activations are substituted in.
        """
        head_index = self.cfg.hook_point_head_index

        def standard_replacement_hook(activations, hook):
            activations = self.forward(activations)[0].to(activations.dtype)
            return activations

        def head_replacement_hook(activations, hook):
            new_actions = self.forward(activations[:, :, head_index])[0].to(
                activations.dtype
            )
            activations[:, :, head_index] = new_actions
            return activations

        replacement_hook = (
            standard_replacement_hook if head_index is None else head_replacement_hook
        )

        ce_loss_with_recons = model.run_with_hooks(
            batch_tokens,
            return_type="loss",
            fwd_hooks=[(self.cfg.hook_point, replacement_hook)],
        )

        return ce_loss_with_recons

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Update grads so that they remove the parallel component
            (d_sae, d_in) shape
        """

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )

        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )

    def save_model(self, path: str):
        """
        Basic save function for the model. Saves the model's state_dict and the config used to train it.
        """

        # check if path exists
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)

        state_dict = {"cfg": self.cfg, "state_dict": self.state_dict()}

        if path.endswith(".pt"):
            torch.save(state_dict, path)
        elif path.endswith("pkl.gz"):
            with gzip.open(path, "wb") as f:
                pickle.dump(state_dict, f)
        else:
            raise ValueError(
                f"Unexpected file extension: {path}, supported extensions are .pt and .pkl.gz"
            )

        print(f"Saved model to {path}")

    @classmethod
    def load_from_pretrained(cls, path: str):
        """
        Load function for the model. Loads the model's state_dict and the config used to train it.
        This method can be called directly on the class, without needing an instance.
        """

        # Ensure the file exists
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No file found at specified path: {path}")

        # Load the state dictionary
        if path.endswith(".pt"):
            try:
                if torch.backends.mps.is_available():
                    state_dict = torch.load(path, map_location="mps")
                    state_dict["cfg"].device = "mps"
                else:
                    state_dict = torch.load(path)
            except Exception as e:
                raise IOError(f"Error loading the state dictionary from .pt file: {e}")

        elif path.endswith(".pkl.gz"):
            try:
                with gzip.open(path, "rb") as f:
                    state_dict = pickle.load(f)
            except Exception as e:
                raise IOError(
                    f"Error loading the state dictionary from .pkl.gz file: {e}"
                )
        elif path.endswith(".pkl"):
            try:
                with open(path, "rb") as f:
                    state_dict = pickle.load(f)
            except Exception as e:
                raise IOError(f"Error loading the state dictionary from .pkl file: {e}")
        else:
            raise ValueError(
                f"Unexpected file extension: {path}, supported extensions are .pt, .pkl, and .pkl.gz"
            )

        # Ensure the loaded state contains both 'cfg' and 'state_dict'
        if "cfg" not in state_dict or "state_dict" not in state_dict:
            raise ValueError(
                "The loaded state dictionary must contain 'cfg' and 'state_dict' keys"
            )

        # Create an instance of the class using the loaded configuration
        instance = cls(cfg=state_dict["cfg"])
        instance.load_state_dict(state_dict["state_dict"])

        return instance

    def get_name(self):
        sae_type = getattr(self.cfg, "sae_type", "standard")
        if isinstance(self.cfg, ViTSAERunnerConfig):
            sae_name = f"sae_{sae_type}_{self.cfg.model_name}_{self.cfg.block_layer}_{self.cfg.module_name}_{self.cfg.d_sae}"
        else:
            sae_name = f"sae_{sae_type}_{self.cfg.model_name}_{self.cfg.hook_point}_{self.cfg.d_sae}"
        return sae_name
