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
import numpy as np
import torch
import torch.nn.functional as F
from geom_median.torch import compute_geometric_median
from jaxtyping import Float
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from tqdm import tqdm
from transformer_lens.hook_points import HookedRootModule, HookPoint

from src.sae_training.config import ViTSAERunnerConfig


class _JumpReLUSTE(torch.autograd.Function):
    """Straight-through gradient estimator for the JumpReLU activation
    (Rajamanoharan et al. 2024, "Jumping Ahead"). The forward pass is an
    exact hard threshold (zero gradient almost everywhere), so backward
    substitutes a rectangle-kernel density estimate of how the threshold
    boundary moves, letting gradients reach both the pre-activation and the
    learned per-unit threshold.
    """

    @staticmethod
    def forward(ctx, hidden_pre_relu, threshold, bandwidth):
        ctx.save_for_backward(hidden_pre_relu, threshold)
        ctx.bandwidth = bandwidth
        mask = (hidden_pre_relu > threshold).to(hidden_pre_relu.dtype)
        return hidden_pre_relu * mask

    @staticmethod
    def backward(ctx, grad_output):
        hidden_pre_relu, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        mask = (hidden_pre_relu > threshold).to(hidden_pre_relu.dtype)
        grad_hidden_pre = grad_output * mask
        in_kernel = (torch.abs(hidden_pre_relu - threshold) < bandwidth / 2).to(hidden_pre_relu.dtype)
        kernel = in_kernel / bandwidth
        # Uses `threshold` (not `hidden_pre_relu`) as the coefficient here --
        # this is not a product-rule derivative of z*H(z-theta) (which would
        # naively suggest using z), it's the paper's own stated STE (Eq. for
        # d/dtheta JumpReLU_theta(z) := -theta/bandwidth * K((z-theta)/bandwidth)),
        # confirmed against SAELens's reference implementation
        # (jumprelu_sae.py's `ste = (threshold / bandwidth) * rectangle(...)`)
        # -- a deliberate near-boundary approximation since the kernel only
        # has support where z is already within `bandwidth` of theta.
        grad_threshold = grad_output * (-threshold) * kernel
        return grad_hidden_pre, grad_threshold, None


class _StepSTE(torch.autograd.Function):
    """Straight-through estimator for the Heaviside step H(z-theta), used to
    compute the JumpReLU SAE's L0 sparsity term (Rajamanoharan et al. 2024).

    Forward returns the exact (hard, non-differentiable) 0/1 indicator, so
    the resulting L0 value is a true sparsity count rather than a smooth
    proxy; backward substitutes the same rectangle-kernel density estimate
    as _JumpReLUSTE, confirmed against SAELens's reference `Step` class
    (jumprelu_sae.py): `ste = (1/bandwidth) * rectangle(...) * grad_output`
    for both the pre-activation and (negated) threshold gradients.
    """

    @staticmethod
    def forward(ctx, hidden_pre_relu, threshold, bandwidth):
        ctx.save_for_backward(hidden_pre_relu, threshold)
        ctx.bandwidth = bandwidth
        return (hidden_pre_relu > threshold).to(hidden_pre_relu.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        hidden_pre_relu, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        in_kernel = (torch.abs(hidden_pre_relu - threshold) < bandwidth / 2).to(hidden_pre_relu.dtype)
        kernel = in_kernel / bandwidth
        grad_hidden_pre = grad_output * kernel
        grad_threshold = grad_output * (-kernel)
        return grad_hidden_pre, grad_threshold, None


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

        if getattr(self.cfg, "jumprelu_sae", False):
            import math as _math

            self.log_threshold = nn.Parameter(
                torch.full(
                    (self.d_sae,),
                    _math.log(self.cfg.jumprelu_init_threshold),
                    dtype=self.dtype,
                    device=self.device,
                )
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

        self.setup()  # Required for `HookedRootModule`s

    def forward(self, x, dead_neuron_mask=None):
        if self.cfg.gated_sae:
            return self.forward_gated(x, dead_neuron_mask)
        elif getattr(self.cfg, "topk_sae", False):
            return self.forward_topk(x, dead_neuron_mask)
        elif getattr(self.cfg, "jumprelu_sae", False):
            return self.forward_jumprelu(x, dead_neuron_mask)
        else:
            return self.forward_standard(x, dead_neuron_mask)

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

    def forward_topk(self, x, dead_neuron_mask=None):
        """Top-K SAE (Gao et al. 2024, "Scaling and evaluating sparse
        autoencoders"). Sparsity is enforced structurally by keeping only the
        top cfg.topk_k ReLU'd pre-activations per token, so there is no L1
        penalty; l1_loss is reported as 0.0 for loss_dict schema
        compatibility with the other variants (trainer/logging code reads it
        unconditionally).

        Dead-latent revival uses the paper's own AuxK auxiliary loss, NOT
        ghost-grads' exp()-based trick used by forward_standard/forward_gated:
        AuxK reconstructs the current residual (x - sae_out) via the top
        cfg.topk_aux_k *dead* latents' own (plain, ReLU'd) activations --
        confirmed against the paper's stated formula, L_aux = ||e - e_hat||^2
        where e_hat = W_dec @ z using the top-k_aux dead latents, added to
        the loss with coefficient alpha (paper default 1/32). Ghost-grads'
        `exp(hidden_pre[dead])` was tried here first and produced inf/NaN
        during a real training run within a few thousand steps: with no L1
        penalty bounding hidden_pre's magnitude (unlike forward_standard),
        exponentiating raw pre-activations overflows readily -- exactly the
        numerical-stability failure mode the paper's AuxK design (plain,
        bounded activations, no exponential) avoids.
        """
        x = x.to(self.dtype)
        sae_in = self.hook_sae_in(x - self.b_dec)

        hidden_pre = self.hook_hidden_pre(
            einops.einsum(sae_in, self.W_enc, "... d_in, d_in d_sae -> ... d_sae") + self.b_enc
        )
        hidden_pre_relu = torch.nn.functional.relu(hidden_pre)
        k = min(self.cfg.topk_k, hidden_pre_relu.shape[-1])
        topk_vals, topk_idx = torch.topk(hidden_pre_relu, k=k, dim=-1)
        feature_acts = torch.zeros_like(hidden_pre_relu).scatter(-1, topk_idx, topk_vals)
        feature_acts = self.hook_hidden_post(feature_acts)

        sae_out = self.hook_sae_out(
            einops.einsum(feature_acts, self.W_dec, "... d_sae, d_sae d_in -> ... d_in") + self.b_dec
        )

        mse_loss = (
            torch.pow((sae_out - x.float()), 2) / (x**2).sum(dim=-1, keepdim=True).sqrt()
        )

        if len(mse_loss.size()) == 3 and self.training:
            mse_loss[:, 0, :] = mse_loss[:, 0, :] * self.cfg.mse_cls_coefficient
        mse_loss = mse_loss.mean()

        aux_loss = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        if self.training and dead_neuron_mask is not None and dead_neuron_mask.sum() > 0:
            n_dead = int(dead_neuron_mask.sum().item())
            k_aux = min(getattr(self.cfg, "topk_aux_k", 512), n_dead)
            if k_aux > 0:
                dead_idx = torch.nonzero(dead_neuron_mask, as_tuple=True)[0]
                dead_pre = hidden_pre_relu[..., dead_idx]
                aux_vals, aux_local_idx = torch.topk(dead_pre, k=k_aux, dim=-1)
                aux_idx = dead_idx[aux_local_idx]
                aux_acts = torch.zeros_like(hidden_pre_relu).scatter(-1, aux_idx, aux_vals)
                aux_recon = einops.einsum(
                    aux_acts, self.W_dec, "... d_sae, d_sae d_in -> ... d_in"
                )
                residual = (x - sae_out).detach()
                aux_loss = torch.pow(aux_recon - residual, 2).sum(dim=-1).mean()

        aux_coefficient = getattr(self.cfg, "topk_aux_coefficient", 1.0 / 32)
        l1_loss = torch.zeros((), dtype=self.dtype, device=self.device)
        loss = mse_loss + aux_coefficient * aux_loss

        loss_dict = {
            "mse_loss": mse_loss,
            "l1_loss": l1_loss,
            # repurposed key: Top-K's AuxK dead-latent loss, kept here for
            # loss_dict/logging schema compatibility with the other variants.
            "mse_loss_ghost_resid": aux_loss,
            "loss": loss.mean(),
        }

        return sae_out, feature_acts, loss_dict

    def forward_jumprelu(self, x, dead_neuron_mask=None):
        """JumpReLU SAE (Rajamanoharan et al. 2024, "Jumping Ahead"). A
        learned per-unit threshold replaces the fixed zero threshold of
        forward_standard's ReLU, trained via the straight-through estimator
        in _JumpReLUSTE. Sparsity is driven by an L0 loss rather than L1
        (L1 would also shrink the *magnitude* of units already exactly where
        the threshold wants them). The L0 term uses _StepSTE so its forward
        value is the true (hard) sparsity count, not a smooth proxy -- only
        its backward is relaxed, via the same rectangle-kernel STE as the
        activation itself (confirmed against SAELens's reference `Step`
        class, which computes L0 as `sum(Step.apply(hidden_pre, threshold,
        bandwidth))`, not a sigmoid relaxation).

        No ghost-grad-style dead-latent revival is used here (unlike
        forward_standard/forward_gated): the paper reports JumpReLU SAEs
        "consistently have few dead features, without the need for
        resampling," and ghost-grad's revival term exponentiates raw
        pre-activations (`exp(hidden_pre[dead])`) which are not L1-bounded
        for this variant (same as Top-K) -- confirmed empirically to
        overflow to inf/NaN during a real training run.
        """
        x = x.to(self.dtype)
        sae_in = self.hook_sae_in(x - self.b_dec)

        hidden_pre = self.hook_hidden_pre(
            einops.einsum(sae_in, self.W_enc, "... d_in, d_in d_sae -> ... d_sae") + self.b_enc
        )
        hidden_pre_relu = torch.nn.functional.relu(hidden_pre)
        threshold = torch.exp(self.log_threshold)
        feature_acts = self.hook_hidden_post(
            _JumpReLUSTE.apply(hidden_pre_relu, threshold, self.cfg.jumprelu_bandwidth)
        )

        sae_out = self.hook_sae_out(
            einops.einsum(feature_acts, self.W_dec, "... d_sae, d_sae d_in -> ... d_in") + self.b_dec
        )

        mse_loss = (
            torch.pow((sae_out - x.float()), 2) / (x**2).sum(dim=-1, keepdim=True).sqrt()
        )
        mse_loss_ghost_resid = torch.tensor(0.0, dtype=self.dtype, device=self.device)

        if len(mse_loss.size()) == 3 and self.training:
            mse_loss[:, 0, :] = mse_loss[:, 0, :] * self.cfg.mse_cls_coefficient
        mse_loss = mse_loss.mean()

        l0_indicator = _StepSTE.apply(hidden_pre_relu, threshold, self.cfg.jumprelu_bandwidth)
        l0_loss = self.cfg.jumprelu_l0_coefficient * l0_indicator.sum(dim=-1).mean()

        loss = mse_loss + l0_loss + mse_loss_ghost_resid

        loss_dict = {
            "mse_loss": mse_loss,
            # repurposed key: JumpReLU's L0 loss, kept here for
            # loss_dict/logging schema compatibility with the other variants.
            "l1_loss": l0_loss,
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
        """Gated SAE (Rajamanoharan et al. 2024). Shares the encoder direction
        W_enc/b_enc between two branches: a binary gate pi_gate = W_enc(x-b_dec)+b_gate
        (using b_enc as b_gate) that decides *which* features fire, and a magnitude
        branch pi_mag = (W_enc * exp(r_mag))(x-b_dec) + b_mag that decides *how much*.
        Final features are the gate (hard threshold) times the magnitude (ReLU).

        The gate is a Heaviside step, so it carries no useful gradient on its own;
        instead the L1 sparsity penalty and an auxiliary frozen-decoder
        reconstruction loss are both applied to relu(pi_gate), which is what
        actually trains W_enc/b_enc to make good gating decisions (without the aux
        loss, the gate sub-network can shrink towards zero to cheat the sparsity
        penalty without learning anything -- see Sec 3.2 / Fig 3 of the paper).
        """
        x = x.to(self.dtype)
        sae_in = self.hook_sae_in(x - self.b_dec)

        pi_gate = self.hook_hidden_pre(
            einops.einsum(sae_in, self.W_enc, "... d_in, d_in d_sae -> ... d_sae")
            + self.b_enc
        )
        feature_magnitudes_via_gate = F.relu(pi_gate)
        feature_acts_gate = (pi_gate > 0).to(self.dtype)

        W_mag = self.W_enc * torch.exp(self.r_mag)
        pi_mag = (
            einops.einsum(sae_in, W_mag, "... d_in, d_in d_sae -> ... d_sae")
            + self.b_mag
        )
        feature_acts_mag = F.relu(pi_mag)

        feature_acts = self.hook_hidden_post(feature_acts_gate * feature_acts_mag)

        sae_out = self.hook_sae_out(
            einops.einsum(
                feature_acts, self.W_dec, "... d_sae, d_sae d_in -> ... d_in"
            )
            + self.b_dec
        )

        mse_loss = (
            torch.pow((sae_out - x.float()), 2)
            / (x**2).sum(dim=-1, keepdim=True).sqrt()
        )

        # Auxiliary loss: reconstruct via the gate branch's own (differentiable)
        # magnitudes, through a FROZEN copy of the decoder, so the gate
        # sub-network gets a real training signal independent of the live
        # decoder/magnitude branch.
        via_gate_reconstruction = (
            einops.einsum(
                feature_magnitudes_via_gate,
                self.W_dec.detach(),
                "... d_sae, d_sae d_in -> ... d_in",
            )
            + self.b_dec.detach()
        )
        aux_loss = (
            torch.pow((via_gate_reconstruction - x.float()), 2)
            / (x**2).sum(dim=-1, keepdim=True).sqrt()
        )

        mse_loss_ghost_resid = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        if self.cfg.use_ghost_grads and self.training and dead_neuron_mask is not None and dead_neuron_mask.sum() > 0:
            residual = x - sae_out
            l2_norm_residual = torch.norm(residual, dim=-1)

            if len(pi_gate.size()) == 3:
                feature_acts_dead_neurons_only = torch.exp(pi_gate[:, :, dead_neuron_mask])
            else:
                feature_acts_dead_neurons_only = torch.exp(pi_gate[:, dead_neuron_mask])
            ghost_out = feature_acts_dead_neurons_only @ self.W_dec[dead_neuron_mask, :]
            l2_norm_ghost_out = torch.norm(ghost_out, dim=-1)
            norm_scaling_factor = l2_norm_residual / (1e-6 + l2_norm_ghost_out * 2)
            if len(pi_gate.size()) == 3:
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

        if len(mse_loss.size()) == 3 and self.training:
            mse_loss[:, 0, :] = mse_loss[:, 0, :] * self.cfg.mse_cls_coefficient
        mse_loss = mse_loss.mean()
        aux_loss = aux_loss.mean()

        sparsity = torch.abs(feature_magnitudes_via_gate).sum(dim=-1).mean(dim=(0,))
        l1_loss = self.l1_coefficient * sparsity
        loss = mse_loss + l1_loss + mse_loss_ghost_resid + aux_loss

        loss_dict = {
            "mse_loss": mse_loss,
            "l1_loss": l1_loss.mean(),
            "mse_loss_ghost_resid": mse_loss_ghost_resid,
            "aux_loss": aux_loss,
            "loss": loss.mean(),
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

    def save_model(self, path: str, training_metadata=None):
        """
        Basic save function for the model. Saves the model's state_dict and the config used to train it.
        """

        # check if path exists
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)

        state_dict = {"cfg": self.cfg, "state_dict": self.state_dict()}
        experiment_metadata = getattr(self.cfg, "experiment_metadata", None)
        if experiment_metadata is not None:
            # Duplicate provenance at the checkpoint top level so audits can
            # inspect experimental factors without depending on cfg internals.
            state_dict["experiment_metadata"] = experiment_metadata
        if training_metadata is not None:
            state_dict["training_metadata"] = training_metadata

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
        if isinstance(self.cfg, ViTSAERunnerConfig):
            sae_name = f"sparse_autoencoder_{self.cfg.model_name}_{self.cfg.block_layer}_{self.cfg.module_name}_{self.cfg.d_sae}"
        else:
            sae_name = f"sparse_autoencoder_{self.cfg.model_name}_{self.cfg.hook_point}_{self.cfg.d_sae}"
        return sae_name
