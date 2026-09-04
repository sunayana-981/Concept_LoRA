"""
# Portions of this file are based on code from the “jbloomAus/SAELens” and "HugoFry/mats_sae_training_for_ViTs" repositories (MIT-licensed):
    https://github.com/jbloomAus/SAELens/blob/main/sae_lens/config.py
    https://github.com/HugoFry/mats_sae_training_for_ViTs/blob/main/sae_training/config.py
"""

from dataclasses import dataclass
from typing import Optional

import torch
import wandb


class Config:
    def __init__(self, config_dict):
        if not isinstance(config_dict, dict):
            config_dict = config_dict.__dict__

        for key, value in config_dict.items():
            if isinstance(value, dict):
                # Recursively convert nested dictionaries
                value = Config(value)
            setattr(self, key, value)


@dataclass
class ViTSAERunnerConfig:
    """
    Configuration for training a sparse autoencoder on a vision transformer.
    """

    # Data Generating Function (Model + Training Distibuion)
    custom_clip_ckpt_path: str = None
    class_token: bool = True
    image_width: int = 224
    image_height: int = 224
    model_name: str = "openai/clip-vit-base-patch32"
    module_name: str = "resid"
    block_layer: int = 10
    dataset_path: str = "evanarlian/imagenet_1k_resized_256"
    image_key: str = "image"
    label_key: str = "label"
    use_cached_activations: bool = False
    cached_activations_path: Optional[str] = (
        None  # Defaults to "activations/{dataset}/{model}/{full_hook_name}_{hook_point_head_index}"
    )

    # SAE Parameters
    d_in: int = 768

    # Activation Store Parameters
    # Historical name retained for checkpoint compatibility. ViT trainers
    # increment this counter by the leading image-batch dimension, so its unit
    # is training examples/images rather than patch activation tokens.
    total_training_tokens: int = 2_000_000
    training_examples: Optional[int] = None
    activation_vectors_per_example: Optional[int] = None
    n_batches_in_store: int = 32
    store_size: Optional[int] = None
    max_batch_size_for_vit_forward_pass: int = 1024
    create_dataloader: bool = True

    # Misc
    device: str = "cpu"
    seed: int = 42
    dtype: torch.dtype = torch.float32

    # SAE Parameters
    b_dec_init_method: str = "geometric_median"
    expansion_factor: int = 4
    from_pretrained_path: Optional[str] = None
    gated_sae: bool = False

    # Top-K SAE (Gao et al. 2024, "Scaling and evaluating sparse
    # autoencoders"): keep only the top-k pre-activations per token instead of
    # thresholding at zero; sparsity is enforced structurally so no L1 penalty
    # is used. Dead-latent revival uses the paper's AuxK auxiliary loss
    # (reconstruct the residual via the top topk_aux_k dead latents' own
    # activations), not ghost-grads -- see forward_topk's docstring.
    topk_sae: bool = False
    topk_k: int = 32
    topk_aux_k: int = 512
    topk_aux_coefficient: float = 1.0 / 32

    # JumpReLU SAE (Rajamanoharan et al. 2024, "Jumping Ahead"): a learned
    # per-unit threshold with a straight-through gradient estimator, trained
    # against an L0 penalty (hard count in the forward pass, STE-relaxed in
    # the backward pass -- see forward_jumprelu's docstring) instead of L1.
    jumprelu_sae: bool = False
    jumprelu_bandwidth: float = 0.001
    jumprelu_init_threshold: float = 0.001
    jumprelu_l0_coefficient: float = 1e-3

    # Matryoshka SAE (Bussmann et al. 2024): nested dictionary prefixes share
    # one encoder/decoder; the training loss averages reconstruction over
    # increasing prefix sizes so early latents must be independently useful.
    matryoshka_sae: bool = False
    matryoshka_levels: int = 1
    matryoshka_min_group_fraction: float = 1.0 / 64

    # Training Parameters
    l1_coefficient: float = 1e-3
    lr: float = 3e-4
    lr_scheduler_name: str = "constant"  # constant, constantwithwarmup, linearwarmupdecay, cosineannealing, cosineannealingwarmup
    lr_warm_up_steps: int = 500
    batch_size: int = 4096
    mse_cls_coefficient: float = 1.0

    # Resampling protocol args
    use_ghost_grads: bool = True
    feature_sampling_window: int = (
        2000  # May need to change this since by default I will use ghost grads
    )
    feature_sampling_method: str = "anthropic"  # None or Anthropic
    resample_batches: int = 32
    feature_reinit_scale: float = 0.2
    dead_feature_window: int = 1000  # unless this window is larger feature sampling,
    dead_feature_estimation_method: str = "no_fire"
    dead_feature_threshold: float = 1e-8

    # WANDB
    log_to_wandb: bool = True
    wandb_project: str = "mats-hugo"
    wandb_entity: str = None
    wandb_log_frequency: int = 10

    # Misc
    n_checkpoints: int = 0
    checkpoint_path: str = "checkpoints"
    experiment_metadata: Optional[dict] = None

    image_key = "image"
    label_key = "label"

    def __post_init__(self):
        if self.training_examples is None:
            self.training_examples = self.total_training_tokens
        else:
            self.total_training_tokens = self.training_examples

        self.store_size = self.n_batches_in_store * self.batch_size

        # Autofill cached_activations_path unless the user overrode it
        if self.cached_activations_path is None:
            self.cached_activations_path = f"activations/{self.dataset_path.replace('/', '_')}/{self.model_name.replace('/', '_')}/{self.block_layer}_{self.module_name}"

        self.d_sae = self.d_in * self.expansion_factor

        self.run_name = f"{self.d_sae}-L1-{self.l1_coefficient}-LR-{self.lr}-Tokens-{self.total_training_tokens:3.3e}"

        if self.feature_sampling_method not in [None, "l2", "anthropic"]:
            raise ValueError(
                f"feature_sampling_method must be None, l2, or anthropic. Got {self.feature_sampling_method}"
            )

        if self.b_dec_init_method not in ["geometric_median", "mean", "zeros"]:
            raise ValueError(
                f"b_dec_init_method must be geometric_median, mean, or zeros. Got {self.b_dec_init_method}"
            )
        if self.b_dec_init_method == "zeros":
            print(
                "Warning: We are initializing b_dec to zeros. This is probably not what you want."
            )

        self.device = torch.device(self.device)

        unique_id = wandb.util.generate_id()
        self.checkpoint_path = f"{self.checkpoint_path}/{unique_id}"

        print(
            f"Run name: {self.d_sae}-L1-{self.l1_coefficient}-LR-{self.lr}-Tokens-{self.total_training_tokens:3.3e}"
        )
        # Print out some useful info:

        total_training_steps = self.total_training_tokens // self.batch_size
        print(f"Total training steps: {total_training_steps}")

        total_wandb_updates = total_training_steps // self.wandb_log_frequency
        print(f"Total wandb updates: {total_wandb_updates}")

        # how many times will we sample dead neurons?
        # assert self.dead_feature_window <= self.feature_sampling_window, "dead_feature_window must be smaller than feature_sampling_window"
        n_dead_feature_samples = total_training_steps // self.dead_feature_window
        n_feature_window_samples = total_training_steps // self.feature_sampling_window
        print(
            f"n_tokens_per_feature_sampling_window (millions): {(self.feature_sampling_window * self.batch_size) / 10**6}"
        )
        print(
            f"n_tokens_per_dead_feature_window (millions): {(self.dead_feature_window * self.batch_size) / 10**6}"
        )
        if self.feature_sampling_method is not None:
            print(f"We will reset neurons {n_dead_feature_samples} times.")

        if self.use_ghost_grads:
            print("Using Ghost Grads.")

        print(
            f"We will reset the sparsity calculation {n_feature_window_samples} times."
        )
        print(
            f"Number of tokens when resampling: {self.resample_batches * self.batch_size}"
        )
        # print("Number tokens in dead feature calculation window: ", self.dead_feature_window * self.train_batch_size)
        print(
            f"Number tokens in sparsity calculation window: {self.feature_sampling_window * self.batch_size:.2e}"
        )
