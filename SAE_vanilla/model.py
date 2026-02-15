# sae_model.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

class SparseAutoencoder(nn.Module):
    def __init__(self, in_dims, h_dims, sparsity_lambda=1e-4, sparsity_target=0.05, xavier_norm_init=True):
        super().__init__()
        self.in_dims = in_dims
        self.h_dims = h_dims
        self.sparsity_lambda = sparsity_lambda
        self.sparsity_target = sparsity_target

        self.encoder = nn.Sequential(nn.Linear(in_dims, h_dims), nn.Sigmoid())
        self.decoder = nn.Sequential(nn.Linear(h_dims, in_dims), nn.Tanh())

        if xavier_norm_init:
            nn.init.xavier_uniform_(self.encoder[0].weight); nn.init.constant_(self.encoder[0].bias, 0)
            nn.init.xavier_uniform_(self.decoder[0].weight); nn.init.constant_(self.decoder[0].bias, 0)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

    def sparsity_penalty(self, z):
        rho_hat = torch.mean(z, dim=0).clamp(1e-8, 1-1e-8)
        rho = self.sparsity_target
        kl = rho*torch.log(rho/rho_hat) + (1-rho)*torch.log((1-rho)/(1-rho_hat))
        return self.sparsity_lambda * kl.sum()

    def loss_function(self, x_hat, x, z):
        return F.mse_loss(x_hat, x) + self.sparsity_penalty(z)
    
def get_base_model(model_name, model_path=None, d_in=None):
    """
    Returns (model, preprocess). model_name can be a descriptive name (e.g. "clip_vit-b-32")
    or an architecture hint containing 'vit' (will map to "ViT-B-32"). If model_path is
    provided and exists, it will be passed as the pretrained argument to open_clip so custom
    weights are loaded. Otherwise falls back to the default 'openai' pretrained weights.
    """
    # infer architecture from model_name (keep simple mapping for common case)
    if "vit" in model_name.lower():
        arch = "ViT-B-32"
    else:
        arch = model_name

    # if a custom path is supplied and exists, pass it to open_clip as pretrained
    if model_path and os.path.exists(model_path):
        pretrained_arg = model_path
    else:
        pretrained_arg = "openai"

    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained_arg)
    return model, preprocess