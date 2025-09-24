# sae_model.py
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
    
def get_model(model_name, d_in=None):
    if model_name == "sae":
        model = SparseAutoencoder(
            in_dims=d_in,
            h_dims=8192,                  # 8k codebook for a simple run
            sparsity_lambda=1e-4,
            sparsity_target=0.05,
            xavier_norm_init=True
        )
    
    if "clip" in model_name:
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    return model