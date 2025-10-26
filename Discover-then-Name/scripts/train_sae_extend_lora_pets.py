#!/usr/bin/env python3
"""
Train-time dictionary growth for a pretrained SAE:
- Freeze old atoms (encoder/decoder rows & biases).
- Add k new atoms with LoRA low-rank parameterization (ΔE, ΔD) ONLY on new rows/cols.
- Optimize on Oxford-IIIT Pets CLIP activations; CC3M replay enforces no-forgetting.
- Group-L1 sparsity on new-atom activations.

Assumptions:
- You have a pretrained SAE checkpoint compatible with SparseAutoencoder below (or import yours).
- We'll extract CLIP layer activations online via open_clip for Pets.
"""

import os
import math
import json
import random
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# --------- CONFIG (edit these) ----------
CKPT_SAE = "/home/sunayana/Documents/Concept_LoRA/Discover-then-Name/pretrained/Checkpoints/clip_ViT-B:16_sparse_autoencoder_final.pt"         # your CC3M-trained SAE checkpoint
OUTPUT_DIR = "./runs_sae_extend_pets"
MODEL_NAME = "ViT-B-16"                          # CLIP visual backbone for activations
PRETRAIN = "openai"
CLIP_LAYER = "blocks.10.mlp.fc2"                 # layer to hook (example for ViT-B/16)
IMSIZE = 224
BATCH = 64
EPOCHS = 10
LR = 2e-3                                        # optimizer lr for new LoRA params only
WEIGHT_DECAY = 0.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# New atoms + LoRA ranks
K_NEW = 128
RANK_E = 8
RANK_D = 8

# Loss weights
LAMBDA_L1_NEW = 5e-4          # group sparsity on new atoms (use L1 on z_new)
LAMBDA_CC3M_REPLAY = 1.0      # recon loss on CC3M buffer (new atoms masked)
CC3M_REPLAY_EVERY = 2         # steps
CC3M_BUFFER_PATH = "/path/to/cc3m_clip_activations.pt"  # tensor [N, in_dim] or memmap-like
CC3M_BUFFER_NUM = 8192        # how many to load (subset if large)

# Oxford-IIIT Pets dataset root (torchvision-style)
PETS_ROOT = "/path/to/oxford-iiit-pet"
NUM_WORKERS = 6
# ----------------------------------------


# ===== Minimal SAE skeleton (use yours if available) =====
class SparseAutoencoder(nn.Module):
    """
    Simple SAE: Linear encoder -> nonlinearity -> Linear decoder
    Your version likely has different details; adapt shapes but keep weight names.
    """
    def __init__(self, n_input_features: int, n_learned_features: int,
                 activation="relu"):
        super().__init__()
        self.n_input = n_input_features
        self.n_hidden = n_learned_features
        self.encoder = nn.Linear(n_input_features, n_learned_features, bias=True)
        self.decoder = nn.Linear(n_learned_features, n_input_features, bias=True)
        self.act = nn.ReLU() if activation == "relu" else nn.Sigmoid()

    def forward(self, x):
        z = self.act(self.encoder(x))   # [B, H]
        x_hat = self.decoder(z)         # [B, D]
        return z, x_hat


# ===== LoRA-extended wrapper for NEW atoms only =====
class SAELoRAGrow(nn.Module):
    """
    Wraps a pretrained SAE:
      - Freezes old encoder/decoder weights.
      - Adds K new atoms whose weights are low-rank ΔW (LoRA).
      - Forward returns concatenated z=[z_old, z_new] and recon via [old+new] decoder.
    """
    def __init__(self, sae: SparseAutoencoder, k_new: int, rank_e: int, rank_d: int):
        super().__init__()
        self.sae = sae.eval()
        for p in self.sae.parameters():
            p.requires_grad_(False)

        self.n_in = sae.n_input
        self.n_old = sae.n_hidden
        self.n_new = k_new

        # ----- Encoder new rows (shape: k_new x n_in) with LoRA ΔW = A_e @ B_e^T -----
        self.enc_A = nn.Parameter(torch.zeros(self.n_new, rank_e))
        self.enc_B = nn.Parameter(torch.zeros(self.n_in,  rank_e))
        nn.init.kaiming_uniform_(self.enc_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.enc_B, a=math.sqrt(5))
        self.enc_bias = nn.Parameter(torch.zeros(self.n_new))

        # ----- Decoder new columns (shape: n_in x k_new) with LoRA ΔW = A_d @ B_d^T -----
        self.dec_A = nn.Parameter(torch.zeros(self.n_in,  rank_d))
        self.dec_B = nn.Parameter(torch.zeros(self.n_new, rank_d))
        nn.init.kaiming_uniform_(self.dec_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.dec_B, a=math.sqrt(5))
        self.dec_bias = nn.Parameter(torch.zeros(self.n_in))  # we’ll add on top of frozen decoder bias

        # Nonlinearity to match SAE
        self.act = self.sae.act

    @property
    def E_old(self):
        return self.sae.encoder.weight  # [n_old, n_in]

    @property
    def bE_old(self):
        return self.sae.encoder.bias    # [n_old]

    @property
    def D_old(self):
        return self.sae.decoder.weight  # [n_in, n_old]

    @property
    def bD_old(self):
        return self.sae.decoder.bias    # [n_in]

    def encoder_new_weight(self):
        # [k_new, n_in]
        return self.enc_A @ self.enc_B.T

    def decoder_new_weight(self):
        # [n_in, k_new]
        return self.dec_A @ self.dec_B.T

    def forward(self, x, mask_new: Optional[torch.Tensor] = None):
        """
        x: [B, n_in]
        mask_new: Optional [B, 1] boolean to zero-out new atoms (for CC3M replay)
        """
        # Old path
        pre_old = F.linear(x, self.E_old, self.bE_old)             # [B, n_old]
        z_old = self.act(pre_old)

        # New path (LoRA-only rows)
        Wnew_e = self.encoder_new_weight()                         # [k_new, n_in]
        pre_new = F.linear(x, Wnew_e, self.enc_bias)               # [B, n_new]
        z_new = self.act(pre_new)

        if mask_new is not None:
            z_new = z_new * (~mask_new)  # if mask_new True -> zero new atoms

        # Decode with concatenated codes
        # Old contribution
        x_old = F.linear(z_old, self.D_old.T, None)                # [B, n_in]
        # New contribution
        Wnew_d = self.decoder_new_weight()                         # [n_in, k_new]
        x_new = F.linear(z_new, Wnew_d.T, None)                    # [B, n_in]

        # Bias: frozen old + trainable delta
        x_hat = x_old + x_new + (self.bD_old + self.dec_bias)

        return (z_old, z_new), x_hat


# ===== Datasets: Oxford-IIIT Pets activations via CLIP; CC3M replay buffer =====
class CLIPActivations(Dataset):
    def __init__(self, root, split="trainval", imsize=224, model_name="ViT-B-16", pretrain="openai",
                 layer=CLIP_LAYER, device=DEVICE):
        from torchvision import transforms, datasets
        import open_clip

        self.device = device
        self.tform = transforms.Compose([
            transforms.Resize((imsize, imsize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711)),
        ])
        # Oxford-IIIT Pets torchvision wrapper
        # We’ll just use images; labels optional (not needed for unsupervised SAE recon)
        self.ds = datasets.OxfordIIITPet(root=root, split=split, download=False, transform=self.tform)

        # CLIP
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrain, device=self.device)
        self.model.eval()

        # Hook to grab hidden layer activations
        self.layer_name = layer
        self.buffer = []

        def hook_fn(module, inp, out):
            with torch.no_grad():
                self.buffer.append(out.detach())  # [B, D]

        # register hook
        target = self._resolve_layer(self.model.visual)
        self.hook = target.register_forward_hook(hook_fn)

    def _resolve_layer(self, visual):
        # Example for ViT-B/16: blocks.10.mlp.fc2
        mod = visual
        for attr in self.layer_name.split("."):
            mod = getattr(mod, attr)
        return mod

    def __len__(self):
        return len(self.ds)

    @torch.no_grad()
    def __getitem__(self, idx):
        img, _ = self.ds[idx]
        img = img.to(self.device)
        self.buffer.clear()
        _ = self.model.encode_image(img.unsqueeze(0))  # forward
        assert len(self.buffer) == 1, "Hook failed to capture activations"
        act = self.buffer[0].squeeze(0).cpu()         # [D]
        return act


class CC3MReplay(Dataset):
    """
    Expects a .pt tensor file [N, D] of CLIP activations from CC3M.
    During replay we mask new atoms ==> the model must reconstruct with old atoms only.
    """
    def __init__(self, path, take_first=CC3M_BUFFER_NUM):
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            # try common keys
            for k in ("acts", "activations", "X", "tensor"):
                if k in obj and isinstance(obj[k], torch.Tensor):
                    obj = obj[k]; break
        assert isinstance(obj, torch.Tensor) and obj.ndim == 2
        if take_first is not None and take_first < obj.shape[0]:
            obj = obj[:take_first]
        self.X = obj.float().contiguous()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


# ====== Training utils ======
def reconstruction_loss(x_hat, x):
    return F.mse_loss(x_hat, x)

def group_l1(z_new):
    # simple elementwise L1; for group-L1 by (sample), keep this (sums act magnitudes)
    return z_new.abs().mean()

def set_seed(seed=0):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ====== Main ======
def main():
    set_seed(0)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Load pretrained SAE (CC3M)
    ckpt = torch.load(CKPT_SAE, map_location="cpu")
    n_in = ckpt.get("n_input", None)
    n_old = ckpt.get("n_hidden", None)
    if n_in is None or n_old is None:
        # Try infer from weights
        w_enc = ckpt["encoder.weight"] if "encoder.weight" in ckpt else ckpt["state_dict"]["encoder.weight"]
        n_old, n_in = w_enc.shape

    sae = SparseAutoencoder(n_input_features=n_in, n_learned_features=n_old).to(DEVICE)
    # Flexible load
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sae.load_state_dict(sd, strict=False)
    sae.eval()

    # 2) Wrap with LoRA growth
    model = SAELoRAGrow(sae, k_new=K_NEW, rank_e=RANK_E, rank_d=RANK_D).to(DEVICE)

    # 3) Optimizer: ONLY new LoRA params + new biases
    trainable = [model.enc_A, model.enc_B, model.enc_bias, model.dec_A, model.dec_B, model.dec_bias]
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)

    # 4) Data
    pets = CLIPActivations(PETS_ROOT, split="trainval", imsize=IMSIZE,
                           model_name=MODEL_NAME, pretrain=PRETRAIN, layer=CLIP_LAYER, device=DEVICE)
    dl = DataLoader(pets, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    cc3m = CC3MReplay(CC3M_BUFFER_PATH, take_first=CC3M_BUFFER_NUM)
    dl_replay = DataLoader(cc3m, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS)

    # 5) Train
    step = 0
    best = float("inf")
    for epoch in range(EPOCHS):
        for xb in dl:
            xb = xb.to(DEVICE, non_blocking=True)
            # forward on Pets
            (z_old, z_new), x_hat = model(xb)
            loss_recon = reconstruction_loss(x_hat, xb)
            loss_sparsity = LAMBDA_L1_NEW * group_l1(z_new)

            loss = loss_recon + loss_sparsity

            # CC3M replay (mask new atoms)
            if step % CC3M_REPLAY_EVERY == 0:
                try:
                    rb = next(cc3m_iter)
                except:
                    cc3m_iter = iter(dl_replay)
                    rb = next(cc3m_iter)
                rb = rb.to(DEVICE, non_blocking=True)
                mask_new = torch.ones((rb.size(0), 1), dtype=torch.bool, device=DEVICE)
                (_, _), rb_hat = model(rb, mask_new=mask_new)
                loss += LAMBDA_CC3M_REPLAY * reconstruction_loss(rb_hat, rb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
            opt.step()

            if step % 50 == 0:
                print(f"ep {epoch} step {step} | recon {loss_recon.item():.4e} | L1new {loss_sparsity.item():.4e} | total {loss.item():.4e}")

            # save best by Pets recon
            if loss_recon.item() < best:
                best = loss_recon.item()
                save_path = Path(OUTPUT_DIR) / "sae_grown_lora.pt"
                torch.save({
                    "state_dict_frozen_sae": sae.state_dict(),    # for reference
                    "state_dict_new": {
                        "enc_A": model.enc_A.detach().cpu(),
                        "enc_B": model.enc_B.detach().cpu(),
                        "enc_bias": model.enc_bias.detach().cpu(),
                        "dec_A": model.dec_A.detach().cpu(),
                        "dec_B": model.dec_B.detach().cpu(),
                        "dec_bias": model.dec_bias.detach().cpu(),
                        "n_input": model.n_in,
                        "n_old": model.n_old,
                        "n_new": model.n_new,
                        "rank_e": RANK_E,
                        "rank_d": RANK_D,
                        "clip_layer": CLIP_LAYER,
                        "model_name": MODEL_NAME,
                        "pretrain": PRETRAIN,
                    }
                }, save_path)
            step += 1

    print("Done.")


if __name__ == "__main__":
    main()
