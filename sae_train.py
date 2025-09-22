# pip install open_clip_torch torchvision torch
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import open_clip

# ---- 1) Load CLIP + CIFAR-100 (with CLIP's preprocess) ----
def load_clip(model_name="ViT-B-32", pretrained="openai", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()
    return model, preprocess, device

def cifar100_loader(preprocess, split="train", batch_size=256):
    ds = datasets.CIFAR100(root="./datasets/cub2002011", train=(split=="train"), download=True, transform=preprocess)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# ---- 2) Collect CLIP global embeddings (one per image) ----
@torch.no_grad()
def collect_clip_embeddings(model, dl, device):
    embs = []
    for imgs, _ in dl:
        imgs = imgs.to(device, non_blocking=True)
        x = model.encode_image(imgs)            # [B, Dproj] (e.g., Dproj=512 for ViT-B-32)
        x = F.normalize(x, dim=-1)              # already normalized by CLIP, but keep it safe
        embs.append(x.cpu())
    X = torch.cat(embs, dim=0)                  # [N, Dproj]
    return X

def standardize(X):
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (X - mean) / std, mean, std

# ---- 3) Train your existing SAE on embeddings ----
def train_sae(sae_model, Xn, device, epochs=10, batch_tokens=4096, lr=3e-4, l2_decay=1e-6):
    sae_model.to(device).train()
    opt = torch.optim.AdamW(sae_model.parameters(), lr=lr, weight_decay=l2_decay)
    dl = DataLoader(TensorDataset(Xn), batch_size=batch_tokens, shuffle=True, pin_memory=True)
    for ep in range(epochs):
        tot = 0.0
        for (batch,) in dl:
            batch = batch.to(device, non_blocking=True)
            enc, dec = sae_model(batch)
            loss = sae_model.loss_function(dec, batch, enc)  # your KL sparsity + MSE
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(sae_model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        print(f"Epoch {ep+1}/{epochs} | Loss {tot/len(dl):.4f}")

if __name__ == "__main__":
    # 0) import your SAE class from your file, or paste it above
    from scripts.sae_model import SparseAutoencoder  # <- change if needed

    model, preprocess, device = load_clip("ViT-B-32", "openai")
    train_dl = cifar100_loader(preprocess, "train", batch_size=256)

    # 1) Get CLIP embeddings for CIFAR-100 train
    X = collect_clip_embeddings(model, train_dl, device)     # [N, D] e.g., D=512
    Xn, mean, std = standardize(X)

    # 2) Build SAE (d_in from embeddings; pick a modest code size)
    d_in = Xn.shape[1]
    sae = SparseAutoencoder(
        in_dims=d_in,
        h_dims=8192,                  # 8k codebook for a simple run
        sparsity_lambda=1e-4,
        sparsity_target=0.05,
        xavier_norm_init=True
    )

    # 3) Train SAE on embeddings
    train_sae(sae, Xn, device, epochs=10, batch_tokens=8192, lr=3e-4)

    # Optional: save
    torch.save({"state_dict": sae.state_dict(), "mean": mean, "std": std}, "sae_clip_cifar100.pth")
    print("Saved to sae_clip_cifar100.pth")
