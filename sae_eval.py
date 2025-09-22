# pip install torch torchvision open_clip_torch matplotlib
import argparse, os, csv
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import open_clip

# ---- SAE loader (rebuild from checkpoint shapes) ----
def build_sae_from_state_dict(state_dict):
    import torch.nn as nn
    # infer dims from decoder weight: [in_dim, h_dim]
    W = state_dict["decoder.0.weight"]
    in_dim, h_dim = W.shape
    class SAE(nn.Module):
        def __init__(self, in_dims, h_dims):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(in_dims, h_dims), nn.Sigmoid())
            self.decoder = nn.Sequential(nn.Linear(h_dims, in_dims), nn.Tanh())

        def forward(self, x):
            z = self.encoder(x)
            x_hat = self.decoder(z)
            return z, x_hat

        def loss_function(self, x_hat, x, z):  # not used here
            rho_hat = z.mean(dim=0).clamp(1e-8, 1-1e-8)
            # dummy sparsity to satisfy signature
            return F.mse_loss(x_hat, x) + 0.0 * (rho_hat.sum())
    sae = SAE(in_dim, h_dim)
    sae.load_state_dict(state_dict, strict=True)
    return sae

# ---- CLIP + CIFAR100 ----
def load_clip(model_name="ViT-B-32", pretrained="openai", device="cuda"):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()
    return model, preprocess

def cifar100_loader(preprocess, split="train", batch_size=256):
    ds = datasets.CIFAR100(root="./data", train=(split=="train"), download=True, transform=preprocess)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return ds, dl

@torch.no_grad()
def collect_clip_embeddings(model, dl, device):
    embs = []
    for imgs, _ in dl:
        imgs = imgs.to(device, non_blocking=True)
        x = model.encode_image(imgs).float()   # [B, D]
        x = F.normalize(x, dim=-1)
        embs.append(x.cpu())
    X = torch.cat(embs, dim=0)                # [N, D]
    return X

def standardize_with_stats(X, mean, std):
    return (X - mean) / std.clamp_min(1e-6)

# ---- Concept inspection helpers ----
def topk_indices_for_neuron(Z, j, k):
    return torch.topk(Z[:, j], k).indices.tolist()

def save_topk_grid(ds_raw, indices, out_path, ncol=8):
    imgs = [ds_raw[i][0] for i in indices]  # already preprocessed tensors
    grid = vutils.make_grid(imgs, nrow=ncol, normalize=True, scale_each=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(ncol*1.5, max(1, len(indices)//ncol)*1.5))
    plt.axis("off"); plt.imshow(grid.permute(1, 2, 0))
    plt.savefig(out_path, bbox_inches="tight", dpi=200); plt.close()

@torch.no_grad()
def text_align_atoms(clip_model, labels, atoms, device):
    tok = open_clip.get_tokenizer("ViT-B-32")
    prompts = [f"a photo of a {t}" for t in labels]
    txt = tok(prompts).to(device)

    T = clip_model.encode_text(txt).float()     # [L, D] on device
    T = F.normalize(T, dim=-1)

    atoms = atoms.to(device)                    # <<< move to same device
    atoms = F.normalize(atoms, dim=0)           # [D, H] on device

    S = T @ atoms                               # [L, H]
    return S

@torch.no_grad()
def class_prototype_align(Xn, y, atoms, num_classes=100):
    # Xn is standardized + (ideally) L2-normalized embeddings
    Xn_u = F.normalize(Xn, dim=-1)
    protos = []
    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=True)[0]
        protos.append(F.normalize(Xn_u[idx].mean(dim=0, keepdim=True), dim=-1))
    P = torch.cat(protos, dim=0)            # [C, D]
    atoms = F.normalize(atoms, dim=0)       # [D, H]
    return P @ atoms                        # [C, H]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="sae_clip_cifar100.pth", help="Checkpoint with state_dict, mean, std")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model", type=str, default="ViT-B-32")
    ap.add_argument("--pretrained", type=str, default="openai")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--neuron", type=int, default=0, help="Neuron index to inspect")
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--outdir", type=str, default="concept_inspect")
    ap.add_argument("--dump_csv", action="store_true", help="Dump per-neuron summaries for all neurons")
    args = ap.parse_args()

    # 1) Load SAE checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sae = build_sae_from_state_dict(ckpt["state_dict"]).to(args.device).eval()
    mean = ckpt["mean"]; std = ckpt["std"]
    in_dim = mean.shape[1]; h_dim = sae.decoder[0].weight.shape[1]
    print(f"[SAE] in_dim={in_dim}, h_dim={h_dim}")

    # 2) CLIP + CIFAR100
    clip_model, preprocess = load_clip(args.model, args.pretrained, args.device)
    ds, dl = cifar100_loader(preprocess, "train", args.batch_size)
    class_names = ds.classes  # CIFAR-100 label names

    # 3) Collect CLIP embeddings; standardize with saved stats
    X = collect_clip_embeddings(clip_model, dl, args.device)        # [N, D]
    assert X.shape[1] == in_dim, f"Embedding dim mismatch: {X.shape[1]} vs {in_dim}"
    Xn = standardize_with_stats(X, mean, std)

    # 4) Codes and atoms
    with torch.no_grad():
        Z, _ = sae(Xn.to(args.device))  # [N, H]
    Z = Z.detach().cpu()
    atoms = sae.decoder[0].weight.data.cpu()  # [D, H]

    # 5) Top-activating images for one neuron + grid
    for j in range(h_dim):
        idxs = topk_indices_for_neuron(Z, j, args.topk)
        grid_path = os.path.join(args.outdir, f"neuron_{j}_top{args.topk}.png")
        save_topk_grid(ds, idxs, grid_path, ncol=min(8, args.topk))
        if j % 100 == 0:  # log every 100 neurons
            print(f"[Neuron {j}] saved top-{args.topk} image grid → {grid_path}")

    # 6) Text alignment against CIFAR-100 names
    S_txt = text_align_atoms(clip_model, class_names, atoms, args.device)  # [100, H]
    top_txt = torch.topk(S_txt[:, j], 5).indices.tolist()
    print(f"[Neuron {j}] top text labels:", [class_names[i] for i in top_txt])

    # 7) Class-prototype alignment (image side)
    # Need labels in the same order as X; CIFAR loader gives that implicitly.
    # Re-iterate once to collect labels:
    ys = []
    for _, yb in dl:
        ys.append(yb)
    y = torch.cat(ys, dim=0)
    S_cls = class_prototype_align(Xn, y, atoms, num_classes=len(class_names))  # [C, H]
    top_cls = torch.topk(S_cls[:, j], 5).indices.tolist()
    print(f"[Neuron {j}] top image classes:", [class_names[i] for i in top_cls])

    # 8) Optional: dump a CSV summary for ALL neurons (top-3 text & classes)
    if args.dump_csv:
        csv_path = os.path.join(args.outdir, "neurons_summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["neuron", "top_text_1", "top_text_2", "top_text_3",
                             "top_class_1", "top_class_2", "top_class_3"])
            for jj in range(h_dim):
                t_idx = torch.topk(S_txt[:, jj], 3).indices.tolist()
                c_idx = torch.topk(S_cls[:, jj], 3).indices.tolist()
                writer.writerow([jj] +
                                [class_names[i] for i in t_idx] +
                                [class_names[i] for i in c_idx])
        print(f"[ALL] CSV dumped → {csv_path}")

if __name__ == "__main__":
    main()
