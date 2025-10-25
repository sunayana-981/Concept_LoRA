# sae_eval.py
# pip install torch torchvision open_clip_torch matplotlib
import argparse, os, csv, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import open_clip

from SAE.dataloader import get_dataloader
from SAE.utils import collect_clip_embeddings


# -----------------------
# SAE loader (rebuild from checkpoint shapes)
# -----------------------
def build_sae_from_state_dict(state_dict):
    W = state_dict["decoder.0.weight"]   # [in_dim, h_dim]
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


# -----------------------
# CLIP + datasets
# -----------------------
def load_clip(model_name="ViT-B-32", pretrained="openai", device="cuda"):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()
    return model, preprocess


def cifar100_loader(preprocess, split="train", batch_size=256):
    ds = datasets.CIFAR100(root="./data", train=(split == "train"),
                           download=True, transform=preprocess)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=4, pin_memory=True)
    return ds, dl


def standardize_with_stats(X, mean, std):
    return (X - mean) / std.clamp_min(1e-6)


# -----------------------
# Concept inspection helpers
# -----------------------
def topk_indices_for_neuron(Z, j, k):
    return torch.topk(Z[:, j], k).indices.tolist()


def save_topk_grid(ds_raw, indices, out_path, ncol=8):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imgs = [ds_raw[i][0] for i in indices]  # tensors already transformed
    grid = vutils.make_grid(imgs, nrow=ncol, normalize=True, scale_each=True)
    plt.figure(figsize=(ncol * 1.5, max(1, (len(indices) + ncol - 1) // ncol) * 1.5))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0))
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()


@torch.no_grad()
def text_align_atoms(clip_model, labels, atoms, device):
    tok = open_clip.get_tokenizer("ViT-B-32")
    prompts = [f"a photo of a {t}" for t in labels]
    txt = tok(prompts).to(device)

    T = clip_model.encode_text(txt).float()     # [L, D]
    T = F.normalize(T, dim=-1)

    atoms = atoms.to(device)                    # [D, H]
    atoms = F.normalize(atoms, dim=0)

    S = T @ atoms                               # [L, H]
    return S


@torch.no_grad()
def class_prototype_align(Xn, y, atoms, num_classes):
    # Xn standardized; L2-normalize before prototypes
    Xn_u = F.normalize(Xn, dim=-1)
    protos = []
    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            # empty class in the subset; use zero proto
            proto = torch.zeros((1, Xn_u.shape[1]), dtype=Xn_u.dtype, device=Xn_u.device)
        else:
            proto = F.normalize(Xn_u[idx].mean(dim=0, keepdim=True), dim=-1)
        protos.append(proto)
    P = torch.cat(protos, dim=0)            # [C, D]
    atoms = F.normalize(atoms, dim=0)       # [D, H]
    return P @ atoms                        # [C, H]


def get_class_names(path_or_dir):
    # COCO JSON
    if str(path_or_dir).lower().endswith(".json"):
        with open(path_or_dir, "r") as f:
            data = json.load(f)
        return [c["name"] for c in data["categories"]]

    # CUB: classes.txt
    if str(path_or_dir).lower().endswith("classes.txt"):
        names = []
        with open(path_or_dir, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                names.append(" ".join(parts[1:]).replace("_", " "))
        return names

    # If directory is passed, try classes.txt inside it
    if os.path.isdir(path_or_dir):
        cand = os.path.join(path_or_dir, "classes.txt")
        if os.path.exists(cand):
            return get_class_names(cand)

    raise ValueError(f"Unsupported annotations file for class names: {path_or_dir}")


def safe_load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # older torch without weights_only
        return torch.load(path, map_location="cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="weights/SAE/base/cifar100/exp_2025:10:13-05:35:21/sae_model_epoch_20.pt",
                    help="Checkpoint with keys: state_dict, mean, std")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model", type=str, default="ViT-B-32")
    ap.add_argument("--dataset", type=str, default="cifar100", choices=["cifar100", "mscoco", "cub"])
    ap.add_argument("--pretrained", type=str, default="openai")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--neuron", type=int, default=0, help="Neuron index to inspect")
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--outdir", type=str, default="plots/concept_inspect_lora_cub")
    ap.add_argument("--dump_csv", action="store_true", help="Dump per-neuron summaries")
    ap.add_argument("--construct_b1", action="store_true",  help="Construct the b1 matrix")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load SAE checkpoint
    ckpt = safe_load_checkpoint(args.ckpt)
    assert "state_dict" in ckpt and "mean" in ckpt and "std" in ckpt, \
        "Checkpoint must contain state_dict, mean, std"
    sae = build_sae_from_state_dict(ckpt["state_dict"]).to(args.device).eval()
    mean = ckpt["mean"]
    std = ckpt["std"]
    in_dim = mean.shape[1]
    h_dim = sae.decoder[0].weight.shape[1]
    print(f"[SAE] in_dim={in_dim}, h_dim={h_dim}")

    # 2) CLIP + dataset
    clip_model, preprocess = load_clip(args.model, args.pretrained, args.device)

    if args.dataset == "cifar100":
        ds, dl = cifar100_loader(preprocess, "train", args.batch_size)
        class_names = ds.classes

    elif args.dataset == "mscoco":
        images_dir = "/data1/ai22resch11001/projects/data/mscoco/train2017"
        annotations_file = "/data1/ai22resch11001/projects/data/mscoco/annotations/instances_train2017.json"
        ds, dl = get_dataloader("mscoco", images_dir, annotations_file,
                                subset=1.0, transform=preprocess, batch_size=args.batch_size)
        class_names = get_class_names(annotations_file)

    elif args.dataset == "cub":
        images_dir = "./datasets/cub2002011"  # root with images/ and PKLs + classes.txt
        annotations_file = os.path.join(images_dir, "classes.txt")
        ds, dl = get_dataloader("cub", images_dir, annotations_file,
                                subset=1.0, transform=preprocess, batch_size=args.batch_size)
        class_names = get_class_names(annotations_file)

    # 3) Gather CLIP embeddings and standardize with saved stats
    X = collect_clip_embeddings(clip_model, dl, args.device)  # [N, D]
    assert X.shape[1] == in_dim, f"Embedding dim mismatch: {X.shape[1]} vs {in_dim}"
    Xn = standardize_with_stats(X, mean, std)

    # 4) Codes and atoms
    with torch.no_grad():
        Z, _ = sae(Xn.to(args.device))  # [N, H]
    Z = Z.detach().cpu()
    atoms = sae.decoder[0].weight.data.cpu()  # [D, H]

    # 5) Per-neuron top-k image grids (+ optional b1)
    b1_cols = []
    for j in range(h_dim):
        idxs = topk_indices_for_neuron(Z, j, args.topk)
        grid_path = os.path.join(args.outdir, f"neuron_{j}_top{args.topk}.png")
        save_topk_grid(ds, idxs, grid_path, ncol=min(8, args.topk))
        if args.construct_b1:
            b1_cols.append(X[idxs].mean(dim=0))  # mean image-embedding for neuron j
        if j % 100 == 0:
            print(f"[Neuron {j}] saved top-{args.topk} grid → {grid_path}")

    if args.construct_b1:
        b1 = torch.stack(b1_cols, dim=1)   # [D, H]
        torch.save(b1, os.path.join(args.outdir, f"b1_matrix_{args.dataset}.pt"))
        print("b1 shape:", b1.shape)

    # 6) Text-label alignment
    S_txt = text_align_atoms(clip_model, class_names, atoms, args.device)  # [C, H]
    idx = int(args.neuron)
    top_txt = torch.topk(S_txt[:, idx], 5).indices.tolist()
    print(f"[Neuron {idx}] top text labels:", [class_names[i] for i in top_txt])

    # 7) Class-prototype alignment (image-side) — only for datasets with labels
    if args.dataset in {"cifar100", "cub"}:
        ys = []
        for _, yb in dl:
            ys.append(yb)
        y = torch.cat(ys, dim=0)
        S_cls = class_prototype_align(Xn, y, atoms, num_classes=len(class_names))  # [C, H]
        top_cls = torch.topk(S_cls[:, idx], 5).indices.tolist()
        print(f"[Neuron {idx}] top image classes:", [class_names[i] for i in top_cls])
    else:
        print("[Info] Skipping class-prototype alignment for MSCOCO (no labels in loader).")

    # 8) Optional: dump CSV for all neurons
    if args.dump_csv:
        csv_path = os.path.join(args.outdir, "neurons_summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["neuron", "top_text_1", "top_text_2", "top_text_3",
                             "top_class_1", "top_class_2", "top_class_3"])
            # For datasets without labels, we only fill text columns.
            for jj in range(h_dim):
                t_idx = torch.topk(S_txt[:, jj], 3).indices.tolist()
                if args.dataset in {"cifar100", "cub"}:
                    c_idx = torch.topk(S_cls[:, jj], 3).indices.tolist()
                    row = [jj] + [class_names[i] for i in t_idx] + [class_names[i] for i in c_idx]
                else:
                    row = [jj] + [class_names[i] for i in t_idx] + ["", "", ""]
                writer.writerow(row)
        print(f"[ALL] CSV dumped → {csv_path}")


if __name__ == "__main__":
    main()
