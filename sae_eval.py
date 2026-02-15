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

from torchvision.datasets import Caltech101, OxfordIIITPet
from torchvision.transforms import ToTensor, Resize, Compose, Lambda

from model import SparseAutoencoder
from integrate_mono import weighted_pairwise_cosine, per_class_monosemanticity, dataset_monosemanticity, topk_neurons_overall


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
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # older torch without weights_only
        return torch.load(path, map_location="cpu")
    
# def safe_load_checkpoint2(path):
#     try:
#         # Load the checkpoint
#         ckpt = torch.load(path, map_location="cpu")
#         state_dict = ckpt["state_dict"]
        
#         # Extract dimensions from the checkpoint
#         h_dims, in_dims = state_dict["W_enc"].shape
#         print('1')
#         # Initialize the SparseAutoencoder model with the correct dimensions
#         model = SparseAutoencoder(in_dims, h_dims)
#         print('2')
#         # Map the keys to match SparseAutoencoder's layer names
#         mapped_state_dict = {
#             "encoder.0.weight": state_dict["W_enc"],
#             "encoder.0.bias": state_dict["b_enc"],
#             "decoder.0.weight": state_dict["W_dec"],
#             "decoder.0.bias": state_dict["b_dec"],
#         }
        
#         # Load the mapped weights
#         model.load_state_dict(mapped_state_dict)
#         model.eval()
#         return model
#     except Exception as e:
#         print(f"Error loading checkpoint: {e}")
#         raise
        
def safe_load_checkpoint2(path):
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt["state_dict"]

    # checkpoint uses math convention:
    # W_enc: [D, H]
    # b_enc: [H]

    D, H = sd["W_enc"].shape

    print("Input dim:", D)
    print("Hidden dim:", H)

    model = SparseAutoencoder(D, H)

    mapped = {
        # transpose weights for PyTorch
        "encoder.0.weight": sd["W_enc"].T,
        "encoder.0.bias": sd["b_enc"],
        "decoder.0.weight": sd["W_dec"].T,
        "decoder.0.bias": sd["b_dec"],
    }

    model.load_state_dict(mapped, strict=True)
    model.eval()
    return model






def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="/DATA/cs22btech11053/Concept_Lora/out.pt",
                    help="Checkpoint with keys: state_dict, mean, std")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model", type=str, default="ViT-B-16")
    ap.add_argument("--dataset", type=str, default="OxfordIIITPet", choices=["cifar100", "mscoco", "cub", "imagenet", "caltech101", "OxfordIIITPet"])
    ap.add_argument("--pretrained", type=str, default="openai")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--neuron", type=int, default=0, help="Neuron index to inspect")
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--outdir", type=str, default="plots/concept_inspect_lora_imagenet")
    ap.add_argument("--dump_csv", action="store_true", help="Dump per-neuron summaries")
    ap.add_argument("--construct_b1", action="store_true",  help="Construct the b1 matrix")
    ap.add_argument("--bool_test", type=bool, default=False, help="Run a quick test with a small subset of data")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load SAE checkpoint
    ckpt = safe_load_checkpoint(args.ckpt)
    # assert "state_dict" in ckpt and "mean" in ckpt and "std" in ckpt, \
    #     "Checkpoint must contain state_dict, mean, std"
    # sae = build_sae_from_state_dict(ckpt["state_dict"]).to(args.device).eval()
    # mean = ckpt["mean"]
    # std = ckpt["std"]
    # in_dim = mean.shape[1]
    # h_dim = sae.decoder[0].weight.shape[1]
    # print(f"[SAE] in_dim={in_dim}, h_dim={h_dim}")

    sae = safe_load_checkpoint2(args.ckpt).to(args.device).eval()
    in_dim = sae.encoder[0].weight.shape[1]
    h_dim = sae.encoder[0].weight.shape[0]
    mean = torch.zeros((1, in_dim))
    std = torch.ones((1, in_dim))
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

    elif args.dataset == "imagenet":
        images_dir = "/DATA/cs22btech11053/Concept_Lora/sae-for-vlm/data/imagenet/"
        annotations_file = "/DATA/cs22btech11053/Concept_Lora/Concept_LoRA/classes.txt"
        ds, dl = get_dataloader("imagenet", images_dir, annotations_file,
                                subset=1.0, transform=preprocess, batch_size=args.batch_size)
        class_names = get_class_names(annotations_file)

    elif args.dataset == "caltech101":
        images_dir = "/DATA/cs22btech11053/Concept_Lora/Concept_LoRA/data/Caltech/"
        transform = Compose([Resize((224, 224)), Lambda(lambda img: img.convert("RGB")), ToTensor()])
        ds = Caltech101(root=images_dir, download=False, transform=transform)
        # train_size = int(0.8 * len(full_dataset))
        # val_size = int(0.2 * len(full_dataset))
        # train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        # test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        class_dir = "/DATA/cs22btech11053/Concept_Lora/Concept_LoRA/data/Caltech/caltech101/101_ObjectCategories"
        class_names = [name for name in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, name))]

    elif args.dataset == "OxfordIIITPet":
        images_dir = "/DATA/cs22btech11053/Concept_Lora/Concept_LoRA/data/OxfordIIITPet"
        transform = Compose([Resize((224, 224)), Lambda(lambda img: img.convert("RGB")), ToTensor()])
        ds = OxfordIIITPet(root=images_dir, split="trainval", target_types="category",
                           transform=transform, download=False)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        class_names = ds.classes

    ys = []
    for _, yb in dl:
        ys.append(yb)
    y = torch.cat(ys, dim=0)

    print("Hi-1")

    # 3) Gather CLIP embeddings and standardize with saved stats
    X = collect_clip_embeddings(clip_model, dl, args.device)  # [N, D]
    assert X.shape[1] == in_dim, f"Embedding dim mismatch: {X.shape[1]} vs {in_dim}"
    Xn = standardize_with_stats(X, mean, std)

    # 4) Codes and atoms
    with torch.no_grad():
        Z, _ = sae(Xn.to(args.device))  # [N, H]
    Z = Z.detach().cpu()
    atoms = sae.decoder[0].weight.data.cpu()  # [D, H]

    num_classes = len(class_names)

    per_class_top10 = per_class_monosemanticity(
        X, Z, y,
        num_classes=num_classes,
        k=10
    )

    print("Hi-2")

    # Store the per-class top-10 monosemanticity scores in a CSV
    csv_string = "per_class_top10_monosemanticity_{}_{}.csv".format(args.dataset, "fine" if args.bool_test else "norm")
    csv_path = os.path.join(args.outdir, csv_string)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "neuron_idx", "monosemanticity_score"])
        for class_id, topk_list in per_class_top10.items():
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            for neuron_idx, score in topk_list:
                writer.writerow([class_id, class_name, neuron_idx, score])
    print(f"Per-class top-10 monosemanticity scores saved to {csv_path}")

    ms_all, dataset_avg = dataset_monosemanticity(X, Z)

    top10_idx, top10_scores = topk_neurons_overall(ms_all, k=10)

    print("Top-10 neurons (global):")
    for i, s in zip(top10_idx.tolist(), top10_scores.tolist()):
        print(f"Neuron {i:4d} | MS = {s:.4f}")

    print("Hi-3")


    mono_scores = weighted_pairwise_cosine(X, Z, pair_batch_size=100)
    print("Mean monosemanticity scores per neuron:", mono_scores.mean().item())
    exit(0)

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
