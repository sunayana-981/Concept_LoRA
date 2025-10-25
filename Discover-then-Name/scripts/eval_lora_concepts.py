# scripts/eval_lora_concepts.py
import os, os.path as osp, argparse, re, torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, Subset
from tqdm import tqdm

from dncbm import arg_parser
from dncbm.utils import common_init
from sparse_autoencoder import SparseAutoencoder

# LoRA bits from training
from scripts.lora_finetune_sae import LoRALinear, LoRABroadcastParametrization
import torch.nn.utils.parametrize as parametrize

# --------- Optional CLIP feature extraction (CUB) ----------
import clip
from torchvision import transforms
from torchvision.datasets import ImageFolder

def get_cub_dataloader(data_root, split='train', batch_size=256, image_size=224, num_workers=4):
    image_dir = os.path.join(data_root, "images")
    image_txt = os.path.join(data_root, "images.txt")
    split_txt = os.path.join(data_root, "train_test_split.txt")

    image_df = pd.read_csv(image_txt, sep=' ', header=None, names=['img_id', 'img_path'])
    split_df = pd.read_csv(split_txt, sep=' ', header=None, names=['img_id', 'is_train'])

    is_train = int(split == 'train')
    split_ids = split_df[split_df['is_train'] == is_train]['img_id'].values
    split_img_paths = image_df[image_df['img_id'].isin(split_ids)]['img_path'].tolist()

    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    full_dataset = ImageFolder(image_dir, transform=tfm)

    img_path_to_idx = {os.path.relpath(path, image_dir): idx for idx, (path, _) in enumerate(full_dataset.samples)}

    selected_indices, missing = [], []
    for rel_path in split_img_paths:
        norm_path = os.path.normpath(rel_path.strip())
        if norm_path in img_path_to_idx:
            selected_indices.append(img_path_to_idx[norm_path])
        else:
            missing.append(norm_path)

    if missing:
        raise KeyError(f"{len(missing)} metadata paths not found in ImageFolder. First: {missing[0]}")

    subset = Subset(full_dataset, selected_indices)
    return DataLoader(
        subset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers,
        pin_memory=True, persistent_workers=(num_workers > 0)
    )

@torch.no_grad()
def extract_clip_features(clip_model, loader, device, l2_normalize: bool = False) -> torch.Tensor:
    feats = []
    for x, _ in tqdm(loader, desc="Extracting CLIP feats"):
        x = x.to(device, non_blocking=True)
        f = clip_model.encode_image(x)
        if l2_normalize:
            f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        feats.append(f.detach().cpu())
    return torch.cat(feats, dim=0)

# ---------- Core utils ----------
def infer_dims_from_ckpt(state):
    enc_w = state["encoder._weight"]  # shape [*, latent, in]
    return int(enc_w.shape[-1]), int(enc_w.shape[-2])

def _owner_and_local(root, full_path):
    parts = full_path.split(".")
    owner = root
    for p in parts[:-1]:
        owner = getattr(owner, p)
    return owner, parts[-1]

def _ensure_param_lora(owner, local, A, B, scale):
    """
    Ensure a LoRA parametrization exists on owner.<local> and load A/B/scale.
    Moves A/B & the parametrization module to the SAME device/dtype as the base param.
    Replaces parameters if shapes mismatch (no in-place resizing of nn.Parameter).
    """
    import torch.nn as nn

    try:
        base_param = owner.get_parameter(local)
    except AttributeError:
        base_param = getattr(owner, local)
    dev = base_param.device
    dty = base_param.dtype

    A = A.to(device=dev, dtype=dty, copy=False)
    B = B.to(device=dev, dtype=dty, copy=False)

    r = A.shape[1]
    alpha = float(scale) * r
    in_f, out_f = A.shape[0], B.shape[0]

    def _load_into_pmod(pmod):
        pmod.to(dev)
        with torch.no_grad():
            if pmod.A.shape != A.shape:
                pmod.A = nn.Parameter(A.clone(), requires_grad=True)
            else:
                pmod.A.data.copy_(A)
            if pmod.B.shape != B.shape:
                pmod.B = nn.Parameter(B.clone(), requires_grad=True)
            else:
                pmod.B.data.copy_(B)
        pmod.scale = float(scale)

    if hasattr(owner, "parametrizations") and local in owner.parametrizations:
        for pmod in owner.parametrizations[local]:
            if isinstance(pmod, LoRABroadcastParametrization):
                _load_into_pmod(pmod)
                return True

    p = LoRABroadcastParametrization(in_f, out_f, r=r, alpha=alpha).to(dev)
    with torch.no_grad():
        if p.A.shape != A.shape:
            p.A = nn.Parameter(A.clone(), requires_grad=True)
        else:
            p.A.data.copy_(A)
        if p.B.shape != B.shape:
            p.B = nn.Parameter(B.clone(), requires_grad=True)
        else:
            p.B.data.copy_(B)
    parametrize.register_parametrization(owner, local, p)

    for pmod in owner.parametrizations[local]:
        if isinstance(pmod, LoRABroadcastParametrization):
            pmod.scale = float(scale)
            return True
    return False

def attach_lora_from_file(sae, lora_path: str):
    state = torch.load(lora_path, map_location="cpu")
    attached_mod, attached_param = 0, 0

    # 1) New format dict entries: "mod:<module_path>" / "param:<full_param_name>"
    for k, v in list(state.items()):
        if not isinstance(v, dict):
            continue
        if k.startswith("mod:"):
            mod_path = k[4:]
            parent = sae
            parts = mod_path.split(".") if mod_path else []
            for p in parts[:-1]:
                parent = getattr(parent, p)
            child = parts[-1] if parts else None
            m = getattr(parent, child) if child else parent
            if isinstance(m, torch.nn.Linear):
                wrapped = LoRALinear(m, r=v["A"].shape[1], alpha=v["scale"] * v["A"].shape[1], dropout=0.0)
                setattr(parent, child, wrapped)
                m = wrapped
            if isinstance(m, LoRALinear):
                with torch.no_grad():
                    m.A.copy_(v["A"]); m.B.copy_(v["B"]); m.scale = float(v["scale"])
                attached_mod += 1
        elif k.startswith("param:"):
            full_param = k[6:]
            owner, local = _owner_and_local(sae, full_param)
            if local.endswith("_bias") or local == "bias":
                print(f"Skipping bias adapter for {full_param} (not supported).")
                continue
            ok = _ensure_param_lora(owner, local, v["A"], v["B"], v["scale"])
            attached_param += int(ok)

    # 2) Legacy flat format: "<prefix>.A/.B/.scale" (e.g., "ParametrizedLinearEncoder.<id>/_weight.A")
    groups = {}
    for k, v in state.items():
        if isinstance(v, dict):
            continue
        if k.endswith(".A") or k.endswith(".B") or k.endswith(".scale"):
            prefix = k.rsplit(".", 1)[0]
            groups.setdefault(prefix, {})[k.split(".")[-1]] = v

    model_param_names = [name for name, _ in sae.named_parameters(recurse=True)]

    def maybe_attach_by_param_name(param_name, pack):
        if {"A","B","scale"} - set(pack.keys()):
            return 0
        if param_name.endswith("_bias") or param_name.endswith(".bias"):
            print(f"Skipping bias adapter for {param_name} (not supported).")
            return 0
        owner, local = _owner_and_local(sae, param_name)
        return int(_ensure_param_lora(owner, local, pack["A"], pack["B"], float(pack["scale"])))

    leftovers = []

    # Pass 1: exact matches ("encoder._weight", "decoder._weight")
    for prefix, pack in groups.items():
        if prefix in model_param_names:
            attached_param += maybe_attach_by_param_name(prefix, pack)
        else:
            leftovers.append((prefix, pack))

    # Pass 2a: strip "ClassName(id)/..."  -> param name
    pat_paren = re.compile(r"^[^/]+?\(\d+\)/(.+)$")
    still = []
    for prefix, pack in leftovers:
        m = pat_paren.match(prefix)
        if m and m.group(1) in model_param_names:
            attached_param += maybe_attach_by_param_name(m.group(1), pack)
        else:
            still.append((prefix, pack))

    # Pass 2b: strip "ClassName.id/..."  -> param name  (legacy variant)
    pat_dotid = re.compile(r"^[^/]+?\.\d+/(.+)$")
    still2 = []
    for prefix, pack in still:
        m = pat_dotid.match(prefix)
        mapped = m.group(1) if m else None
        if mapped:
            # Heuristic: map to encoder/decoder weight by original class hint
            if "_weight" in mapped and "Encoder" in prefix and "encoder._weight" in model_param_names:
                mapped = "encoder._weight"
            elif "_weight" in mapped and "Decoder" in prefix and "decoder._weight" in model_param_names:
                mapped = "decoder._weight"
            if mapped in model_param_names:
                attached_param += maybe_attach_by_param_name(mapped, pack)
            else:
                still2.append((prefix, pack))
        else:
            still2.append((prefix, pack))

    # Pass 3: suffix fallback (unique endswith)
    for prefix, pack in still2:
        suffix = prefix.split("/")[-1]  # e.g., "_weight" or "encoder._weight"
        if suffix in model_param_names:
            attached_param += maybe_attach_by_param_name(suffix, pack)
            continue
        candidates = [n for n in model_param_names if n.endswith(suffix)]
        if len(candidates) == 1:
            attached_param += maybe_attach_by_param_name(candidates[0], pack)
        elif len(candidates) > 1:
            prio = [n for n in candidates if n in ("encoder._weight", "decoder._weight")]
            if len(prio) == 1:
                attached_param += maybe_attach_by_param_name(prio[0], pack)
            else:
                print(f"Ambiguous legacy key '{prefix}' -> {candidates}; skipping.")

    print(f"Attached LoRA: {attached_mod} module-level, {attached_param} param-level")
    for p in sae.parameters(): p.requires_grad = False
    if attached_mod + attached_param == 0:
        print("WARNING: No LoRA adapters attached. Ensure the adapter file matches this SAE or re-save with the new saver.")
    return sae

# ---------- encode & summarize ----------
@torch.no_grad()
def encode_latents(sae, feats, device, bs=4096):
    ds = TensorDataset(feats); loader = DataLoader(ds, batch_size=bs, shuffle=False)
    Z = None
    for (x,) in tqdm(loader, desc="Encoding"):
        x = x.to(device, non_blocking=True)
        z, recon = sae(x)
        if z.ndim == 3 and z.size(1) == 1: z = z.squeeze(1)
        if recon.ndim == 3 and recon.size(1) == 1: recon = recon.squeeze(1)
        z = z.detach().cpu()
        Z = z if Z is None else torch.vstack([Z, z])
    return Z

def summarize(Z: torch.Tensor, tau=1e-3):
    absZ = Z.abs()
    nz = (absZ > tau).float().mean(dim=0)
    mean_abs = absZ.mean(dim=0)
    share = mean_abs / (mean_abs.sum() + 1e-12)
    df = pd.DataFrame({
        "concept_id": torch.arange(Z.shape[1]).numpy(),
        "activation_rate": nz.numpy(),
        "mean_abs": mean_abs.numpy(),
        "energy_share": share.numpy(),
    }).sort_values("energy_share", ascending=False, ignore_index=True)
    return df

# ---------- Visualization helpers (opt-in) ----------
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def plot_topk_energy(df, k=20, out_dir="."):
    top = df.nlargest(k, "energy_share").reset_index(drop=True)
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(top)), top["energy_share"].values)
    plt.xticks(range(len(top)), top["concept_id"].astype(int).tolist(), rotation=45, ha="right")
    plt.ylabel("Energy share"); plt.title(f"Top-{k} concepts by energy share")
    plt.tight_layout()
    p = osp.join(out_dir, f"concept_energy_top{k}.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"Saved {p}")

def plot_cumulative_energy(df, out_dir="."):
    s = df.sort_values("energy_share", ascending=False)["energy_share"].cumsum().values
    x = range(1, len(s)+1)
    plt.figure(figsize=(8,4))
    plt.plot(x, s)
    plt.axhline(0.8, ls="--"); plt.axhline(0.9, ls="--")
    plt.xlabel("# concepts"); plt.ylabel("Cumulative energy share")
    plt.title("Cumulative energy capture")
    plt.tight_layout()
    p = osp.join(out_dir, "cumulative_energy.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"Saved {p}")

def plot_activation_rate_hist(df, out_dir="."):
    plt.figure(figsize=(6,4))
    plt.hist(df["activation_rate"].values, bins=50)
    plt.xlabel("Activation rate (> τ)"); plt.ylabel("Count of concepts")
    plt.title("Concept sparsity (activation rate)")
    plt.tight_layout()
    p = osp.join(out_dir, "activation_rate_hist.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"Saved {p}")

def _collect_subset_paths(cub_root, split):
    image_dir = os.path.join(cub_root, "images")
    image_txt = os.path.join(cub_root, "images.txt")
    split_txt = os.path.join(cub_root, "train_test_split.txt")
    image_df = pd.read_csv(image_txt, sep=' ', header=None, names=['img_id', 'img_path'])
    split_df = pd.read_csv(split_txt, sep=' ', header=None, names=['img_id', 'is_train'])
    is_train = int(split == 'train')
    split_ids = split_df[split_df['is_train'] == is_train]['img_id'].values
    split_img_paths = image_df[image_df['img_id'].isin(split_ids)]['img_path'].tolist()
    return [os.path.join(image_dir, os.path.normpath(p)) for p in split_img_paths]

def make_contact_sheet(img_paths, grid=(4,4), thumb_size=224, out_path="contact_sheet.png"):
    rows, cols = grid
    W = cols * thumb_size; H = rows * thumb_size
    canvas = Image.new("RGB", (W, H), (255,255,255))
    for i, p in enumerate(img_paths[:rows*cols]):
        try:
            im = Image.open(p).convert("RGB")
            im = im.resize((thumb_size, thumb_size))
            r, c = divmod(i, cols)
            canvas.paste(im, (c*thumb_size, r*thumb_size))
        except Exception:
            pass
    canvas.save(out_path)
    print(f"Saved {out_path}")

def save_topk_images_for_concept(Z, concept_id, paths_in_order, k=16, out_dir="."):
    zc = Z[:, concept_id].numpy()
    top_idx = np.argsort(-zc)[:k]
    sel_paths = [paths_in_order[i] for i in top_idx]
    os.makedirs(out_dir, exist_ok=True)
    g = int(np.sqrt(k)); g = max(1, g)
    make_contact_sheet(sel_paths, grid=(g, g), thumb_size=224,
                       out_path=osp.join(out_dir, f"concept_{concept_id}_top{k}.png"))

def plot_class_concept_heatmap(Z, y, class_names=None, topk_concepts=50, out_dir="."):
    Zabs = Z.abs().numpy()
    C = int(y.max()) + 1
    by_class = []
    for c in range(C):
        mask = (y.numpy() == c)
        if mask.sum() == 0:
            by_class.append(np.zeros((1, Zabs.shape[1]), dtype=Zabs.dtype))
        else:
            by_class.append(Zabs[mask].mean(axis=0, keepdims=True))
    M = np.vstack(by_class)  # C x K
    keep = np.argsort(-M.mean(axis=0))[:topk_concepts]
    M = M[:, keep]
    plt.figure(figsize=(min(20, topk_concepts*0.3+4), 8))
    plt.imshow(M, aspect="auto")
    plt.colorbar(label="mean |z|")
    plt.xlabel(f"Top-{topk_concepts} concepts"); plt.ylabel("Class")
    if class_names:
        plt.yticks(range(C), class_names)
    plt.tight_layout()
    p = osp.join(out_dir, "class_concept_heatmap.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"Saved {p}")

# ---------- main ----------
def main():
    parser = arg_parser.get_common_parser()
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--sae_base_ckpt", type=str, required=True)
    parser.add_argument("--features_path", type=str, default="", help="Direct path to (N,D) features .pt")
    parser.add_argument("--mode", choices=["from_tensors", "from_cub"], default="from_tensors",
                        help="If from_cub, extract CLIP features on-the-fly")
    parser.add_argument("--cub_root", type=str, default="/home/sunayana/Documents/Concept_LoRA/datasets/cub2002011")
    parser.add_argument("--cub_split", choices=["train","test"], default="train")
    parser.add_argument("--clip_model", type=str, default="ViT-B/16")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--norm_features", action="store_true", help="L2-normalize CLIP features when extracting.")

    parser.add_argument("--batch_size_eval", type=int, default=4096)
    parser.add_argument("--nz_thresh", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, default="")

    # ---- NEW: visualization flags (all optional) ----
    parser.add_argument("--viz", action="store_true", help="Save summary plots (top-k energy, cumulative, activation hist).")
    parser.add_argument("--viz_topk", type=int, default=20, help="Top-k concepts for energy bar plot and contact sheets.")
    parser.add_argument("--viz_contact_sheets", action="store_true", help="Save contact sheets for top-k concepts (CUB only).")
    parser.add_argument("--contact_k", type=int, default=16, help="Images per concept contact sheet (must be a square like 9,16,25...).")
    parser.add_argument("--viz_heatmap", action="store_true", help="Plot class vs concept heatmap (requires --labels_path).")
    parser.add_argument("--labels_path", type=str, default="", help="Path to labels tensor (N,) aligned with features.")
    parser.add_argument("--class_names_csv", type=str, default="", help="Optional CSV with one class name per line.")

    args = parser.parse_args()
    common_init(args)

    # -- Build SAE from checkpoint metadata
    base_state = torch.load(args.sae_base_ckpt, map_location="cpu")
    in_dim, latent_dim = infer_dims_from_ckpt(base_state)
    n_components = len(args.hook_points)
    print(f"Building SAE: in_dim={in_dim}, latent_dim={latent_dim}, n_components={n_components}")
    sae = SparseAutoencoder(n_input_features=in_dim, n_learned_features=latent_dim, n_components=n_components).to(args.device)
    print(f"Loading base SAE from: {args.sae_base_ckpt}")
    sae.load_state_dict(base_state, strict=True)

    # -- Attach LoRA adapters
    sae = attach_lora_from_file(sae, args.lora_path)
    sae.eval()

    # -- Features
    if args.mode == "from_tensors":
        if not args.features_path:
            raise ValueError("--features_path is required when --mode=from_tensors")
        feats = torch.load(args.features_path, map_location="cpu")
        if isinstance(feats, dict) and "features" in feats:
            feats = feats["features"]
        if feats.dim() != 2:
            raise ValueError(f"Expected (N,D) features, got {tuple(feats.shape)}")
    else:
        print(f"Extracting CLIP features from CUB ({args.cub_split})...")
        clip_model, _ = clip.load(args.clip_model, device=args.device)
        clip_model.eval()
        loader = get_cub_dataloader(args.cub_root, split=args.cub_split,
                                    batch_size=256, image_size=args.image_size, num_workers=args.num_workers)
        feats = extract_clip_features(clip_model, loader, device=args.device, l2_normalize=args.norm_features)

    if feats.size(1) != in_dim:
        raise ValueError(f"Feature dim {feats.size(1)} != SAE input dim {in_dim}. Make sure features/CLIP model match the SAE.")

    # -- Encode & summarize
    Z = encode_latents(sae, feats, args.device, bs=args.batch_size_eval)
    out_root = args.out_dir if args.out_dir else osp.join(args.probe_cs_save_dir, args.probe_split)
    os.makedirs(out_root, exist_ok=True)
    torch.save(Z, osp.join(out_root, "all_concepts_lora.pth"))
    df = summarize(Z, tau=args.nz_thresh)
    df.to_csv(osp.join(out_root, "concepts_summary_lora.csv"), index=False)

    print("\nTop-10 by energy_share:")
    print(df.loc[:9, ["concept_id", "energy_share", "activation_rate", "mean_abs"]])

    # ---- Visualizations (opt-in) ----
    if args.viz:
        plot_topk_energy(df, k=args.viz_topk, out_dir=out_root)
        plot_cumulative_energy(df, out_dir=out_root)
        plot_activation_rate_hist(df, out_dir=out_root)

    if args.viz_contact_sheets and args.mode == "from_cub":
        cub_paths = _collect_subset_paths(args.cub_root, args.cub_split)
        top_concepts = df.nlargest(args.viz_topk, "energy_share")["concept_id"].astype(int).tolist()
        viz_dir = osp.join(out_root, "concept_examples")
        os.makedirs(viz_dir, exist_ok=True)
        k = args.contact_k
        # force square grid (closest square ≤ k)
        g = int(np.sqrt(k)); g = max(1, g); k_sq = g*g
        for cid in top_concepts:
            save_topk_images_for_concept(Z, cid, cub_paths, k=k_sq, out_dir=viz_dir)

    if args.viz_heatmap and args.labels_path:
        y = torch.load(args.labels_path, map_location="cpu")
        if isinstance(y, dict) and "labels" in y: y = y["labels"]
        if y.ndim != 1 or y.shape[0] != Z.shape[0]:
            raise ValueError(f"labels shape {tuple(y.shape)} must be (N,) matching Z N={Z.shape[0]}")
        class_names = None
        if args.class_names_csv and osp.isfile(args.class_names_csv):
            class_names = [r.strip() for r in open(args.class_names_csv, "r", encoding="utf-8").read().splitlines() if r.strip()]
        plot_class_concept_heatmap(Z, y, class_names=class_names, topk_concepts=min(args.viz_topk, Z.shape[1]), out_dir=out_root)

if __name__ == "__main__":
    main()
