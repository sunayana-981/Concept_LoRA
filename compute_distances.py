"""
Compute FID, MMD², and Bhattacharyya distance between each dataset and ImageNet.

Two backbones:
  • InceptionV3 pool3  (2048-dim, unnormalized) — standard for FID
  • CLIP ViT-L-14/openai (768-dim, L2-normalized) — semantic distances

Reference: ImageNet-1k val (50 000 images, all 1000 classes) from
  ~/.cache/huggingface/hub/datasets--evanarlian--imagenet_1k_resized_256

Features cached to CACHE_DIR; re-runs skip extraction.
"""

import os, glob, io, warnings, csv, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import scipy.linalg
import scipy.stats
from tqdm import tqdm
import clip                   # openai/CLIP — uses ~/.cache/clip/ViT-L-14.pt

warnings.filterwarnings("ignore")

# ── Paths & config ─────────────────────────────────────────────────────────────
IMAGENET_SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/"
    "datasets--evanarlian--imagenet_1k_resized_256/snapshots/"
    "8de107e08d1f61884d5a4ec2dea8740667264a60/data"
)
DATA_ROOT   = "/home/sunayana/Documents/Concept_LoRA/data"
CACHE_DIR   = "/home/sunayana/Documents/Concept_LoRA/feature_cache"
RESULTS_DIR = "/home/sunayana/Documents/Concept_LoRA"

BATCH_SIZE  = 256
MMD_N       = 2000    # subsample size for O(n²) MMD kernel
BASELINE_N  = 2000    # each half for split-half baseline
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
PCA_CUMVAR  = 0.99    # keep enough PCA dims to explain this fraction of variance

os.makedirs(CACHE_DIR, exist_ok=True)

# ── Datasets ───────────────────────────────────────────────────────────────────
DATASETS = {
    "eurosat":       os.path.join(DATA_ROOT, "eurosat/2750"),
    "dtd":           os.path.join(DATA_ROOT, "dtd/images"),
    "imagenet-r":    os.path.join(DATA_ROOT, "imagenet-r"),
    "imagenetv2":    os.path.join(DATA_ROOT, "imagenetv2/imagenetv2-matched-frequency-format-val"),
    "caltech-101":   os.path.join(DATA_ROOT, "caltech-101"),
    "flowers102":    os.path.join(DATA_ROOT, "flowers102_imagefolder"),
    "imagenet-a":    os.path.join(DATA_ROOT, "imagenet-a"),
    "imagenet-o":    os.path.join(DATA_ROOT, "imagenet-o"),
    "sketch":        os.path.join(DATA_ROOT, "sketch"),
    "pathmnist":     os.path.join(DATA_ROOT, "pathmnist_imagefolder"),
    "cub200":        os.path.join(DATA_ROOT, "cub2002011"),
    "UCF101":        os.path.join(DATA_ROOT, "UCF101"),
}

PARQUET_DATASETS = {
    "stanford_cars": sorted(glob.glob(os.path.join(DATA_ROOT, "stanford_cars/data/train*.parquet"))),
    "food101":       sorted(glob.glob(os.path.join(DATA_ROOT, "food101/data/train*.parquet"))),
}

PACS_DOMAINS = {
    "pacs/art":     os.path.join(DATA_ROOT, "pacs_imagefolder/no_art_painting"),
    "pacs/cartoon": os.path.join(DATA_ROOT, "pacs_imagefolder/no_cartoon"),
    "pacs/photo":   os.path.join(DATA_ROOT, "pacs_imagefolder/no_photo"),
    "pacs/sketch":  os.path.join(DATA_ROOT, "pacs_imagefolder/no_sketch"),
}

OH_DOMAINS = {
    "oh/Art":      os.path.join(DATA_ROOT, "OfficeHomeDataset_10072016/Art"),
    "oh/Clipart":  os.path.join(DATA_ROOT, "OfficeHomeDataset_10072016/Clipart"),
    "oh/Product":  os.path.join(DATA_ROOT, "OfficeHomeDataset_10072016/Product"),
    "oh/Real":     os.path.join(DATA_ROOT, "OfficeHomeDataset_10072016/Real World"),
}

HF_DATASETS = {
    "oxford_pets": ("pcuenq/oxford-pets", "train"),
}

ALL_DATASETS = {**DATASETS, **PACS_DOMAINS, **OH_DOMAINS}
ALL_PARQUET_DATASETS = {**PARQUET_DATASETS}

# ── Feature extractors ─────────────────────────────────────────────────────────
class InceptionFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        inc = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        inc.eval()
        self.net = nn.Sequential(
            inc.Conv2d_1a_3x3, inc.Conv2d_2a_3x3, inc.Conv2d_2b_3x3,
            nn.MaxPool2d(3, 2),
            inc.Conv2d_3b_1x1, inc.Conv2d_4a_3x3,
            nn.MaxPool2d(3, 2),
            inc.Mixed_5b, inc.Mixed_5c, inc.Mixed_5d,
            inc.Mixed_6a, inc.Mixed_6b, inc.Mixed_6c, inc.Mixed_6d, inc.Mixed_6e,
            inc.Mixed_7a, inc.Mixed_7b, inc.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1).squeeze(-1)


class CLIPFeatures(nn.Module):
    def __init__(self, model_name="ViT-L/14"):
        super().__init__()
        self.model, self.preprocess = clip.load(model_name, device="cpu")
        self.model.eval()

    def forward(self, x):
        feats = self.model.encode_image(x).float()
        return F.normalize(feats, dim=-1)   # L2-normalised: unit hypersphere


INCEPTION_TRANSFORM = transforms.Compose([
    transforms.Resize(342),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Image dataset helpers ──────────────────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FolderDataset(Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.paths = [
            os.path.join(dp, f)
            for dp, _, fnames in os.walk(root)
            for f in fnames
            if os.path.splitext(f.lower())[1] in IMAGE_EXTS
        ]

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (299, 299))
        return self.transform(img)


class ParquetDataset(Dataset):
    """HuggingFace parquet shards with an 'image' column of dict{'bytes':...}."""
    def __init__(self, parquet_files, transform):
        import pandas as pd
        self.transform = transform
        df = pd.concat([pd.read_parquet(f, columns=["image"]) for f in parquet_files],
                       ignore_index=True)
        self.images = df["image"].tolist()

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        raw = self.images[idx]
        try:
            byt = raw["bytes"] if isinstance(raw, dict) else raw
            img = Image.open(io.BytesIO(byt)).convert("RGB")
        except Exception:
            img = Image.new("RGB", (299, 299))
        return self.transform(img)


class HFDataset(Dataset):
    """HuggingFace datasets Hub dataset with a PIL 'image' column."""
    def __init__(self, hf_path, split, transform):
        from datasets import load_dataset
        self.transform = transform
        self.ds = load_dataset(hf_path, split=split, trust_remote_code=True)

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        try:
            img = self.ds[idx]["image"].convert("RGB")
        except Exception:
            img = Image.new("RGB", (299, 299))
        return self.transform(img)


# ── Feature extraction & caching ──────────────────────────────────────────────
def _run_extraction(model, dataset, desc):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4,
                        pin_memory=True, shuffle=False)
    feats = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  {desc}", leave=False):
            feats.append(model(batch.to(DEVICE)).cpu().float().numpy())
    return np.concatenate(feats, axis=0)


def get_features(model, backbone, name, dataset):
    path = os.path.join(CACHE_DIR, f"{backbone}_{name.replace('/', '_')}.npy")
    if os.path.exists(path):
        arr = np.load(path)
        print(f"  [{backbone}] {name}: loaded {arr.shape[0]} from cache")
        return arr
    arr = _run_extraction(model, dataset, f"[{backbone}] {name}")
    np.save(path, arr)
    print(f"  [{backbone}] {name}: extracted {arr.shape[0]} → cached")
    return arr


# ── Distance metrics ───────────────────────────────────────────────────────────
def fit_gaussian(feats, eps=1e-6):
    mu = feats.mean(0)
    c  = feats - mu
    cov = (c.T @ c) / (len(feats) - 1) + eps * np.eye(feats.shape[1])
    return mu, cov


def compute_fid(mu1, s1, mu2, s2):
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(s1 @ s2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def compute_mmd(X, Y):
    """Unbiased MMD² with RBF kernel; bandwidth = median heuristic on cross-distances."""
    X, Y = X.astype(np.float32), Y.astype(np.float32)
    n, m = len(X), len(Y)

    def sq_dist(A, B):
        return ((A**2).sum(1, keepdims=True) + (B**2).sum(1, keepdims=True).T
                - 2 * A @ B.T)

    Dxy  = sq_dist(X, Y)
    pos  = Dxy[Dxy > 0]
    sigma = float(np.sqrt(np.median(pos) / 2)) if len(pos) else 1.0

    Kxx = np.exp(-sq_dist(X, X) / (2 * sigma**2))
    Kyy = np.exp(-sq_dist(Y, Y) / (2 * sigma**2))
    Kxy = np.exp(-Dxy         / (2 * sigma**2))
    np.fill_diagonal(Kxx, 0); np.fill_diagonal(Kyy, 0)

    return float(Kxx.sum()/(n*(n-1)) + Kyy.sum()/(m*(m-1)) - 2*Kxy.mean())


def pca_basis(feats, cumvar_threshold=PCA_CUMVAR):
    """Return (V, k) where V[:, :k] are the top-k eigenvectors of feats' covariance."""
    mu = feats.mean(0)
    c  = feats - mu
    cov = (c.T @ c) / (len(feats) - 1)
    vals, vecs = np.linalg.eigh(cov)          # ascending order
    vals, vecs = vals[::-1], vecs[:, ::-1]    # descending
    cumvar = vals.cumsum() / vals.sum()
    k = int(np.searchsorted(cumvar, cumvar_threshold)) + 1
    k = max(k, 16)
    return vecs[:, :k].astype(np.float32), k, float(cumvar[k-1]*100)


def compute_bhattacharyya(ref_feats, test_feats, V):
    """
    Bhattacharyya distance in the PCA subspace defined by V (columns = eigenvecs).
    D_B = (1/8) Δμᵀ Σ̄⁻¹ Δμ  +  (1/2) ln(|Σ̄| / sqrt(|Σ₁||Σ₂|))
    """
    # Project
    ref_p  = ref_feats  @ V    # (N_ref, k)
    test_p = test_feats @ V    # (N_test, k)

    mu1, s1 = fit_gaussian(ref_p,  eps=1e-6)
    mu2, s2 = fit_gaussian(test_p, eps=1e-6)
    s_bar   = (s1 + s2) / 2

    delta   = mu1 - mu2
    s_bar_inv = np.linalg.pinv(s_bar)
    term1   = 0.125 * float(delta @ s_bar_inv @ delta)

    _, ld1    = np.linalg.slogdet(s1)
    _, ld2    = np.linalg.slogdet(s2)
    _, ld_bar = np.linalg.slogdet(s_bar)
    term2   = 0.5 * (ld_bar - 0.5 * (ld1 + ld2))

    return float(term1 + term2)


def subsample(feats, n, seed=42):
    rng = np.random.default_rng(seed)
    if len(feats) <= n:
        return feats
    return feats[rng.choice(len(feats), n, replace=False)]


# ── Per-backbone analysis ──────────────────────────────────────────────────────
def run_backbone(backbone_name, model, transform, ref_parquet_files):
    print(f"\n{'='*70}")
    print(f"  Backbone: {backbone_name}")
    print(f"{'='*70}\n")

    # ── Reference features ────────────────────────────────────────────────────
    print("Loading ImageNet val reference…")
    ref_ds    = ParquetDataset(ref_parquet_files, transform)
    ref_feats = get_features(model, backbone_name, "imagenet_val", ref_ds)
    print(f"  Reference: {ref_feats.shape[0]} images, {ref_feats.shape[1]}-dim\n")

    # PCA basis from full reference
    V, k, pct = pca_basis(ref_feats)
    print(f"  PCA: top {k} components explain {pct:.1f}% variance (threshold={PCA_CUMVAR*100:.0f}%)\n")

    mu_ref, sigma_ref = fit_gaussian(ref_feats)

    # ── Split-half baseline (noise floor at N=BASELINE_N) ─────────────────────
    rng  = np.random.default_rng(0)
    idx  = rng.permutation(len(ref_feats))
    half = len(ref_feats) // 2
    hA   = subsample(ref_feats[idx[:half]],  BASELINE_N)
    hB   = subsample(ref_feats[idx[half:]], BASELINE_N)

    mu_A, s_A = fit_gaussian(hA); mu_B, s_B = fit_gaussian(hB)
    baseline_fid  = compute_fid(mu_A, s_A, mu_B, s_B)
    baseline_mmd  = compute_mmd(hA, hB)
    baseline_bhat = compute_bhattacharyya(hA, hB, V)
    print(f"  *** Split-half baseline (N={BASELINE_N} each) ***")
    print(f"      FID={baseline_fid:.3f}  MMD²={baseline_mmd:.6f}  Bhat={baseline_bhat:.4f}\n")

    # ── Per-dataset metrics ───────────────────────────────────────────────────
    col = 22
    header = (f"{'Dataset':<{col}} {'N':>7}  {'FID':>9}  {'MMD²':>12}  {'Bhat':>9}")
    print(header); print("─"*len(header))

    results = []

    def _process(name, ds):
        if len(ds) == 0:
            print(f"  {name:<{col}} — no images found, skipping")
            return
        feats = get_features(model, backbone_name, name, ds)
        N     = len(feats)
        mu_t, s_t = fit_gaussian(feats)
        fid  = compute_fid(mu_ref, sigma_ref, mu_t, s_t)
        mmd  = compute_mmd(subsample(ref_feats, MMD_N), subsample(feats, MMD_N))
        bhat = compute_bhattacharyya(ref_feats, feats, V)
        results.append({"dataset": name, "N": N, "FID": fid, "MMD2": mmd, "Bhat": bhat})
        print(f"  {name:<{col}} {N:>7}  {fid:>9.2f}  {mmd:>12.6f}  {bhat:>9.4f}")

    for name, path in ALL_DATASETS.items():
        _process(name, FolderDataset(path, transform))

    for name, parquet_files in ALL_PARQUET_DATASETS.items():
        _process(name, ParquetDataset(parquet_files, transform))

    for name, (hf_path, split) in HF_DATASETS.items():
        _process(name, HFDataset(hf_path, split, transform))

    # ── Spearman rank correlations ────────────────────────────────────────────
    if len(results) >= 3:
        vals = {k: np.array([r[k] for r in results]) for k in ("FID","MMD2","Bhat")}
        print(f"\n  Spearman ρ between metrics ({len(results)} datasets):")
        pairs = [("FID","MMD2"), ("FID","Bhat"), ("MMD2","Bhat")]
        for a, b in pairs:
            rho, pval = scipy.stats.spearmanr(vals[a], vals[b])
            print(f"    {a} vs {b}: ρ={rho:.3f}  p={pval:.3e}")

    return results, {"fid": baseline_fid, "mmd": baseline_mmd, "bhat": baseline_bhat}


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}\n")

    val_files = sorted(glob.glob(os.path.join(IMAGENET_SNAP, "val-*.parquet")))
    assert val_files, f"No val parquet files found in {IMAGENET_SNAP}"

    backbones = {}

    # ── InceptionV3 ───────────────────────────────────────────────────────────
    print("Loading InceptionV3…")
    inc_model = InceptionFeatures().to(DEVICE).eval()
    inc_results, inc_baseline = run_backbone(
        "inception", inc_model, INCEPTION_TRANSFORM, val_files
    )
    backbones["inception"] = inc_results
    del inc_model; torch.cuda.empty_cache()

    # ── CLIP ViT-L-14 (openai/CLIP, uses ~/.cache/clip/ViT-L-14.pt) ─────────────
    print("\nLoading CLIP ViT-L-14 (openai)…")
    clip_obj = CLIPFeatures("ViT-L/14").to(DEVICE).eval()
    # Use the official CLIP preprocessing (bicubic 224×224, CLIP normalisation)
    clip_transform = clip_obj.preprocess
    clip_results, clip_baseline = run_backbone(
        "clip", clip_obj, clip_transform, val_files
    )
    backbones["clip"] = clip_results
    del clip_obj; torch.cuda.empty_cache()

    # ── Save combined CSV ─────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, "distance_results.csv")
    fields = ["dataset", "N",
              "FID_inception", "MMD2_inception", "Bhat_inception",
              "FID_clip",      "MMD2_clip",      "Bhat_clip"]

    inc_map  = {r["dataset"]: r for r in inc_results}
    clip_map = {r["dataset"]: r for r in clip_results}
    all_names = list(dict.fromkeys(
        [r["dataset"] for r in inc_results] + [r["dataset"] for r in clip_results]
    ))

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        # Noise-floor row first
        w.writerow({"dataset": "**imagenet_split_half**",
                    "N": BASELINE_N,
                    "FID_inception":  round(inc_baseline["fid"],  4),
                    "MMD2_inception": round(inc_baseline["mmd"],  6),
                    "Bhat_inception": round(inc_baseline["bhat"], 4),
                    "FID_clip":       round(clip_baseline["fid"], 4),
                    "MMD2_clip":      round(clip_baseline["mmd"], 6),
                    "Bhat_clip":      round(clip_baseline["bhat"],4)})
        for name in all_names:
            ri = inc_map.get(name, {})
            rc = clip_map.get(name, {})
            w.writerow({"dataset": name,
                        "N":              ri.get("N", rc.get("N", "")),
                        "FID_inception":  round(ri["FID"],  2) if ri else "",
                        "MMD2_inception": round(ri["MMD2"], 6) if ri else "",
                        "Bhat_inception": round(ri["Bhat"], 4) if ri else "",
                        "FID_clip":       round(rc["FID"],  2) if rc else "",
                        "MMD2_clip":      round(rc["MMD2"], 6) if rc else "",
                        "Bhat_clip":      round(rc["Bhat"], 4) if rc else ""})

    print(f"\n\nResults saved → {out_path}")
    print("Noise-floor rows are labelled **imagenet_split_half** at the top.")


if __name__ == "__main__":
    main()
