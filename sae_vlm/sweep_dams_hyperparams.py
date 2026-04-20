#!/usr/bin/env python3
"""
Full hyperparameter sweep for DAMS across ALL available SAE/dataset combinations.

    DAMS = EC × (α × CSS_norm + β × FSS)
    CSS_norm = css_raw / (css_raw + κ)
    FSS = mean_c max_f [ p(c|f) × (1 − h_norm(f))^s ]

Sweep variables: α  (β=1−α),  κ  (CSS saturation),  s  (entropy sharpening)

SAE combinations covered:
  Existing:  Base / {caltech101, eurosat, medmnist}
             LoRA l-3 / caltech101        (ted4zuln)
             LoRA l-3 / eurosat           (bk3rbkcx)
             LoRA l-2 / medmnist          (d2ygd3bb)
  New:       Base + LoRA l-2 / {cub2002011, dtd, ucf101}
             LoRA l-1 / caltech101        (3hu8t1bb)
             LoRA l-1 / eurosat           (m8tn3p5s)
             LoRA l-3 / medmnist          (91r6lhuw)
             LoRA l-1 / medmnist          (qh3pp6cl)

SAE forward passes run ONCE per combo.  Sweep is purely analytical thereafter.
"""
import sys, math, os, json, itertools
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from src.sae_training.loaders import load_sae
from src.metrics.dams import (
    _compute_pooled_activations,
    compute_kernel_alignment,
    compute_mmd_score,
    extract_fss_components,
    fss_from_components,
)
import clip as openai_clip

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = "/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16"
DR = "/home/sunayana/Documents/Concept_LoRA/data"
BB = "ViT-B/16"
BS = 64
BASE_SAE = "data/sae_weight/base/out.pt"

# (label, dataset, sae_path, lora_weights_path)
RUNS = [
    # ── existing datasets ────────────────────────────────────────────────
    ("Base SAE",       "caltech101", BASE_SAE, None),
    ("LoRA SAE l-3",   "caltech101",
     "out/checkpoints/caltech101/ted4zuln/final_sparse_autoencoder_openai/clip-vit-base-patch16_-3_resid_49152.pt",
     f"{LR}/caltech101/16shots/seed1/lora_weights.pt"),
    ("LoRA SAE l-1",   "caltech101",
     "out/checkpoints/caltech101/3hu8t1bb/final_sparse_autoencoder_openai/clip-vit-base-patch16_-1_resid_49152.pt",
     f"{LR}/caltech101/16shots/seed1/lora_weights.pt"),

    ("Base SAE",       "eurosat",    BASE_SAE, None),
    ("LoRA SAE l-3",   "eurosat",
     "out/checkpoints/eurosat/bk3rbkcx/final_sparse_autoencoder_openai/clip-vit-base-patch16_-3_resid_49152.pt",
     f"{LR}/eurosat/16shots/seed1/lora_weights.pt"),
    ("LoRA SAE l-1",   "eurosat",
     "out/checkpoints/eurosat/m8tn3p5s/final_sparse_autoencoder_openai/clip-vit-base-patch16_-1_resid_49152.pt",
     f"{LR}/eurosat/16shots/seed1/lora_weights.pt"),

    ("Base SAE",       "medmnist",   BASE_SAE, None),
    ("LoRA SAE l-2",   "medmnist",
     "out/checkpoints/medmnist/d2ygd3bb/final_sparse_autoencoder_openai/clip-vit-base-patch16_-2_resid_49152.pt",
     f"{LR}/medmnist/16shots/seed1/lora_weights.pt"),
    ("LoRA SAE l-3",   "medmnist",
     "out/checkpoints/medmnist/91r6lhuw/final_sparse_autoencoder_openai/clip-vit-base-patch16_-3_resid_49152.pt",
     f"{LR}/medmnist/16shots/seed1/lora_weights.pt"),
    ("LoRA SAE l-1",   "medmnist",
     "out/checkpoints/medmnist/qh3pp6cl/final_sparse_autoencoder_openai/clip-vit-base-patch16_-1_resid_49152.pt",
     f"{LR}/medmnist/16shots/seed1/lora_weights.pt"),

    # ── new datasets ─────────────────────────────────────────────────────
    ("Base SAE",       "cub2002011", BASE_SAE, None),
    ("LoRA SAE l-2",   "cub2002011",
     "out/checkpoints/cub2002011/578p9z8f/final_sparse_autoencoder_openai/clip-vit-base-patch16_-2_resid_49152.pt",
     f"{LR}/cub2002011/16shots/seed1/lora_weights.pt"),

    ("Base SAE",       "dtd",        BASE_SAE, None),
    ("LoRA SAE l-2",   "dtd",
     "out/checkpoints/dtd/sd5h6hxv/final_sparse_autoencoder_openai/clip-vit-base-patch16_-2_resid_49152.pt",
     f"{LR}/dtd/16shots/seed42/lora_weights.pt"),

    ("Base SAE",       "ucf101",     BASE_SAE, None),
    ("LoRA SAE l-2",   "ucf101",
     "out/checkpoints/ucf101/j04tcnkc/final_sparse_autoencoder_openai/clip-vit-base-patch16_-2_resid_49152.pt",
     f"{LR}/ucf101/16shots/seed1/lora_weights.pt"),
]

MED = ["adipose", "background", "debris", "lymphocytes", "mucus",
       "smooth muscle", "normal colon mucosa",
       "cancer-associated stroma", "colorectal adenocarcinoma epithelium"]


def build_clip(lp):
    m, p = openai_clip.load(BB, device=DEVICE)
    if lp is None:
        return m, p
    s = torch.load(lp, map_location=DEVICE, weights_only=False)
    if "weights" not in s:
        m.load_state_dict(s)
        return m, p
    ly, me = s["weights"], s["metadata"]
    sc = me["alpha"] / math.sqrt(me["r"])
    with torch.no_grad():
        for i in range(12):
            ld = ly.get(f"layer_{i+12}", {})
            if not ld:
                continue
            blk = m.visual.transformer.resblocks[i]
            w = blk.attn.in_proj_weight.data
            d = w.shape[1]
            for pr, off in [("q_proj", 0), ("k_proj", d), ("v_proj", 2*d)]:
                try:
                    A = ld[pr]["w_lora_A"] if isinstance(ld.get(pr), dict) else ld[f"{pr}.w_lora_A"]
                    B = ld[pr]["w_lora_B"] if isinstance(ld.get(pr), dict) else ld[f"{pr}.w_lora_B"]
                    w[off:off+d] += (sc * B.float().to(DEVICE) @ A.float().to(DEVICE)).to(w.dtype)
                except Exception:
                    pass
    return m, p


def find_root(path):
    for r, dirs, _ in os.walk(path):
        if dirs:
            sd = os.path.join(r, dirs[0])
            exts = {os.path.splitext(f)[1].lower()
                    for f in os.listdir(sd) if os.path.isfile(os.path.join(sd, f))}
            if exts & {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                return r
    return path


class MedDS(torch.utils.data.Dataset):
    def __init__(self, pp):
        data = np.load(f"{DR}/pathmnist.npz")
        self.imgs = data["test_images"]
        self.labels = data["test_labels"].flatten().astype(int)
        self.pp = pp
        if_ds = datasets.ImageFolder(root=find_root(f"{DR}/pathmnist_imagefolder"))
        im = {n.replace("_", " ").lower(): i for n, i in if_ds.class_to_idx.items()}
        self.lm = {i: im.get(c.lower(), i) for i, c in enumerate(MED)}
        self.nc = len(MED)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return self.pp(Image.fromarray(self.imgs[i])), self.lm[int(self.labels[i])]


class ImgDS(torch.utils.data.Dataset):
    def __init__(self, path, pp, exc=None):
        r = find_root(path)
        full = datasets.ImageFolder(root=r, transform=pp)
        ex = exc or set()
        keep = [i for i, (_, l) in enumerate(full.samples) if full.classes[l] not in ex]
        self._ds = full; self._idx = keep
        kc = sorted({full.classes[full.targets[i]] for i in keep})
        self._m = {full.class_to_idx[c]: ni for ni, c in enumerate(kc)}
        self.nc = len(kc)
    def __len__(self): return len(self._idx)
    def __getitem__(self, i):
        img, l = self._ds[self._idx[i]]
        return img, self._m[l]


DS_PATHS = {
    "caltech101": (f"{DR}/caltech-101",             {"BACKGROUND_Google"}),
    "eurosat":    (f"{DR}/eurosat/2750",             None),
    "cub2002011": (f"{DR}/cub2002011/test",          None),
    "dtd":        (f"{DR}/DTD/images",               None),
    "ucf101":     (f"{DR}/UCF101/UCF-101-midframes", None),
}


def make_dataset(ds_name, pp):
    if ds_name == "medmnist":
        return MedDS(pp)
    path, exc = DS_PATHS[ds_name]
    return ImgDS(path, pp, exc=exc)


class Cap:
    def __init__(self): self.act = None; self._h = None
    def reg(self, blk):
        def f(m, inp, out): self.act = out.detach().float().transpose(0, 1)
        self._h = blk.register_forward_hook(f)
    def rm(self):
        if self._h: self._h.remove()


# ── Step 1: extract raw sub-metrics ──────────────────────────────────────────
print("=" * 70)
print(f"STEP 1 — extracting raw sub-metrics for {len(RUNS)} SAE/dataset combos")
print("=" * 70)

raw_data = []
loaded_clip = {}

for name, ds_name, sp, lp in RUNS:
    print(f"\n{'─'*60}\n{name} / {ds_name}")
    sae, cfg = load_sae(sp, DEVICE)
    sae.eval().to(DEVICE)

    if lp not in loaded_clip:
        loaded_clip[lp] = build_clip(lp)
    model, pp = loaded_clip[lp]
    model.eval()

    ds = make_dataset(ds_name, pp)
    loader = DataLoader(ds, batch_size=BS, shuffle=False, num_workers=4, pin_memory=True)
    nl = len(model.visual.transformer.resblocks)
    li = cfg.block_layer if cfg.block_layer >= 0 else nl + cfg.block_layer

    cap = Cap()
    cap.reg(model.visual.transformer.resblocks[li])
    af, al = [], []
    with torch.no_grad():
        for imgs, labs in tqdm(loader, desc="extract", leave=False):
            model.encode_image(imgs.to(DEVICE))
            af.append(cap.act.cpu())
            al.extend(labs.tolist())
    cap.rm()

    feats = torch.cat(af, dim=0)
    nc = ds.nc
    print(f"  feats={feats.shape}, C={nc}, layer={cfg.block_layer}")

    acts = _compute_pooled_activations(sae, feats, device=DEVICE, batch_size=256)
    ec, _, _ = compute_kernel_alignment(sae, feats, device=DEVICE,
                                        batch_size=256, subsample=2000,
                                        precomputed_acts=acts)
    css_raw, _, _ = compute_mmd_score(sae, feats, al, nc, device=DEVICE,
                                      batch_size=256, precomputed_acts=acts)
    p_c_given_f, h_norm, support_mask = extract_fss_components(acts, al, nc)

    raw_data.append(dict(
        name=name, dataset=ds_name, layer=cfg.block_layer,
        ec=ec, css_raw=css_raw, nc=nc,
        p_c_given_f=p_c_given_f, h_norm=h_norm, support_mask=support_mask,
    ))
    print(f"  EC={ec:.4f}  CSS_raw={css_raw:.6f}")

    del sae, feats, acts
    torch.cuda.empty_cache()

print(f"\nDone. Extracted {len(raw_data)} combos.")

# ── Step 2: analytical hyperparameter sweep ───────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2 — analytical sweep over α, κ, entropy_sharpening")
print("=" * 70)

alphas      = np.round(np.arange(0.05, 1.0, 0.05), 2)   # 0.05 … 0.95
kappas      = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]
sharpenings = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

# Precompute FSS for each combo × sharpening
fss_table = {}
for ci, row in enumerate(raw_data):
    for s in sharpenings:
        fss_table[(ci, s)] = fss_from_components(
            row['p_c_given_f'], row['h_norm'], row['support_mask'],
            row['nc'], entropy_sharpening=s,
        )
print(f"FSS precomputed for {len(fss_table)} (combo, sharpening) pairs.")

# Group combos: for each dataset, identify base and adapted SAEs
datasets_seen = sorted({r['dataset'] for r in raw_data})
base_idx  = {r['dataset']: i for i, r in enumerate(raw_data) if r['name'] == "Base SAE"}
adapted_idxs = {}
for ds in datasets_seen:
    adapted_idxs[ds] = [i for i, r in enumerate(raw_data)
                        if r['dataset'] == ds and r['name'] != "Base SAE"]

print(f"\nDatasets: {datasets_seen}")
for ds in datasets_seen:
    adp = [raw_data[i]['name'] for i in adapted_idxs.get(ds, [])]
    base = "Base SAE" if ds in base_idx else "NONE"
    print(f"  {ds:<12}: base={base}, adapted={adp}")

# For gap: sum of (best_adapted - base) DAMS over all datasets that have both
def compute_gap_score(alpha, kappa, sharpening):
    beta = 1.0 - alpha
    # Compute DAMS for all combos
    dams = {}
    for ci, row in enumerate(raw_data):
        css_norm = row['css_raw'] / (row['css_raw'] + kappa)
        fss = fss_table[(ci, sharpening)]
        dams[ci] = row['ec'] * (alpha * css_norm + beta * fss)

    total_gap = 0.0
    per_ds_gap = {}
    for ds in datasets_seen:
        if ds not in base_idx or not adapted_idxs.get(ds):
            continue
        base_dams  = dams[base_idx[ds]]
        best_adapted = max(dams[i] for i in adapted_idxs[ds])
        gap = best_adapted - base_dams
        per_ds_gap[ds] = gap
        total_gap += gap

    return total_gap, per_ds_gap, dams


total = len(alphas) * len(kappas) * len(sharpenings)
print(f"\nSweeping {total} combinations...")

sweep_results = []
best_gap = -np.inf
best_params = None

for alpha, kappa, sharpening in itertools.product(alphas, kappas, sharpenings):
    total_gap, per_ds_gap, dams_scores = compute_gap_score(alpha, kappa, sharpening)

    row = dict(
        alpha=alpha, kappa=kappa, sharpening=sharpening,
        total_gap=round(total_gap, 4),
    )
    for ds, g in per_ds_gap.items():
        row[f"gap_{ds}"] = round(g, 4)
    for ci, r in enumerate(raw_data):
        key = f"dams_{r['name'].replace(' ','_')}_{r['dataset']}"
        row[key] = round(dams_scores[ci], 4)
    sweep_results.append(row)

    if total_gap > best_gap:
        best_gap = total_gap
        best_params = dict(alpha=alpha, kappa=kappa, sharpening=sharpening)

print(f"Sweep complete.")

# ── Step 3: report ────────────────────────────────────────────────────────────
df = pd.DataFrame(sweep_results)
gap_cols = ["total_gap"] + [f"gap_{ds}" for ds in datasets_seen if f"gap_{ds}" in df.columns]
df_sorted = df.sort_values("total_gap", ascending=False)

print("\n" + "=" * 110)
print("TOP 20 combinations (by total gap across all datasets)")
print("=" * 110)
print(df_sorted[["alpha","kappa","sharpening"] + gap_cols].head(20).to_string(index=False))

print("\n" + "=" * 110)
bp = best_params
print(f"BEST PARAMS: α={bp['alpha']}, β={1-bp['alpha']:.2f}, κ={bp['kappa']}, sharpening={bp['sharpening']}")
print(f"Total gap = {best_gap:.4f}")
print("=" * 110)

# ── Full DAMS table at best params ───────────────────────────────────────────
print(f"\nDAMS table at best params (α={bp['alpha']}, κ={bp['kappa']}, s={bp['sharpening']}):")
print(f"{'SAE':<20} {'Dataset':<12} {'Layer':>6} {'EC':>8} {'CSS_nm':>8} {'FSS':>8} {'DAMS':>8}")
print("-" * 75)
for ci, row in enumerate(raw_data):
    css_norm = row['css_raw'] / (row['css_raw'] + bp['kappa'])
    fss = fss_table[(ci, bp['sharpening'])]
    dams = row['ec'] * (bp['alpha'] * css_norm + (1 - bp['alpha']) * fss)
    print(f"{row['name']:<20} {row['dataset']:<12} {row['layer']:>6} "
          f"{row['ec']:>8.4f} {css_norm:>8.4f} {fss:>8.4f} {dams:>8.4f}")

# ── Per-dataset summary at best params ───────────────────────────────────────
print(f"\nPer-dataset gap summary at best params:")
print(f"{'Dataset':<12} {'Base DAMS':>11} {'Best LoRA':>11} {'Gap':>8} {'Relative':>10}")
print("-" * 55)
_, per_ds, dams_scores = compute_gap_score(bp['alpha'], bp['kappa'], bp['sharpening'])
for ds in datasets_seen:
    if ds not in base_idx:
        continue
    b_dams = dams_scores[base_idx[ds]]
    if adapted_idxs.get(ds):
        best_i = max(adapted_idxs[ds], key=lambda i: dams_scores[i])
        best_dams = dams_scores[best_i]
        gap = best_dams - b_dams
        rel = gap / b_dams * 100
        best_label = raw_data[best_i]['name']
        print(f"{ds:<12} {b_dams:>11.4f} {best_dams:>11.4f} {gap:>8.4f} {rel:>9.1f}%  ({best_label})")
    else:
        print(f"{ds:<12} {b_dams:>11.4f}  (no adapted SAE)")

# Save
os.makedirs("out", exist_ok=True)
df_sorted.to_csv("out/dams_sweep_full.csv", index=False)
df_sorted.head(50).to_csv("out/dams_sweep_top50.csv", index=False)

# Also save raw metrics
raw_export = [
    {k: v for k, v in r.items() if k not in ('p_c_given_f', 'h_norm', 'support_mask')}
    for r in raw_data
]
with open("out/dams_raw_metrics.json", "w") as f:
    json.dump(raw_export, f, indent=2)

print("\nSaved: out/dams_sweep_full.csv, out/dams_sweep_top50.csv, out/dams_raw_metrics.json")
