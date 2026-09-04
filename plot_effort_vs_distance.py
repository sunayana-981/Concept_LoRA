"""
"Same training budget — bigger payoff at larger domain distance."

Panel A: Training curves (EVR vs the legacy example counter) for all SAE runs,
         coloured by CLIP-FID tier, line style by architecture (layers),
         with a vertical dashed line at the 2M-example budget.

Panel B: DAMS gain (Specific SAE − Generic SAE) vs CLIP FID,
         bubble area ∝ training examples used.

The trainer's historical ``details/n_training_tokens`` field increments by
input images, not by patch-token activation vectors. Labels below use the
actual unit of that counter.
"""

import os, glob, json, yaml, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
from wandb.sdk.internal import datastore
import wandb.proto.wandb_internal_pb2 as pb

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

WANDB_DIR   = "/home/sunayana/Documents/Concept_LoRA/sae_vlm/wandb"
RESULTS_CSV = "/home/sunayana/Documents/Concept_LoRA/distance_results.csv"
DAMS_CSV    = "/home/sunayana/Documents/Concept_LoRA/sae_vlm/out/dams_sweep_full.csv"
OUT_DIR     = "/home/sunayana/Documents/Concept_LoRA/figures"
SMOOTH_WIN  = 20

DATASET_MAP = {
    "eurosat":    "eurosat",
    "caltech101": "caltech-101",
    "medmnist":   "pathmnist",
    "dtd":        "dtd",
    "ucf101":     "UCF101",
    "cub2002011": "cub200",
    "imagenet":   "imagenetv2",
}

DS_NORM = {
    "caltech101": "caltech-101",
    "eurosat":    "eurosat",
    "medmnist":   "pathmnist",
    "cub2002011": "cub200",
    "dtd":        "dtd",
    "ucf101":     "UCF101",
}

DS_LABEL = {
    "caltech-101": "Caltech-101",
    "eurosat":     "EuroSAT",
    "pathmnist":   "MedMNIST",
    "cub200":      "CUB-200",
    "dtd":         "DTD",
    "UCF101":      "UCF-101",
    "imagenetv2":  "ImageNet-v2",
}

TIER_BOUNDS = [0.0, 0.05, 0.25, 0.45, 1.0]
TIER_LABELS = ["Near-IID", "Low shift", "High shift", "OOD"]
TIER_COLORS = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"]

def tier_color(fid):
    for i, (lo, hi) in enumerate(zip(TIER_BOUNDS[:-1], TIER_BOUNDS[1:])):
        if lo <= fid < hi:
            return TIER_COLORS[i]
    return TIER_COLORS[-1]

os.makedirs(OUT_DIR, exist_ok=True)

# ── Parse wandb runs ──────────────────────────────────────────────────────────
def read_history(run_dir):
    wb_files = glob.glob(os.path.join(run_dir, "*.wandb"))
    if not wb_files:
        return None
    ds_store = datastore.DataStore()
    ds_store.open_for_scan(wb_files[0])
    rows = []
    while True:
        data = ds_store.scan_data()
        if data is None:
            break
        try:
            rec = pb.Record()
            rec.ParseFromString(data)
            if rec.WhichOneof("record_type") != "history":
                continue
            row = {}
            for item in rec.history.item:
                key = "/".join(item.nested_key) if item.nested_key else item.key
                if not key:
                    continue
                try:
                    row[key] = json.loads(item.value_json)
                except Exception:
                    row[key] = item.value_json
            if row:
                rows.append(row)
        except Exception:
            pass
    return pd.DataFrame(rows) if rows else None

def read_config(run_dir):
    p = os.path.join(run_dir, "files", "config.yaml")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        raw = yaml.safe_load(f)
    return {k: v.get("value") if isinstance(v, dict) and "value" in v else v
            for k, v in raw.items() if k != "_wandb"}

print("Parsing wandb runs…")
KEY_TOKS = "details/n_training_tokens"
KEY_EVR  = "metrics/explained_variance"

runs_data = []
for run_dir in sorted(glob.glob(os.path.join(WANDB_DIR, "run-*"))):
    cfg = read_config(run_dir)
    dataset_wandb = cfg.get("dataset")
    if dataset_wandb not in DATASET_MAP:
        continue
    df = read_history(run_dir)
    if df is None or df.empty:
        continue
    if KEY_TOKS not in df.columns or KEY_EVR not in df.columns:
        continue
    df = df.dropna(subset=[KEY_TOKS]).copy()
    df = df.sort_values(KEY_TOKS).reset_index(drop=True)
    block_layers = str(cfg.get("block_layers", "unknown"))
    n_max = df[KEY_TOKS].max()
    runs_data.append({
        "run_id":        os.path.basename(run_dir),
        "dataset_wandb": dataset_wandb,
        "dataset":       DATASET_MAP[dataset_wandb],
        "block_layers":  block_layers,
        "n_tokens_max":  n_max,
        "df":            df,
    })
    print(f"  {os.path.basename(run_dir):40s}  ds={dataset_wandb:12s}  "
          f"layers={block_layers:12s}  tokens={n_max:>9,.0f}")

# Best run per dataset = most training tokens
best_runs = {}
for r in runs_data:
    ds = r["dataset"]
    if ds not in best_runs or r["n_tokens_max"] > best_runs[ds]["n_tokens_max"]:
        best_runs[ds] = r

print(f"\nBest run per dataset:")
for ds, r in sorted(best_runs.items(), key=lambda x: x[1]["n_tokens_max"]):
    print(f"  {ds:15s}  layers={r['block_layers']:12s}  tokens={r['n_tokens_max']:>9,.0f}")

# ── Distance data ─────────────────────────────────────────────────────────────
dist_df = pd.read_csv(RESULTS_CSV)
dist_df = dist_df[dist_df.dataset != "**imagenet_split_half**"]
fid_map = dict(zip(dist_df.dataset, dist_df.FID_clip))

# ── DAMS data ─────────────────────────────────────────────────────────────────
dams_sweep = pd.read_csv(DAMS_CSV)
best_row   = dams_sweep.loc[dams_sweep["total_gap"].idxmax()]

def best_lora_dams(ds_key, row):
    cols = [c for c in row.index if "LoRA_SAE" in c and ds_key in c]
    vals = [row[c] for c in cols if not pd.isna(row[c])]
    return max(vals) if vals else np.nan

dams_data = {}
for ds_wandb, ds_norm in DS_NORM.items():
    base_col = f"dams_Base_SAE_{ds_wandb}"
    if base_col not in best_row.index:
        continue
    base = best_row[base_col]
    lora = best_lora_dams(ds_wandb, best_row)
    fid  = fid_map.get(ds_norm, np.nan)
    if not np.isnan(base) and not np.isnan(fid):
        dams_data[ds_norm] = {
            "base":  base,
            "lora":  lora,
            "gain":  (lora - base) if not np.isnan(lora) else np.nan,
            "fid":   fid,
            "label": DS_LABEL.get(ds_norm, ds_norm),
        }

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13.5, 5.4))
gs  = GridSpec(1, 2, figure=fig, width_ratios=[1.15, 0.85], wspace=0.40)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# ── Panel A: Training curves ───────────────────────────────────────────────────
for ds, r in sorted(best_runs.items(), key=lambda x: fid_map.get(x[0], 99)):
    fid = fid_map.get(ds, np.nan)
    if math.isnan(fid):
        continue

    sub = r["df"][[KEY_TOKS, KEY_EVR]].dropna().sort_values(KEY_TOKS)
    if sub.empty:
        continue

    evr_smooth = sub[KEY_EVR].rolling(SMOOTH_WIN, min_periods=1).mean()
    toks_m     = sub[KEY_TOKS] / 1e6
    color      = tier_color(fid)
    label_text = DS_LABEL.get(ds, ds)

    # Solid = [-3,-1], dashed = [-2], dotted = [-1] only
    bl = r["block_layers"]
    if "-3" in bl:
        ls, lw = "-", 2.0
    elif "-2" in bl and "-3" not in bl:
        ls, lw = "--", 2.0
    else:
        ls, lw = ":", 1.6

    ax1.plot(toks_m, evr_smooth, color=color, alpha=0.88,
             lw=lw, ls=ls, zorder=3)

    # Label at endpoint (right side)
    x_end = float(toks_m.iloc[-1])
    y_end = float(evr_smooth.iloc[-1])
    ax1.annotate(f"{label_text}\n(FID={fid:.2f})",
                 xy=(x_end, y_end),
                 xytext=(5, 0), textcoords="offset points",
                 fontsize=7, color=color, va="center")

# 2M budget line
ax1.axvline(2.0, color="black", lw=1.1, ls=":", alpha=0.6, zorder=2)
ax1.text(2.08, 0.10, "2M\nbudget", fontsize=7.5, color="black",
         va="bottom", ha="left", alpha=0.65)

# EVR threshold
ax1.axhline(0.95, color="grey", lw=0.8, ls="--", alpha=0.5, zorder=1)
ax1.text(0.05, 0.955, "EVR = 0.95", fontsize=7, color="grey", va="bottom")

ax1.set_xlabel("Training images/examples (millions)", labelpad=6)
ax1.set_ylabel("Explained Variance Ratio (EVR)", labelpad=6)
ax1.set_title("A   SAE quality converges quickly regardless of domain distance",
              loc="left", fontweight="bold", fontsize=10.5, pad=8)
ax1.set_ylim(0, 1.07)
ax1.set_xlim(left=0)

tier_patches = [mpatches.Patch(color=TIER_COLORS[i], label=TIER_LABELS[i], alpha=0.85)
                for i in range(len(TIER_LABELS))]
arch_lines = [
    plt.Line2D([0], [0], color="grey", ls="-",  lw=1.8, label="Layers [−3,−1]"),
    plt.Line2D([0], [0], color="grey", ls="--", lw=1.8, label="Layer  [−2]"),
    plt.Line2D([0], [0], color="grey", ls=":",  lw=1.4, label="Layer  [−1]"),
]
ax1.legend(handles=tier_patches + arch_lines,
           fontsize=7.5, loc="lower right", framealpha=0.92,
           ncol=2, columnspacing=0.8, handlelength=1.8)

# ── Panel B: DAMS gain vs CLIP FID — bubble = tokens ─────────────────────────
fid_vals, gain_vals, tok_vals = [], [], []
labels_b, colors_b = [], []

for ds_norm, vals in dams_data.items():
    gain = vals["gain"]
    fid  = vals["fid"]
    if np.isnan(gain):
        continue
    n_tokens = best_runs[ds_norm]["n_tokens_max"] if ds_norm in best_runs else 2e6
    fid_vals.append(fid)
    gain_vals.append(gain)
    tok_vals.append(n_tokens)
    labels_b.append(vals["label"])
    colors_b.append(tier_color(fid))

max_tok = max(tok_vals) if tok_vals else 5e6
# area ∝ tokens^0.65 for legibility
sizes = [350 * (t / max_tok) ** 0.65 for t in tok_vals]

sc = ax2.scatter(fid_vals, gain_vals, s=sizes, c=colors_b,
                 alpha=0.82, edgecolors="white", linewidth=0.8, zorder=4)

for x, y, lbl, c, n in zip(fid_vals, gain_vals, labels_b, colors_b, tok_vals):
    # Nudge label above for positive gain, below for negative
    offset_y = 8 if y >= 0 else -12
    ax2.annotate(lbl, (x, y), xytext=(4, offset_y), textcoords="offset points",
                 fontsize=8, color=c, va="bottom" if y >= 0 else "top")

ax2.axhline(0, color="grey", lw=0.7, ls=":", zorder=1)
ax2.text(0.08, 0.02, "No gain", fontsize=7.5, color="grey", va="bottom")

# Quadratic fit through gain vs FID
if len(fid_vals) >= 3:
    xs, ys = np.array(fid_vals), np.array(gain_vals)
    try:
        def quad(x, a, b, c): return a * x**2 + b * x + c
        popt, _ = curve_fit(quad, xs, ys, p0=[1, -0.5, 0])
        xf = np.linspace(0.10, 0.80, 200)
        yf = quad(xf, *popt)
        ax2.plot(xf, yf, color="#2980b9", lw=1.8, ls="--", alpha=0.75)
        ax2.fill_between(xf, 0, np.clip(yf, 0, None),
                         alpha=0.08, color="#2980b9")
    except Exception as e:
        print(f"  Curve fit failed: {e}")

# Bubble size legend (reference circles)
for tok_m, lbl_s in [(2.0, "2M examples"), (5.0, "5M examples")]:
    sz = 350 * (tok_m * 1e6 / max_tok) ** 0.65
    ax2.scatter([], [], s=sz, color="grey", alpha=0.65,
                edgecolors="white", linewidth=0.8, label=lbl_s)

ax2.set_xlabel("CLIP FID (domain distance from ImageNet)", labelpad=6)
ax2.set_ylabel("DAMS gain  (Specific − Generic SAE)", labelpad=6)
ax2.set_title("B   Measured DAMS gain versus domain distance",
              loc="left", fontweight="bold", fontsize=10.5, pad=8)
ax2.legend(fontsize=8, loc="upper left", framealpha=0.92,
           title="Bubble ∝ training examples", title_fontsize=7.5)
ax2.set_xlim(0.08, 0.82)

# Callout annotation
# ── Footer ────────────────────────────────────────────────────────────────────
fig.text(0.5, -0.03,
         "Domain distance: CLIP ViT-L/14 FID vs ImageNet-1k val.  "
         "Bubble area ∝ training examples used.  Dashed curve: exploratory quadratic fit.",
         ha="center", fontsize=8, color="grey", style="italic")

plt.savefig(os.path.join(OUT_DIR, "effort_vs_distance.pdf"),
            bbox_inches="tight", dpi=150)
plt.savefig(os.path.join(OUT_DIR, "effort_vs_distance.png"),
            bbox_inches="tight", dpi=180)
plt.close(fig)
print("Saved effort_vs_distance.{pdf,png} →", OUT_DIR)

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"{'Dataset':<14} {'FID':>7} {'Examples (M)':>12} {'Base DAMS':>10} "
      f"{'Lora DAMS':>10} {'Gain':>8}")
print("─"*65)
for ds_norm in sorted(dams_data, key=lambda d: dams_data[d]["fid"]):
    v = dams_data[ds_norm]
    n_tok = best_runs[ds_norm]["n_tokens_max"] / 1e6 if ds_norm in best_runs else float("nan")
    gain  = v["gain"] if not np.isnan(v["gain"]) else float("nan")
    print(f"  {v['label']:<12} {v['fid']:>7.3f} {n_tok:>12.2f} "
          f"{v['base']:>10.4f} {v['lora']:>10.4f} {gain:>8.4f}")
