"""
Training effort vs MMD plot.

Reads all local wandb run histories and distance_results.csv, then produces:
  1. training_curves.pdf   – EVR & MSE loss vs the legacy training counter, one curve per
                             dataset, coloured by CLIP MMD tier
  2. scatter_mmd_vs_evr.pdf – final EVR vs CLIP MMD² scatter
  3. scatter_mmd_vs_tokens.pdf – examples-to-threshold vs CLIP MMD²

The trainer records ``details/n_training_tokens`` for backward compatibility,
but increments it by the leading image-batch dimension. It is therefore a
count of input images/examples, not ViT activation vectors.
"""

import os, glob, json, yaml, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from wandb.sdk.internal import datastore
import wandb.proto.wandb_internal_pb2 as pb

# ── Config ─────────────────────────────────────────────────────────────────────
WANDB_DIR    = "/home/sunayana/Documents/Concept_LoRA/sae_vlm/wandb"
RESULTS_CSV  = "/home/sunayana/Documents/Concept_LoRA/distance_results.csv"
OUT_DIR      = "/home/sunayana/Documents/Concept_LoRA/figures"
EVR_THRESH   = 0.95     # "converged" threshold for tokens-to-threshold scatter
SMOOTH_WIN   = 20       # rolling-mean window for training curves

DATASET_MAP = {
    "eurosat":    "eurosat",
    "caltech101": "caltech-101",
    "medmnist":   "pathmnist",
    "dtd":        "dtd",
    "ucf101":     "UCF101",
    "cub2002011": "cub200",
    "imagenet":   "imagenetv2",
}

os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Parse wandb histories ───────────────────────────────────────────────────
def read_history(run_dir):
    wb_files = glob.glob(os.path.join(run_dir, "*.wandb"))
    if not wb_files:
        return None
    ds = datastore.DataStore()
    ds.open_for_scan(wb_files[0])
    rows = []
    while True:
        data = ds.scan_data()
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
runs_data = []
for run_dir in sorted(glob.glob(os.path.join(WANDB_DIR, "run-*"))):
    cfg = read_config(run_dir)
    dataset_wandb = cfg.get("dataset")
    if dataset_wandb not in DATASET_MAP:
        continue
    df = read_history(run_dir)
    if df is None or df.empty:
        continue
    # Keep only rows with n_training_tokens
    if "details/n_training_tokens" not in df.columns:
        continue
    df = df.dropna(subset=["details/n_training_tokens"]).copy()
    df = df.sort_values("details/n_training_tokens").reset_index(drop=True)
    runs_data.append({
        "run_id":        os.path.basename(run_dir),
        "dataset_wandb": dataset_wandb,
        "dataset":       DATASET_MAP[dataset_wandb],
        "block_layers":  str(cfg.get("block_layers")),
        "n_tokens_max":  df["details/n_training_tokens"].max(),
        "df":            df,
    })
    print(f"  {os.path.basename(run_dir):40s} dataset={dataset_wandb:12s} "
          f"tokens={df['details/n_training_tokens'].max():>9,.0f}  "
          f"steps={len(df):>6d}")

# ── 2. Select best run per dataset ────────────────────────────────────────────
# "Best" = most training tokens; ties broken by run recency (sorted dir name)
best_runs = {}
for r in runs_data:
    ds = r["dataset"]
    if ds not in best_runs or r["n_tokens_max"] > best_runs[ds]["n_tokens_max"]:
        best_runs[ds] = r

print(f"\nBest run per dataset:")
for ds, r in sorted(best_runs.items()):
    print(f"  {ds:15s}  {r['run_id']}  tokens={r['n_tokens_max']:>9,.0f}")

# ── 3. Load distance results (CLIP MMD as primary ordering) ───────────────────
dist_df = pd.read_csv(RESULTS_CSV)
dist_df = dist_df[dist_df.dataset != "**imagenet_split_half**"]
mmd_map  = dict(zip(dist_df.dataset, dist_df.MMD2_clip))
fid_map  = dict(zip(dist_df.dataset, dist_df.FID_clip))
bhat_map = dict(zip(dist_df.dataset, dist_df.Bhat_clip))

# ── 4. Assign MMD tiers & colours ─────────────────────────────────────────────
# Tiers from earlier analysis (CLIP FID):
# near-IID ≤ 0.05, low 0.05–0.25, mid 0.25–0.45, high > 0.45
TIER_BOUNDS = [0.0, 0.05, 0.25, 0.45, 1.0]
TIER_LABELS = ["near-IID", "low shift", "mid shift", "high shift"]
TIER_COLORS = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"]

def get_tier(fid):
    for i, (lo, hi) in enumerate(zip(TIER_BOUNDS[:-1], TIER_BOUNDS[1:])):
        if lo <= fid < hi:
            return i
    return len(TIER_LABELS) - 1

# ── 5. Training curves ─────────────────────────────────────────────────────────
print("\nPlotting training curves…")
KEY_EVR  = "metrics/explained_variance"
KEY_MSE  = "losses/mse_loss"
KEY_L0   = "metrics/l0"
KEY_TOKS = "details/n_training_tokens"

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ds, r in sorted(best_runs.items(), key=lambda x: fid_map.get(x[0], 99)):
    df  = r["df"]
    fid = fid_map.get(ds, np.nan)
    mmd = mmd_map.get(ds, np.nan)
    tier = get_tier(fid)
    color = TIER_COLORS[tier]
    label = f"{ds}  (MMD²={mmd:.3f})"

    sub = df[[KEY_TOKS, KEY_EVR, KEY_MSE]].dropna()
    if sub.empty:
        continue
    sub = sub.sort_values(KEY_TOKS)
    # smooth
    evr = sub[KEY_EVR].rolling(SMOOTH_WIN, min_periods=1).mean()
    mse = sub[KEY_MSE].rolling(SMOOTH_WIN, min_periods=1).mean()
    toks = sub[KEY_TOKS] / 1e6  # millions

    axes[0].plot(toks, evr, color=color, alpha=0.85, linewidth=1.5, label=label)
    axes[1].plot(toks, mse, color=color, alpha=0.85, linewidth=1.5, label=label)

axes[0].set_xlabel("Training images/examples (millions)")
axes[0].set_ylabel("Explained Variance Ratio")
axes[0].set_title("SAE quality vs training effort")
axes[0].set_ylim(0, 1.05)
axes[0].axhline(EVR_THRESH, color="black", linewidth=0.8, linestyle="--",
                label=f"EVR={EVR_THRESH} threshold")
axes[0].legend(fontsize=7, loc="lower right")
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel("Training images/examples (millions)")
axes[1].set_ylabel("MSE loss")
axes[1].set_title("Reconstruction loss vs training effort")
axes[1].set_yscale("log")
axes[1].legend(fontsize=7, loc="upper right")
axes[1].grid(True, alpha=0.3)

# Tier legend patches
from matplotlib.patches import Patch
tier_patches = [Patch(color=TIER_COLORS[i], label=TIER_LABELS[i])
                for i in range(len(TIER_LABELS))]
fig.legend(handles=tier_patches, title="Domain shift tier (CLIP FID)",
           loc="lower center", ncol=4, fontsize=8,
           bbox_to_anchor=(0.5, -0.04))
plt.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(os.path.join(OUT_DIR, "training_curves.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "training_curves.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
print("  Saved training_curves.{pdf,png}")

# ── 6. Scatter: final EVR vs CLIP MMD² ────────────────────────────────────────
print("Plotting scatter: final EVR vs CLIP MMD²…")
fig, ax = plt.subplots(figsize=(7, 5))

scatter_data = []
for ds, r in best_runs.items():
    df  = r["df"]
    mmd = mmd_map.get(ds, np.nan)
    fid = fid_map.get(ds, np.nan)
    if math.isnan(mmd):
        continue
    # Final EVR = median of last 10% of training
    sub = df[KEY_EVR].dropna()
    if sub.empty:
        continue
    tail_n   = max(1, len(sub) // 10)
    final_evr = float(sub.iloc[-tail_n:].mean())
    tier  = get_tier(fid)
    color = TIER_COLORS[tier]
    scatter_data.append((ds, mmd, fid, final_evr, tier, color))
    ax.scatter(mmd, final_evr, color=color, s=80, zorder=3, edgecolors="white", linewidth=0.6)
    ax.annotate(ds, (mmd, final_evr), textcoords="offset points",
                xytext=(5, 3), fontsize=7, color=color)

ax.set_xlabel("CLIP MMD² (distribution distance from ImageNet)")
ax.set_ylabel("Final Explained Variance Ratio")
ax.set_title("SAE quality vs domain shift")
ax.axhline(EVR_THRESH, color="grey", linewidth=0.8, linestyle="--",
           label=f"EVR={EVR_THRESH}")
ax.legend(handles=tier_patches + [plt.Line2D([0],[0],color="grey",linestyle="--",
          label=f"EVR={EVR_THRESH}")],
          title="Tier", fontsize=8, loc="lower left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "scatter_mmd_vs_evr.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "scatter_mmd_vs_evr.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
print("  Saved scatter_mmd_vs_evr.{pdf,png}")

# ── 7. Scatter: tokens to EVR threshold vs CLIP MMD² ─────────────────────────
print("Plotting scatter: tokens-to-threshold vs CLIP MMD²…")
fig, ax = plt.subplots(figsize=(7, 5))

for ds, mmd, fid, final_evr, tier, color in scatter_data:
    df   = best_runs[ds]["df"]
    sub  = df[[KEY_TOKS, KEY_EVR]].dropna().sort_values(KEY_TOKS)
    evr_smooth = sub[KEY_EVR].rolling(SMOOTH_WIN, min_periods=1).mean()
    reached = sub[KEY_TOKS][evr_smooth >= EVR_THRESH]

    if reached.empty:
        # Never reached — plot at max tokens with open marker
        max_toks = sub[KEY_TOKS].max() / 1e6
        ax.scatter(mmd, max_toks, color=color, s=100, marker="^", zorder=3,
                   edgecolors="white", linewidth=0.6, alpha=0.5)
        ax.annotate(f"{ds}*", (mmd, max_toks), textcoords="offset points",
                    xytext=(5, 3), fontsize=7, color=color)
    else:
        toks_m = float(reached.iloc[0]) / 1e6
        ax.scatter(mmd, toks_m, color=color, s=80, zorder=3,
                   edgecolors="white", linewidth=0.6)
        ax.annotate(ds, (mmd, toks_m), textcoords="offset points",
                    xytext=(5, 3), fontsize=7, color=color)

ax.set_xlabel("CLIP MMD² (distribution distance from ImageNet)")
ax.set_ylabel(f"Images/examples to reach EVR ≥ {EVR_THRESH} (millions)")
ax.set_title(f"Training effort to convergence (EVR ≥ {EVR_THRESH}) vs domain shift\n"
             f"(^ = never reached threshold)")
ax.legend(handles=tier_patches, title="Tier", fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "scatter_mmd_vs_tokens.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "scatter_mmd_vs_tokens.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
print("  Saved scatter_mmd_vs_tokens.{pdf,png}")

# ── 8. Print summary table ─────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"{'Dataset':<14} {'CLIP MMD²':>10} {'CLIP FID':>9} {'Final EVR':>10} "
      f"{'Examples@thresh (M)':>20} {'Tier':<12}")
print("─"*70)
for ds, mmd, fid, final_evr, tier, _ in sorted(scatter_data, key=lambda x: x[1]):
    df   = best_runs[ds]["df"]
    sub  = df[[KEY_TOKS, KEY_EVR]].dropna().sort_values(KEY_TOKS)
    evr_s = sub[KEY_EVR].rolling(SMOOTH_WIN, min_periods=1).mean()
    reached = sub[KEY_TOKS][evr_s >= EVR_THRESH]
    tok_str  = f"{reached.iloc[0]/1e6:.2f}" if not reached.empty else "DNF"
    print(f"  {ds:<12} {mmd:>10.4f} {fid:>9.4f} {final_evr:>10.4f} "
          f"{tok_str:>20} {TIER_LABELS[tier]}")
