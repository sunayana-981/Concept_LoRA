#!/usr/bin/env python3
"""
Tables and figures for the rebuttal PDF, built from the CSVs Tasks 1/2/6
produce.

NOTE on input paths: run_rebuttal_evals.sh (Task 6) writes the new-domain and
old-domain matrices to separate directories (out/rebuttal/matrix/new_domain/
and out/rebuttal/matrix/old_domain/) rather than a single results.csv,
because they need different --lora_checkpoints/--sae_paths overrides (the
old-domain run reuses pathmnist's LoRA + FT-SAE on ImageNet). This script
takes both as separate CLI args accordingly.

Produces:
  Table 1 (out/rebuttal/tables/table1_accuracy.tex):
      rows=dataset (pets -> eurosat -> pathmnist, increasing domain shift),
      cols=ZS, FT, FT+G-SAE, FT+FT-SAE, (FT+FullFT-SAE for pathmnist),
      each SAE column annotated with Delta vs FT.
  Table 2 (out/rebuttal/tables/table2_sae_quality.tex):
      rows=(dataset, sae_condition), cols=L0, dead_frac, recon_cosine, FVE,
      label_entropy.
  Table 3 (out/rebuttal/tables/table3_old_domain.tex):
      single row (imagenet_subset), cols=ZS, FT, ZS+G-SAE, FT+G-SAE, FT+FT-SAE.
  Figure 1 (out/rebuttal/figures/fig1_accuracy_drop.{pdf,png}):
      line plot, x=dataset (domain-shift order), y=FT_acc - (FT+SAE)_acc,
      one line per sae_condition.
  Figure 2 (out/rebuttal/figures/fig2_steering.{pdf,png}):
      bar chart, mean ON-gain / OFF-drop per (dataset, sae_condition),
      averaged over class and k.

Each table/figure is skipped (with a printed reason) if its required source
rows aren't present yet -- this script is meant to be re-run as more of
Tasks 4/5/6 finish, not only once everything is done.

Usage:
    python tasks/make_rebuttal_figures.py
"""

import argparse
import os
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tasks.eval_steering import build_summary as steering_build_summary

DOMAIN_SHIFT_ORDER = ["pets", "eurosat", "pathmnist"]
SAE_LABELS = {
    "gsae": "G-SAE",
    "ftsae": "FT-SAE",
    "scratchsae": "Scratch-SAE",
    "fullftsae": "FullFT-SAE",
}
SAE_ORDER = ["gsae", "ftsae", "scratchsae", "fullftsae"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--new_domain_csv", type=str,
                    default="out/rebuttal/matrix/new_domain/results.csv")
    p.add_argument("--old_domain_csv", type=str,
                    default="out/rebuttal/matrix/old_domain/results.csv")
    p.add_argument("--steering_csv", type=str,
                    default="out/rebuttal/steering/steering_results.csv")
    p.add_argument("--tables_dir", type=str, default="out/rebuttal/tables")
    p.add_argument("--figures_dir", type=str, default="out/rebuttal/figures")
    return p.parse_args()


def load_csv(path):
    if not os.path.exists(path):
        print(f"[SKIP] {path} does not exist yet")
        return None
    df = pd.read_csv(path)
    if df.empty:
        print(f"[SKIP] {path} is empty")
        return None
    return df


def cell_value(df, dataset, vit_type, sae_condition, col="zeroshot_acc"):
    row = df[(df["dataset"] == dataset) & (df["vit_type"] == vit_type) &
             (df["sae_condition"] == sae_condition) & (~df["skipped"])]
    if row.empty:
        return None
    return row.iloc[0][col]


# ═════════════════════════════════════════════════════════════════════════
# LaTeX helpers
# ═════════════════════════════════════════════════════════════════════════

def esc(s):
    return str(s).replace("_", "\\_")


def write_booktabs(path, header, rows, caption, label):
    ncols = len(header)
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{l" + "c" * (ncols - 1) + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def fmt_acc(v):
    return "--" if v is None else f"{v:.1f}"


def fmt_acc_delta(v, ref):
    if v is None:
        return "--"
    if ref is None:
        return f"{v:.1f}"
    return f"{v:.1f} ({v - ref:+.1f})"


def fmt_metric(v, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v * 100:.1f}\\%" if pct else f"{v:.2f}"


# ═════════════════════════════════════════════════════════════════════════
# Table 1: accuracy
# ═════════════════════════════════════════════════════════════════════════

def make_table1(df, tables_dir):
    if df is None:
        print("[SKIP] Table 1: no new-domain results")
        return
    datasets = [d for d in DOMAIN_SHIFT_ORDER if d in df["dataset"].unique()]
    if not datasets:
        print("[SKIP] Table 1: none of pets/eurosat/pathmnist present")
        return

    has_fullft = any(cell_value(df, d, "lora", "fullftsae") is not None for d in datasets)
    has_scratch = any(cell_value(df, d, "lora", "scratchsae") is not None for d in datasets)
    header = ["Dataset", "ZS", "FT", "FT+G-SAE", "FT+FT-SAE"]
    if has_scratch:
        header.append("FT+Scratch-SAE")
    if has_fullft:
        header.append("FT+FullFT-SAE")

    rows = []
    for d in datasets:
        zs = cell_value(df, d, "base", "none")
        ft = cell_value(df, d, "lora", "none")
        gsae = cell_value(df, d, "lora", "gsae")
        ftsae = cell_value(df, d, "lora", "ftsae")
        row = [esc(d), fmt_acc(zs), fmt_acc(ft), fmt_acc_delta(gsae, ft), fmt_acc_delta(ftsae, ft)]
        if has_scratch:
            scratch = cell_value(df, d, "lora", "scratchsae")
            row.append(fmt_acc_delta(scratch, ft))
        if has_fullft:
            fullft = cell_value(df, d, "lora", "fullftsae")
            row.append(fmt_acc_delta(fullft, ft))
        rows.append(row)

    write_booktabs(
        os.path.join(tables_dir, "table1_accuracy.tex"), header, rows,
        caption="Zero-shot accuracy across increasing domain shift (pets $\\to$ eurosat $\\to$ pathmnist). "
                "SAE columns show $\\Delta$ vs.\\ FT in parentheses.",
        label="tab:rebuttal-accuracy")


# ═════════════════════════════════════════════════════════════════════════
# Table 2: SAE quality
# ═════════════════════════════════════════════════════════════════════════

def make_table2(df, tables_dir):
    if df is None:
        print("[SKIP] Table 2: no new-domain results")
        return
    datasets = [d for d in DOMAIN_SHIFT_ORDER if d in df["dataset"].unique()]
    header = ["Dataset", "SAE", "L0", "Dead frac.", "Recon.\\ cos.", "FVE", "Label entropy"]
    rows = []
    for d in datasets:
        for sae_condition in SAE_ORDER:
            r = df[(df["dataset"] == d) & (df["vit_type"] == "lora") &
                   (df["sae_condition"] == sae_condition) & (~df["skipped"])]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append([
                esc(d), SAE_LABELS[sae_condition],
                fmt_metric(r["l0"]), fmt_metric(r["dead_frac"], pct=True),
                fmt_metric(r["recon_cosine"]), fmt_metric(r["fve"]),
                fmt_metric(r["label_entropy_mean"]),
            ])
    if not rows:
        print("[SKIP] Table 2: no SAE-condition rows with quality metrics")
        return

    write_booktabs(
        os.path.join(tables_dir, "table2_sae_quality.tex"), header, rows,
        caption="SAE reconstruction faithfulness and feature quality, per dataset and SAE condition "
                "(LoRA model only; layer $-2$, expansion factor 64, $L_1{=}8{\\times}10^{-5}$ throughout).",
        label="tab:rebuttal-sae-quality")


# ═════════════════════════════════════════════════════════════════════════
# Table 3: old-domain (ImageNet)
# ═════════════════════════════════════════════════════════════════════════

def make_table3(df, tables_dir):
    if df is None:
        print("[SKIP] Table 3: no old-domain results")
        return
    if "imagenet_subset" not in df["dataset"].unique():
        print("[SKIP] Table 3: no imagenet_subset rows")
        return

    header = ["ZS", "FT", "ZS+G-SAE", "FT+G-SAE", "FT+FT-SAE"]
    zs = cell_value(df, "imagenet_subset", "base", "none")
    ft = cell_value(df, "imagenet_subset", "lora", "none")
    zs_gsae = cell_value(df, "imagenet_subset", "base", "gsae")
    ft_gsae = cell_value(df, "imagenet_subset", "lora", "gsae")
    ft_ftsae = cell_value(df, "imagenet_subset", "lora", "ftsae")
    row = [fmt_acc(zs), fmt_acc(ft), fmt_acc_delta(zs_gsae, zs), fmt_acc_delta(ft_gsae, ft),
           fmt_acc_delta(ft_ftsae, ft)]

    write_booktabs(
        os.path.join(tables_dir, "table3_old_domain.tex"), header, [row],
        caption="Old-domain (ImageNet) accuracy, reusing pathmnist's LoRA weights and FT-SAE. "
                "SAE columns show $\\Delta$ vs.\\ the matching no-SAE baseline in parentheses.",
        label="tab:rebuttal-old-domain")


# ═════════════════════════════════════════════════════════════════════════
# Figure 1: accuracy drop from SAE insertion, by domain shift
# ═════════════════════════════════════════════════════════════════════════

def make_figure1(df, figures_dir):
    if df is None:
        print("[SKIP] Figure 1: no new-domain results")
        return
    datasets = [d for d in DOMAIN_SHIFT_ORDER if d in df["dataset"].unique()]
    if len(datasets) < 2:
        print("[SKIP] Figure 1: fewer than 2 datasets present")
        return

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    x = np.arange(len(datasets))
    any_plotted = False

    for sae_condition in SAE_ORDER:
        ys, xs = [], []
        for i, d in enumerate(datasets):
            ft = cell_value(df, d, "lora", "none")
            sae_acc = cell_value(df, d, "lora", sae_condition)
            if ft is None or sae_acc is None:
                continue
            xs.append(i)
            ys.append(ft - sae_acc)
        if len(xs) >= 1:
            ax.plot(xs, ys, marker="o", label=SAE_LABELS[sae_condition])
            any_plotted = True

    if not any_plotted:
        print("[SKIP] Figure 1: no sae_condition had any data")
        plt.close(fig)
        return

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_xlabel("Dataset (increasing domain shift $\\rightarrow$)")
    ax.set_ylabel("Accuracy drop: FT $-$ (FT+SAE)")
    ax.set_title("SAE insertion cost vs.\\ domain shift")
    ax.legend(fontsize=8)
    fig.tight_layout()

    os.makedirs(figures_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        out_path = os.path.join(figures_dir, f"fig1_accuracy_drop.{ext}")
        fig.savefig(out_path, dpi=150)
        print(f"Wrote {out_path}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════
# Figure 2: steering ON-gain / OFF-drop
# ═════════════════════════════════════════════════════════════════════════

def make_figure2(steering_df, figures_dir):
    if steering_df is None:
        print("[SKIP] Figure 2: no steering results")
        return

    per_k = steering_build_summary(steering_df)
    if not per_k:
        print("[SKIP] Figure 2: steering summary is empty")
        return

    agg = pd.DataFrame(per_k).groupby(["dataset", "sae_condition"], as_index=False).agg(
        mean_on_delta=("mean_on_delta", "mean"),
        mean_off_delta=("mean_off_delta", "mean"),
    )

    datasets = [d for d in DOMAIN_SHIFT_ORDER if d in agg["dataset"].unique()]
    sae_conditions = [s for s in SAE_ORDER if s in agg["sae_condition"].unique()]
    if not datasets or not sae_conditions:
        print("[SKIP] Figure 2: no (dataset, sae_condition) pairs found")
        return

    fig, (ax_on, ax_off) = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=False)
    width = 0.8 / max(len(sae_conditions), 1)
    x = np.arange(len(datasets))

    for i, sae_condition in enumerate(sae_conditions):
        sub = agg[agg["sae_condition"] == sae_condition].set_index("dataset")
        on_vals = [sub["mean_on_delta"].get(d, np.nan) for d in datasets]
        off_vals = [sub["mean_off_delta"].get(d, np.nan) for d in datasets]
        offset = (i - (len(sae_conditions) - 1) / 2) * width
        ax_on.bar(x + offset, on_vals, width=width, label=SAE_LABELS[sae_condition])
        ax_off.bar(x + offset, off_vals, width=width, label=SAE_LABELS[sae_condition])

    for ax, title in ((ax_on, "ON-gain (top-$k$ clamped ON)"), (ax_off, "OFF-drop (top-$k$ clamped OFF)")):
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("Mean $\\Delta$acc vs.\\ baseline")
    ax_on.legend(fontsize=7)
    fig.suptitle("Steering: class-selective feature clamping (mean over class, $k$)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    os.makedirs(figures_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        out_path = os.path.join(figures_dir, f"fig2_steering.{ext}")
        fig.savefig(out_path, dpi=150)
        print(f"Wrote {out_path}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.makedirs(args.tables_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    new_df = load_csv(args.new_domain_csv)
    old_df = load_csv(args.old_domain_csv)
    steering_df = load_csv(args.steering_csv)

    make_table1(new_df, args.tables_dir)
    make_table2(new_df, args.tables_dir)
    make_table3(old_df, args.tables_dir)
    make_figure1(new_df, args.figures_dir)
    make_figure2(steering_df, args.figures_dir)


if __name__ == "__main__":
    main()
