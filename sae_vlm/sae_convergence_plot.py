"""
Find SAE convergence step per wandb run, then plot
convergence_step vs distance-from-ImageNet, one point per dataset.

Usage:
    pip install wandb pandas matplotlib
    wandb login
    python sae_convergence_plot.py \
        --entity sunayana1233 --project maple_clip_sae \
        --metric loss/reconstruction \
        --md-csv md_table.csv \
        --out conv_vs_md.png
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import wandb


# ---------------------------------------------------------------------------
# Convergence detection
# ---------------------------------------------------------------------------

def convergence_step(
    steps: np.ndarray,
    values: np.ndarray,
    tol: float = 0.01,
    smooth_frac: float = 0.05,
    tail_frac: float = 0.10,
) -> int | None:
    """
    First step where smoothed metric enters a `tol`-band around its final
    smoothed value AND stays inside that band through the end.

    steps:  (T,) monotonic step indices.
    values: (T,) metric values (lower-is-better assumed; works for MSE, L0).
    tol:    relative tolerance (0.01 = 1%).
    smooth_frac: window for moving average, as fraction of total length.
    tail_frac:   fraction of the run defining "final value" (median).

    Returns: step index, or None if never converged.
    """
    T = len(values)
    if T < 20:
        return None

    # Clip ghost-grad spikes at P99 before smoothing.
    # This removes extreme transients (e.g. MSE briefly → 13) without
    # distorting the normal early-training descent.
    clip_thresh = float(np.percentile(values, 99))
    clipped = np.minimum(values, clip_thresh)

    w = max(5, int(smooth_frac * T))
    smooth = pd.Series(clipped).rolling(w, min_periods=1, center=True).mean().to_numpy()

    tail_n = max(5, int(tail_frac * T))
    final = float(np.median(smooth[-tail_n:]))
    # Absolute floor so logging-granularity noise (SAE MSE ~0.001–0.005)
    # doesn't make the band impossibly narrow.
    band = max(tol * abs(final), 3e-4)

    in_band = np.abs(smooth - final) <= band

    # Find the first step after which the smoothed curve stays in-band.
    # Walk backwards: last out-of-band index + 1.
    if in_band.all():
        return int(steps[0])
    last_false = int(np.where(~in_band)[0][-1])
    if last_false == T - 1:
        return None  # never fully settled
    return int(steps[last_false + 1])


# ---------------------------------------------------------------------------
# wandb scraping
# ---------------------------------------------------------------------------

# Matches both "eurosat_maple_sae_layer-2" and "eurosat_sae_layer-3"
DATASET_RE = re.compile(r"^([a-z0-9]+(?:_[a-z0-9]+)?)_(?:maple_)?sae", re.IGNORECASE)

def parse_dataset(run_name: str) -> str | None:
    """
    'eurosat_maple_sae_layer-2' -> 'eurosat'
    'caltech101_sae_layer-3'    -> 'caltech101'
    'caltech101_maple_sae_layer-2' -> 'caltech101'
    """
    m = DATASET_RE.match(run_name)
    if not m:
        return None
    name = m.group(1).lower()
    # strip _maple suffix so maple and lora runs map to the same dataset key
    if name.endswith("_maple"):
        name = name[: -len("_maple")]
    return name


def fetch_runs(entity: str, projects: list[str], metric: str,
               only_finished: bool = False) -> dict[str, list[dict]]:
    """
    Returns {dataset_name: [run_records]}. Queries multiple wandb projects
    so both lora_clip_sae and maple_clip_sae runs are captured.
    only_finished=False by default so crashed/interrupted runs are included.
    """
    api = wandb.Api(timeout=60)
    grouped = defaultdict(list)

    for project in projects:
        print(f"  querying {entity}/{project} ...")
        try:
            runs = api.runs(f"{entity}/{project}")
        except Exception as e:
            print(f"  [warn] could not fetch {project}: {e}")
            continue

        for r in runs:
            if only_finished and r.state != "finished":
                continue
            dset = parse_dataset(r.name)
            if dset is None:
                print(f"  [skip] cannot parse dataset from {r.name!r}")
                continue
            # skip generic/test runs that aren't real datasets
            if dset in ("lora", "sae", "test", "debug"):
                print(f"  [skip] generic run name {r.name!r}")
                continue
            # wandb >=0.16 dropped the pandas= kwarg; history() always returns a DataFrame
            try:
                hist = r.history(keys=["_step", metric], samples=100_000)
            except TypeError:
                hist = r.history(keys=["_step", metric], samples=100_000, pandas=True)
            hist = hist.dropna(subset=[metric]).sort_values("_step")
            if len(hist) < 20:
                print(f"  [skip] {r.name}: only {len(hist)} points for {metric!r}")
                continue
            grouped[dset].append({
                "run_id":   r.id,
                "name":     r.name,
                "project":  project,
                "created":  r.created_at,
                "steps":    hist["_step"].to_numpy(),
                "values":   hist[metric].to_numpy(),
            })
    return grouped


def pick_canonical_run(runs: list[dict]) -> dict:
    """Pick the longest (most steps) run if duplicates exist."""
    return max(runs, key=lambda r: r["steps"][-1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", default="sunayana1233-iiit-hyderabad")
    p.add_argument("--projects", nargs="+",
                   default=["lora_clip_sae", "maple_clip_sae"],
                   help="one or more wandb project names to query")
    p.add_argument("--metric", default="losses/mse_loss",
                   help="wandb key for convergence; also try 'metrics/l0'")
    p.add_argument("--tol", type=float, default=0.01)
    p.add_argument("--md-csv", type=Path,
                   default=Path(__file__).parent / "md_table.csv",
                   help="CSV with columns: dataset, md_mean (or fid)")
    p.add_argument("--md-col", default="md_mean")
    p.add_argument("--out", default="conv_vs_md.png")
    p.add_argument("--csv-out", default="conv_vs_md.csv")
    args = p.parse_args()

    print(f"[+] Fetching runs from {args.entity} / {args.projects}")
    grouped = fetch_runs(args.entity, args.projects, args.metric)

    print(f"[+] Computing convergence step (tol={args.tol})")
    conv = {}
    for dset, runs in grouped.items():
        run = pick_canonical_run(runs)
        c = convergence_step(run["steps"], run["values"], tol=args.tol)
        conv[dset] = {
            "run_name": run["name"],
            "total_steps": int(run["steps"][-1]),
            "conv_step": c,
            "frac_of_run": (c / run["steps"][-1]) if c is not None else None,
        }
        print(f"  {dset:20s} conv@{c} of {run['steps'][-1]} "
              f"({len(runs)} runs found)")

    # ----- Join with MD table -----
    md_df = pd.read_csv(args.md_csv)
    md_df["dataset_key"] = md_df["dataset"].str.lower().str.replace("-", "")
    # match keys: 'eurosat', 'imagenetv2', 'imageneta', etc.
    rows = []
    for dset, info in conv.items():
        if info["conv_step"] is None:
            continue
        m = md_df[md_df["dataset_key"].str.contains(dset.replace("_", ""))]
        if m.empty:
            print(f"  [warn] no MD entry for {dset}")
            continue
        rows.append({
            "dataset": dset,
            "md": float(m.iloc[0][args.md_col]),
            "conv_step": info["conv_step"],
            "total_steps": info["total_steps"],
            "frac_of_run": info["frac_of_run"],
        })

    out_df = pd.DataFrame(rows).sort_values("md")
    out_df.to_csv(args.csv_out, index=False)
    print(f"[+] Wrote {args.csv_out}")
    print(out_df.to_string(index=False))

    # ----- Plot -----
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(out_df["md"], out_df["conv_step"], s=60, color="#2b8cbe",
               edgecolor="k", linewidth=0.5)
    for _, row in out_df.iterrows():
        ax.annotate(row["dataset"], (row["md"], row["conv_step"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    # trend line
    if len(out_df) >= 3:
        slope, intercept = np.polyfit(out_df["md"], out_df["conv_step"], 1)
        xs = np.linspace(out_df["md"].min(), out_df["md"].max(), 50)
        ax.plot(xs, slope * xs + intercept, "--", color="gray",
                label=f"linear fit (slope={slope:.1f})")
        # also Spearman, since linear fit is iffy on small N
        from scipy.stats import spearmanr
        rho, pval = spearmanr(out_df["md"], out_df["conv_step"])
        ax.set_title(f"Convergence step vs MD from ImageNet  "
                     f"(Spearman ρ={rho:.2f}, p={pval:.3f}, n={len(out_df)})")
        ax.legend()
    ax.set_xlabel(f"Distance from ImageNet ({args.md_col})")
    ax.set_ylabel(f"Convergence step (tol={args.tol}, metric={args.metric})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[+] Wrote {args.out}")


if __name__ == "__main__":
    main()