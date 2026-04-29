#!/usr/bin/env python3
"""
Unified SAE evaluation entry point.

Runs any combination of evaluation tasks in a single script with one forward
pass per task per dataset.  Select tasks with --tasks (default: feature_stats).

Tasks
-----
feature_stats   Per-feature stats: max-activating images, sparsity, L0, MSE,
                explained variance, class-wise activation counts, selectivity,
                top-1 fraction, label entropy.  Saves .pt/.npy artifacts +
                4 visualisation plots.

classification  Zero-shot CLIP accuracy (baseline + optional SAE modes).
                CLIP-family backbones only.  Controlled by --sae_modes.

linear_probe    Logistic regression on [N, d_sae] SAE latents + optional CLIP
                baseline.  Fits on --probe_train_split, evaluates on
                --probe_test_split.  Works with any backbone.

feature_masking Causal ablation: clamp top-K SAE features ON or OFF per class,
                measure accuracy change.  CLIP-family only.
                Requires cls_sae_cnt (run feature_stats first or pass both).

compare         Compare two backbone instances (e.g. base vs LoRA) on
                embedding drift: per-sample SAE cosine/L2, CLIP cosine/L2,
                sparsity.  Requires --compare_model_b.

embedding_mono  Weighted pairwise cosine monosemanticity in embedding space
                (MS_h from integrate_mono.py).  Summarises top-K neurons and
                per-class top-K neurons.  Controlled by --embedding_mono_k.

Usage examples
--------------
# Full feature stats on two datasets:
python eval_sae.py \\
    --sae_path out/checkpoints/<run>/final_sparse_autoencoder_*.pt \\
    --model dinov2_base \\
    --datasets imagenet caltech101

# Feature stats + zero-shot classification (CLIP only):
python eval_sae.py \\
    --sae_path out/checkpoints/<run>/final_sparse_autoencoder_*.pt \\
    --model clip_vit_b16 \\
    --datasets imagenet oxford_flowers \\
    --tasks feature_stats classification \\
    --sae_modes preserve_cls

# Embedding comparison (base CLIP vs LoRA-merged CLIP):
python eval_sae.py \\
    --sae_path out/checkpoints/<run>/final_sparse_autoencoder_*.pt \\
    --model clip_vit_b16 \\
    --compare_model_b clip_vit_b16 \\
    --compare_weights_b /path/to/lora_merged.pt \\
    --datasets imagenet \\
    --tasks compare

# Regenerate plots from saved artifacts (no re-eval):
python eval_sae.py \\
    --sae_path out/checkpoints/<run>/final_sparse_autoencoder_*.pt \\
    --model dinov2_base \\
    --datasets imagenet \\
    --replot
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

_root = str(Path(__file__).resolve().parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import torch

from src.models.registry import get_backbone, load_model_cfg
from src.data.dataset_registry import get_dataset, get_classnames, get_label_key, load_registry
from src.eval.feature_stats import compute_feature_stats
from src.eval.monosemanticity import compute_monosemanticity
from src.eval.classification import (
    classification_suite, linear_probe_accuracy, feature_masking_accuracy,
)
from src.eval.compare import (
    compare_representations, collect_embeddings_and_activations, embedding_monosemanticity
)
from src.eval import viz
from src.sae_training.loaders import load_sae


# ── I/O ───────────────────────────────────────────────────────────────────────

def save_artifacts(artifacts: dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(artifacts["max_activating_image_indices"],
               os.path.join(save_dir, "max_activating_image_indices.pt"))
    torch.save(artifacts["max_activating_image_values"],
               os.path.join(save_dir, "max_activating_image_values.pt"))
    torch.save(artifacts["sae_mean_acts"],
               os.path.join(save_dir, "sae_mean_acts.pt"))
    torch.save(artifacts["sae_sparsity"],
               os.path.join(save_dir, "sae_sparsity.pt"))
    np.save(os.path.join(save_dir, "cls_sae_cnt.npy"),    artifacts["cls_sae_cnt"])
    np.save(os.path.join(save_dir, "selectivity.npy"),    artifacts.get("selectivity",   np.array([])))
    np.save(os.path.join(save_dir, "top1_fraction.npy"),  artifacts.get("top1_fraction", np.array([])))
    np.save(os.path.join(save_dir, "label_entropy_norm.npy"),
            artifacts.get("label_entropy_norm", np.array([])))
    print(f"  Artifacts → {save_dir}/")


def load_saved_artifacts(save_dir: str) -> dict:
    def _t(name): return torch.load(os.path.join(save_dir, name), map_location="cpu")
    def _n(name):
        p = os.path.join(save_dir, name)
        return np.load(p) if os.path.exists(p) else np.array([])
    return {
        "max_activating_image_indices": _t("max_activating_image_indices.pt"),
        "max_activating_image_values":  _t("max_activating_image_values.pt"),
        "sae_mean_acts":               _t("sae_mean_acts.pt"),
        "sae_sparsity":                _t("sae_sparsity.pt"),
        "cls_sae_cnt":                 _n("cls_sae_cnt.npy"),
        "selectivity":                 _n("selectivity.npy"),
        "top1_fraction":               _n("top1_fraction.npy"),
        "label_entropy_norm":          _n("label_entropy_norm.npy"),
    }


def _scalar(v):
    """Return True if v is JSON-serialisable as a scalar."""
    return isinstance(v, (int, float, str, bool))


def save_metrics(metrics: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({k: v for k, v in metrics.items() if _scalar(v)}, f, indent=2)


def print_feature_stats(ds_name: str, metrics: dict, n_feat: int):
    print(f"\n{'='*65}")
    print(f"FEATURE STATS — {ds_name}")
    print(f"{'='*65}")
    print(f"  Images           : {metrics.get('n_images', '?'):,}")
    print(f"  L0               : {metrics.get('l0', 0):.2f}")
    print(f"  Recon MSE        : {metrics.get('recon_mse', 0):.5f}")
    print(f"  Expl. Variance   : {metrics.get('explained_variance', 0):.4f}")
    print(f"  Dead             : {metrics.get('dead_pct', 0):.1f}%  ({metrics.get('n_dead',0):,}/{n_feat:,})")
    print(f"  Alive            : {metrics.get('alive_pct', 0):.1f}%  ({metrics.get('n_alive',0):,})")
    print(f"  Noisy (top-5%)   : {metrics.get('noisy_pct', 0):.1f}%  ({metrics.get('n_noisy',0):,})")
    print(f"  Usable           : {metrics.get('usable_pct', 0):.1f}%  ({metrics.get('n_usable',0):,})")
    print(f"  Selectivity      : {metrics.get('selectivity_mean',0):.4f}  "
          f"({metrics.get('frac_sel_gt_0.5',0)*100:.1f}% > 0.5)")
    print(f"  Top-1 fraction   : {metrics.get('top1_fraction_mean',0):.4f}")
    print(f"  LabelH norm(wt)  : {metrics.get('label_entropy_weighted_norm',0):.4f}  "
          f"(0=mono, 1=poly)")
    print(f"{'='*65}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified SAE evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core
    parser.add_argument("--sae_path",   required=True)
    parser.add_argument("--model",      required=True,
                        help="Model config name, e.g. clip_vit_b16 / dinov2_base / align_base")
    parser.add_argument("--datasets",   nargs="+", default=["imagenet"])
    parser.add_argument("--layer",      type=int, default=-1)
    parser.add_argument("--tasks",      nargs="+",
                        default=["feature_stats"],
                        choices=["feature_stats", "classification", "linear_probe",
                                 "feature_masking", "compare", "embedding_mono"],
                        help="Which eval tasks to run")

    # feature_stats
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--max_images",  type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--replot",      action="store_true",
                        help="Skip eval; reload saved artifacts and regenerate plots only")

    # classification (CLIP only)
    parser.add_argument("--sae_modes",   nargs="+",
                        default=["preserve_cls"],
                        choices=["preserve_cls", "original", "blend"],
                        help="SAE reconstruction modes for classification task")
    parser.add_argument("--blend_alpha", type=float, default=0.2)

    # linear_probe
    parser.add_argument("--probe_train_split", default="train",
                        help="Dataset split to use as linear probe training set")
    parser.add_argument("--probe_test_split",  default="validation",
                        help="Dataset split to use as linear probe test set")
    parser.add_argument("--probe_max_iter",    type=int, default=1000,
                        help="Max iterations for SGDClassifier in linear probe")

    # feature_masking
    parser.add_argument("--masking_topk",   nargs="+", type=int,
                        default=[1, 2, 5, 10, 50, 100, 500, 1000],
                        help="Top-K values to use in feature masking ablation")
    parser.add_argument("--masking_batch_size", type=int, default=32)

    # compare
    parser.add_argument("--compare_model_b",    default=None,
                        help="Config name for backbone B (default: same as --model)")
    parser.add_argument("--compare_weights_b",  default=None,
                        help="Optional: path to merged state-dict to load into backbone B")

    # embedding_mono
    parser.add_argument("--embedding_mono_k",   type=int, default=20,
                        help="Top-K neurons to report for embedding monosemanticity")
    parser.add_argument("--embedding_mono_max",  type=int, default=50_000,
                        help="Max images to use for embedding monosemanticity (memory)")

    # Output / misc
    parser.add_argument("--out_root",   default="out")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--clipood_checkpoint", default=None,
                        help="Path to CLIPood fine-tuned state dict (.pt). "
                             "Requires --model clipood_vit_b16.")

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU.")
        device = "cpu"

    tasks = set(args.tasks)

    # ── Load SAE ──────────────────────────────────────────────────────────────
    print(f"[INFO] Loading SAE from {args.sae_path}")
    sae, sae_cfg = load_sae(args.sae_path, device)
    n_feat = sae.W_enc.shape[1]
    print(f"[INFO] SAE: d_in={sae_cfg.d_in}, d_sae={n_feat}, layer={sae_cfg.block_layer}")

    model_cfg = load_model_cfg(args.model)
    run_tag   = Path(args.sae_path).parent.name
    registry  = load_registry()

    # ── Load backbone A ───────────────────────────────────────────────────────
    if not args.replot:
        print(f"[INFO] Loading backbone '{args.model}'...")
        backbone = get_backbone(args.model, device=device)
        if args.clipood_checkpoint:
            backbone.model_cfg["clipood_checkpoint_path"] = args.clipood_checkpoint
        backbone = backbone.load()

    # ── Load backbone B (for compare task) ───────────────────────────────────
    backbone_b = None
    if "compare" in tasks and not args.replot:
        b_name = args.compare_model_b or args.model
        print(f"[INFO] Loading backbone B '{b_name}'...")
        backbone_b = get_backbone(b_name, device=device).load()
        if args.compare_weights_b:
            print(f"[INFO] Loading weights for backbone B from {args.compare_weights_b}")
            sd = torch.load(args.compare_weights_b, map_location=device, weights_only=False)
            backbone_b.model.load_state_dict(sd, strict=False)

    # ── Per-dataset loop ──────────────────────────────────────────────────────
    all_metrics = {}

    for ds_name in args.datasets:
        print(f"\n{'─'*60}")
        print(f"DATASET: {ds_name}")
        print(f"{'─'*60}")

        feat_dir = os.path.join(args.out_root, "feature_data",
                                model_cfg["model_id"].replace("/", "_"), ds_name)
        out_dir  = os.path.join(args.out_root, "eval", run_tag, ds_name)
        os.makedirs(feat_dir, exist_ok=True)
        os.makedirs(out_dir,  exist_ok=True)

        # Classnames
        try:
            ds_probe   = get_dataset(ds_name, registry=registry, max_samples=10)
            classnames = get_classnames(ds_name, dataset=ds_probe, registry=registry)
        except Exception as e:
            print(f"  [WARN] classnames: {e}")
            classnames = [str(i) for i in range(registry[ds_name].get("n_classes", 1000))]
        n_classes = len(classnames)

        ds_metrics = {}
        dataset    = None   # populated lazily; shared across tasks within this dataset

        def _ensure_dataset():
            nonlocal dataset
            if dataset is None:
                dataset = get_dataset(ds_name, registry=registry, max_samples=args.max_images)
            return dataset

        # ── feature_stats ─────────────────────────────────────────────────────
        if "feature_stats" in tasks:
            if args.replot:
                print(f"  [replot] Loading artifacts from {feat_dir}/")
                artifacts = load_saved_artifacts(feat_dir)
                mono = compute_monosemanticity(
                    artifacts["cls_sae_cnt"],
                    artifacts["sae_mean_acts"].numpy(),
                    artifacts["sae_sparsity"].numpy(),
                )
                artifacts.update(mono)
                mpath = os.path.join(out_dir, "metrics.json")
                fm = json.load(open(mpath)) if os.path.exists(mpath) else {}
                fm.update({k: v for k, v in mono.items() if _scalar(v)})
            else:
                dataset   = get_dataset(ds_name, registry=registry,
                                        max_samples=args.max_images)
                label_key = get_label_key(ds_name, dataset, registry=registry)
                print(f"  Dataset: {len(dataset):,} images, label_key='{label_key}'")

                artifacts = compute_feature_stats(
                    backbone=backbone, sae=sae, dataset=dataset,
                    label_key=label_key, n_classes=n_classes,
                    layer=args.layer, batch_size=args.batch_size,
                    device=device, max_images=args.max_images,
                    num_workers=args.num_workers,
                )
                mono = compute_monosemanticity(
                    artifacts["cls_sae_cnt"],
                    artifacts["sae_mean_acts"].numpy(),
                    artifacts["sae_sparsity"].numpy(),
                )
                artifacts.update(mono)
                artifacts["n_images"] = artifacts["metrics"]["n_images"]

                fm = {**artifacts["metrics"], **{k: v for k, v in mono.items() if _scalar(v)}}
                save_artifacts(artifacts, feat_dir)
                save_metrics(fm, os.path.join(out_dir, "metrics.json"))

            print_feature_stats(ds_name, fm, n_feat)
            ds_metrics.update({k: v for k, v in fm.items() if _scalar(v)})

            # Plots
            model_tag = model_cfg["model_id"].split("/")[-1]
            _ensure_dataset()
            viz.plot_top_activating_grid(
                dataset, artifacts, classnames, out_dir,
                model_tag=model_tag, dataset_tag=ds_name)
            viz.plot_monosemanticity_histograms(
                artifacts, out_dir, model_tag=model_tag, dataset_tag=ds_name)
            viz.plot_class_feature_heatmap(artifacts, classnames, out_dir)
            viz.plot_mean_acts_and_sparsity(
                artifacts, out_dir, tag=f"{model_tag} on {ds_name}")

        # ── classification ────────────────────────────────────────────────────
        if "classification" in tasks:
            if args.replot:
                print("  [skip] classification does not support --replot")
            elif model_cfg.get("family") != "clip":
                print(f"  [skip] classification requires CLIP backbone, got {model_cfg.get('family')}")
            else:
                _ensure_dataset()
                label_key = get_label_key(ds_name, dataset, registry=registry)

                print(f"  Running classification suite (modes: {args.sae_modes})...")
                clf_results = classification_suite(
                    backbone=backbone, dataset=dataset,
                    classnames=classnames, label_key=label_key,
                    sae=sae, sae_modes=args.sae_modes,
                    blend_alpha=args.blend_alpha, layer=args.layer,
                    batch_size=args.batch_size, device=device,
                    num_workers=args.num_workers,
                )
                print(f"  Classification results:")
                for mode_name, res in clf_results.items():
                    drop = res.get("accuracy_drop_pct", 0.0)
                    print(f"    {mode_name:<22}: {res['accuracy']:.2f}%"
                          + (f"  (drop: {drop:.1f}%)" if drop else ""))
                    ds_metrics[f"clf_{mode_name}_acc"] = res["accuracy"]
                    if "recon_mse" in res:
                        ds_metrics[f"clf_{mode_name}_recon_mse"] = res["recon_mse"]

                save_metrics({"classification": clf_results},
                             os.path.join(out_dir, "classification.json"))

        # ── linear_probe ─────────────────────────────────────────────────────
        if "linear_probe" in tasks:
            if args.replot:
                print("  [skip] linear_probe does not support --replot")
            else:
                print(f"  Running linear probe "
                      f"(train={args.probe_train_split}, test={args.probe_test_split})...")
                try:
                    train_ds = get_dataset(ds_name, registry=registry,
                                           max_samples=args.max_images,
                                           split=args.probe_train_split)
                    test_ds  = get_dataset(ds_name, registry=registry,
                                           split=args.probe_test_split)
                    lp_label_key = get_label_key(ds_name, train_ds, registry=registry)
                    lp_results = linear_probe_accuracy(
                        backbone=backbone, train_dataset=train_ds, test_dataset=test_ds,
                        sae=sae, label_key=lp_label_key, layer=args.layer,
                        batch_size=args.batch_size, device=device,
                        num_workers=args.num_workers, probe_max_iter=args.probe_max_iter,
                        also_clip_features=(model_cfg.get("family") == "clip"),
                    )
                    for subset_name, res in lp_results.items():
                        print(f"    {subset_name:<20}: train={res['train_acc']:.2f}%  "
                              f"test={res['test_acc']:.2f}%  d={res['feature_dim']}")
                        ds_metrics[f"probe_{subset_name}_test_acc"]  = res["test_acc"]
                        ds_metrics[f"probe_{subset_name}_train_acc"] = res["train_acc"]
                    save_metrics({"linear_probe": lp_results},
                                 os.path.join(out_dir, "linear_probe.json"))
                except Exception as e:
                    print(f"  [WARN] linear_probe failed: {e}")

        # ── feature_masking ───────────────────────────────────────────────────
        if "feature_masking" in tasks:
            if args.replot:
                print("  [skip] feature_masking does not support --replot")
            elif model_cfg.get("family") != "clip":
                print(f"  [skip] feature_masking requires CLIP backbone")
            else:
                # Load cls_sae_cnt from already-computed artifacts or disk
                if "feature_stats" in tasks and "cls_sae_cnt" in artifacts:
                    cnt = artifacts["cls_sae_cnt"]
                else:
                    cnt_path = os.path.join(feat_dir, "cls_sae_cnt.npy")
                    if not os.path.exists(cnt_path):
                        print(f"  [skip] feature_masking: cls_sae_cnt.npy not found "
                              f"at {cnt_path}. Run feature_stats first.")
                        cnt = None
                    else:
                        cnt = np.load(cnt_path)

                if cnt is not None:
                    _ensure_dataset()
                    fm_label_key = get_label_key(ds_name, dataset, registry=registry)
                    print(f"  Running feature masking ablation "
                          f"(topk={args.masking_topk})...")
                    fm_results = feature_masking_accuracy(
                        backbone=backbone, sae=sae, dataset=dataset,
                        classnames=classnames, cls_sae_cnt=cnt,
                        label_key=fm_label_key, topk_list=args.masking_topk,
                        layer=args.layer, batch_size=args.masking_batch_size,
                        device=device, num_workers=args.num_workers,
                    )
                    # Summarise: mean accuracy per condition across classes
                    cond_accs = {}
                    for cls_res in fm_results.values():
                        for cond, acc in cls_res.items():
                            cond_accs.setdefault(cond, []).append(acc)
                    print("  Feature masking (mean accuracy across classes):")
                    for cond, accs in sorted(cond_accs.items()):
                        mean_acc = sum(accs) / len(accs)
                        print(f"    {cond:<15}: {mean_acc:.2f}%")
                        ds_metrics[f"mask_{cond}_mean_acc"] = mean_acc
                    save_metrics({"feature_masking": {
                        str(k): v for k, v in fm_results.items()
                    }}, os.path.join(out_dir, "feature_masking.json"))

        # ── compare ──────────────────────────────────────────────────────────
        if "compare" in tasks and backbone_b is not None:
            if args.replot:
                print("  [skip] compare does not support --replot")
            else:
                _ensure_dataset()
                print("  Running embedding comparison (backbone A vs B)...")
                stats = compare_representations(
                    backbone_a=backbone, backbone_b=backbone_b,
                    sae_a=sae, sae_b=sae,
                    dataset=dataset, dataset_name=ds_name,
                    layer=args.layer, batch_size=args.batch_size,
                    device=device, num_workers=args.num_workers,
                )
                print(f"  SAE cosine:  {stats.sae_cosine_mean:.4f} ± {stats.sae_cosine_std:.4f}")
                print(f"  CLIP cosine: {stats.clip_cosine_mean:.4f} ± {stats.clip_cosine_std:.4f}")
                print(f"  SAE L2:      {stats.sae_l2_mean:.4f} ± {stats.sae_l2_std:.4f}")

                compare_dict = asdict(stats)
                save_metrics(compare_dict, os.path.join(out_dir, "compare.json"))
                ds_metrics.update({f"compare_{k}": v for k, v in compare_dict.items()
                                   if _scalar(v)})

        # ── embedding_mono ────────────────────────────────────────────────────
        if "embedding_mono" in tasks:
            if args.replot:
                print("  [skip] embedding_mono does not support --replot")
            else:
                _ensure_dataset()
                print(f"  Collecting embeddings for embedding_mono "
                      f"(max={args.embedding_mono_max})...")
                embeds, acts = collect_embeddings_and_activations(
                    backbone=backbone, sae=sae, dataset=dataset,
                    layer=args.layer, batch_size=args.batch_size,
                    device=device, num_workers=args.num_workers,
                    max_samples=args.embedding_mono_max,
                )

                print(f"  Computing weighted pairwise cosine mono "
                      f"(N={embeds.shape[0]}, H={acts.shape[1]})...")
                ms = embedding_monosemanticity(embeds, acts, device=device)

                valid   = ms[~ms.isnan()]
                top_idx = ms.nan_to_num(-1e9).topk(args.embedding_mono_k).indices.tolist()
                print(f"  MS: mean={valid.mean():.4f}, median={valid.median():.4f}, "
                      f"valid={len(valid)}/{len(ms)}")
                print(f"  Top-{args.embedding_mono_k} neurons: {top_idx}")

                mono_path = os.path.join(feat_dir, "embedding_mono_scores.pt")
                torch.save(ms, mono_path)

                em_metrics = {
                    "embed_mono_mean":   float(valid.mean())   if len(valid) > 0 else 0.0,
                    "embed_mono_median": float(valid.median()) if len(valid) > 0 else 0.0,
                    "embed_mono_top_neurons": top_idx,
                }
                save_metrics(em_metrics, os.path.join(out_dir, "embedding_mono.json"))
                ds_metrics.update({k: v for k, v in em_metrics.items() if _scalar(v)})

        all_metrics[ds_name] = ds_metrics

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_path = os.path.join(args.out_root, "eval", run_tag, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if _scalar(vv)}
                   for k, v in all_metrics.items()}, f, indent=2)
    print(f"\n[INFO] Summary → {summary_path}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
