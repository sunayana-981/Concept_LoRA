import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path so "src" and "tasks" packages resolve.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import matplotlib
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image

from tasks.compute_sae_feature_data import main as compute_sae_feature_data_main
from tasks.utils import DATASET_INFO, detect_label_key, get_classnames

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


IMAGE_KEY_CANDIDATES = ("image", "jpg", "img", "pixel_values")


def resolve_dataset(dataset_name: str, seed: int):
    dataset = load_dataset(**DATASET_INFO[dataset_name])
    if isinstance(dataset, dict):
        split = DATASET_INFO[dataset_name].get("split", "train")
        dataset = dataset[split]
    return dataset.shuffle(seed=seed)


def detect_image_key(dataset) -> str:
    for key in IMAGE_KEY_CANDIDATES:
        if key in dataset.features:
            return key

    sample = dataset[0]
    for key, value in sample.items():
        if isinstance(value, Image.Image):
            return key
        if isinstance(value, (np.ndarray, torch.Tensor)):
            arr = np.asarray(value)
            if arr.ndim in (2, 3):
                return key

    raise ValueError(
        f"Could not detect an image field in dataset. Available keys: {list(dataset.features.keys())}"
    )


def to_pil_image(image_obj) -> Image.Image:
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")

    if isinstance(image_obj, torch.Tensor):
        arr = image_obj.detach().cpu().numpy()
    else:
        arr = np.asarray(image_obj)

    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.0:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        else:
            arr = np.clip(arr, 0.0, 255.0)
        arr = arr.astype(np.uint8)

    return Image.fromarray(arr).convert("RGB")


def compute_feature_directory(
    root_dir: Path, feature_save_name: str, sae_path: Path, vit_type: str, dataset_name: str
) -> Path:
    sae_run_name = sae_path.resolve().parent.name
    return root_dir / feature_save_name / f"sae_{sae_run_name}" / vit_type / dataset_name


def compute_output_directory(
    root_dir: Path, output_root: str, sae_path: Path, vit_type: str, dataset_name: str
) -> Path:
    sae_run_name = sae_path.resolve().parent.name
    return root_dir / output_root / f"sae_{sae_run_name}" / vit_type / dataset_name


def save_top_neuron_summary_plot(
    top_indices: torch.Tensor,
    top_mean_acts: torch.Tensor,
    top_sparsity: torch.Tensor,
    output_path: Path,
) -> None:
    top_indices_np = top_indices.cpu().numpy()
    top_mean_np = top_mean_acts.cpu().numpy()
    top_sparsity_np = top_sparsity.cpu().numpy()
    x = np.arange(len(top_indices_np))
    labels = [str(int(idx)) for idx in top_indices_np]

    fig, (ax_mean, ax_sparse) = plt.subplots(
        2, 1, figsize=(max(12, len(labels) * 0.5), 9), constrained_layout=True
    )

    ax_mean.bar(x, top_mean_np, color="#1f77b4")
    ax_mean.set_title("Top Neurons by Mean Activation")
    ax_mean.set_ylabel("Mean Activation")
    ax_mean.set_xticks(x)
    ax_mean.set_xticklabels(labels, rotation=60, ha="right")

    ax_sparse.bar(x, top_sparsity_np, color="#ff7f0e")
    ax_sparse.set_title("Activation Frequency of Top Mean-Activation Neurons")
    ax_sparse.set_ylabel("Activation Frequency")
    ax_sparse.set_xlabel("Neuron Index")
    ax_sparse.set_xticks(x)
    ax_sparse.set_xticklabels(labels, rotation=60, ha="right")

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def safe_to_int(value) -> Optional[int]:
    if isinstance(value, torch.Tensor):
        value = value.item()
    if isinstance(value, (np.integer, int)):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def format_label(label_value, classnames: Optional[list[str]]) -> str:
    label_int = safe_to_int(label_value)
    if label_int is None:
        return "label: ?"
    if classnames and 0 <= label_int < len(classnames):
        return f"label: {label_int} ({classnames[label_int]})"
    return f"label: {label_int}"


def save_neuron_image_grid(
    dataset,
    image_key: str,
    label_key: Optional[str],
    classnames: Optional[list[str]],
    neuron_idx: int,
    image_indices: torch.Tensor,
    image_values: Optional[torch.Tensor],
    images_per_grid: int,
    output_path: Path,
    image_size: int = 224,
) -> None:
    image_indices = image_indices[:images_per_grid].to(torch.long).cpu()
    if image_values is not None:
        image_values = image_values[:images_per_grid].float().cpu()

    n_images = len(image_indices)
    if n_images == 0:
        return

    n_cols = min(5, n_images)
    n_rows = int(math.ceil(n_images / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.6 * n_cols, 3.9 * n_rows), constrained_layout=True
    )
    axes = np.array(axes).reshape(-1)

    for panel_idx, axis in enumerate(axes):
        if panel_idx >= n_images:
            axis.axis("off")
            continue

        dataset_index = int(image_indices[panel_idx].item())
        sample = dataset[dataset_index]
        image = to_pil_image(sample[image_key]).resize((image_size, image_size))
        axis.imshow(image)
        axis.axis("off")

        title_lines = [f"rank {panel_idx + 1} | ds idx {dataset_index}"]
        if image_values is not None:
            title_lines.append(f"act={float(image_values[panel_idx]):.4f}")
        if label_key and label_key in sample:
            title_lines.append(format_label(sample[label_key], classnames))
        axis.set_title("\n".join(title_lines), fontsize=8)

    fig.suptitle(f"Neuron {neuron_idx} Top Activating Images", fontsize=13)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_summary_csv(
    output_path: Path,
    top_indices: torch.Tensor,
    top_mean_acts: torch.Tensor,
    top_sparsity: torch.Tensor,
):
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["rank", "neuron_idx", "mean_activation", "activation_frequency"])
        for rank, (idx, mean_act, sparsity) in enumerate(
            zip(top_indices.tolist(), top_mean_acts.tolist(), top_sparsity.tolist()),
            start=1,
        ):
            writer.writerow([rank, int(idx), float(mean_act), float(sparsity)])


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export top-activating neuron plots and top-activating image grids "
            "for a trained SAE checkpoint."
        )
    )
    parser.add_argument("--sae_path", type=str, required=True, help="Path to SAE checkpoint (.pt)")
    parser.add_argument("--dataset_name", type=str, required=True, choices=sorted(DATASET_INFO.keys()))
    parser.add_argument("--vit_type", type=str, default="base", choices=["base", "maple"])
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--feature_save_name", type=str, default="out/feature_data")
    parser.add_argument("--output_root", type=str, default="out/top_activations")
    parser.add_argument("--backbone", type=str, default="openai/clip-vit-base-patch16")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_top_images_per_neuron", type=int, default=25)
    parser.add_argument("--num_neurons_to_plot", type=int, default=20)
    parser.add_argument("--images_per_neuron_grid", type=int, default=10)
    parser.add_argument("--skip_feature_compute", action="store_true")
    args = parser.parse_args()

    sae_path = Path(args.sae_path).expanduser().resolve()
    root_dir = Path(args.root_dir).expanduser().resolve()

    if not sae_path.exists():
        raise FileNotFoundError(f"SAE checkpoint does not exist: {sae_path}")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    feature_dir = compute_feature_directory(
        root_dir, args.feature_save_name, sae_path, args.vit_type, args.dataset_name
    )

    n_top_images = max(args.num_top_images_per_neuron, args.images_per_neuron_grid)
    if not args.skip_feature_compute:
        print(f"[INFO] Computing SAE feature data at: {feature_dir}")
        compute_sae_feature_data_main(
            sae_path=str(sae_path),
            vit_type=args.vit_type,
            device=args.device,
            dataset_name=args.dataset_name,
            root_dir=str(root_dir),
            save_name=args.feature_save_name,
            backbone=args.backbone,
            number_of_max_activating_images=n_top_images,
            seed=args.seed,
            batch_size=args.batch_size,
            model_path=args.model_path,
            config_path=args.config_path,
        )

    required_files = [
        feature_dir / "sae_mean_acts.pt",
        feature_dir / "sae_sparsity.pt",
        feature_dir / "max_activating_image_indices.pt",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required feature-data files:\n" + "\n".join(missing)
        )

    mean_acts = torch.load(feature_dir / "sae_mean_acts.pt", map_location="cpu").float()
    sparsity = torch.load(feature_dir / "sae_sparsity.pt", map_location="cpu").float()
    max_image_indices = torch.load(
        feature_dir / "max_activating_image_indices.pt", map_location="cpu"
    ).to(torch.long)
    max_image_values = None
    if (feature_dir / "max_activating_image_values.pt").exists():
        max_image_values = torch.load(
            feature_dir / "max_activating_image_values.pt", map_location="cpu"
        ).float()

    mean_acts = torch.nan_to_num(mean_acts, nan=0.0, posinf=0.0, neginf=0.0)
    sparsity = torch.nan_to_num(sparsity, nan=0.0, posinf=0.0, neginf=0.0)

    top_k = min(args.num_neurons_to_plot, mean_acts.numel())
    if top_k <= 0:
        raise ValueError("No neurons available to plot.")

    top_mean_acts, top_indices = torch.topk(mean_acts, k=top_k)
    top_sparsity = sparsity[top_indices]

    output_dir = compute_output_directory(
        root_dir, args.output_root, sae_path, args.vit_type, args.dataset_name
    )
    top_images_dir = output_dir / "top_images"
    top_images_dir.mkdir(parents=True, exist_ok=True)

    save_top_neuron_summary_plot(
        top_indices, top_mean_acts, top_sparsity, output_dir / "top_neurons_summary.png"
    )
    save_summary_csv(output_dir / "top_neurons_summary.csv", top_indices, top_mean_acts, top_sparsity)

    dataset = resolve_dataset(args.dataset_name, seed=args.seed)
    image_key = detect_image_key(dataset)
    try:
        label_key = detect_label_key(dataset)
    except ValueError:
        label_key = None
    try:
        classnames = get_classnames(args.dataset_name, dataset)
    except ValueError:
        classnames = None

    for neuron_idx in top_indices.tolist():
        image_values = max_image_values[neuron_idx] if max_image_values is not None else None
        save_neuron_image_grid(
            dataset=dataset,
            image_key=image_key,
            label_key=label_key,
            classnames=classnames,
            neuron_idx=int(neuron_idx),
            image_indices=max_image_indices[neuron_idx],
            image_values=image_values,
            images_per_grid=args.images_per_neuron_grid,
            output_path=top_images_dir / f"neuron_{int(neuron_idx)}.png",
        )

    print(f"[INFO] Top-activation artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
