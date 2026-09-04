#!/usr/bin/env python3
"""Run the controlled G-SAE / FT-SAE / scratch-SAE initialization ablation.

Definitions enforced by this runner:

* ``gsae``: the frozen ImageNet SAE checkpoint (never retrained here);
* ``ftsae``: that exact G-SAE initializes an all-parameter target-domain run
  (``protect_frac=0``);
* ``scratchsae``: random initialization with every non-initialization factor
  matched to ``ftsae``.

The default headline sweep uses three domain tiers and three SAE-training
seeds. Evaluation/probe seeds are recorded separately and remain fixed; they
are not treated as SAE-training replicates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.sae_training.provenance import (  # noqa: E402
    PROVENANCE_SCHEMA_VERSION,
    infer_activation_vectors_per_example,
    sha256_file,
)


TRAIN_DATASET_KEYS = {
    "pets": "oxford_pets",
    "pathmnist": "medmnist",
}
DEFAULT_DATASETS = ["caltech101", "dtd", "pathmnist"]
DEFAULT_DOMAIN_TIERS = ["near", "mid", "far"]
TRAINED_ARMS = ("ftsae", "scratchsae")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument(
        "--domain_tiers",
        nargs="+",
        default=DEFAULT_DOMAIN_TIERS,
        help="One near/mid/far (or custom) tier per --datasets entry.",
    )
    p.add_argument("--arms", nargs="+", choices=TRAINED_ARMS, default=list(TRAINED_ARMS))
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
        help="SAE initialization/data-order seeds (training variance).",
    )
    p.add_argument(
        "--evaluation_seed",
        type=int,
        default=2026,
        help="Fixed seed for evaluation-example selection; not a training replicate.",
    )
    p.add_argument(
        "--probe_seed",
        type=int,
        default=2026,
        help="Fixed downstream probe/steering seed; not a training replicate.",
    )
    p.add_argument("--dataset_registry", default="configs/rebuttal_datasets.json")
    p.add_argument("--gsae_path", default="data/sae_weight/base/out.pt")
    p.add_argument("--checkpoint_root", default="out/checkpoints/init_ablation")
    p.add_argument("--log_root", default="out/logs/init_ablation")
    p.add_argument("--registry", default="out/rebuttal/sae_registry.json")
    p.add_argument(
        "--manifest",
        default="out/rebuttal/sae_initialization_ablation_manifest.json",
    )
    p.add_argument("--training_examples", type=int, default=100_000)
    p.add_argument("--block_layer", type=int, default=-2)
    p.add_argument("--expansion_factor", type=int, default=64)
    p.add_argument("--l1_coefficient", type=float, default=0.00008)
    p.add_argument("--lr", type=float, default=0.0004)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr_warm_up_steps", type=int, default=500)
    p.add_argument("--model_name", default="openai/clip-vit-base-patch16")
    p.add_argument("--clip_dim", type=int, default=768)
    p.add_argument("--image_width", type=int, default=224)
    p.add_argument("--image_height", type=int, default=224)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gated_sae", action="store_true")
    p.add_argument("--log_to_wandb", action="store_true")
    p.add_argument("--wandb_project", default="sae_initialization_ablation")
    p.add_argument("--n_checkpoints", type=int, default=1)
    p.add_argument(
        "--print_only",
        action="store_true",
        help="Print the fully resolved plan and commands; do not read/write outputs.",
    )
    p.add_argument(
        "--rerun",
        action="store_true",
        help="Train a fresh run even when a completed checkpoint already exists.",
    )
    p.add_argument("--python", default=sys.executable)
    args = p.parse_args(argv)

    if len(args.datasets) != len(args.domain_tiers):
        p.error("--domain_tiers must have exactly one entry per --datasets entry")
    if len(set(args.datasets)) != len(args.datasets):
        p.error("--datasets contains duplicates")
    if len(set(args.seeds)) != len(args.seeds):
        p.error("--seeds contains duplicates")
    if args.training_examples <= 0:
        p.error("--training_examples must be positive")
    return args


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _project_root / path


def load_dataset_registry(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def inventory_path_size_sha256(root: Path) -> tuple[str, int, int]:
    """Hash a deterministic relative-path/size inventory without reading images."""
    digest = hashlib.sha256()
    n_files = 0
    total_bytes = 0
    if root.is_file():
        files = [root]
        base = root.parent
    else:
        files = sorted(path for path in root.rglob("*") if path.is_file())
        base = root
    for path in files:
        relative = path.relative_to(base).as_posix()
        size = path.stat().st_size
        digest.update(f"{relative}\0{size}\n".encode("utf-8"))
        n_files += 1
        total_bytes += size
    return digest.hexdigest(), n_files, total_bytes


def build_target_data_identity(
    dataset: str,
    registry_entry: dict[str, Any],
    *,
    compute_inventory: bool,
) -> dict[str, Any]:
    train_key = TRAIN_DATASET_KEYS.get(dataset, dataset)
    recipe = (
        f"tasks.utils.DATASET_INFO[{train_key!r}], split='train'; "
        f"rebuttal_registry_entry_sha256={canonical_json_sha256(registry_entry)}"
    )
    root_value = (
        registry_entry.get("imagefolder_root")
        if registry_entry.get("type") == "medmnist_npz"
        else registry_entry.get("path")
    )
    identity = {
        "recipe_identifier": recipe,
        "split": "train",
        "inventory_method": "relative_path_and_file_size_sha256",
        "inventory_root": root_value,
        "inventory_sha256": None,
        "n_files": None,
        "total_bytes": None,
    }
    if not compute_inventory:
        identity["inventory_status"] = "not_computed_print_only"
        return identity
    if not root_value or not Path(root_value).exists():
        identity["inventory_status"] = "unavailable_recipe_recorded"
        return identity
    digest, n_files, total_bytes = inventory_path_size_sha256(Path(root_value))
    identity.update({
        "inventory_status": "computed",
        "inventory_sha256": digest,
        "n_files": n_files,
        "total_bytes": total_bytes,
    })
    return identity


def inspect_checkpoint_gated(path: Path) -> bool:
    """Read the checkpoint config and return its gated/standard architecture."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("cfg", checkpoint.get("config"))
    if isinstance(cfg, dict):
        gated = bool(cfg.get("gated_sae", False))
    else:
        gated = bool(getattr(cfg, "gated_sae", False))
    del checkpoint
    return gated


def validate_checkpoint_arm(
    path: Path,
    *,
    condition: str,
    training_seed: int,
    gsae_sha256: str,
) -> None:
    """Verify an idempotently reused checkpoint belongs to the requested arm."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("experiment_metadata")
    if metadata is None:
        cfg = checkpoint.get("cfg", checkpoint.get("config"))
        metadata = (
            cfg.get("experiment_metadata")
            if isinstance(cfg, dict)
            else getattr(cfg, "experiment_metadata", None)
        )
    if metadata is not None and not isinstance(metadata, dict):
        metadata = vars(metadata)
    del checkpoint
    expected_init = "checkpoint" if condition == "ftsae" else "scratch_random"
    errors = []
    if not metadata:
        errors.append("missing experiment_metadata")
    else:
        if metadata.get("condition") != condition:
            errors.append(
                f"condition={metadata.get('condition')!r}, expected {condition!r}"
            )
        if metadata.get("sae_initialization") != expected_init:
            errors.append(
                f"sae_initialization={metadata.get('sae_initialization')!r}, "
                f"expected {expected_init!r}"
            )
        if int(metadata.get("data_order_seed", -1)) != training_seed:
            errors.append(
                f"data_order_seed={metadata.get('data_order_seed')!r}, "
                f"expected {training_seed}"
            )
        if condition == "ftsae" and (
            metadata.get("initialization_checkpoint_sha256") != gsae_sha256
        ):
            errors.append("G-SAE initialization SHA-256 mismatch")
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors))


def build_command(
    args,
    *,
    dataset: str,
    condition: str,
    training_seed: int,
    lora_checkpoint: Path,
    checkpoint_dir: Path,
    gsae_path: Path,
    target_data_identity: dict[str, Any] | None = None,
) -> list[str]:
    train_key = TRAIN_DATASET_KEYS.get(dataset, dataset)
    command = [
        args.python,
        "tasks/train_sae_lora_clip.py",
        "--model_name",
        args.model_name,
        "--clip_dim",
        str(args.clip_dim),
        "--image_width",
        str(args.image_width),
        "--image_height",
        str(args.image_height),
        "--patch_size",
        str(args.patch_size),
        "--lora_checkpoint_path",
        str(lora_checkpoint),
        "--block_layers",
        str(args.block_layer),
        "--dataset",
        train_key,
        "--target_dataset",
        dataset,
        "--activation_data_role",
        "target",
        "--sae_condition",
        condition,
        "--protect_frac",
        "0",
        "--expansion_factor",
        str(args.expansion_factor),
        "--l1_coefficient",
        str(args.l1_coefficient),
        "--lr",
        str(args.lr),
        "--batch_size",
        str(args.batch_size),
        "--lr_warm_up_steps",
        str(args.lr_warm_up_steps),
        "--training_examples",
        str(args.training_examples),
        "--use_ghost_grads",
        "--checkpoint_path",
        str(checkpoint_dir),
        "--n_checkpoints",
        str(args.n_checkpoints),
        "--seed",
        str(training_seed),
        "--device",
        args.device,
        "--skip_activation_verify",
        "--run_name",
        f"init_ablation_{dataset}_{condition}_seed{training_seed}",
    ]
    if target_data_identity:
        command.extend([
            "--target_data_recipe",
            target_data_identity["recipe_identifier"],
        ])
        inventory_sha = target_data_identity.get("inventory_sha256")
        if inventory_sha:
            command.extend([
                "--target_data_inventory_sha256",
                inventory_sha,
            ])
    if condition == "ftsae":
        command.extend([
            "--sae_initialization",
            "checkpoint",
            "--sae_checkpoint_path",
            str(gsae_path),
        ])
    else:
        command.extend(["--sae_initialization", "scratch"])
    if args.gated_sae:
        command.append("--gated_sae")
    if args.log_to_wandb:
        command.extend(["--log_to_wandb", "--wandb_project", args.wandb_project])
    return command


def final_checkpoints(checkpoint_dir: Path, block_layer: int) -> list[Path]:
    pattern = f"*/final_sparse_autoencoder_*/*_{block_layer}_resid_*.pt"
    return sorted(checkpoint_dir.glob(pattern), key=lambda p: p.stat().st_mtime)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temp_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temp_path, path)


def upsert_registry_record(registry_path: Path, record: dict[str, Any]) -> None:
    records = []
    if registry_path.exists():
        with registry_path.open() as handle:
            records = json.load(handle)
    records = [
        old for old in records
        if not (
            old.get("dataset") == record["dataset"]
            and old.get("condition") == record["condition"]
            and old.get("training_seed") == record.get("training_seed")
        )
    ]
    records.append(record)
    write_json_atomic(registry_path, records)


def expected_cells(args) -> list[dict[str, Any]]:
    cells = []
    for dataset, tier in zip(args.datasets, args.domain_tiers):
        cells.append({
            "dataset": dataset,
            "domain_tier": tier,
            "condition": "gsae",
            "training_seed": None,
        })
        for condition in args.arms:
            for seed in args.seeds:
                cells.append({
                    "dataset": dataset,
                    "domain_tier": tier,
                    "condition": condition,
                    "training_seed": seed,
                })
    return cells


def make_manifest(
    args,
    gsae_path: Path,
    gsae_sha256: str,
    target_data_identities: dict[str, dict[str, Any]],
    adapted_model_sha256: dict[str, str],
) -> dict[str, Any]:
    vectors_per_example = infer_activation_vectors_per_example(
        image_width=args.image_width,
        image_height=args.image_height,
        patch_size=args.patch_size,
        class_token_only=False,
    )
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "experiment": "sae_initialization_ablation",
        "definitions": {
            "gsae": "frozen ImageNet SAE",
            "ftsae": "G-SAE init + target adapted activations + protect_frac=0",
            "scratchsae": "random init + target adapted activations",
        },
        "fixed_factors": {
            "gsae_path": str(gsae_path),
            "gsae_sha256": gsae_sha256,
            "block_layer": args.block_layer,
            "d_in": args.clip_dim,
            "expansion_factor": args.expansion_factor,
            "architecture": "gated" if args.gated_sae else "standard",
            "l1_coefficient": args.l1_coefficient,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "lr_warm_up_steps": args.lr_warm_up_steps,
            "training_examples_requested": args.training_examples,
            "activation_vectors_per_example": vectors_per_example,
            "derived_activation_vector_exposure_requested": (
                args.training_examples * vectors_per_example
            ),
            "legacy_trainer_counter_name": "total_training_tokens",
            "legacy_trainer_counter_unit": "images/examples",
            "evaluation_seed": args.evaluation_seed,
            "probe_seed": args.probe_seed,
        },
        "training_seeds": args.seeds,
        "target_data_identities": target_data_identities,
        "adapted_model_checkpoint_sha256": adapted_model_sha256,
        "expected_cells": expected_cells(args),
        "runs": [],
    }


def register_gsae_controls(
    args,
    manifest,
    dataset_registry,
    registry_path: Path,
    gsae_path: Path,
) -> None:
    fixed = manifest["fixed_factors"]
    for dataset, tier in zip(args.datasets, args.domain_tiers):
        record = {
            "dataset": dataset,
            "domain_tier": tier,
            "vit_type": "lora",
            "condition": "gsae",
            "checkpoint_path": str(gsae_path),
            "frozen": True,
            "sae_initialization": "pretrained_imagenet_control",
            "initialization_checkpoint": str(gsae_path),
            "initialization_checkpoint_sha256": fixed["gsae_sha256"],
            "training_seed": None,
            "evaluation_seed": args.evaluation_seed,
            "probe_seed": args.probe_seed,
            "adapted_model_checkpoint": dataset_registry[dataset]["lora_checkpoint"],
            "adapted_model_checkpoint_sha256": manifest[
                "adapted_model_checkpoint_sha256"
            ][dataset],
            "target_data_identity": manifest["target_data_identities"][dataset],
            "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        upsert_registry_record(registry_path, record)


def stream_subprocess(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=_project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
        return process.wait()


def main(argv=None):
    args = parse_args(argv)
    dataset_registry_path = resolve_path(args.dataset_registry)
    gsae_path = resolve_path(args.gsae_path)
    checkpoint_root = resolve_path(args.checkpoint_root)
    log_root = resolve_path(args.log_root)
    registry_path = resolve_path(args.registry)
    manifest_path = resolve_path(args.manifest)

    if dataset_registry_path.is_file():
        dataset_registry = load_dataset_registry(dataset_registry_path)
    elif args.print_only:
        # Planning must not depend on mounted datasets/checkpoints. Placeholders
        # make every unresolved input explicit in the printed commands.
        dataset_registry = {
            dataset: {
                "type": "recipe",
                "lora_checkpoint": f"<LORA_CHECKPOINT:{dataset}>",
            }
            for dataset in args.datasets
        }
    else:
        raise SystemExit(
            f"[FATAL] dataset registry not found: {dataset_registry_path}"
        )
    missing_datasets = [name for name in args.datasets if name not in dataset_registry]
    if missing_datasets:
        raise SystemExit(
            f"[FATAL] dataset(s) absent from {dataset_registry_path}: {missing_datasets}"
        )

    target_data_identities = {
        dataset: build_target_data_identity(
            dataset,
            dataset_registry[dataset],
            compute_inventory=not args.print_only,
        )
        for dataset in args.datasets
    }

    commands = []
    for dataset in args.datasets:
        lora_value = dataset_registry[dataset].get("lora_checkpoint")
        if not lora_value:
            raise SystemExit(f"[FATAL] no LoRA checkpoint configured for {dataset}")
        lora_path = Path(lora_value)
        for condition in args.arms:
            for seed in args.seeds:
                checkpoint_dir = checkpoint_root / dataset / condition / f"seed{seed}"
                commands.append((
                    dataset,
                    condition,
                    seed,
                    lora_path,
                    checkpoint_dir,
                    build_command(
                        args,
                        dataset=dataset,
                        condition=condition,
                        training_seed=seed,
                        lora_checkpoint=lora_path,
                        checkpoint_dir=checkpoint_dir,
                        gsae_path=gsae_path,
                        target_data_identity=target_data_identities[dataset],
                    ),
                ))

    if args.print_only:
        print("Definitions: gsae=frozen; ftsae=G-SAE init+p=0; scratchsae=random init")
        print(f"Training seeds: {args.seeds}")
        print(
            f"Fixed evaluation_seed={args.evaluation_seed}, "
            f"probe_seed={args.probe_seed}"
        )
        for dataset, condition, seed, _lora, _out, command in commands:
            print(f"\n[{dataset}/{condition}/train-seed-{seed}]")
            print(shlex.join(command))
        return 0

    if not gsae_path.is_file():
        raise SystemExit(f"[FATAL] G-SAE checkpoint not found: {gsae_path}")
    gsae_sha256 = sha256_file(gsae_path)
    gsae_is_gated = inspect_checkpoint_gated(gsae_path)
    if gsae_is_gated != bool(args.gated_sae):
        requested = "gated" if args.gated_sae else "standard"
        observed = "gated" if gsae_is_gated else "standard"
        raise SystemExit(
            f"[FATAL] --gated_sae requests {requested} architecture but "
            f"{gsae_path} is {observed}"
        )
    adapted_model_sha256 = {}
    for dataset, _condition, _seed, lora_path, _out, _command in commands:
        if not lora_path.is_file():
            raise SystemExit(
                f"[FATAL] LoRA checkpoint not found for {dataset}: {lora_path}"
            )
        if dataset not in adapted_model_sha256:
            adapted_model_sha256[dataset] = sha256_file(lora_path)

    manifest = make_manifest(
        args,
        gsae_path.resolve(),
        gsae_sha256,
        target_data_identities,
        adapted_model_sha256,
    )
    write_json_atomic(manifest_path, manifest)
    register_gsae_controls(
        args, manifest, dataset_registry, registry_path, gsae_path.resolve()
    )

    failures = []
    tier_by_dataset = dict(zip(args.datasets, args.domain_tiers))
    for dataset, condition, seed, lora_path, checkpoint_dir, command in commands:
        existing = final_checkpoints(checkpoint_dir, args.block_layer)
        if existing and not args.rerun:
            checkpoint_path = existing[-1].resolve()
            try:
                validate_checkpoint_arm(
                    checkpoint_path,
                    condition=condition,
                    training_seed=seed,
                    gsae_sha256=gsae_sha256,
                )
            except ValueError as exc:
                raise SystemExit(
                    f"[FATAL] refusing to reuse mismatched checkpoint: {exc}. "
                    "Pass --rerun to create a fresh run in a new run-id directory."
                ) from exc
            status = "completed_existing"
            exit_code = 0
            print(f"[SKIP] {dataset}/{condition}/seed{seed}: {checkpoint_path}")
        else:
            started_at = time.time()
            log_path = log_root / dataset / condition / f"seed{seed}.log"
            print(f"[RUN] {dataset}/{condition}/seed{seed}")
            exit_code = stream_subprocess(command, log_path)
            new_checkpoints = [
                path for path in final_checkpoints(checkpoint_dir, args.block_layer)
                if path.stat().st_mtime >= started_at
            ]
            checkpoint_path = new_checkpoints[-1].resolve() if new_checkpoints else None
            status = "completed" if exit_code == 0 and checkpoint_path else "failed"

        run_record = {
            "dataset": dataset,
            "domain_tier": tier_by_dataset[dataset],
            "condition": condition,
            "training_seed": seed,
            "evaluation_seed": args.evaluation_seed,
            "probe_seed": args.probe_seed,
            "status": status,
            "exit_code": exit_code,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "checkpoint_sha256": (
                sha256_file(checkpoint_path) if checkpoint_path else None
            ),
            "sae_initialization": (
                "checkpoint" if condition == "ftsae" else "scratch_random"
            ),
            "initialization_checkpoint": (
                str(gsae_path.resolve()) if condition == "ftsae" else None
            ),
            "initialization_checkpoint_sha256": (
                gsae_sha256 if condition == "ftsae" else None
            ),
            "activation_dataset": TRAIN_DATASET_KEYS.get(dataset, dataset),
            "target_dataset": dataset,
            "activation_data_role": "target",
            "adapted_model_checkpoint": str(lora_path),
            "adapted_model_checkpoint_sha256": adapted_model_sha256[dataset],
            "target_data_identity": target_data_identities[dataset],
            "protect_frac": 0.0,
            "architecture": manifest["fixed_factors"]["architecture"],
            "training_examples": args.training_examples,
            "activation_vectors_per_example": manifest["fixed_factors"][
                "activation_vectors_per_example"
            ],
            "derived_activation_vector_exposure_requested": manifest[
                "fixed_factors"
            ]["derived_activation_vector_exposure_requested"],
            "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        manifest["runs"].append(run_record)
        write_json_atomic(manifest_path, manifest)

        if status.startswith("completed"):
            registry_record = {
                **run_record,
                "vit_type": "lora",
                "layer": args.block_layer,
                "checkpoint_path": str(checkpoint_path),
            }
            upsert_registry_record(registry_path, registry_record)
        else:
            failures.append((dataset, condition, seed))

    if failures:
        print(f"[FAIL] incomplete training cells: {failures}")
        return 1
    print(f"[OK] all {len(commands)} trained cells complete")
    print(f"Manifest: {manifest_path}")
    print(f"Registry: {registry_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
