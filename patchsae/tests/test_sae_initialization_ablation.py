from pathlib import Path

import pytest

from src.sae_training.provenance import (
    build_experiment_metadata,
    build_training_metadata,
    infer_activation_vectors_per_example,
)
from tasks.registry_to_sae_paths import build_sae_paths
from tasks.run_sae_initialization_ablation import (
    build_command,
    expected_cells,
    parse_args,
)


def test_exposure_metadata_names_examples_and_derives_patch_vectors():
    vectors = infer_activation_vectors_per_example(
        image_width=224,
        image_height=224,
        patch_size=16,
        class_token_only=False,
    )
    assert vectors == 197

    metadata = build_training_metadata(
        examples_seen=112,
        training_steps=7,
        requested_examples=100,
        activation_vectors_per_example=vectors,
    )
    assert metadata["counter_unit"] == "images/examples"
    assert metadata["n_training_examples"] == 112
    assert metadata["derived_activation_vector_exposure"] == 112 * 197


def test_controlled_condition_metadata_rejects_mislabeled_scratch_ftsae():
    metadata = build_experiment_metadata(
        condition="scratchsae",
        initialization="scratch_random",
        initialization_checkpoint=None,
        initialization_checkpoint_sha256=None,
        activation_dataset="medmnist",
        target_dataset="pathmnist",
        activation_data_role="target",
        adapted_model_checkpoint="/tmp/lora.pt",
        seed=42,
        block_layer=-2,
        module_name="resid",
        d_in=768,
        expansion_factor=64,
        gated_sae=False,
        training_examples_requested=100,
        activation_vectors_per_example=197,
        protect_frac=0,
    )
    assert metadata["sae_initialization"] == "scratch_random"
    assert metadata["legacy_trainer_counter_unit"] == "images/examples"

    with pytest.raises(ValueError, match="checkpoint initialization"):
        build_experiment_metadata(
            **{
                **{
                    "condition": "ftsae",
                    "initialization": "checkpoint",
                    "initialization_checkpoint": None,
                    "initialization_checkpoint_sha256": None,
                    "activation_dataset": "medmnist",
                    "target_dataset": "pathmnist",
                    "activation_data_role": "target",
                    "adapted_model_checkpoint": "/tmp/lora.pt",
                    "seed": 42,
                    "block_layer": -2,
                    "module_name": "resid",
                    "d_in": 768,
                    "expansion_factor": 64,
                    "gated_sae": False,
                    "training_examples_requested": 100,
                    "activation_vectors_per_example": 197,
                    "protect_frac": 0,
                }
            }
        )


def test_runner_commands_match_except_initialization_factor():
    args = parse_args([
        "--datasets", "eurosat",
        "--domain_tiers", "far",
        "--seeds", "7",
        "--print_only",
    ])
    common = dict(
        args=args,
        dataset="eurosat",
        training_seed=7,
        lora_checkpoint=Path("/checkpoints/lora.pt"),
        checkpoint_dir=Path("/out"),
        gsae_path=Path("/checkpoints/gsae.pt"),
    )
    warm = build_command(condition="ftsae", **common)
    scratch = build_command(condition="scratchsae", **common)

    assert "--sae_initialization" in warm
    assert warm[warm.index("--sae_initialization") + 1] == "checkpoint"
    assert "--sae_checkpoint_path" in warm
    assert scratch[scratch.index("--sae_initialization") + 1] == "scratch"
    assert "--sae_checkpoint_path" not in scratch
    for flag in (
        "--training_examples",
        "--seed",
        "--dataset",
        "--block_layers",
        "--protect_frac",
        "--lr",
        "--l1_coefficient",
    ):
        assert warm[warm.index(flag) + 1] == scratch[scratch.index(flag) + 1]


def test_default_headline_plan_has_three_tiers_and_three_training_seeds():
    args = parse_args(["--print_only"])
    cells = expected_cells(args)
    assert len([cell for cell in cells if cell["condition"] == "gsae"]) == 3
    assert len([cell for cell in cells if cell["condition"] == "ftsae"]) == 9
    assert len([cell for cell in cells if cell["condition"] == "scratchsae"]) == 9


def test_registry_migrates_legacy_ftsae_and_refuses_seed_collapse():
    legacy = [{
        "dataset": "eurosat",
        "condition": "ftsae",
        "checkpoint_path": "/legacy/random.pt",
    }]
    assert build_sae_paths(legacy) == {
        "eurosat": {"scratchsae": "/legacy/random.pt"}
    }

    seeded = [
        {
            "dataset": "eurosat",
            "condition": "ftsae",
            "sae_initialization": "checkpoint",
            "training_seed": seed,
            "checkpoint_path": f"/warm/{seed}.pt",
        }
        for seed in (42, 43)
    ]
    with pytest.raises(ValueError, match="--training_seed"):
        build_sae_paths(seeded)
    assert build_sae_paths(seeded, training_seed=43) == {
        "eurosat": {"ftsae": "/warm/43.pt"}
    }
