import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PATCHSAE_ROOT = Path(__file__).resolve().parents[1]
if str(PATCHSAE_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCHSAE_ROOT))

from analysis.compute_cached_clip_mmd import compute_mmd2_rbf_unbiased
from analysis.mmd_degradation_predictor import (
    ENDPOINT_ID,
    PredictorError,
    extract_endpoint_from_matrix,
    fit_and_freeze,
    score_frozen_predictions,
)


def _write_split(path, train, holdout):
    path.write_text(
        json.dumps(
            {
                "endpoint_id": ENDPOINT_ID,
                "planned_training_n": len(train),
                "planned_development_datasets": train,
                "current_training_datasets": train,
                "heldout_datasets": holdout,
            }
        )
    )


def _synthetic_inputs(tmp_path):
    train = [f"train{i}" for i in range(6)]
    holdout = ["officehome", "kitti"]
    split = tmp_path / "split.json"
    _write_split(split, train, holdout)

    distances = pd.DataFrame(
        {
            "dataset": train + holdout,
            "MMD2_clip": [0.02, 0.05, 0.08, 0.12, 0.18, 0.27, 0.10, 0.23],
        }
    )
    distance_path = tmp_path / "distances.csv"
    distances.to_csv(distance_path, index=False)

    x = distances.iloc[: len(train)]["MMD2_clip"].to_numpy()
    y = 2.0 + 40.0 * x + np.asarray([0.2, -0.1, 0.1, -0.2, 0.15, -0.15])
    outcomes = pd.DataFrame(
        {
            "dataset": train,
            "endpoint_id": ENDPOINT_ID,
            "degradation_pp": y,
        }
    )
    outcome_path = tmp_path / "train_outcomes.csv"
    outcomes.to_csv(outcome_path, index=False)
    return train, holdout, split, distance_path, outcome_path


def test_fit_freezes_blind_predictions_and_scores_without_refit(tmp_path):
    train, holdout, split, distances, train_outcomes = _synthetic_inputs(tmp_path)
    blind = tmp_path / "blind"
    manifest = fit_and_freeze(
        distance_paths=[distances],
        train_outcomes_path=train_outcomes,
        split_config_path=split,
        out_dir=blind,
        cohort="planned",
    )
    assert manifest["status"] == "frozen_blind_predictions"
    assert (
        manifest["analysis_status"]
        == "confirmatory_frozen_planned_development_cohort"
    )
    predictions = pd.read_csv(blind / "holdout_predictions.csv")
    assert set(predictions["dataset"]) == set(holdout)
    assert not predictions["outcome_revealed"].any()
    assert (predictions["prediction_interval95_upper_pp"]
            >= predictions["prediction_interval95_lower_pp"]).all()

    observed = pd.DataFrame(
        {
            "dataset": holdout,
            "endpoint_id": ENDPOINT_ID,
            "degradation_pp": [6.1, 11.0],
        }
    )
    observed_path = tmp_path / "holdout_outcomes.csv"
    observed.to_csv(observed_path, index=False)
    scored_dir = tmp_path / "scored"
    score_manifest = score_frozen_predictions(
        artifact_dir=blind,
        holdout_outcomes_path=observed_path,
        out_dir=scored_dir,
    )
    assert score_manifest["refit_performed"] is False
    scored = pd.read_csv(scored_dir / "holdout_scored.csv")
    assert scored["outcome_revealed"].all()


def test_fit_rejects_holdout_outcome_leakage(tmp_path):
    _, _, split, distances, train_outcomes = _synthetic_inputs(tmp_path)
    frame = pd.read_csv(train_outcomes)
    frame.loc[len(frame)] = ["kitti", ENDPOINT_ID, 99.0]
    leaked = tmp_path / "leaked.csv"
    frame.to_csv(leaked, index=False)
    with pytest.raises(PredictorError, match="LEAKAGE GUARD"):
        fit_and_freeze(
            distance_paths=[distances],
            train_outcomes_path=leaked,
            split_config_path=split,
            out_dir=tmp_path / "blind",
            cohort="planned",
        )


def test_score_rejects_tampered_blind_predictions(tmp_path):
    _, holdout, split, distances, train_outcomes = _synthetic_inputs(tmp_path)
    blind = tmp_path / "blind"
    fit_and_freeze(
        distance_paths=[distances],
        train_outcomes_path=train_outcomes,
        split_config_path=split,
        out_dir=blind,
        cohort="planned",
    )
    predictions_path = blind / "holdout_predictions.csv"
    predictions = pd.read_csv(predictions_path)
    predictions.loc[0, "predicted_degradation_pp"] += 100
    predictions.to_csv(predictions_path, index=False)

    outcomes = pd.DataFrame(
        {
            "dataset": holdout,
            "endpoint_id": ENDPOINT_ID,
            "degradation_pp": [5.0, 10.0],
        }
    )
    outcome_path = tmp_path / "holdout.csv"
    outcomes.to_csv(outcome_path, index=False)
    with pytest.raises(PredictorError, match="hash mismatch"):
        score_frozen_predictions(
            artifact_dir=blind,
            holdout_outcomes_path=outcome_path,
            out_dir=tmp_path / "scored",
        )


def test_extract_matrix_uses_paired_accuracy_difference(tmp_path):
    matrix = pd.DataFrame(
        [
            {
                "dataset": "Caltech-101",
                "vit_type": "lora",
                "sae_condition": "none",
                "zeroshot_acc": 80.0,
                "n_images": 100,
                "skipped": False,
            },
            {
                "dataset": "Caltech-101",
                "vit_type": "lora",
                "sae_condition": "gsae",
                "zeroshot_acc": 65.5,
                "n_images": 100,
                "skipped": False,
            },
        ]
    )
    path = tmp_path / "matrix.csv"
    matrix.to_csv(path, index=False)
    endpoint = extract_endpoint_from_matrix(path, datasets=["caltech101"])
    assert endpoint.loc[0, "dataset"] == "caltech101"
    assert endpoint.loc[0, "degradation_pp"] == pytest.approx(14.5)


def test_cached_mmd_estimator_is_symmetric():
    rng = np.random.default_rng(7)
    left = rng.normal(size=(20, 5)).astype(np.float32)
    right = rng.normal(loc=0.5, size=(17, 5)).astype(np.float32)
    forward, sigma_forward = compute_mmd2_rbf_unbiased(left, right)
    reverse, sigma_reverse = compute_mmd2_rbf_unbiased(right, left)
    assert forward == pytest.approx(reverse, abs=1e-7)
    assert sigma_forward == pytest.approx(sigma_reverse, abs=1e-7)
