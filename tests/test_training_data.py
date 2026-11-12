import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from test_open_pool import make_pool, small_config

from ntl.data import fold_table
from ntl.open_pool import clean_open_pool
from ntl.training_data import sample_training_pool, training_folds


@pytest.fixture
def frames():
    train = make_pool()[["id", "text", "model", "prompt_group"]].rename(
        columns={"prompt_group": "prompt_id"}
    )
    train["id"] = "competition-" + train["id"]
    train["text"] = "Competition passage " + train["text"]
    pool, _ = clean_open_pool(make_pool(), small_config())
    return train, fold_table(train, n_splits=2), pool


def test_sampling_is_reproducible_with_reordered_inputs(frames):
    _, _, pool = frames
    selected = sample_training_pool(pool, max_rows_per_class=2, seed=7)
    reordered = sample_training_pool(pool.iloc[::-1], max_rows_per_class=2, seed=7)
    pd.testing.assert_frame_equal(selected, reordered)
    assert selected.groupby("model").size().eq(2).all()
    pd.testing.assert_frame_equal(pool, sample_training_pool(pool, max_rows_per_class=100))
    with pytest.raises(ValueError, match="positive"):
        sample_training_pool(pool, max_rows_per_class=0)


def test_folds_keep_validation_isolated_and_align_weights(frames):
    train, folds, pool = frames
    selected = sample_training_pool(pool, max_rows_per_class=2)
    validation_ids = []
    for _, training, validation in training_folds(train, folds, selected, pool_weight_ratio=0.5):
        competition = training[training.data_source.eq("competition")]
        auxiliary = training[training.data_source.eq("pool")]
        assert set(training.id).isdisjoint(validation.id)
        assert set(competition.prompt_id).isdisjoint(validation.prompt_id)
        assert auxiliary.id.tolist() == selected.id.tolist()
        assert competition.sample_weight.eq(1).all()
        assert auxiliary.sample_weight.sum() == pytest.approx(len(competition) * 0.5)
        assert auxiliary.groupby("model").sample_weight.sum().nunique() == 1
        expected = train.set_index("id").loc[validation.id].reset_index()
        pd.testing.assert_frame_equal(validation, expected[validation.columns])
        validation_ids.extend(validation.id)
    assert sorted(validation_ids) == sorted(train.id)


@pytest.mark.parametrize("column", ["id", "text"])
def test_folds_reject_pool_overlap_with_any_competition_fold(frames, column):
    train, folds, pool = frames
    pool.loc[0, column] = train.loc[0, column]
    with pytest.raises(ValueError, match="overlap"):
        list(training_folds(train, folds, pool))


@pytest.mark.parametrize("competition_only", [False, True])
def test_cli_exports_ready_to_train_files(frames, tmp_path, competition_only):
    train, folds, pool = frames
    train.to_csv(tmp_path / "train.csv", index=False)
    folds.to_parquet(tmp_path / "folds.parquet", index=False)
    pool.to_parquet(tmp_path / "pool.parquet", index=False)
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable, str(root / "scripts" / "prepare_training_data.py"),
        "--train", str(tmp_path / "train.csv"),
        "--folds", str(tmp_path / "folds.parquet"),
        "--pool", str(tmp_path / "pool.parquet"),
        "--max-pool-rows-per-class", "2",
        "--out", str(tmp_path / "prepared"),
    ]
    if competition_only:
        command.append("--competition-only")
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    report = json.loads((tmp_path / "prepared" / "manifest.json").read_text())
    assert report["competition_only"] == competition_only
    assert report["pool_rows"] == (0 if competition_only else 36)
    assert len(report["inputs"]["train"]["sha256"]) == 64
    for fold, expected_train, expected_validation in training_folds(
        train, folds, None if competition_only else sample_training_pool(pool, max_rows_per_class=2)
    ):
        directory = tmp_path / "prepared" / f"fold_{fold}"
        pd.testing.assert_frame_equal(
            pd.read_parquet(directory / "train.parquet").fillna(""), expected_train.fillna("")
        )
        pd.testing.assert_frame_equal(
            pd.read_parquet(directory / "validation.parquet"), expected_validation
        )
    # Re-running with a different configuration must not mix exported artifacts.
    repeated = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    assert repeated.returncode != 0
    assert "FileExistsError" in repeated.stderr
