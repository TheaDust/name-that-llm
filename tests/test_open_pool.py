import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from ntl.constants import ID_COLUMN, LABEL_COLUMN, MODEL_LABELS, TEXT_COLUMN
from ntl.open_pool import (
    FOLD_COLUMN,
    POOL_PROMPT_COLUMN,
    PoolCleanConfig,
    clean_open_pool,
    pool_sample_weights,
    validate_clean_pool,
)


def long_text(prompt: int, model: str, k: int = 0) -> str:
    words = [f"word{i}" for i in range(42)]
    return " ".join([*words, f"prompt{prompt}", model.replace("-", ""), f"k{k}"])


def make_pool(
    n_prompts: int = 6,
    models: tuple[str, ...] = MODEL_LABELS,
    per_pair: int = 1,
) -> pd.DataFrame:
    rows = []
    for prompt in range(n_prompts):
        for model in models:
            for k in range(per_pair):
                rows.append(
                    {
                        ID_COLUMN: f"pool-{prompt:03d}-{model}-{k}",
                        POOL_PROMPT_COLUMN: f"q{prompt:03d}",
                        TEXT_COLUMN: long_text(prompt, model, k),
                        LABEL_COLUMN: model,
                        "source_dataset": f"source{prompt % 3}",
                        "source_license": "CC0-1.0",
                        "style_score": 0.1,
                        "match_weight": 1.0,
                        "flag_kb_proxy": False,
                        "flag_kb_proxy_after_ruling": False,
                    }
                )
    return pd.DataFrame(rows)


def small_config(**overrides) -> PoolCleanConfig:
    defaults = {"n_splits": 2, "seed": 3}
    return PoolCleanConfig(**{**defaults, **overrides})


def test_clean_pool_keeps_core_rows_and_all_labels():
    cleaned, steps = clean_open_pool(make_pool(), small_config())

    assert len(cleaned) == 6 * len(MODEL_LABELS)
    assert set(cleaned[LABEL_COLUMN]) == set(MODEL_LABELS)
    assert steps.iloc[-1]["remaining"] == len(cleaned)


def test_clean_pool_enforces_length_bounds():
    pool = make_pool()
    short = pool.iloc[[0]].copy()
    short[ID_COLUMN] = "too-short"
    short[TEXT_COLUMN] = "only a few words"
    long = pool.iloc[[1]].copy()
    long[ID_COLUMN] = "too-long"
    long[TEXT_COLUMN] = " ".join(["word"] * 300)
    pool = pd.concat([pool, short, long], ignore_index=True)

    cleaned, steps = clean_open_pool(pool, small_config())

    assert "too-short" not in set(cleaned[ID_COLUMN])
    assert "too-long" not in set(cleaned[ID_COLUMN])
    assert int(steps.loc[steps["step"] == "length", "delta"].item()) == -2


def test_clean_pool_drops_proxy_unknown_and_empty_rows():
    pool = make_pool()
    pool.loc[0, "flag_kb_proxy"] = True
    pool.loc[1, "flag_kb_proxy_after_ruling"] = True
    pool.loc[2, LABEL_COLUMN] = "not-a-model"
    pool.loc[3, TEXT_COLUMN] = "   "

    cleaned, _ = clean_open_pool(pool, small_config())

    assert len(cleaned) == 6 * len(MODEL_LABELS) - 4


def test_clean_pool_dedupes_normalized_text():
    pool = make_pool()
    duplicate = pool.iloc[[0]].copy()
    duplicate[ID_COLUMN] = "duplicate-id"
    duplicate[TEXT_COLUMN] = f"  {pool.iloc[0][TEXT_COLUMN]}\n"
    pool = pd.concat([pool, duplicate], ignore_index=True)

    cleaned, _ = clean_open_pool(pool, small_config())

    assert len(cleaned) == 6 * len(MODEL_LABELS)
    assert cleaned[TEXT_COLUMN].duplicated().sum() == 0


def test_clean_pool_drops_text_with_conflicting_labels():
    pool = make_pool()
    conflict = pool.iloc[[0]].copy()
    conflict[ID_COLUMN] = "conflict-id"
    conflict[LABEL_COLUMN] = MODEL_LABELS[1]
    pool = pd.concat([pool, conflict], ignore_index=True)

    cleaned, _ = clean_open_pool(pool, small_config())

    assert pool.iloc[0][TEXT_COLUMN] not in set(cleaned[TEXT_COLUMN])
    assert len(cleaned) == 6 * len(MODEL_LABELS) - 1


def test_clean_pool_drops_prompts_with_too_few_models():
    pool = make_pool()
    thin = []
    for model in MODEL_LABELS[:3]:
        thin.append(
            {
                ID_COLUMN: f"thin-{model}",
                POOL_PROMPT_COLUMN: "q999",
                TEXT_COLUMN: long_text(999, model),
                LABEL_COLUMN: model,
                "source_dataset": "source0",
                "source_license": "CC0-1.0",
                "style_score": 0.1,
                "match_weight": 1.0,
                "flag_kb_proxy": False,
                "flag_kb_proxy_after_ruling": False,
            }
        )
    pool = pd.concat([pool, pd.DataFrame(thin)], ignore_index=True)

    cleaned, _ = clean_open_pool(pool, small_config())

    assert "q999" not in set(cleaned[POOL_PROMPT_COLUMN])


def test_clean_pool_caps_rows_per_prompt_model_pair():
    cleaned, _ = clean_open_pool(
        make_pool(n_prompts=3, per_pair=5), small_config(max_rows_per_pair=2)
    )

    per_pair = cleaned.groupby([POOL_PROMPT_COLUMN, LABEL_COLUMN]).size()
    assert per_pair.max() == 2
    assert len(cleaned) == 3 * len(MODEL_LABELS) * 2


def test_clean_pool_retains_only_prompts_meeting_model_coverage():
    rare = "gemini-3.7-flash"
    pool = make_pool(n_prompts=5, models=tuple(m for m in MODEL_LABELS if m != rare))
    thin = []
    for prompt in range(5):
        for model in (rare, MODEL_LABELS[0], MODEL_LABELS[1], MODEL_LABELS[2]):
            thin.append(
                {
                    ID_COLUMN: f"thin-{prompt}-{model}",
                    POOL_PROMPT_COLUMN: f"t{prompt:03d}",
                    TEXT_COLUMN: long_text(100 + prompt, model),
                    LABEL_COLUMN: model,
                    "source_dataset": "source0",
                    "source_license": "CC0-1.0",
                    "style_score": 0.1,
                    "match_weight": 1.0,
                    "flag_kb_proxy": False,
                    "flag_kb_proxy_after_ruling": False,
                }
            )
    pool = pd.concat([pool, pd.DataFrame(thin)], ignore_index=True)

    cleaned, _ = clean_open_pool(pool, small_config())

    assert rare not in set(cleaned[LABEL_COLUMN])
    assert cleaned[POOL_PROMPT_COLUMN].str.startswith("q").all()
    assert cleaned.groupby(POOL_PROMPT_COLUMN)[LABEL_COLUMN].nunique().ge(8).all()


def test_pool_weights_balance_selected_classes_and_limit_total_mass():
    pool = make_pool()
    # Simulate unequal sampling and preserve a nonconsecutive, reordered index.
    selected = pool.iloc[[19, 0, 18, 1, 36]]
    weights = pool_sample_weights(selected, train_rows=100)
    totals = pd.Series(weights).groupby(selected[LABEL_COLUMN].to_numpy()).sum()

    assert (weights > 0).all()
    assert weights.sum() == pytest.approx(100)
    assert totals.tolist() == pytest.approx([50, 50])
    assert weights.tolist() == pytest.approx([25, 100 / 6, 100 / 6, 25, 100 / 6])


def test_pool_weights_recomputed_after_class_balanced_sampling():
    pool = make_pool()
    selected = pool.groupby(LABEL_COLUMN, sort=False).head(2)
    weights = pool_sample_weights(selected, train_rows=90, pool_weight_ratio=0.5)

    assert weights.sum() == pytest.approx(45)
    assert weights.tolist() == pytest.approx([45 / len(selected)] * len(selected))


@pytest.mark.parametrize("ratio", [0, -1, float("nan"), float("inf")])
def test_pool_weights_reject_invalid_ratio(ratio):
    with pytest.raises(ValueError, match="pool_weight_ratio"):
        pool_sample_weights(make_pool(), train_rows=100, pool_weight_ratio=ratio)


def test_pool_weights_require_selected_rows_and_training_rows():
    with pytest.raises(ValueError, match="empty"):
        pool_sample_weights(make_pool().iloc[:0], train_rows=100)
    with pytest.raises(ValueError, match="train_rows"):
        pool_sample_weights(make_pool(), train_rows=0)


def test_clean_pool_folds_keep_each_prompt_together():
    cleaned, _ = clean_open_pool(make_pool(), small_config())

    per_prompt = cleaned.groupby(POOL_PROMPT_COLUMN)[FOLD_COLUMN].nunique()
    assert (per_prompt == 1).all()
    assert set(cleaned[FOLD_COLUMN]) == {0, 1}


def test_validate_clean_pool_rejects_duplicate_ids():
    cleaned, _ = clean_open_pool(make_pool(), small_config())
    cleaned.loc[1, ID_COLUMN] = cleaned.loc[0, ID_COLUMN]

    with pytest.raises(ValueError, match="duplicated ids"):
        validate_clean_pool(cleaned)


def cross_source_pool():
    pool = make_pool()
    pool["prompt_canon"] = pool[POOL_PROMPT_COLUMN].map(lambda value: f"Question {value}")
    # Each source-local group has only one label; canonical panels have all 18.
    pool[POOL_PROMPT_COLUMN] += ":" + pool[LABEL_COLUMN]
    return pool


def test_canonical_panels_recover_models_across_source_local_groups():
    pool = cross_source_pool()
    pool.loc[0, "prompt_canon"] = "  Question\nq000  "
    cleaned, _ = clean_open_pool(pool, small_config())

    assert len(cleaned) == len(pool)
    assert set(cleaned[LABEL_COLUMN]) == set(MODEL_LABELS)
    assert cleaned[POOL_PROMPT_COLUMN].nunique() == 6
    assert cleaned.groupby(POOL_PROMPT_COLUMN)[LABEL_COLUMN].nunique().eq(18).all()
    assert cleaned.groupby(POOL_PROMPT_COLUMN)[FOLD_COLUMN].nunique().eq(1).all()
    shuffled, _ = clean_open_pool(pool.sample(frac=1, random_state=1), small_config())
    pd.testing.assert_frame_equal(cleaned, shuffled)


def test_canonical_pair_cap_applies_across_sources():
    pool = cross_source_pool()
    extra = pool.copy()
    extra[ID_COLUMN] += "-extra"
    extra[POOL_PROMPT_COLUMN] += "-other-source"
    extra[TEXT_COLUMN] += " extra"
    cleaned, _ = clean_open_pool(
        pd.concat([pool, extra], ignore_index=True), small_config(max_rows_per_pair=1)
    )
    assert len(cleaned) == len(pool)
    assert cleaned.groupby([POOL_PROMPT_COLUMN, LABEL_COLUMN]).size().eq(1).all()


def test_canonical_coverage_excludes_filtered_rows():
    pool = cross_source_pool()
    # One panel falls from 8 to 7 labels after proxy removal.
    panel = pool["prompt_canon"].eq("Question q000")
    pool = pool[~panel | pool[LABEL_COLUMN].isin(MODEL_LABELS[:8])].copy()
    pool.loc[pool.index[0], "flag_kb_proxy"] = True
    cleaned, _ = clean_open_pool(pool, small_config())
    assert not cleaned[ID_COLUMN].str.startswith("pool-000-").any()


def test_empty_canonical_values_do_not_form_a_shared_panel():
    pool = cross_source_pool()
    pool["prompt_canon"] = [None, "", "  "] * (len(pool) // 3)
    # Keep enough valid legacy groups for fold assignment.
    legacy = make_pool(n_prompts=3)
    legacy[ID_COLUMN] += "-legacy"
    legacy[TEXT_COLUMN] += " legacy"
    pool = pd.concat([pool, legacy], ignore_index=True)
    cleaned, _ = clean_open_pool(pool, small_config())
    assert len(cleaned) == 3 * len(MODEL_LABELS)
    assert cleaned[POOL_PROMPT_COLUMN].str.startswith("source:q").all()


def test_canonical_normalization_preserves_case_distinctions():
    pool = cross_source_pool()
    first_panel = pool["prompt_canon"].eq("Question q000")
    pool.loc[first_panel & pool[LABEL_COLUMN].isin(MODEL_LABELS[:9]), "prompt_canon"] = (
        "question q000"
    )
    cleaned, _ = clean_open_pool(pool, small_config(min_prompt_models=12))
    assert not cleaned[ID_COLUMN].str.startswith("pool-000-").any()


@pytest.mark.parametrize("value", [None, "", " "])
def test_clean_pool_rejects_missing_source_group(value):
    pool = make_pool()
    pool.loc[0, POOL_PROMPT_COLUMN] = value
    with pytest.raises(ValueError, match="null/empty prompt groups"):
        clean_open_pool(pool, small_config())


def test_prepare_cli_rejects_missing_classes_unless_explicitly_allowed(tmp_path):
    pool = make_pool(models=MODEL_LABELS[:8])
    input_path = tmp_path / "raw.parquet"
    pool.to_parquet(input_path, index=False)
    output_path = tmp_path / "clean.parquet"
    report_path = tmp_path / "report.json"
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(root / "scripts/prepare_open_pool.py"),
        "--input",
        str(input_path),
        "--out",
        str(output_path),
        "--report",
        str(report_path),
        "--train",
        str(tmp_path / "absent.csv"),
        "--n-splits",
        "2",
    ]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "cleaning removed all rows for classes" in result.stderr
    assert not output_path.exists()
    assert not report_path.exists()
    result = subprocess.run(
        [*command, "--allow-missing-classes"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text())
    assert set(report["missing_classes"]) == set(MODEL_LABELS[8:])
    assert report["allow_missing_classes"] is True
    assert len(pd.read_parquet(output_path)) == len(pool)
