"""Train Qwen3 QLoRA on one prepared fold or the complete shared CV split."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import shutil
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import EarlyStoppingCallback, TrainingArguments, set_seed

from experiments.c_qlora.qlora import (
    PassageDataset,
    WeightedCollator,
    WeightedTrainer,
    load_model,
    probability_table,
)
from ntl.constants import MODEL_LABELS
from ntl.data import grouped_cv_splits, load_test, load_train
from ntl.metrics import evaluate
from ntl.submission import argmax_proba, write_proba

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, help="train one fold; omit for full OOF")
    parser.add_argument("--output-dir", type=Path, help="override config output_dir")
    return parser.parse_args()


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def compute_metrics(prediction):
    labels = np.asarray(MODEL_LABELS)
    return {
        "balanced_accuracy": evaluate(
            labels[prediction.label_ids], labels[prediction.predictions.argmax(axis=1)]
        )["balanced_accuracy"]
    }


def main():
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a CUDA GPU")
    model_config, training_config = config["model"], dict(config["training"])
    patience = training_config.pop("early_stopping_patience")
    if model_config["compute_dtype"] == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support bf16; set model.compute_dtype to float16")
    data = config["data"]
    train = load_train(data["train"])
    test = load_test(data["test"])[["id", "text"]]
    folds = pd.read_parquet(data["folds"])
    splits = list(grouped_cv_splits(train, folds))
    selected = list(range(len(splits))) if args.fold is None else [args.fold]
    if not set(selected).issubset(range(len(splits))):
        raise ValueError("--fold is outside the shared fold range")
    prepared_dir = Path(data["splits_dir"])
    output = args.output_dir or Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "log.txt")],
    )
    config["output_dir"] = str(output)
    config["selected_folds"] = selected
    (output / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    shutil.copyfile(data["folds"], output / "folds.parquet")
    shutil.copyfile(prepared_dir / "manifest.json", output / "data_manifest.json")
    environment = {
        "packages": {
            name: version(name)
            for name in ("torch", "transformers", "peft", "bitsandbytes", "accelerate")
        },
        "gpu": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda,
    }
    (output / "environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    provenance = {}
    for name in ("train", "test", "folds"):
        path = Path(data[name])
        provenance[name] = {"path": str(path.resolve()), "sha256": digest(path)}
    validation_probas, test_probas, fold_scores = [], [], []
    for fold in selected:
        directory = prepared_dir / f"fold_{fold}"
        training = pd.read_parquet(directory / "train.parquet")
        validation = pd.read_parquet(directory / "validation.parquet")
        expected_training, expected_validation = splits[fold]
        # The shared assignment, rather than directory names, owns the split.
        expected = train.iloc[expected_validation].set_index("id").sort_index()
        pd.testing.assert_frame_equal(
            validation.set_index("id").sort_index()[expected.columns], expected
        )
        competition = training[training["data_source"].eq("competition")]
        expected = train.iloc[expected_training].set_index("id").sort_index()
        pd.testing.assert_frame_equal(
            competition.set_index("id").sort_index()[expected.columns], expected
        )
        if training["id"].duplicated().any() or set(training["id"]) & set(validation["id"]):
            raise ValueError("training ids must be unique and disjoint from validation")
        for name in ("train", "validation"):
            path = directory / f"{name}.parquet"
            provenance[f"fold_{fold}/{name}"] = {"sha256": digest(path)}
        (output / "inputs.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        set_seed(training_config["seed"])
        model, tokenizer = load_model(
            model_config, gradient_checkpointing=training_config["gradient_checkpointing"]
        )
        max_length = model_config["max_length"]
        fold_dir = output / f"fold_{fold}"
        trainer = WeightedTrainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(fold_dir / "checkpoints"),
                eval_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=1,
                load_best_model_at_end=True,
                metric_for_best_model="balanced_accuracy",
                greater_is_better=True,
                bf16=model_config["compute_dtype"] == "bfloat16",
                fp16=model_config["compute_dtype"] == "float16",
                optim="adamw_torch",
                remove_unused_columns=False,
                label_names=["labels"],
                report_to="none",
                gradient_checkpointing_kwargs={"use_reentrant": False},
                **training_config,
            ),
            train_dataset=PassageDataset(training, tokenizer, max_length, normalize_weights=True),
            eval_dataset=PassageDataset(validation, tokenizer, max_length),
            processing_class=tokenizer,
            data_collator=WeightedCollator(tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
        )
        trainer.train()
        (fold_dir / "history.json").write_text(
            json.dumps(trainer.state.log_history, indent=2), encoding="utf-8"
        )
        trainer.save_model(str(fold_dir / "adapter"))
        tokenizer.save_pretrained(fold_dir / "adapter")
        # The adapter stores the trained classification head and label order.
        model.config.save_pretrained(fold_dir / "adapter")
        val_proba = probability_table(
            validation["id"], trainer.predict(trainer.eval_dataset).predictions
        )
        test_proba = probability_table(
            test["id"], trainer.predict(PassageDataset(test, tokenizer, max_length)).predictions
        )
        write_proba(val_proba, fold_dir / "oof_proba.parquet", ids=validation["id"])
        write_proba(test_proba, fold_dir / "test_proba.parquet", ids=test["id"])
        score = evaluate(validation["model"], argmax_proba(val_proba)["model"])
        (fold_dir / "metrics.json").write_text(json.dumps(score, indent=2), encoding="utf-8")
        LOGGER.info("fold %s balanced accuracy: %.4f", fold, score["balanced_accuracy"])
        validation_probas.append(val_proba)
        test_probas.append(test_proba[list(MODEL_LABELS)].to_numpy())
        fold_scores.append({"fold": fold, **score})
        del trainer, model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    if args.fold is None:
        oof = pd.concat(validation_probas).set_index("id").loc[train["id"]].reset_index()
        write_proba(oof, output / "oof_proba.parquet", ids=train["id"])
        averaged = pd.DataFrame(np.mean(test_probas, axis=0), columns=list(MODEL_LABELS))
        averaged.insert(0, "id", test["id"].to_numpy())
        write_proba(averaged, output / "test_proba.parquet", ids=test["id"])
        score = evaluate(train["model"], argmax_proba(oof)["model"])
        score["folds"] = fold_scores
        (output / "metrics.json").write_text(json.dumps(score, indent=2), encoding="utf-8")
        LOGGER.info("full OOF balanced accuracy: %.4f", score["balanced_accuracy"])


if __name__ == "__main__":
    main()
