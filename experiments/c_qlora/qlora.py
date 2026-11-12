"""Single-GPU Qwen3 QLoRA classification using prepared, weighted CV data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
)

from ntl.constants import MODEL_LABELS
from ntl.submission import validate_proba

LABEL_TO_ID = {label: index for index, label in enumerate(MODEL_LABELS)}


class PassageDataset(Dataset):
    """Tokenize passage text only; retain labels and loss weights separately."""

    def __init__(self, frame, tokenizer, max_length, *, normalize_weights=False):
        self.tokens = tokenizer(
            frame["text"].tolist(), truncation=True, max_length=max_length, padding=False
        )
        self.labels = None
        self.weights = None
        if "model" in frame:
            self.labels = [LABEL_TO_ID[label] for label in frame["model"]]
            weights = (
                frame["sample_weight"].to_numpy(dtype=np.float32)
                if "sample_weight" in frame else np.ones(len(frame), dtype=np.float32)
            )
            if not np.isfinite(weights).all() or not (weights > 0).all():
                raise ValueError("sample_weight must be finite and positive")
            # One global scale preserves pool/competition ratios while keeping
            # the average loss scale stable across different pool sample sizes.
            self.weights = weights / weights.mean() if normalize_weights else weights

    def __len__(self):
        return len(self.tokens["input_ids"])

    def __getitem__(self, index):
        row = {key: value[index] for key, value in self.tokens.items()}
        if self.labels is not None:
            row["labels"] = self.labels[index]
            row["sample_weight"] = float(self.weights[index])
        return row


class WeightedCollator:
    def __init__(self, tokenizer):
        self.padding = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    def __call__(self, features):
        rows = [dict(feature) for feature in features]
        labelled = "labels" in rows[0]
        if labelled:
            labels = [row.pop("labels") for row in rows]
            weights = [row.pop("sample_weight") for row in rows]
        batch = self.padding(rows)
        if labelled:
            batch["labels"] = torch.tensor(labels, dtype=torch.long)
            batch["sample_weight"] = torch.tensor(weights, dtype=torch.float32)
        return batch


class WeightedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Trainer handles gradient accumulation for the mean microbatch loss.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        weights = inputs.pop("sample_weight")
        outputs = model(**inputs)
        losses = F.cross_entropy(outputs.logits.float(), labels, reduction="none")
        loss = (losses * weights).mean()
        return (loss, outputs) if return_outputs else loss


def load_model(config, *, gradient_checkpointing):
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[config["compute_dtype"]]
    location = config["name_or_path"]
    kwargs = {
        "revision": config["revision"],
        "local_files_only": config["local_files_only"],
    }
    tokenizer = AutoTokenizer.from_pretrained(location, padding_side="right", **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        location,
        num_labels=len(MODEL_LABELS),
        id2label=dict(enumerate(MODEL_LABELS)),
        label2id=LABEL_TO_ID,
        pad_token_id=tokenizer.pad_token_id,
        problem_type="single_label_classification",
        torch_dtype=dtype,
        device_map={"": torch.cuda.current_device()},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
            llm_int8_skip_modules=["score"],
        ),
        **kwargs,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        modules_to_save=["score"],
        bias="none",
    ))
    model.print_trainable_parameters()
    return model, tokenizer


def probability_table(ids, logits):
    values = torch.as_tensor(logits, dtype=torch.float32).softmax(dim=-1).numpy()
    frame = pd.DataFrame(values, columns=list(MODEL_LABELS))
    frame.insert(0, "id", list(ids))
    return validate_proba(frame, ids=ids)
