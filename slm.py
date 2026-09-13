import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "distilbert/distilbert-base-uncased"
LABELS = ["Read", "Write", "Execute", "Destructive", "Financial", "Other"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def build_text(row):
    name = str(row.get("name") or "[MISSING]").strip()
    description = str(row.get("description") or "[MISSING]").strip()
    schema = row.get("input_schema") or "[MISSING]"
    if isinstance(schema, dict):
        schema = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return f"[NAME] {name}\n[DESCRIPTION] {description}\n[SCHEMA] {schema}"


class ToolDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length, labeled=True):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labeled = labeled

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        item = self.tokenizer(
            build_text(self.rows[index]),
            truncation=True,
            max_length=self.max_length,
        )
        if self.labeled:
            item["labels"] = LABEL2ID[self.rows[index]["category"]]
        return item


def compute_metrics(prediction):
    predictions = np.argmax(prediction.predictions, axis=1)
    labels = prediction.label_ids
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=range(len(LABELS)), zero_division=0
    )
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "weighted_f1": f1_score(labels, predictions, average="weighted"),
    }
    for i, label in enumerate(LABELS):
        metrics[f"{label}_precision"] = precision[i]
        metrics[f"{label}_recall"] = recall[i]
        metrics[f"{label}_f1"] = f1[i]
        metrics[f"{label}_support"] = support[i]
    return metrics


def save_evaluation(rows, predictions, output_dir, extra_metrics):
    true_ids = np.array([LABEL2ID[row["category"]] for row in rows])
    pred_ids = np.asarray(predictions)
    report = classification_report(
        true_ids,
        pred_ids,
        labels=range(len(LABELS)),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(true_ids, pred_ids, labels=range(len(LABELS)))

    metrics = {
        "primary_metric": "macro_f1",
        "macro_f1": f1_score(true_ids, pred_ids, average="macro"),
        "weighted_f1": f1_score(true_ids, pred_ids, average="weighted"),
        "accuracy": accuracy_score(true_ids, pred_ids),
        "invalid_output_rate": 0.0,
        "per_class": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": int(report[label]["support"]),
            }
            for label in LABELS
        },
        "confusion_matrix": matrix.tolist(),
        **extra_metrics,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    np.savetxt(
        output_dir / "confusion_matrix.csv",
        matrix,
        fmt="%d",
        delimiter=",",
        header=",".join(LABELS),
        comments="",
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Validation Confusion Matrix")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    return metrics


def model_size_mb(path):
    return sum(file.stat().st_size for file in Path(path).rglob("*") if file.is_file()) / 1e6


def training_arguments(args):
    common = dict(
        output_dir=str(Path(args.output_dir) / "checkpoints"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
        fp16=False,
        dataloader_num_workers=0,
    )
    try:
        return TrainingArguments(**common)
    except TypeError:
        common["evaluation_strategy"] = common.pop("eval_strategy")
        return TrainingArguments(**common)


def train(args):
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(args.train)
    validation_rows = load_jsonl(args.validation)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for split_name, rows in [
        ("train", train_rows),
        ("validation", validation_rows),
    ]:
        unknown = sorted({
            row.get("category")
            for row in rows
            if row.get("category") not in LABEL2ID
        })

        if unknown:
            raise ValueError(
                f"{split_name} contains unknown labels: {unknown}. "
                f"Expected only: {LABELS}"
            )

        encoded = [LABEL2ID[row["category"]] for row in rows]

        print(
            split_name,
            "labels:",
            sorted(set(row["category"] for row in rows)),
            "encoded range:",
            min(encoded),
            max(encoded),
        )

        assert min(encoded) >= 0
        assert max(encoded) < len(LABELS)

    train_dataset = ToolDataset(train_rows, tokenizer, args.max_length)
    validation_dataset = ToolDataset(validation_rows, tokenizer, args.max_length)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    trainer = Trainer(
    model=model,
    args=training_arguments(args),
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    sample = DataCollatorWithPadding(tokenizer)(
    [train_dataset[i] for i in range(4)]
)

    print("Input ID range:", sample["input_ids"].min().item(),
        sample["input_ids"].max().item())
    print("Vocabulary size:", model.config.vocab_size)
    print("Labels:", sample["labels"].tolist())

    assert sample["input_ids"].min() >= 0
    assert sample["input_ids"].max() < model.config.vocab_size
    assert sample["labels"].min() >= 0
    assert sample["labels"].max() < model.config.num_labels
    trainer.train()
    training_seconds = time.perf_counter() - start

    model_dir = output_dir / "model"
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    start = time.perf_counter()
    result = trainer.predict(validation_dataset)
    inference_seconds = time.perf_counter() - start
    pred_ids = np.argmax(result.predictions, axis=1)

    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / 1024**2
        if torch.cuda.is_available()
        else psutil.Process(os.getpid()).memory_info().rss / 1024**2
    )
    metrics = save_evaluation(
        validation_rows,
        pred_ids,
        output_dir,
        {
            "model_name": MODEL_NAME,
            "model_size_mb": model_size_mb(model_dir),
            "peak_memory_mb": peak_memory_mb,
            "training_seconds": training_seconds,
            "validation_inference_seconds": inference_seconds,
            "latency_ms_per_record": inference_seconds * 1000 / len(validation_rows),
            "throughput_records_per_second": len(validation_rows) / inference_seconds,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
    )

    print(f"\nMacro F1:   {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:{metrics['weighted_f1']:.4f}")
    print(f"Accuracy:   {metrics['accuracy']:.4f}")
    print(f"Model saved:{model_dir}")
    print(f"Metrics:    {output_dir / 'metrics.json'}")


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return tokenizer, model, device


def predict_rows(rows, model_path, max_length, batch_size=16):
    tokenizer, model, device = load_model(model_path)
    predictions = []

    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            encoded = tokenizer(
                [build_text(row) for row in batch_rows],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            pred_ids = model(**encoded).logits.argmax(dim=1).cpu().tolist()
            predictions.extend(ID2LABEL[index] for index in pred_ids)

    return predictions


def predict_file(args):
    rows = load_jsonl(args.input)
    predictions = predict_rows(
        rows, args.model_path, args.max_length, args.eval_batch_size
    )
    with open(args.output, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["record_id", "category"])
        writer.writerows(
            (row["record_id"], prediction)
            for row, prediction in zip(rows, predictions)
        )
    print(f"Saved {len(predictions)} predictions to {args.output}")


def predict_text(args):
    row = {
        "name": args.name or "[MISSING]",
        "description": args.description,
        "input_schema": args.schema or "[MISSING]",
    }
    prediction = predict_rows([row], args.model_path, args.max_length, 1)[0]
    print(f"Predicted category: {prediction}")


def parser():
    root = argparse.ArgumentParser()
    subparsers = root.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--train", default="data/train.jsonl")
    train_parser.add_argument("--validation", default="data/validation.jsonl")
    train_parser.add_argument("--output-dir", default="slm_outputs")
    train_parser.add_argument("--max-length", type=int, default=256)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--eval-batch-size", type=int, default=8)
    train_parser.add_argument("--gradient-accumulation", type=int, default=4)
    train_parser.add_argument("--epochs", type=float, default=3)
    train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.set_defaults(function=train)

    file_parser = subparsers.add_parser("predict-file")
    file_parser.add_argument("--model-path", default="slm_outputs/model")
    file_parser.add_argument("--input", required=True)
    file_parser.add_argument("--output", default="predictions.csv")
    file_parser.add_argument("--max-length", type=int, default=256)
    file_parser.add_argument("--eval-batch-size", type=int, default=8)
    file_parser.set_defaults(function=predict_file)

    text_parser = subparsers.add_parser("predict-text")
    text_parser.add_argument("--model-path", default="slm_outputs/model")
    text_parser.add_argument("--description", required=True)
    text_parser.add_argument("--name", default="")
    text_parser.add_argument("--schema", default="")
    text_parser.add_argument("--max-length", type=int, default=256)
    text_parser.set_defaults(function=predict_text)

    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.function(arguments)