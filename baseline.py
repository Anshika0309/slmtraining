"""Train and evaluate a TF-IDF + LinearSVC baseline for MCP tool classification."""

import argparse
import json
import os
import time
from pathlib import Path

import joblib
import psutil
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

LABELS = ["Read", "Write", "Execute", "Destructive", "Financial", "Other"]


def load_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSON at {path}:{line_number}") from error
            if "category" not in row:
                raise ValueError(f"Missing category at {path}:{line_number}")
            rows.append(row)
    return rows


def build_text(row):
    name = row.get("name") or "[MISSING]"
    description = row.get("description") or "[MISSING]"
    schema = row.get("input_schema") or "[MISSING]"
    return f"[NAME] {name} [DESCRIPTION] {description} [SCHEMA] {schema}"


def memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def save_confusion_matrix(matrix, output_dir: Path):
    csv_path = output_dir / "baseline_confusion_matrix.csv"
    with csv_path.open("w", encoding="utf-8") as file:
        file.write("actual\\predicted," + ",".join(LABELS) + "\n")
        for label, row in zip(LABELS, matrix):
            file.write(label + "," + ",".join(map(str, row)) + "\n")

    try:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(8, 6))
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_xticks(range(len(LABELS)), LABELS, rotation=35, ha="right")
        axis.set_yticks(range(len(LABELS)), LABELS)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_title("TF-IDF + LinearSVC Confusion Matrix")
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                axis.text(j, i, matrix[i, j], ha="center", va="center")
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(output_dir / "baseline_confusion_matrix.png", dpi=160)
        plt.close(figure)
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("data/validation.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("baseline_outputs"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(args.train)
    validation_rows = load_jsonl(args.validation)
    x_train = [build_text(row) for row in train_rows]
    y_train = [row["category"] for row in train_rows]
    x_validation = [build_text(row) for row in validation_rows]
    y_validation = [row["category"] for row in validation_rows]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("classifier", LinearSVC(C=2.0, random_state=42)),
    ])

    start_memory = memory_mb()
    train_start = time.perf_counter()
    model.fit(x_train, y_train)
    training_seconds = time.perf_counter() - train_start
    peak_memory_mb = memory_mb()

    inference_start = time.perf_counter()
    predictions = model.predict(x_validation)
    inference_seconds = time.perf_counter() - inference_start

    precision, recall, class_f1, support = precision_recall_fscore_support(
        y_validation, predictions, labels=LABELS, zero_division=0
    )
    matrix = confusion_matrix(y_validation, predictions, labels=LABELS)

    model_path = args.output_dir / "baseline_model.joblib"
    joblib.dump(model, model_path, compress=3)
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    metrics = {
        "primary_metric": "macro_f1",
        "macro_f1": f1_score(y_validation, predictions, average="macro"),
        "weighted_f1": f1_score(y_validation, predictions, average="weighted"),
        "accuracy": accuracy_score(y_validation, predictions),
        "invalid_output_rate": 0.0,
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(class_f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(LABELS)
        },
        "confusion_matrix": matrix.tolist(),
        "efficiency": {
            "training_seconds": training_seconds,
            "inference_seconds": inference_seconds,
            "validation_records": len(validation_rows),
            "latency_ms_per_record": 1000 * inference_seconds / len(validation_rows),
            "throughput_records_per_second": len(validation_rows) / inference_seconds,
            "model_size_mb": model_size_mb,
            "peak_process_memory_mb": peak_memory_mb,
            "memory_increase_mb": max(0.0, peak_memory_mb - start_memory),
        },
        "configuration": {
            "text_fields": ["name", "description", "input_schema"],
            "excluded_fields": ["record_id", "server_slug", "category"],
            "features": "word TF-IDF unigrams and bigrams",
            "classifier": "LinearSVC",
            "C": 2.0,
            "class_weight": None,
            "random_state": 42,
        },
    }

    with (args.output_dir / "baseline_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    with (args.output_dir / "baseline_classification_report.txt").open("w", encoding="utf-8") as file:
        file.write(classification_report(
            y_validation, predictions, labels=LABELS, digits=4, zero_division=0
        ))
    save_confusion_matrix(matrix, args.output_dir)

    print("\nTF-IDF + LinearSVC BASELINE")
    print(f"Train / validation rows : {len(train_rows):,} / {len(validation_rows):,}")
    print(f"Macro F1                : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1             : {metrics['weighted_f1']:.4f}")
    print(f"Accuracy                : {metrics['accuracy']:.4f}")
    print(f"Invalid-output rate     : 0.0000 (non-generative classifier)")
    print(f"Model size              : {model_size_mb:.2f} MB")
    print(f"Peak process memory     : {peak_memory_mb:.2f} MB")
    print(f"Inference latency       : {metrics['efficiency']['latency_ms_per_record']:.3f} ms/record")
    print(f"Inference throughput    : {metrics['efficiency']['throughput_records_per_second']:.1f} records/s")
    print("\nPer-class metrics")
    for label in LABELS:
        values = metrics["per_class"][label]
        print(f"{label:11s} P={values['precision']:.4f} R={values['recall']:.4f} "
              f"F1={values['f1']:.4f} N={values['support']}")
    print("\nAnalysis: Macro F1 is lower than weighted F1 when minority classes remain difficult.")
    print("The confusion matrix and per-class scores should guide later SLM improvements.")
    print(f"Outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()