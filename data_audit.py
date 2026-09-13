#!/usr/bin/env python3
"""Minimal data audit for the MCP tool-classification assessment."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

LABELS = ["Read", "Write", "Execute", "Destructive", "Financial", "Other"]
REQUIRED_FIELDS = ["record_id", "server_slug", "name", "description", "input_schema"]


def find_data_dir(explicit: str | None = None) -> Path:
    """Find the supplied data directory from a CLI argument or common locations."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend([Path("data"), Path.cwd() / "data", Path(__file__).resolve().parent / "data"])

    for path in candidates:
        if (path / "train.jsonl").exists() and (path / "validation.jsonl").exists():
            return path.resolve()
    raise FileNotFoundError("Could not find data/train.jsonl and data/validation.jsonl. Use --data-dir.")


def load_jsonl(path: Path, labeled: bool) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Load JSONL while counting malformed JSON, missing fields, and invalid schemas."""
    records: list[dict[str, Any]] = []
    stats = Counter()

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stats["total_lines"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["malformed_json_rows"] += 1
                continue

            if not isinstance(row, dict):
                stats["non_object_rows"] += 1
                continue

            for field in REQUIRED_FIELDS + (["category"] if labeled else []):
                if field not in row:
                    stats[f"missing_{field}"] += 1
                elif row[field] is None or (isinstance(row[field], str) and not row[field].strip()):
                    stats[f"empty_{field}"] += 1

            schema_text = row.get("input_schema", "")
            if isinstance(schema_text, str) and schema_text.strip():
                try:
                    schema = json.loads(schema_text)
                    if not isinstance(schema, dict):
                        stats["schema_not_object"] += 1
                except json.JSONDecodeError:
                    stats["malformed_input_schema"] += 1
            else:
                stats["unusable_input_schema"] += 1

            if labeled and row.get("category") not in LABELS:
                stats["invalid_category"] += 1

            row["_line_number"] = line_number
            records.append(row)

    return records, dict(stats)


def canonical_schema(schema_text: Any) -> str:
    """Canonicalize valid JSON schemas; retain malformed text instead of dropping it."""
    if not isinstance(schema_text, str) or not schema_text.strip():
        return "[MISSING]"
    try:
        schema = json.loads(schema_text)
        return json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except json.JSONDecodeError:
        return schema_text.strip()


def build_text(row: dict[str, Any]) -> str:
    """Recommended model input: explicit field boundaries without IDs or server slugs."""
    name = str(row.get("name") or "[MISSING]").strip()
    description = str(row.get("description") or "[MISSING]").strip()
    schema = canonical_schema(row.get("input_schema"))
    return f"[NAME] {name}\n[DESCRIPTION] {description}\n[SCHEMA] {schema}"


def normalized_text(text: str, aggressive: bool = False) -> str:
    """Normalize text for exact and conservative near-duplicate fingerprints."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    if aggressive:
        text = re.sub(r"https?://\S+", "<url>", text)
        text = re.sub(r"\b[0-9a-f]{8,}\b", "<hex>", text)
        text = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", text)
        text = re.sub(r"[^a-z0-9_<>{}\[\]:,.-]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
    return text


def fingerprint(text: str, aggressive: bool = False) -> str:
    return hashlib.sha1(normalized_text(text, aggressive).encode("utf-8")).hexdigest()


def label_audit(splits: dict[str, list[dict[str, Any]]], output_dir: Path) -> dict[str, Any]:
    """Count label imbalance and save train/validation bar charts."""
    result: dict[str, Any] = {}
    for split_name in ("train", "validation"):
        counts = Counter(row.get("category") for row in splits[split_name])
        total = sum(counts.values())
        valid_counts = {label: counts[label] for label in LABELS}
        result[split_name] = {
            "rows": total,
            "counts": valid_counts,
            "percentages": {label: round(100 * valid_counts[label] / total, 3) if total else 0 for label in LABELS},
            "imbalance_ratio": round(max(valid_counts.values()) / max(1, min(valid_counts.values())), 2),
        }

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for axis, split_name in zip(axes, ("train", "validation")):
            counts = [result[split_name]["counts"][label] for label in LABELS]
            bars = axis.bar(LABELS, counts, color="#4776E6")
            axis.set_title(f"{split_name.title()} label distribution")
            axis.set_ylabel("Records")
            axis.tick_params(axis="x", rotation=30)
            for bar, count in zip(bars, counts):
                axis.text(bar.get_x() + bar.get_width() / 2, count, str(count), ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        chart_path = output_dir / "label_distribution.png"
        fig.savefig(chart_path, dpi=160)
        plt.close(fig)
        result["chart"] = str(chart_path)
    except ImportError:
        result["chart"] = "Not generated: install matplotlib"

    return result


def field_audit(load_stats: dict[str, dict[str, int]]) -> dict[str, Any]:
    """Summarize missing and malformed fields for all splits."""
    return load_stats


def duplicate_audit(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Measure exact and conservative normalized duplicate risks."""
    exact_maps: dict[str, dict[str, list[dict[str, Any]]]] = {}
    near_maps: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for split_name, rows in splits.items():
        exact_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        near_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            text = build_text(row)
            exact_groups[fingerprint(text)].append(row)
            near_groups[fingerprint(text, aggressive=True)].append(row)
        exact_maps[split_name] = exact_groups
        near_maps[split_name] = near_groups

    def repeated_rows(groups: dict[str, list[dict[str, Any]]]) -> int:
        return sum(len(group) for group in groups.values() if len(group) > 1)

    train_exact = set(exact_maps["train"])
    train_near = set(near_maps["train"])
    result = {
        split: {
            "exact_duplicate_rows_within_split": repeated_rows(exact_maps[split]),
            "normalized_duplicate_rows_within_split": repeated_rows(near_maps[split]),
        }
        for split in splits
    }

    for split in ("validation", "test"):
        result[split]["rows_exactly_seen_in_train"] = sum(
            len(rows) for key, rows in exact_maps[split].items() if key in train_exact
        )
        result[split]["rows_normalized_seen_in_train"] = sum(
            len(rows) for key, rows in near_maps[split].items() if key in train_near
        )

    conflicting = 0
    combined_groups: dict[str, set[str]] = defaultdict(set)
    for split in ("train", "validation"):
        for row in splits[split]:
            combined_groups[fingerprint(build_text(row), aggressive=True)].add(str(row.get("category")))
    conflicting = sum(1 for labels in combined_groups.values() if len(labels) > 1)
    result["normalized_groups_with_conflicting_labels"] = conflicting
    return result


def leakage_audit(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Check grouping leakage and obvious category-name leakage in model fields."""
    server_sets = {name: {str(row.get("server_slug")) for row in rows} for name, rows in splits.items()}
    overlaps = {
        "train_validation_server_overlap": len(server_sets["train"] & server_sets["validation"]),
        "train_test_server_overlap": len(server_sets["train"] & server_sets["test"]),
        "validation_test_server_overlap": len(server_sets["validation"] & server_sets["test"]),
    }

    label_mentions = Counter()
    for split in ("train", "validation"):
        for row in splits[split]:
            category = str(row.get("category", "")).lower()
            searchable = " ".join(str(row.get(field) or "") for field in ("name", "description", "input_schema")).lower()
            if category and re.search(rf"\b{re.escape(category)}\b", searchable):
                label_mentions[split] += 1

    record_ids = {name: [str(row.get("record_id")) for row in rows] for name, rows in splits.items()}
    return {
        **overlaps,
        "duplicate_record_ids": {name: len(ids) - len(set(ids)) for name, ids in record_ids.items()},
        "rows_containing_their_label_word": dict(label_mentions),
        "excluded_from_model_text": ["record_id", "server_slug", "category"],
        
    }


def representation_audit(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Describe the chosen structured representation and basic length statistics."""
    lengths = []
    for row in splits["train"]:
        lengths.append(len(build_text(row).split()))
    lengths.sort()

    def percentile(p: float) -> int:
        if not lengths:
            return 0
        index = min(len(lengths) - 1, round((len(lengths) - 1) * p))
        return lengths[index]

    return {
        "chosen_format": "[NAME] <name>\\n[DESCRIPTION] <description>\\n[SCHEMA] <canonical JSON schema>",
        "included_fields": ["name", "description", "input_schema"],
        "excluded_fields": ["record_id", "server_slug", "category"],
        "train_word_length": {
            "min": lengths[0] if lengths else 0,
            "median": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": lengths[-1] if lengths else 0,
        },
    }


def print_section(title: str, data: Any, analysis: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Analysis: {analysis}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the MCP classification dataset.")
    parser.add_argument("--data-dir", default=None, help="Directory containing train.jsonl, validation.jsonl, and test_unlabeled.jsonl")
    parser.add_argument("--output-dir", default="audit_outputs", help="Directory for audit_report.json and diagrams")
    args = parser.parse_args()

    data_dir = find_data_dir(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_specs = {
        "train": (data_dir / "train.jsonl", True),
        "validation": (data_dir / "validation.jsonl", True),
        "test": (data_dir / "test_unlabeled.jsonl", False),
    }

    splits: dict[str, list[dict[str, Any]]] = {}
    load_stats: dict[str, dict[str, int]] = {}
    for split_name, (path, labeled) in file_specs.items():
        splits[split_name], load_stats[split_name] = load_jsonl(path, labeled)

    report = {
        "label_distribution": label_audit(splits, output_dir),
        "missing_or_malformed_fields": field_audit(load_stats),
        "duplicate_or_near_duplicate_risks": duplicate_audit(splits),
        "potential_target_leakage": leakage_audit(splits),
        "chosen_text_representation": representation_audit(splits),
    }

    report_path = output_dir / "audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    train_ratio = report["label_distribution"]["train"]["imbalance_ratio"]
    print_section(
        "1. LABEL DISTRIBUTION AND IMBALANCE",
        report["label_distribution"],
        f"The data is severely imbalanced: the largest-to-smallest train-class ratio is {train_ratio}:1. "
        "Accuracy will be dominated by Read/Write, so macro F1 and per-class F1 must drive model selection.",
    )
    print_section(
        "2. MISSING OR MALFORMED FIELDS",
        report["missing_or_malformed_fields"],
        "Malformed JSONL rows cannot be used; malformed input_schema values should be retained as raw text rather than silently discarded. "
        "Missing fields should receive an explicit [MISSING] marker so preprocessing remains deterministic.",
    )
    print_section(
        "3. DUPLICATE OR NEAR-DUPLICATE RISKS",
        report["duplicate_or_near_duplicate_risks"],
        "Repeated tool descriptions can inflate validation scores through memorization even though servers are disjoint. "
        "Report both overall and novel-input performance later; do not remove supplied rows without a documented experiment.",
    )
    print_section(
        "4. POTENTIAL TARGET LEAKAGE",
        report["potential_target_leakage"],
        "Zero server overlap is required by the assessment design. record_id, raw server_slug, and category are excluded because they can encode identity or the target rather than tool semantics.",
    )
    print_section(
        "5. CHOSEN TEXT REPRESENTATION",
        report["chosen_text_representation"],
        "The structured Name + Description + Schema representation is the strongest default because it combines action cues, natural-language intent, and argument semantics. "
        "Field markers help both TF-IDF and transformer models distinguish these information sources.",
    )

    print(f"\nSaved report: {report_path.resolve()}")
    print(f"Saved chart:  {(output_dir / 'label_distribution.png').resolve()}")


if __name__ == "__main__":
    main()