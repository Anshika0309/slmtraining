import argparse, csv, json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["Read", "Write", "Execute", "Destructive", "Financial", "Other"]

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def build_text(row):
    name = str(row.get("name") or "[MISSING]").strip()
    description = str(row.get("description") or "[MISSING]").strip()
    schema = row.get("input_schema") or "[MISSING]"
    if isinstance(schema, dict):
        schema = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return f"[NAME] {name}\n[DESCRIPTION] {description}\n[SCHEMA] {schema}"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    rows = load_jsonl(args.input)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    predictions = []
    with torch.inference_mode():
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i:i+args.batch_size]
            enc = tok([build_text(r) for r in batch], padding=True, truncation=True,
                      max_length=args.max_length, return_tensors="pt").to(device)
            ids = model(**enc).logits.argmax(dim=1).cpu().tolist()
            predictions.extend(LABELS[j] for j in ids)

    assert len(predictions) == len(rows)
    assert all(p in LABELS for p in predictions)
    assert all("record_id" in r for r in rows)
    assert len({r["record_id"] for r in rows}) == len(rows)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["record_id","category"])
        w.writerows((r["record_id"], p) for r,p in zip(rows,predictions))

    print(f"Device: {device}")
    print(f"Rows: {len(rows)}")
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()