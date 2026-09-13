import argparse, csv, json
from collections import Counter
from pathlib import Path
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

def predict(rows, model_path, max_length, batch_size):
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]
            enc = tok([build_text(r) for r in batch], padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt").to(device)
            probs = torch.softmax(model(**enc).logits, dim=1)
            top2 = torch.topk(probs, 2, dim=1)
            for j in range(len(batch)):
                a, b = int(top2.indices[j,0]), int(top2.indices[j,1])
                out.append({
                    "predicted": LABELS[a],
                    "confidence": float(top2.values[j,0]),
                    "second": LABELS[b],
                    "margin": float(top2.values[j,0] - top2.values[j,1])
                })
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="slm_outputs/model")
    p.add_argument("--validation", default="data/validation.jsonl")
    p.add_argument("--output-dir", default="error_analysis")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    rows = load_jsonl(args.validation)
    preds = predict(rows, args.model_path, args.max_length, args.batch_size)

    errors = []
    for row, pred in zip(rows, preds):
        if pred["predicted"] != row["category"]:
            errors.append({
                "record_id": row.get("record_id",""),
                "name": row.get("name",""),
                "description": row.get("description",""),
                "true": row["category"],
                **pred
            })

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fields = ["record_id","name","description","true","predicted","confidence","second","margin"]
    with open(out/"validation_errors.csv","w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(errors)

    conf = Counter((e["true"], e["predicted"]) for e in errors)
    write_execute = [e for e in errors if {e["true"], e["predicted"]} == {"Write","Execute"}]
    minority = [e for e in errors if e["true"] in {"Execute","Financial","Other"}]

    selected = []
    for group in [minority, write_execute,
                  sorted(errors,key=lambda x:x["confidence"], reverse=True),
                  sorted(errors,key=lambda x:x["margin"]),
                  errors]:
        for e in group:
            if e not in selected:
                selected.append(e)
            if len(selected) == 20: break
        if len(selected) == 20: break

    with open(out/"review_20_errors.csv","w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(selected)

    report = [
        "# Validation Error Analysis",
        f"- Total validation records: {len(rows)}",
        f"- Total errors: {len(errors)}",
        f"- Error rate: {len(errors)/len(rows):.4f}",
        "",
        "## Top confusion pairs",
    ]
    for (true,pred),count in conf.most_common(10):
        report.append(f"- {true} -> {pred}: {count}")

    report += [
        "",
        "## Recurring failure modes",
        f"1. Majority-class pull: {sum(e['predicted'] in {'Read','Write'} for e in errors)} errors were predicted as Read/Write. Improvement: moderate class-weighted loss.",
        f"2. Write vs Execute ambiguity: {len(write_execute)} direct confusions. Improvement: emphasize state change versus actual code/command execution.",
        f"3. Minority-class weakness: {len(minority)} errors came from Execute/Financial/Other. Improvement: targeted weighting and minority-focused examples.",
        "",
        "## 20 errors for manual review",
    ]
    for i,e in enumerate(selected,1):
        desc = str(e["description"]).replace("\n"," ")[:220]
        report.append(f"{i}. `{e['name']}` | true={e['true']} | pred={e['predicted']} | conf={e['confidence']:.4f} | {desc}")

    (out/"error_analysis.md").write_text("\n".join(report), encoding="utf-8")

    print(f"Validation records: {len(rows)}")
    print(f"Errors: {len(errors)}")
    print(f"Error rate: {len(errors)/len(rows):.4f}")
    print("Top confusion pairs:")
    for (true,pred),count in conf.most_common(5):
        print(f"  {true} -> {pred}: {count}")
    print(f"Saved: {out/'validation_errors.csv'}")
    print(f"Saved: {out/'review_20_errors.csv'}")
    print(f"Saved: {out/'error_analysis.md'}")

if __name__ == "__main__":
    main()