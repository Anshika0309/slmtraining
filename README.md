# AI Engineer Take-Home Assessment: Small Language Model Classification

## Objective

Build and evaluate a small language model that classifies an MCP tool into exactly one action category:

- `Read`
- `Write`
- `Execute`
- `Destructive`
- `Financial`
- `Other`

The classifier should infer the label from the tool's natural-language and schema fields. This exercise tests practical model development, data judgment, evaluation rigor, and production thinking—not only the final metric.

## Timebox and compute budget

- Recommended effort: **6 hours**; hard cap: **8 hours**.
- Use an open-weight language model with **3 billion parameters or fewer**.
- The complete experiment must be feasible on a single commodity GPU with **24 GB VRAM or less**. Parameter-efficient fine-tuning such as LoRA or QLoRA is encouraged.
- Training a model from scratch is not expected.
- External labeled datasets are not allowed. Public pretrained model weights and standard open-source libraries are allowed.

## Provided files

- `data/train.jsonl`: labeled training records.
- `data/validation.jsonl`: labeled validation records.
- `data/test_unlabeled.jsonl`: held-out test records without labels.
- `data/servers_public.jsonl`: optional server-level metadata, with direct category labels removed.
- `data/split_summary.json`: split sizes and label counts.

Each tool record contains:

```json
{
  "record_id": "stable identifier",
  "server_slug": "server grouping key",
  "name": "tool/function name",
  "description": "natural-language tool description",
  "input_schema": "JSON schema serialized as text",
  "category": "present only in train and validation"
}
```

The train, validation, and test sets are isolated by `server_slug`. Do not merge the sets or create a new random row-level split.

## Required work

### 1. Data audit

Document:

- label distribution and imbalance;
- missing or malformed fields;
- duplicate or near-duplicate risks;
- potential target leakage;
- the text representation you chose and why.

### 2. Baseline

Implement at least one non-neural or non-generative baseline, such as TF-IDF plus logistic regression or a linear SVM. Report the same metrics used for the SLM.

### 3. SLM training

Fine-tune or adapt an SLM to predict one of the six labels. You may formulate this as sequence classification or constrained text generation.

Your solution must explain:

- model and tokenizer selection;
- prompt/input format;
- maximum sequence length and truncation policy;
- class-imbalance strategy;
- training configuration;
- checkpoint selection and stopping rule;
- steps taken to make output labels valid and deterministic.

### 4. Evaluation

Use **macro F1** as the primary metric. Also report:

- per-class precision, recall, and F1;
- weighted F1;
- accuracy;
- confusion matrix;
- invalid-output rate, if using generative classification;
- model size, peak memory if available, and inference latency or throughput.

Do not tune on the held-out test set. The employer will score `predictions.csv` against private labels.

### 5. Error analysis

Review at least 20 validation errors. Identify at least three recurring failure modes and propose concrete improvements. Include examples, especially for minority classes and ambiguous `Write` versus `Execute` behavior.

### 6. Inference interface

Provide a command that produces predictions for a JSONL file:

```bash
python predict.py \
  --model-path <path-or-model-id> \
  --input data/test_unlabeled.jsonl \
  --output predictions.csv
```

The output must contain exactly:

```csv
record_id,category
```

Every input record must receive exactly one valid category.

## Deliverables

Submit a repository or archive containing:

1. `README.md` with setup, commands, decisions, and results.
2. Reproducible data-preparation code.
3. Baseline training/evaluation code.
4. SLM training code or notebook.
5. `predict.py` or an equivalent inference entry point.
6. `predictions.csv` for the supplied held-out test set.
7. `metrics.json` and a confusion-matrix image or table.
8. A short model card covering intended use, limitations, and known failure modes.

Do not include large model weights in the submission. Provide a model identifier, adapter artifact, or reproducible checkpoint instructions.

## Evaluation priorities

We value correct experimental design, leakage prevention, reproducibility, thoughtful error analysis, and deployable inference. A clear, honest solution with well-explained tradeoffs is stronger than an opaque solution reporting one high score.

## Follow-up interview

Be prepared for a 45-minute discussion covering:

- why the model improved or failed to improve over the baseline;
- how you would handle new categories or multi-label tools;
- calibration and abstention for high-risk predictions;
- production monitoring and drift;
- how you would reduce cost and latency without materially hurting macro F1.
