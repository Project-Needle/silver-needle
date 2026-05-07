# LLM Vendor Accuracy Audit

A lightweight tool for evaluating and comparing LLM vendor API endpoints on labeled benchmark datasets. Designed for black-box auditing: given a dataset with known answers, it queries the vendor, checks correctness, and produces a Markdown report.

## Files

```
accuracy_probe.py        # Main evaluation script
compare_results.py       # Generate a comparison report from two saved result JSONs
request_llm.py           # Low-level API client (OpenAI-compatible and Anthropic)
dataset/
  audit_sample15.json    # 15-question audit sample (GPQA / GSM8K / C-Eval)
output/                  # Recommended directory for all result files (create before first run)
```

## Requirements

```
pip install requests
```

Python 3.10+ required (uses `dict[...]` type hints).

---

## Usage

### 1. Single vendor — produce one result JSON and report

```python
python accuracy_probe.py \
  --data dataset/audit_sample15.json \
  --base-url https://api.example.com \
  --api-key YOUR_API_KEY \
  --model model-name \
  --output-dir output/
```

Outputs to `output/`:
- `results_<hostname>_<timestamp>.json` — raw per-question results
- `report_<hostname>_<timestamp>.md` — Markdown summary report

Optional flags:
```
--api-style   openai (default) | anthropic
--name        Display name for the vendor (defaults to hostname from --base-url)
--temperature 0.7 (default)
--max-tokens  1024 (default)
--delay       1.0  Seconds between requests (default)
--system      System prompt, Anthropic style only
```

---

### 2. Two vendors in one run — produce a comparison report

Pass a second vendor with `--base-url-b` and related flags. Both vendors are tested on the **same questions in the same order**, ensuring a fair comparison.

```python
python accuracy_probe.py \
  --data dataset/audit_sample15.json \
  --base-url   https://api.reference.com --api-key KEY_A --model model-a --api-style openai \
  --base-url-b https://api.audit.com     --api-key-b KEY_B --model-b model-b --api-style-b anthropic \
  --output-dir output/
```

Each vendor has its own independent `--api-style` flag, so a reference vendor using the OpenAI-compatible endpoint and an audit target using the Anthropic endpoint can be compared in a single run.

Outputs to `output/`:
- `results_<hostname-a>_<timestamp>.json`
- `results_<hostname-b>_<timestamp>.json`
- `comparison_report_<timestamp>.md` — side-by-side accuracy and response-length comparison with audit findings

Full per-vendor flag reference:

| Vendor A | Vendor B | Description |
|---|---|---|
| `--base-url` | `--base-url-b` | API base URL |
| `--api-key` | `--api-key-b` | API key |
| `--model` | `--model-b` | Model name |
| `--api-style` | `--api-style-b` | `openai` (default) or `anthropic` |
| `--name` | `--name-b` | Display name (defaults to hostname) |
| `--system` | `--system-b` | System prompt (Anthropic style only) |

---

### 3. Generate a comparison report from two existing result JSONs

If you tested two vendors separately (at different times, or by hand), use `compare_results.py` to produce a comparison report from the saved JSONs without making any new API calls.

```python
python compare_results.py \
  output/results_vendor_a_20250507_120000.json \
  output/results_vendor_b_20250507_153000.json \
  --output-dir output/
```

The first argument is treated as **Vendor A (reference)** and the second as **Vendor B (audit target)**. This determines the direction of deltas and the wording of audit findings in the report.

To specify an exact output path instead of a directory:
```python
python compare_results.py \
  output/results_a.json \
  output/results_b.json \
  --output output/final_audit_report.md
```

---

## Dataset format

The tool expects a JSON array. Each element must have at minimum:

```json
[
  {
    "id": "q001",
    "type": "multiple_choice",
    "question": "Which of the following is correct?\n\nA) ...\nB) ...\nC) ...\nD) ...",
    "answer": "B",
    "answer_aliases": ["B"]
  },
  {
    "id": "q002",
    "type": "math",
    "question": "Janet has 16 eggs ...",
    "answer": "18",
    "answer_aliases": ["18"]
  }
]
```

Supported `type` values:

| Type | Answer extraction | Notes |
|---|---|---|
| `multiple_choice` | Last A/B/C/D letter found in response | Include choices in the `question` field |
| `math` | Number after `####`, or last number in response | GSM8K-style |
| `text` | First 100 chars of response (exact match) | For open-ended with exact expected output |

`answer_aliases` is a list of accepted strings (case-insensitive). Use it to handle equivalent forms, e.g. `["42", "42.0"]`.

---

## Result JSON format

The raw result files saved by `accuracy_probe.py` contain everything needed to regenerate a report later:

```json
{
  "name": "api.example.com",
  "model": "model-name",
  "n_questions": 15,
  "correct_count": 9,
  "accuracy": 0.6,
  "avg_response_length": 312.4,
  "median_response_length": 287,
  "avg_latency": 2.31,
  "per_type_accuracy": {
    "multiple_choice": 0.7,
    "math": 0.5
  },
  "responses": [
    {
      "question_id": "gpqa_01",
      "question_type": "multiple_choice",
      "content": "...",
      "response_length": 423,
      "latency": 1.84,
      "extracted": "B",
      "correct": true
    }
  ]
}
```

`avg_latency` may be `null` if results were constructed manually — all other fields are required.
