"""
Accuracy probe: evaluate LLM vendor(s) on a labeled dataset.

Runs all questions in order (no random sampling) so results are deterministic.
Supports single-vendor mode (one report) and dual-vendor comparison mode.

Dataset format (JSON array):
[
  {
    "id": "q001",
    "type": "multiple_choice" | "math" | "text",
    "question": "...",
    "answer": "A",
    "answer_aliases": ["A", "a"],
    "choices": ["A) ...", "B) ...", "C) ...", "D) ..."]  # optional
  },
  ...
]

Usage — single vendor:
  python accuracy_probe.py --data dataset.json \\
    --base-url https://api.openai.com --api-key sk-... --model gpt-4

Usage — comparison (vendor A vs vendor B):
  python accuracy_probe.py --data dataset.json \\
    --base-url https://api.ref.com  --api-key sk-... --model gpt-4 \\
    --base-url-b https://api.audit.com --api-key-b sk-... --model-b model-name
"""

import json
import math
import re
import time
import argparse
import statistics
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional

from request_llm import call_api


# ---------------------------------------------------------------------------
# Answer extraction (compatible with difficulty_curve_probe.py)
# ---------------------------------------------------------------------------

def extract_answer(text: str, question: dict) -> Optional[str]:
    qtype = question.get("type", "text")

    if qtype == "multiple_choice":
        matches = re.findall(r'(?<![A-Za-z])([A-D])(?![A-Za-z])', text)
        return matches[-1].upper() if matches else None

    if qtype == "math":
        m = re.search(r'####\s*([\d,./\s]+)', text)
        if m:
            return m.group(1).strip().replace(',', '')
        m = re.search(r'(?:answer is|=)\s*([\d,./]+)', text, re.IGNORECASE)
        if m:
            return m.group(1).strip().replace(',', '')
        nums = re.findall(r'\b\d+(?:[.,/]\d+)*\b', text)
        return nums[-1].replace(',', '') if nums else None

    return text.strip()[:100]


def is_correct(extracted: Optional[str], question: dict) -> bool:
    if extracted is None:
        return False
    aliases = [a.lower().strip() for a in question.get("answer_aliases", [])]
    return extracted.lower().strip() in aliases


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def vendor_name(base_url: str, override: Optional[str] = None) -> str:
    if override:
        return override
    return urlparse(base_url).hostname or base_url


def safe_filename(name: str) -> str:
    """Strip characters unsafe for filenames."""
    return re.sub(r'[^\w\-.]', '_', name)


def fmt_pct(v: float) -> str:
    return f"{v:.1%}"


def fmt_latency(v) -> str:
    return f"{v}s" if v is not None else "N/A"


# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------

def load_questions(data_file: Path) -> list[dict]:
    with open(data_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
    if not isinstance(questions, list):
        raise ValueError(f"{data_file} must contain a JSON array")
    print(f"Loaded {len(questions)} questions from {data_file.name}")
    return questions


# ---------------------------------------------------------------------------
# Query and evaluate
# ---------------------------------------------------------------------------

ANSWER_SUFFIX = (
    "\n\nProvide your final answer clearly at the end. "
    "For multiple choice questions, state the letter (A/B/C/D). "
    "For math problems, state the final number after '####'."
)


def query_question(question: dict, client_cfg: dict, temperature: float) -> dict:
    prompt = question["question"] + ANSWER_SUFFIX
    messages = [{"role": "user", "content": prompt}]

    t0 = time.time()
    content = call_api(messages, client_cfg, temperature)
    latency = time.time() - t0

    extracted = extract_answer(content, question)
    correct = is_correct(extracted, question)

    return {
        "question_id": question.get("id", "unknown"),
        "question_type": question.get("type", "text"),
        "content": content,
        "response_length": len(content),
        "latency": round(latency, 2),
        "extracted": extracted,
        "correct": correct,
    }


def evaluate_vendor(
    name: str,
    questions: list[dict],
    client_cfg: dict,
    temperature: float,
    delay: float,
) -> dict:
    print(f"\n{'='*60}")
    print(f"Vendor: {name}  |  Model: {client_cfg['model']}")
    print(f"{'='*60}")

    responses = []
    for i, q in enumerate(questions):
        resp = query_question(q, client_cfg, temperature)
        responses.append(resp)
        status = "✓" if resp["correct"] else "✗"
        print(f"  [{i+1:3}/{len(questions)}] {q.get('id', '?'):15} {status}  "
              f"extracted={str(resp['extracted'])!r:10}  "
              f"len={resp['response_length']:5}  {resp['latency']:.1f}s")

        if delay > 0 and i < len(questions) - 1:
            time.sleep(delay)

    return {"name": name, "model": client_cfg["model"], "responses": responses}


# ---------------------------------------------------------------------------
# Compute aggregate stats
# ---------------------------------------------------------------------------

def compute_stats(eval_result: dict) -> dict:
    responses = eval_result["responses"]
    n = len(responses)
    if n == 0:
        return {}

    correct_count = sum(r["correct"] for r in responses)
    lengths = [r["response_length"] for r in responses]
    latencies = [r["latency"] for r in responses]

    by_type: dict[str, dict] = {}
    for r in responses:
        qtype = r["question_type"]
        if qtype not in by_type:
            by_type[qtype] = {"correct": 0, "total": 0}
        by_type[qtype]["total"] += 1
        if r["correct"]:
            by_type[qtype]["correct"] += 1

    per_type_accuracy = {
        qtype: round(v["correct"] / v["total"], 4)
        for qtype, v in by_type.items()
    }

    return {
        "name": eval_result["name"],
        "model": eval_result["model"],
        "n_questions": n,
        "correct_count": correct_count,
        "accuracy": round(correct_count / n, 4),
        "avg_response_length": round(statistics.mean(lengths), 1),
        "median_response_length": int(statistics.median(lengths)),
        "avg_latency": round(statistics.mean(latencies), 2),
        "per_type_accuracy": per_type_accuracy,
        "responses": responses,
    }


# ---------------------------------------------------------------------------
# Markdown report: single vendor
# ---------------------------------------------------------------------------

def _question_table_rows(responses: list[dict]) -> list[str]:
    rows = []
    for r in responses:
        status = "✓" if r["correct"] else "✗"
        ext = str(r["extracted"]) if r["extracted"] is not None else "—"
        rows.append(
            f"| {r['question_id']} | {r['question_type']} | {status} "
            f"| `{ext}` | {r['response_length']} | {r['latency']}s |"
        )
    return rows


def generate_report_single(stats: dict, timestamp: str) -> str:
    lines = [
        "# Accuracy Probe Report",
        "",
        f"**Generated:** {timestamp}",
        "",
        "## Vendor",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Name | `{stats['name']}` |",
        f"| Model | `{stats['model']}` |",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total Questions | {stats['n_questions']} |",
        f"| Correct | {stats['correct_count']} |",
        f"| **Accuracy** | **{fmt_pct(stats['accuracy'])}** |",
        f"| Avg Response Length | {stats['avg_response_length']} chars |",
        f"| Median Response Length | {stats['median_response_length']} chars |",
        f"| Avg Latency | {fmt_latency(stats['avg_latency'])} |",
        "",
    ]

    if stats["per_type_accuracy"]:
        lines += [
            "## Per-Type Accuracy",
            "",
            "| Question Type | Accuracy |",
            "|---|---|",
        ]
        for qtype, acc in sorted(stats["per_type_accuracy"].items()):
            lines.append(f"| {qtype} | {fmt_pct(acc)} |")
        lines.append("")

    lines += [
        "## Per-Question Results",
        "",
        "| ID | Type | Correct | Extracted | Length | Latency |",
        "|---|---|---|---|---|---|",
    ]
    lines += _question_table_rows(stats["responses"])

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Markdown report: comparison
# ---------------------------------------------------------------------------

def _audit_findings(stats_a: dict, stats_b: dict) -> list[str]:
    lines = ["## Audit Findings", ""]

    acc_delta = stats_b["accuracy"] - stats_a["accuracy"]
    len_delta = stats_b["avg_response_length"] - stats_a["avg_response_length"]

    if acc_delta < -0.10:
        lines.append(
            f"- ⚠️  **Significant accuracy gap**: Vendor B is **{-acc_delta:.1%} less accurate** "
            f"than Vendor A. Strong signal of model substitution or capability degradation."
        )
    elif acc_delta < -0.05:
        lines.append(
            f"- ⚠️  **Moderate accuracy gap**: Vendor B is {-acc_delta:.1%} less accurate than Vendor A."
        )
    elif acc_delta < 0:
        lines.append(
            f"- Vendor B accuracy is slightly lower ({acc_delta:.1%}) — within noise range."
        )
    else:
        lines.append(
            f"- ✓  Vendor B accuracy matches or exceeds Vendor A ({acc_delta:+.1%})."
        )

    if len_delta < -100:
        lines.append(
            f"- ⚠️  **Response length gap**: Vendor B responses are **{-len_delta:.0f} chars shorter** "
            f"on average. Shorter outputs may indicate a weaker or smaller underlying model."
        )
    elif len_delta < -50:
        lines.append(
            f"- Vendor B responses are {-len_delta:.0f} chars shorter on average — worth monitoring."
        )
    else:
        lines.append(
            f"- Response length is comparable between vendors ({len_delta:+.0f} chars avg)."
        )

    lines.append("")
    return lines


def generate_report_comparison(stats_a: dict, stats_b: dict, timestamp: str) -> str:
    acc_delta = stats_b["accuracy"] - stats_a["accuracy"]
    len_delta = stats_b["avg_response_length"] - stats_a["avg_response_length"]
    lat_a = stats_a["avg_latency"]
    lat_b = stats_b["avg_latency"]
    lat_delta_str = (
        f"{lat_b - lat_a:+.2f}s" if (lat_a is not None and lat_b is not None) else "N/A"
    )

    lines = [
        "# Accuracy Probe — Comparison Report",
        "",
        f"**Generated:** {timestamp}",
        "",
        "## Vendors",
        "",
        "| | Vendor A (reference) | Vendor B (audit target) |",
        "|---|---|---|",
        f"| Name | `{stats_a['name']}` | `{stats_b['name']}` |",
        f"| Model | `{stats_a['model']}` | `{stats_b['model']}` |",
        "",
        "## Summary",
        "",
        "| Metric | Vendor A | Vendor B | Delta (B−A) |",
        "|---|---|---|---|",
        f"| Total Questions | {stats_a['n_questions']} | {stats_b['n_questions']} | — |",
        f"| Correct | {stats_a['correct_count']} | {stats_b['correct_count']} | {stats_b['correct_count'] - stats_a['correct_count']:+d} |",
        f"| **Accuracy** | **{fmt_pct(stats_a['accuracy'])}** | **{fmt_pct(stats_b['accuracy'])}** | **{acc_delta:+.1%}** |",
        f"| Avg Response Length | {stats_a['avg_response_length']} chars | {stats_b['avg_response_length']} chars | {len_delta:+.1f} chars |",
        f"| Median Response Length | {stats_a['median_response_length']} chars | {stats_b['median_response_length']} chars | — |",
        f"| Avg Latency | {fmt_latency(lat_a)} | {fmt_latency(lat_b)} | {lat_delta_str} |",
        "",
    ]

    all_types = sorted(
        set(stats_a["per_type_accuracy"]) | set(stats_b["per_type_accuracy"])
    )
    if all_types:
        lines += [
            "## Per-Type Accuracy",
            "",
            "| Question Type | Vendor A | Vendor B | Delta |",
            "|---|---|---|---|",
        ]
        for qtype in all_types:
            a_acc = stats_a["per_type_accuracy"].get(qtype)
            b_acc = stats_b["per_type_accuracy"].get(qtype)
            a_str = fmt_pct(a_acc) if a_acc is not None else "N/A"
            b_str = fmt_pct(b_acc) if b_acc is not None else "N/A"
            if a_acc is not None and b_acc is not None:
                delta_str = f"{b_acc - a_acc:+.1%}"
            else:
                delta_str = "—"
            lines.append(f"| {qtype} | {a_str} | {b_str} | {delta_str} |")
        lines.append("")

    lines += _audit_findings(stats_a, stats_b)

    # Per-question side-by-side
    b_by_id = {r["question_id"]: r for r in stats_b["responses"]}

    lines += [
        "## Per-Question Results",
        "",
        "| ID | Type | A | B | A Extracted | B Extracted |",
        "|---|---|---|---|---|---|",
    ]
    for r_a in stats_a["responses"]:
        qid = r_a["question_id"]
        r_b = b_by_id.get(qid)
        sa = "✓" if r_a["correct"] else "✗"
        sb = ("✓" if r_b["correct"] else "✗") if r_b else "—"
        ext_a = str(r_a["extracted"]) if r_a["extracted"] is not None else "—"
        ext_b = str(r_b["extracted"]) if (r_b and r_b["extracted"] is not None) else "—"
        lines.append(
            f"| {qid} | {r_a['question_type']} | {sa} | {sb} "
            f"| `{ext_a}` | `{ext_b}` |"
        )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="LLM accuracy probe — single or dual-vendor comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data", required=True, help="Path to dataset JSON file")

    # Vendor A (always required)
    p.add_argument("--base-url", required=True, help="Vendor A base URL")
    p.add_argument("--api-key", required=True, help="Vendor A API key")
    p.add_argument("--model", required=True, help="Vendor A model name")
    p.add_argument("--api-style", default="openai", choices=["openai", "anthropic"],
                   help="API style for vendor A (default: openai)")
    p.add_argument("--system", default=None, help="System prompt for vendor A (anthropic only)")
    p.add_argument("--name", default=None,
                   help="Vendor A display name (default: hostname from --base-url)")

    # Vendor B (optional — enables comparison mode)
    p.add_argument("--base-url-b", default=None,
                   help="Vendor B base URL — enables comparison mode")
    p.add_argument("--api-key-b", default=None, help="Vendor B API key")
    p.add_argument("--model-b", default=None, help="Vendor B model name")
    p.add_argument("--api-style-b", default="openai", choices=["openai", "anthropic"])
    p.add_argument("--system-b", default=None)
    p.add_argument("--name-b", default=None,
                   help="Vendor B display name (default: hostname from --base-url-b)")

    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--delay", type=float, default=1.0,
                   help="Seconds to wait between requests (default: 1.0)")
    p.add_argument("--output-dir", default=".", help="Directory for output files")

    return p.parse_args()


def main():
    args = parse_args()

    data_file = Path(args.data)
    if not data_file.is_file():
        print(f"Error: dataset file not found: {data_file}")
        return

    questions = load_questions(data_file)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_mode = args.base_url_b is not None

    # --- Vendor A ---
    name_a = vendor_name(args.base_url, args.name)
    client_cfg_a = {
        "base_url": args.base_url,
        "api_key": args.api_key,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "api_style": args.api_style,
        "system": args.system,
    }

    eval_a = evaluate_vendor(name_a, questions, client_cfg_a, args.temperature, args.delay)
    stats_a = compute_stats(eval_a)

    raw_a_path = output_dir / f"results_{safe_filename(name_a)}_{timestamp}.json"
    with open(raw_a_path, "w", encoding="utf-8") as f:
        json.dump(stats_a, f, ensure_ascii=False, indent=2)
    print(f"\nVendor A raw results: {raw_a_path}")

    # --- Vendor B (optional) ---
    if comparison_mode:
        if not args.api_key_b or not args.model_b:
            print("Error: --api-key-b and --model-b are required when --base-url-b is set")
            return

        name_b = vendor_name(args.base_url_b, args.name_b)
        client_cfg_b = {
            "base_url": args.base_url_b,
            "api_key": args.api_key_b,
            "model": args.model_b,
            "max_tokens": args.max_tokens,
            "api_style": args.api_style_b,
            "system": args.system_b,
        }

        eval_b = evaluate_vendor(name_b, questions, client_cfg_b, args.temperature, args.delay)
        stats_b = compute_stats(eval_b)

        raw_b_path = output_dir / f"results_{safe_filename(name_b)}_{timestamp}.json"
        with open(raw_b_path, "w", encoding="utf-8") as f:
            json.dump(stats_b, f, ensure_ascii=False, indent=2)
        print(f"Vendor B raw results: {raw_b_path}")

        report = generate_report_comparison(stats_a, stats_b, timestamp)
        report_path = output_dir / f"comparison_report_{timestamp}.md"
    else:
        report = generate_report_single(stats_a, timestamp)
        report_path = output_dir / f"report_{safe_filename(name_a)}_{timestamp}.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report: {report_path}")

    # Console summary
    print(f"\n{'='*60}")
    if comparison_mode:
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Vendor A: ' + name_a:40}  accuracy={fmt_pct(stats_a['accuracy'])}  "
              f"avg_len={stats_a['avg_response_length']:.0f}")
        print(f"  {'Vendor B: ' + name_b:40}  accuracy={fmt_pct(stats_b['accuracy'])}  "
              f"avg_len={stats_b['avg_response_length']:.0f}")
        delta = stats_b["accuracy"] - stats_a["accuracy"]
        print(f"  Accuracy delta (B−A): {delta:+.1%}")
    else:
        print(f"SUMMARY  —  {name_a}")
        print(f"{'='*60}")
        print(f"  Accuracy:            {fmt_pct(stats_a['accuracy'])}  "
              f"({stats_a['correct_count']}/{stats_a['n_questions']})")
        print(f"  Avg response length: {stats_a['avg_response_length']:.0f} chars")
        print(f"  Avg latency:         {fmt_latency(stats_a['avg_latency'])}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
