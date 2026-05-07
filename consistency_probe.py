"""
Consistency probe: ask the same question N times, analyze answer stability.

Metrics:
  - answer_entropy:     Shannon entropy over extracted answers (0 = perfectly consistent)
  - unique_answer_rate: fraction of unique answers (0 = all same, 1 = all different)
  - length_mean/std:    output character length distribution
  - length_cv:          coefficient of variation (std/mean), scale-free length instability
  - ngram_diversity:    mean pairwise Jaccard distance on 3-grams (0 = identical, 1 = totally different)

Probe set: hand-picked from GSM8K, ARC-Challenge, C-Eval.
Selection criteria:
  - requires genuine multi-step reasoning (not pattern matching)
  - has a short, unambiguous extractable answer
  - strong models answer correctly and consistently; weaker models show drift
"""

import re
import math
import time
import json
import argparse
import statistics
from collections import Counter
from typing import Optional

from request_llm import call_api


# ---------------------------------------------------------------------------
# Probe question bank
# ---------------------------------------------------------------------------

PROBES = [
    # --- GSM8K multi-step arithmetic ---
    {
        "id": "gsm_001",
        "source": "gsm8k",
        "type": "math",
        "question": (
            "John drives for 3 hours at a speed of 60 mph and then turns around because he "
            "realizes he forgot something very important. He drives back the way he came at "
            "a speed of 80 mph. How long does it take him to get back to where he started?"
        ),
        "answer": "2.25",  # hours; also accept 9/4
        "answer_aliases": ["2.25", "2 hours 15 minutes", "2h15m", "135 minutes", "9/4"],
    },
    {
        "id": "gsm_002",
        "source": "gsm8k",
        "type": "math",
        "question": (
            "Dana can run at a rate of speed four times faster than she can walk, but she can "
            "skip at a rate of speed that is half as fast as she can run. If she can skip at 3 "
            "miles per hour, how many miles can she travel in six hours if she spends one-third "
            "of the time running and two-thirds of the time walking?"
        ),
        "answer": "18",
        "answer_aliases": ["18"],
    },
    {
        "id": "gsm_003",
        "source": "gsm8k",
        "type": "math",
        "question": (
            "I have 10 liters of orange drink that are two-thirds water and I wish to add it "
            "to 15 liters of pineapple drink that is three-fifths water. But as I pour it, "
            "I spill one liter of the orange drink. How much water is in the remaining 24 liters?"
        ),
        "answer": "15",
        "answer_aliases": ["15"],
    },
    # --- ARC-Challenge science reasoning ---
    {
        "id": "arc_001",
        "source": "arc_challenge",
        "type": "multiple_choice",
        "question": (
            "An astronaut drops a 1.0 kg object and a 5.0 kg object on the Moon. Both objects "
            "fall a total distance of 2.0 m vertically. Which of the following best describes "
            "the objects after they have fallen a distance of 1.0 m?\n"
            "A) They have each lost kinetic energy.\n"
            "B) They have each gained the same amount of potential energy.\n"
            "C) They have each lost the same amount of potential energy.\n"
            "D) They have each gained one-half of their maximum kinetic energy."
        ),
        "answer": "D",
        "answer_aliases": ["D"],
    },
    {
        "id": "arc_002",
        "source": "arc_challenge",
        "type": "multiple_choice",
        "question": (
            "Students heated three objects to different temperatures during a classroom "
            "demonstration. Each object emitted light of a different color:\n"
            "  Object 1: blue light\n"
            "  Object 2: red light\n"
            "  Object 3: orange light\n"
            "Which list presents the objects in order from highest to lowest temperature?\n"
            "A) Object 1, Object 2, Object 3\n"
            "B) Object 1, Object 3, Object 2\n"
            "C) Object 2, Object 1, Object 3\n"
            "D) Object 2, Object 3, Object 1"
        ),
        "answer": "B",
        "answer_aliases": ["B"],
    },
    {
        "id": "arc_003",
        "source": "arc_challenge",
        "type": "multiple_choice",
        "question": (
            "A scientist maps a long region in which earthquakes originate and determines this "
            "region is a transform plate boundary. Which evidence would cause the scientist to "
            "reevaluate this determination?\n"
            "A) Volcanism also characterizes the region.\n"
            "B) Earthquake centers in the region occur at shallow depths.\n"
            "C) The region shows extensive faulting of sediments.\n"
            "D) Equal crust densities are found on opposite sides of the region."
        ),
        "answer": "A",
        "answer_aliases": ["A"],
    },
    # --- C-Eval Chinese reasoning ---
    {
        "id": "ceval_001",
        "source": "ceval",
        "type": "multiple_choice",
        "question": (
            "使用位填充方法，以01111110为帧首flag，数据为011011111111111111110010，"
            "求传送时要添加几个0？\n"
            "A) 1\nB) 2\nC) 3\nD) 4"
        ),
        "answer": "C",
        "answer_aliases": ["C"],
    },
    {
        "id": "ceval_002",
        "source": "ceval",
        "type": "multiple_choice",
        "question": (
            "在推导弹簧弹力做功的表达式时，把整个做功过程划分成很多小段，每一小段近似看作恒力做功，"
            "然后把各小段弹力所做的功相加。这里采用的物理学研究方法是：\n"
            "A) 极限思想法\nB) 控制变量法\nC) 理想实验法\nD) 微元法"
        ),
        "answer": "D",
        "answer_aliases": ["D"],
    },
]


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(text: str, probe: dict) -> Optional[str]:
    """
    Extract the final answer from model output.
    Returns a normalized string, or None if extraction fails.
    """
    if probe["type"] == "multiple_choice":
        # Look for a standalone A/B/C/D, preferring the last occurrence
        # (models often reason before giving the final answer)
        matches = re.findall(r'(?<![A-Za-z])([A-D])(?![A-Za-z])', text)
        if matches:
            return matches[-1].upper()
        return None

    if probe["type"] == "math":
        # Try #### pattern (GSM8K style)
        m = re.search(r'####\s*([\d,./\s]+)', text)
        if m:
            return m.group(1).strip().replace(',', '')
        # Try "the answer is X" pattern
        m = re.search(r'(?:answer is|=)\s*([\d,./]+)', text, re.IGNORECASE)
        if m:
            return m.group(1).strip().replace(',', '')
        # Last number in the text
        nums = re.findall(r'\b\d+(?:[.,/]\d+)*\b', text)
        if nums:
            return nums[-1].replace(',', '')
        return None

    return text.strip()[:100]


def is_correct(extracted: Optional[str], probe: dict) -> bool:
    if extracted is None:
        return False
    aliases = [a.lower().strip() for a in probe["answer_aliases"]]
    return extracted.lower().strip() in aliases


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def shannon_entropy(values: list) -> float:
    """Shannon entropy in bits over a list of discrete values."""
    if not values:
        return 0.0
    counts = Counter(values)
    n = len(values)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def ngram_set(text: str, n: int = 3) -> set:
    tokens = text.lower().split()
    return set(zip(*[tokens[i:] for i in range(n)])) if len(tokens) >= n else set()


def jaccard_distance(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def pairwise_ngram_diversity(texts: list, n: int = 3) -> float:
    """Mean pairwise Jaccard distance on n-grams across all response pairs."""
    sets = [ngram_set(t, n) for t in texts]
    pairs = [(i, j) for i in range(len(sets)) for j in range(i + 1, len(sets))]
    if not pairs:
        return 0.0
    return sum(jaccard_distance(sets[i], sets[j]) for i, j in pairs) / len(pairs)


def compute_metrics(responses: list[dict], probe: dict) -> dict:
    """
    responses: list of {"content": str, "latency": float, "raw": dict}
    """
    contents = [r["content"] for r in responses]
    answers = [extract_answer(c, probe) for c in contents]
    valid_answers = [a for a in answers if a is not None]

    lengths = [len(c) for c in contents]
    length_mean = statistics.mean(lengths) if lengths else 0
    length_std = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
    length_cv = (length_std / length_mean) if length_mean > 0 else 0.0

    answer_counts = Counter(valid_answers)
    correct_count = sum(v for k, v in answer_counts.items() if is_correct(k, probe))

    return {
        "probe_id": probe["id"],
        "n_samples": len(responses),
        "answers": answers,
        "answer_entropy": round(shannon_entropy(valid_answers), 4),
        "unique_answer_rate": round(len(set(valid_answers)) / len(valid_answers), 4) if valid_answers else None,
        "answer_distribution": dict(answer_counts),
        "correct_count": correct_count,
        "correct_rate": round(correct_count / len(responses), 4),
        "length_mean": round(length_mean, 1),
        "length_std": round(length_std, 1),
        "length_cv": round(length_cv, 4),
        "ngram_diversity": round(pairwise_ngram_diversity(contents), 4),
        "latencies": [round(r["latency"], 2) for r in responses],
    }


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_probe(probe: dict, client_cfg: dict, n: int, temperature: float, delay: float) -> list[dict]:
    """
    Call the API n times for a single probe question.
    Returns list of response dicts.
    """
    prompt = (
        probe["question"] + "\n\n"
        "Provide your final answer clearly at the end. "
        "For multiple choice questions, state the letter (A/B/C/D). "
        "For math problems, state the final number after '####'."
    )
    messages = [{"role": "user", "content": prompt}]
    responses = []

    for i in range(n):
        t0 = time.time()
        content = call_api(messages, client_cfg, temperature)
        latency = time.time() - t0

        responses.append({"content": content, "latency": latency, "raw": {}})
        print(f"  [{probe['id']}] sample {i+1}/{n} — {len(content)} chars, {latency:.1f}s")

        if delay > 0 and i < n - 1:
            time.sleep(delay)

    return responses


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(all_metrics: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("CONSISTENCY PROBE REPORT")
    print("=" * 70)

    for m in all_metrics:
        print(f"\n[{m['probe_id']}]  n={m['n_samples']}")
        print(f"  answer_entropy    : {m['answer_entropy']:.4f}  "
              f"(0=perfectly consistent, {math.log2(m['n_samples']):.2f}=max)")
        print(f"  unique_answer_rate: {m['unique_answer_rate']}")
        print(f"  answer_distribution: {m['answer_distribution']}")
        print(f"  correct_rate      : {m['correct_rate']:.0%}")
        print(f"  length mean±std   : {m['length_mean']:.0f} ± {m['length_std']:.0f} chars  "
              f"(CV={m['length_cv']:.3f})")
        print(f"  ngram_diversity   : {m['ngram_diversity']:.4f}  "
              f"(0=identical, 1=totally different)")
        print(f"  latencies (s)     : {m['latencies']}")

    print("\n" + "-" * 70)
    print("AGGREGATE")
    print("-" * 70)
    avg_entropy = statistics.mean(m["answer_entropy"] for m in all_metrics)
    avg_cv = statistics.mean(m["length_cv"] for m in all_metrics)
    avg_diversity = statistics.mean(m["ngram_diversity"] for m in all_metrics)
    avg_correct = statistics.mean(m["correct_rate"] for m in all_metrics)
    print(f"  mean answer_entropy    : {avg_entropy:.4f}")
    print(f"  mean length_cv         : {avg_cv:.4f}")
    print(f"  mean ngram_diversity   : {avg_diversity:.4f}")
    print(f"  mean correct_rate      : {avg_correct:.0%}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LLM consistency probe")
    p.add_argument("--base-url", required=True, help="API base URL, e.g. https://api.openai.com")
    p.add_argument("--api-key", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-style", default="openai", choices=["openai", "anthropic"])
    p.add_argument("--system", default=None, help="System prompt (anthropic only)")
    p.add_argument("--n-samples", type=int, default=7, help="Samples per probe (default 7)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--delay", type=float, default=1.0, help="Seconds between requests (default 1)")
    p.add_argument("--probes", nargs="*", help="Probe IDs to run (default: random 1)")
    p.add_argument("--n-probes", type=int, default=1, help="Number of random probes to select (default 1)")
    p.add_argument("--output", help="Save raw results to this JSON file")
    return p.parse_args()


def main():
    args = parse_args()

    client_cfg = {
        "base_url": args.base_url,
        "api_key": args.api_key,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "api_style": args.api_style,
        "system": args.system,
    }

    if args.probes:
        probes_to_run = [p for p in PROBES if p["id"] in args.probes]
        if not probes_to_run:
            print(f"No probes matched. Available: {[p['id'] for p in PROBES]}")
            return
    else:
        import random
        n = min(args.n_probes, len(PROBES))
        probes_to_run = random.sample(PROBES, n)
        print(f"Randomly selected {n} probe(s): {[p['id'] for p in probes_to_run]}")

    print(f"Running {len(probes_to_run)} probe(s) × {args.n_samples} samples "
          f"on {args.model} @ {args.base_url}")
    print(f"temperature={args.temperature}, delay={args.delay}s\n")

    all_results = []
    all_metrics = []

    for probe in probes_to_run:
        print(f"\n--- Probe: {probe['id']} ({probe['source']}) ---")
        responses = sample_probe(probe, client_cfg, args.n_samples, args.temperature, args.delay)
        metrics = compute_metrics(responses, probe)
        all_metrics.append(metrics)
        all_results.append({"probe": probe, "responses": responses, "metrics": metrics})

    print_report(all_metrics)

    if args.output:
        # Strip raw API responses to keep file size reasonable
        slim = []
        for r in all_results:
            slim.append({
                "probe_id": r["probe"]["id"],
                "metrics": r["metrics"],
                "responses": [{"content": resp["content"], "latency": resp["latency"]}
                               for resp in r["responses"]],
            })
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(slim, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
