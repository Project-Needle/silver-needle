"""
Context window stress test: detect whether a provider truncates or compresses
long contexts by planting a unique token (needle) in the middle of filler text
and checking whether swapping the needle changes the model's answer.

Three requests, all at temperature=0 (determinism is required for comparison):
  R1 (needle_A in haystack) — filler_head + needle_A + filler_tail + question
  R2 (needle_B in haystack) — filler_head + needle_B + filler_tail + question
  R3 (short control)        — needle_A alone + question

Detection signals:
  R1_answer == R2_answer          → HIGH: needle swap made no difference;
                                    the needle position is likely truncated/ignored
  R1_answer != expected_A         → HIGH: correct needle not extracted from long context
  R3_answer == expected_A         → confirms the needle+question pair is answerable alone
  short_succeeds + long_fails     → HIGH: clear truncation fingerprint
  R1/R2 latency ≈ R3 latency     → MEDIUM: suspiciously fast; full context may not be processed
"""

import json
import os
import time
import argparse
from typing import Optional

from request_llm import call_api_raw
from model_db import normalize_model_name, lookup


# ---------------------------------------------------------------------------
# Model specs: max context window and recommended test size
# ---------------------------------------------------------------------------

_SPECS_FILE = os.path.join(os.path.dirname(__file__), "assets", "model_specs.json")


def lookup_model_spec(model_name: str) -> Optional[dict]:
    return lookup(_SPECS_FILE, model_name)


def recommend_context_chars(max_tokens: int) -> int:
    """Fallback formula when the model is not in model_specs.json."""
    if max_tokens <= 8192:
        return 1000
    elif max_tokens <= 32768:
        return 4000
    elif max_tokens <= 131072:
        return 16000
    else:
        return 32000


# ---------------------------------------------------------------------------
# Needle values — unique enough to not appear by coincidence or prior knowledge
# ---------------------------------------------------------------------------

NEEDLE_A = "X7K-DELTA-2941"
NEEDLE_B = "Q3M-SIGMA-8857"

NEEDLE_EMBED_TEMPLATE = (
    "\n\n"
    "==============================\n"
    "INTERNAL PROJECT RECORD\n"
    "Authentication Token: {token}\n"
    "==============================\n"
    "\n"
)

NEEDLE_QUESTION = (
    "Based on the document above, what is the authentication token listed "
    "in the INTERNAL PROJECT RECORD section? "
    "Reply with only the token value, nothing else."
)


# ---------------------------------------------------------------------------
# Filler content
#
# Each paragraph is ~250 characters. generate_filler() cycles through them
# with sequential numbering so repeated paragraphs are not identical strings
# (prevents the model from pattern-matching on the repetition and skipping ahead).
# The head and tail use different starting offsets to avoid identical blocks.
# ---------------------------------------------------------------------------

_FILLER_PARAGRAPHS = [
    "The development of calculus in the 17th century independently by Newton and Leibniz "
    "transformed mathematics and physics, providing tools to describe rates of change and "
    "accumulation that underpin modern science and engineering.",

    "Continental drift, proposed by Alfred Wegener in 1912 and later confirmed by "
    "paleomagnetic data, explains how Earth's lithospheric plates move several centimeters "
    "per year, reshaping continents and ocean basins over geological timescales.",

    "The Byzantine Empire, successor to the eastern Roman Empire, preserved classical "
    "Greco-Roman knowledge through the medieval period and acted as a cultural bridge "
    "between antiquity and the Renaissance, transmitting texts and scholarship westward.",

    "Photosynthesis converts solar energy into chemical energy by splitting water molecules "
    "and fixing atmospheric carbon dioxide into glucose, releasing oxygen as a byproduct "
    "and forming the base of almost all food chains on Earth.",

    "The Silk Road was not a single road but a network of trade routes connecting China "
    "to the Mediterranean across Central Asia, facilitating the exchange of goods, "
    "technologies, religions, and ideas for over a millennium.",

    "Quantum entanglement describes a phenomenon where two particles become correlated "
    "such that the quantum state of one cannot be described independently of the other, "
    "regardless of the spatial separation between them, challenging classical intuitions.",

    "The invention of the printing press by Gutenberg around 1440 accelerated the "
    "dissemination of knowledge, dramatically reduced the cost of books, and contributed "
    "to the Protestant Reformation and the broader Scientific Revolution.",

    "Thermodynamics governs energy conversion and transfer. The second law establishes "
    "that entropy in an isolated system tends to increase, placing fundamental limits "
    "on the efficiency of heat engines, refrigerators, and chemical reactions.",

    "The Amazon basin contains roughly 10% of all species on Earth, generates about "
    "20% of the world's freshwater discharge into the ocean, and plays a critical role "
    "in regulating the global carbon cycle and regional precipitation patterns.",

    "DNA's double-helix structure, elucidated by Watson and Crick in 1953, revealed "
    "how genetic information is stored in complementary base-pair sequences and "
    "faithfully replicated during cell division with remarkably high fidelity.",

    "The Hagia Sophia, completed in 537 AD, features a massive central dome that appears "
    "to float on a ring of windows, creating an illusion of lightness that influenced "
    "both subsequent Islamic architecture and European religious building traditions.",

    "Epidemiological studies rely on cohort and case-control designs to quantify "
    "associations between exposures and outcomes, applying statistical adjustment "
    "for confounders to distinguish causal effects from mere correlation.",

    "The ocean conveyor belt, or thermohaline circulation, drives deep-water formation "
    "in the North Atlantic as cold, saline water sinks, distributing heat globally "
    "and influencing climate on timescales of centuries to millennia.",

    "Compiler design translates high-level source code through lexical analysis, "
    "parsing, semantic analysis, and optimization passes before generating machine "
    "code or an intermediate representation targeting a virtual machine.",

    "Medieval Islamic scholars preserved and extended Greek mathematics, developing "
    "algebra, algorithms, and trigonometric tables that were later translated into "
    "Latin and became the foundation of European Renaissance mathematics.",

    "The Haber-Bosch process, developed in the early 20th century, synthesizes ammonia "
    "from atmospheric nitrogen and hydrogen using an iron catalyst, enabling large-scale "
    "fertilizer production that supports about half of the world's food supply today.",

    "Neuroplasticity refers to the brain's capacity to reorganize itself by forming new "
    "neural connections throughout life, enabling learning, recovery from injury, and "
    "adaptation to new experiences across all developmental stages.",

    "Comparative advantage, articulated by David Ricardo in 1817, explains why nations "
    "benefit from specializing in goods they produce most efficiently and trading for "
    "the rest, even when one nation is more efficient at producing everything.",

    "Binary stars, which make up a majority of star systems in the Milky Way, orbit a "
    "common center of mass and allow astronomers to measure stellar masses directly "
    "from Kepler's laws of orbital mechanics without relying on indirect proxies.",

    "The Treaty of Westphalia in 1648 ended the Thirty Years' War and codified the "
    "principle of state sovereignty that forms the basis of the modern international "
    "order, limiting external interference in the domestic affairs of recognized states.",
]


def generate_filler(target_chars: int, start_offset: int = 0) -> str:
    """
    Generate filler text by cycling through _FILLER_PARAGRAPHS with sequential
    paragraph numbers prepended. start_offset shifts which paragraph we begin
    with so that head and tail fillers are not identical.
    """
    n = len(_FILLER_PARAGRAPHS)
    parts = []
    total = 0
    counter = 1
    i = start_offset % n
    while total < target_chars:
        para = f"[{counter}] {_FILLER_PARAGRAPHS[i % n]}"
        parts.append(para)
        total += len(para) + 2  # +2 for the "\n\n" separator
        i += 1
        counter += 1
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_haystack_message(filler_head: str, needle_embed: str, filler_tail: str) -> str:
    return filler_head + needle_embed + filler_tail + "\n\n" + NEEDLE_QUESTION


def build_short_message(needle_embed: str) -> str:
    return "Read the following record carefully.\n" + needle_embed + "\n" + NEEDLE_QUESTION


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> str:
    """Return the first non-empty line stripped of whitespace."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def answer_matches(text: str, expected: str) -> bool:
    """True if expected token appears anywhere in the response (case-insensitive)."""
    return expected.upper() in text.upper()


# ---------------------------------------------------------------------------
# Single-request wrapper
# ---------------------------------------------------------------------------

def run_request(label: str, user_message: str, client_cfg: dict, timeout: int) -> dict:
    print(f"  Sending {label} (prompt {len(user_message)} chars) ...")
    result = call_api_raw(
        messages=[{"role": "user", "content": user_message}],
        client_cfg=client_cfg,
        temperature=0.0,
        timeout=timeout,
    )
    answer = extract_answer(result["content"])
    print(f"    → latency={result['latency']}s  answer={repr(answer[:80])}")
    return {
        "label": label,
        "latency": result["latency"],
        "prompt_len": len(user_message),
        "content_len": len(result["content"]),
        "answer": answer,
    }


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(r1: dict, r2: dict, r3: dict) -> dict:
    r1_correct = answer_matches(r1["answer"], NEEDLE_A)
    r2_correct = answer_matches(r2["answer"], NEEDLE_B)
    r3_correct = answer_matches(r3["answer"], NEEDLE_A)
    needle_swap_sensitive = r1["answer"].upper() != r2["answer"].upper()
    r3_lat = r3["latency"]
    latency_ratio_r1 = round(r1["latency"] / r3_lat, 2) if r3_lat > 0 else None
    latency_ratio_r2 = round(r2["latency"] / r3_lat, 2) if r3_lat > 0 else None
    return {
        "r1_correct": r1_correct,
        "r2_correct": r2_correct,
        "r3_correct": r3_correct,
        "needle_swap_sensitive": needle_swap_sensitive,
        "short_succeeds_long_fails": r3_correct and not r1_correct,
        "latency_ratio_r1": latency_ratio_r1,
        "latency_ratio_r2": latency_ratio_r2,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(
    r1: dict, r2: dict, r3: dict,
    ev: dict,
    model: str,
    base_url: str,
    context_chars: int,
) -> None:
    print("\n" + "=" * 70)
    print("CONTEXT WINDOW PROBE REPORT")
    print("=" * 70)
    print(f"  Provider       : {base_url}")
    print(f"  Model          : {model}")
    print(f"  Needle A       : {NEEDLE_A}  (planted in R1 and R3)")
    print(f"  Needle B       : {NEEDLE_B}  (planted in R2, replaces A)")
    print(f"  Filler per side: ~{context_chars} chars  (~{context_chars // 4} tokens est.)")
    print(f"  temperature    : 0  (fixed)")
    print()

    for r, expected, label in [
        (r1, NEEDLE_A, "R1 needle_A in haystack"),
        (r2, NEEDLE_B, "R2 needle_B in haystack"),
        (r3, NEEDLE_A, "R3 short control       "),
    ]:
        matched = answer_matches(r["answer"], expected)
        sym = "✓" if matched else "✗"
        print(f"  {sym} [{r['label']}] {label}  "
              f"answer={repr(r['answer'][:50])}  latency={r['latency']}s")

    print()
    print("  Checks")
    print("  " + "-" * 64)

    sym = "✓" if ev["r1_correct"] else "✗"
    print(f"  {sym} R1 extracted correct needle (long context)  "
          f"expected={NEEDLE_A}  got={repr(r1['answer'][:40])}")

    sym = "✓" if ev["r2_correct"] else "✗"
    print(f"  {sym} R2 extracted correct needle (long context)  "
          f"expected={NEEDLE_B}  got={repr(r2['answer'][:40])}")

    sym = "✓" if ev["r3_correct"] else "✗"
    print(f"  {sym} R3 extracted correct needle (short control)  "
          f"expected={NEEDLE_A}  got={repr(r3['answer'][:40])}")

    sym = "✓" if ev["needle_swap_sensitive"] else "✗"
    print(f"  {sym} needle swap changed the answer  "
          f"(R1={repr(r1['answer'][:30])}  R2={repr(r2['answer'][:30])})")
    if not ev["needle_swap_sensitive"]:
        print(f"       → R1 and R2 returned the same answer; needle position likely ignored")

    if ev["short_succeeds_long_fails"]:
        print(f"  ✗ short control passed but long context failed  "
              f"→ truncation/compression of middle region")

    print()
    print("  Latency")
    print("  " + "-" * 64)
    print(f"  R1 / R3 ratio: {ev['latency_ratio_r1']}x  "
          f"({r1['latency']}s / {r3['latency']}s,  "
          f"prompt {r1['prompt_len']} vs {r3['prompt_len']} chars)")
    print(f"  R2 / R3 ratio: {ev['latency_ratio_r2']}x  "
          f"({r2['latency']}s / {r3['latency']}s,  "
          f"prompt {r2['prompt_len']} vs {r3['prompt_len']} chars)")
    if ev["latency_ratio_r1"] is not None and ev["latency_ratio_r1"] < 1.5:
        print(f"  note: R1/R3 ratio is low; long context may not have been fully prefilled")
    if ev["latency_ratio_r2"] is not None and ev["latency_ratio_r2"] < 1.5:
        print(f"  note: R2/R3 ratio is low; long context may not have been fully prefilled")

    print()
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LLM context window stress probe")
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-style", default="openai", choices=["openai", "anthropic"])
    p.add_argument("--system", default=None)
    p.add_argument("--max-tokens", type=int, default=64,
                   help="Max output tokens (answer should be just the token code)")
    p.add_argument("--context-chars", type=int, default=None,
                   help=(
                       "Target character count for each filler half (head and tail). "
                       "If omitted, auto-selected from model_specs.json based on --model. "
                       "Falls back to 8000 if the model is not in the spec file."
                   ))
    p.add_argument("--delay", type=float, default=2.0,
                   help="Seconds to wait between requests")
    p.add_argument("--timeout", type=int, default=300,
                   help="Per-request HTTP timeout in seconds (long contexts need more time)")
    p.add_argument("--output", help="Save full results to this JSON file")
    return p.parse_args()


def main():
    args = parse_args()

    client_cfg = {
        "base_url": args.base_url,
        "api_key": args.api_key,
        "model": args.model,
        "api_style": args.api_style,
        "system": args.system,
        "max_tokens": args.max_tokens,
    }

    # Resolve context_chars: CLI > model_specs.json > fallback
    if args.context_chars is not None:
        context_chars = args.context_chars
        spec_source = "cli"
    else:
        spec = lookup_model_spec(args.model)
        if spec is not None:
            context_chars = spec["recommended_context_chars"]
            spec_source = f"model_specs.json ({spec['notes']}, max_tokens={spec['max_tokens']})"
        else:
            context_chars = 8000
            spec_source = "fallback default (model not found in model_specs.json)"

    print(f"Model          : {args.model}")
    print(f"context_chars  : {context_chars} per side  (source: {spec_source})")
    print()
    print(f"Generating filler (~{context_chars} chars per side) ...")
    filler_head = generate_filler(context_chars, start_offset=0)
    filler_tail = generate_filler(context_chars, start_offset=len(_FILLER_PARAGRAPHS) // 2)

    needle_embed_a = NEEDLE_EMBED_TEMPLATE.format(token=NEEDLE_A)
    needle_embed_b = NEEDLE_EMBED_TEMPLATE.format(token=NEEDLE_B)

    msg_r1 = build_haystack_message(filler_head, needle_embed_a, filler_tail)
    msg_r2 = build_haystack_message(filler_head, needle_embed_b, filler_tail)
    msg_r3 = build_short_message(needle_embed_a)

    print(f"  R1 prompt: {len(msg_r1)} chars  (~{len(msg_r1) // 4} tokens)")
    print(f"  R2 prompt: {len(msg_r2)} chars  (~{len(msg_r2) // 4} tokens)")
    print(f"  R3 prompt: {len(msg_r3)} chars  (~{len(msg_r3) // 4} tokens)")
    print(f"  temperature: 0  (fixed for deterministic comparison)")
    print()

    r1 = run_request("R1", msg_r1, client_cfg, args.timeout)
    time.sleep(args.delay)
    r2 = run_request("R2", msg_r2, client_cfg, args.timeout)
    time.sleep(args.delay)
    r3 = run_request("R3", msg_r3, client_cfg, args.timeout)

    ev = evaluate(r1, r2, r3)
    print_report(r1, r2, r3, ev,
                 model=args.model, base_url=args.base_url,
                 context_chars=context_chars)

    if args.output:
        out = {
            "mode": "context_window",
            "model": args.model,
            "base_url": args.base_url,
            "needle_a": NEEDLE_A,
            "needle_b": NEEDLE_B,
            "context_chars_per_side": context_chars,
            "r1": r1,
            "r2": r2,
            "r3": r3,
            "evaluation": ev,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
