"""
Structure probe: send a small number of requests (simple + complex question),
exhaustively enumerate all fields in the raw response, and surface anomalies.

Two modes:
  non-stream (default): analyze the final JSON response object field by field.
  stream (--stream):    analyze the SSE event sequence without storing full text.
                        Captures: event types, delta field names, reasoning vs content
                        chunk counts, first/last 3 chunks as samples, timing.

Strategy: no hard-coded "expected schema" — instead apply heuristics:
  1. Collect every leaf field path and its value from the raw response.
  2. Compare simple vs complex question responses for the same provider.
  3. Flag any field whose value degrades (non-zero → zero, non-null → null,
     non-empty → empty) when moving from simple to complex question.
  4. Flag numeric fields that are zero on the complex question.
  5. Flag string fields that are empty or null on the complex question.
  6. Report all fields verbatim so the user can inspect anything unusual.

The tool never concludes "fraud" — it surfaces evidence for human judgement.
"""

import re
import json
import time
import argparse
from typing import Any

from request_llm import call_api_raw, chat_completions_stream, anthropic_messages_stream


# ---------------------------------------------------------------------------
# Test prompts
# ---------------------------------------------------------------------------

SIMPLE_PROMPT = "What is 2 + 2?"

COMPLEX_PROMPT = (
    "A snail climbs a 10-meter pole. Each day it climbs 3 meters, "
    "but each night it slides back 2 meters. "
    "On which day does it reach the top? Show your reasoning."
)


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

def flatten(obj: Any, path: str = "") -> dict:
    """
    Recursively flatten a nested JSON object into dot-notation paths.

    Returns: {path: value}
    Arrays are represented by their first element with index [0].
    Empty arrays are represented as an empty-list sentinel.
    """
    result = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            child = f"{path}.{k}" if path else k
            result.update(flatten(v, child))

    elif isinstance(obj, list):
        if obj:
            result.update(flatten(obj[0], f"{path}[0]"))
        else:
            result[f"{path}[]"] = []

    else:
        result[path] = obj

    return result


# ---------------------------------------------------------------------------
# Value classification helpers
# ---------------------------------------------------------------------------

def is_empty(value: Any) -> bool:
    """True if value is absent/null/empty in a semantically meaningful sense."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (int, float)) and value == 0:
        return True
    return False


def value_type_label(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        n = len(value)
        return f"str(len={n})"
    if isinstance(value, list):
        return f"list(len={len(value)})"
    return type(value).__name__


# ---------------------------------------------------------------------------
# Anomaly detection heuristics
# ---------------------------------------------------------------------------

def detect_anomalies(simple_flat: dict, complex_flat: dict) -> list:
    """
    Compare simple vs complex response field maps and apply heuristics.
    Returns a list of anomaly dicts sorted by severity.
    """
    anomalies = []
    all_paths = sorted(set(simple_flat) | set(complex_flat))

    for path in all_paths:
        in_simple = path in simple_flat
        in_complex = path in complex_flat
        s_val = simple_flat.get(path)
        c_val = complex_flat.get(path)

        # Heuristic 1: field present in simple but absent in complex
        if in_simple and not in_complex:
            anomalies.append({
                "severity": "high",
                "heuristic": "field_disappeared",
                "path": path,
                "simple_value": s_val,
                "complex_value": "(absent)",
                "note": "Field present for simple question but missing for complex question"
            })
            continue

        # Heuristic 2: field present in complex but absent in simple (informational)
        if in_complex and not in_simple:
            anomalies.append({
                "severity": "info",
                "heuristic": "field_appeared",
                "path": path,
                "simple_value": "(absent)",
                "complex_value": c_val,
                "note": "Field only appears for complex question"
            })
            continue

        # Both present from here on
        s_empty = is_empty(s_val)
        c_empty = is_empty(c_val)

        # Heuristic 3: value was meaningful in simple, became empty/zero in complex
        # This is the core signal: something that should grow for harder questions shrinks to nothing
        if not s_empty and c_empty:
            anomalies.append({
                "severity": "high",
                "heuristic": "value_degraded",
                "path": path,
                "simple_value": s_val,
                "complex_value": c_val,
                "note": (
                    "Value is non-empty for simple question but empty/zero/null for complex. "
                    "Suspicious: complex questions should generally produce more, not less."
                )
            })
            continue

        # Heuristic 4: numeric field is zero on complex question (even if also zero on simple)
        if isinstance(c_val, (int, float)) and c_val == 0:
            # Only flag if the field name suggests it should be non-zero for complex reasoning
            reasoning_keywords = ("reasoning", "thinking", "thought", "chain")
            if any(kw in path.lower() for kw in reasoning_keywords):
                anomalies.append({
                    "severity": "high",
                    "heuristic": "reasoning_zero",
                    "path": path,
                    "simple_value": s_val,
                    "complex_value": c_val,
                    "note": (
                        "Reasoning-related numeric field is zero for complex question. "
                        "If this model supports extended thinking, this indicates it was suppressed."
                    )
                })

        # Heuristic 5: string field containing reasoning keywords is empty on complex
        if isinstance(c_val, str) and c_val.strip() == "":
            reasoning_keywords = ("reasoning", "thinking", "thought", "chain")
            if any(kw in path.lower() for kw in reasoning_keywords):
                anomalies.append({
                    "severity": "high",
                    "heuristic": "reasoning_empty",
                    "path": path,
                    "simple_value": s_val if isinstance(s_val, str) and len(s_val) < 80 else f"str(len={len(str(s_val))})",
                    "complex_value": "(empty string)",
                    "note": (
                        "Reasoning-related string field is empty for complex question."
                    )
                })

        # Heuristic 6: system_fingerprint is empty string (ucloud pattern)
        if path == "system_fingerprint" and isinstance(c_val, str) and c_val.strip() == "":
            anomalies.append({
                "severity": "medium",
                "heuristic": "empty_fingerprint",
                "path": path,
                "simple_value": s_val,
                "complex_value": c_val,
                "note": (
                    "system_fingerprint is empty. "
                    "This may indicate a different backend node served this request "
                    "(observed pattern: empty fingerprint correlates with disabled reasoning)."
                )
            })

    return sorted(anomalies, key=lambda x: {"high": 0, "medium": 1, "info": 2}.get(x["severity"], 3))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

SEVERITY_LABELS = {"high": "⚠ HIGH", "medium": "~ MED", "info": "  INFO"}


def print_report(
    simple_result: dict,
    complex_result: dict,
    simple_flat: dict,
    complex_flat: dict,
    anomalies: list,
    model: str,
    base_url: str,
) -> None:
    print("\n" + "=" * 70)
    print("STRUCTURE PROBE REPORT")
    print("=" * 70)
    print(f"  Provider : {base_url}")
    print(f"  Model    : {model}")
    print()

    # Per-request summary
    for label, result, flat in [
        ("SIMPLE", simple_result, simple_flat),
        ("COMPLEX", complex_result, complex_flat),
    ]:
        content_len = len(result["content"])
        latency = result["latency"]
        n_fields = len(flat)
        print(f"  [{label}]  latency={latency}s  content_len={content_len}  fields={n_fields}")

    print()
    print("  All fields (simple → complex):")
    print("  " + "-" * 64)
    all_paths = sorted(set(simple_flat) | set(complex_flat))
    for path in all_paths:
        s = simple_flat.get(path, "(absent)")
        c = complex_flat.get(path, "(absent)")
        # Truncate long string values for display
        def fmt(v):
            if isinstance(v, str) and len(v) > 60:
                return f'"{v[:57]}..." ({len(v)} chars)'
            return repr(v)
        changed = "→" if s != c else " "
        print(f"  {changed} {path:<50}  {fmt(s)}  →  {fmt(c)}")

    print()
    print(f"  Anomalies detected: {len(anomalies)}")
    print("  " + "-" * 64)

    if not anomalies:
        print("  ✓ No anomalies detected.")
    else:
        for a in anomalies:
            label = SEVERITY_LABELS.get(a["severity"], a["severity"])
            print(f"\n  [{label}]  {a['heuristic']}  @ {a['path']}")
            print(f"           simple  : {repr(a['simple_value'])[:80]}")
            print(f"           complex : {repr(a['complex_value'])[:80]}")
            print(f"           → {a['note']}")

    print()
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# Streaming analysis
# ---------------------------------------------------------------------------

# How many chunks to capture from the head and tail of the stream
STREAM_SAMPLE_SIZE = 3


def analyze_stream(
    messages: list,
    client_cfg: dict,
    temperature: float,
) -> dict:
    """
    Consume a streaming response and extract diagnostic information
    without storing the full text content.

    Returned dict:
      chunk_count          : total number of SSE data chunks received
      delta_fields_seen    : set of field names that appeared inside delta objects
                             e.g. {"content", "reasoning_content"} or {"text_delta"}
      reasoning_chunk_count: chunks that carried reasoning/thinking text
      content_chunk_count  : chunks that carried final content text
      empty_reasoning_block: True if a thinking block opened and closed with no deltas
                             (Anthropic only)
      event_sequence       : compact list of event type labels in order
                             truncated to first 20 + "..." + last 5 if longer
      first_chunks         : first STREAM_SAMPLE_SIZE raw chunk dicts (for inspection)
      last_chunks          : last  STREAM_SAMPLE_SIZE raw chunk dicts
      finish_reason        : finish_reason value from the final choices chunk (OpenAI)
                             or stop_reason from message_delta (Anthropic)
      latency_first_chunk  : seconds until first chunk arrived
      latency_total        : total seconds until stream finished
    """
    api_style = client_cfg.get("api_style", "openai").lower()
    t_start = time.time()
    t_first = None

    chunk_count = 0
    delta_fields_seen: set = set()
    reasoning_chunk_count = 0
    content_chunk_count = 0
    empty_reasoning_block = False  # Anthropic-specific
    full_event_sequence: list = []
    first_chunks: list = []
    last_chunks_ring: list = []
    finish_reason = None

    # Anthropic: track open thinking blocks
    _thinking_block_open = False
    _thinking_block_had_content = False

    stream_iter = (
        chat_completions_stream(
            base_url=client_cfg["base_url"],
            api_key=client_cfg["api_key"],
            model=client_cfg["model"],
            messages=messages,
            max_tokens=client_cfg.get("max_tokens", 1024),
            temperature=temperature,
        )
        if api_style == "openai"
        else anthropic_messages_stream(
            base_url=client_cfg["base_url"],
            api_key=client_cfg["api_key"],
            model=client_cfg["model"],
            messages=messages,
            max_tokens=client_cfg.get("max_tokens", 1024),
            temperature=temperature,
            system=client_cfg.get("system"),
        )
    )

    for chunk in stream_iter:
        if t_first is None:
            t_first = time.time()

        chunk_count += 1

        # Sample first / last chunks (strip long text to keep output small)
        slim = _slim_chunk(chunk)
        if len(first_chunks) < STREAM_SAMPLE_SIZE:
            first_chunks.append(slim)
        last_chunks_ring.append(slim)
        if len(last_chunks_ring) > STREAM_SAMPLE_SIZE:
            last_chunks_ring.pop(0)

        # --- OpenAI-style parsing ---
        if api_style == "openai":
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                fr = choices[0].get("finish_reason")
                if fr:
                    finish_reason = fr
                    full_event_sequence.append(f"finish:{fr}")
                else:
                    for field, value in delta.items():
                        if field in ("role",):
                            continue
                        delta_fields_seen.add(field)
                        if value:  # non-empty text
                            if field == "reasoning_content":
                                reasoning_chunk_count += 1
                                full_event_sequence.append("R")
                            elif field == "content":
                                content_chunk_count += 1
                                full_event_sequence.append("C")
                            else:
                                full_event_sequence.append(f"?{field}")
            # usage chunk (last chunk with no choices)
            elif "usage" in chunk:
                full_event_sequence.append("usage")

        # --- Anthropic-style parsing ---
        else:
            event_type = chunk.get("_event") or chunk.get("type", "?")

            if event_type == "content_block_start":
                block_type = chunk.get("content_block", {}).get("type", "?")
                full_event_sequence.append(f"block_start:{block_type}")
                if block_type == "thinking":
                    _thinking_block_open = True
                    _thinking_block_had_content = False

            elif event_type == "content_block_delta":
                delta = chunk.get("delta", {})
                dtype = delta.get("type", "?")
                delta_fields_seen.add(dtype)
                has_text = bool(delta.get("thinking") or delta.get("text"))
                if has_text:
                    if dtype == "thinking_delta":
                        reasoning_chunk_count += 1
                        _thinking_block_had_content = True
                        full_event_sequence.append("R")
                    elif dtype == "text_delta":
                        content_chunk_count += 1
                        full_event_sequence.append("C")
                    else:
                        full_event_sequence.append(f"?{dtype}")

            elif event_type == "content_block_stop":
                full_event_sequence.append("block_stop")
                if _thinking_block_open:
                    if not _thinking_block_had_content:
                        empty_reasoning_block = True
                    _thinking_block_open = False

            elif event_type == "message_delta":
                stop = chunk.get("delta", {}).get("stop_reason")
                if stop:
                    finish_reason = stop
                    full_event_sequence.append(f"finish:{stop}")

            elif event_type == "message_start":
                full_event_sequence.append("msg_start")

            elif event_type == "message_stop":
                full_event_sequence.append("msg_stop")

    t_end = time.time()

    # Compact event sequence: keep first 20 + last 5 if long
    if len(full_event_sequence) > 25:
        event_sequence = (
            full_event_sequence[:20]
            + [f"...({len(full_event_sequence) - 25} more)..."]
            + full_event_sequence[-5:]
        )
    else:
        event_sequence = full_event_sequence

    return {
        "chunk_count": chunk_count,
        "delta_fields_seen": sorted(delta_fields_seen),
        "reasoning_chunk_count": reasoning_chunk_count,
        "content_chunk_count": content_chunk_count,
        "empty_reasoning_block": empty_reasoning_block,
        "event_sequence": event_sequence,
        "first_chunks": first_chunks,
        "last_chunks": list(last_chunks_ring),
        "finish_reason": finish_reason,
        "latency_first_chunk": round(t_first - t_start, 3) if t_first else None,
        "latency_total": round(t_end - t_start, 3)
    }


def _slim_chunk(chunk: dict) -> dict:
    """
    Return a copy of chunk with long string values truncated to 80 chars.
    Keeps structure intact for inspection without bloating the saved JSON.
    """
    out = {}
    for k, v in chunk.items():
        if isinstance(v, str) and len(v) > 80:
            out[k] = v[:80] + f"…({len(v)} chars)"
        elif isinstance(v, dict):
            out[k] = _slim_chunk(v)
        elif isinstance(v, list):
            out[k] = [_slim_chunk(i) if isinstance(i, dict) else i for i in v[:3]]
        else:
            out[k] = v
    return out


def detect_stream_anomalies(simple_stream: dict, complex_stream: dict) -> list:
    """
    Heuristics applied to streaming event sequences only.

    Focuses on SSE chunk patterns, delta fields, and reasoning chunks.
    Does NOT analyze metadata fields (id, created, usage, etc.) because
    those naturally differ between requests (timestamps, IDs, token counts).
    For metadata field analysis, run structure_probe in non-stream mode.
    """
    anomalies = []
    reasoning_keywords = ("reasoning", "thinking", "thought")
    # 通过分析最后一个chunk的metadata来判断该模型是支持reasoning的，然后这种情况下复杂问题没有reasoning才能判断为有问题
    # 考虑到供应商对不支持推理的模型强行加上非标准的reasoning_keywords概率很小，但是漏报情况（明明支持但供应商故意消去）
    # 审计工具设计上接受漏报，这是无法确认的
    flatten_simple_metadata = flatten(simple_stream["last_chunks"][-1])
    flatten_complex_metadata = flatten(complex_stream["last_chunks"][-1])
    all_paths = sorted(set(flatten_simple_metadata) | set(flatten_complex_metadata))
    support_reasoning = any([any(kw in path.lower() for kw in reasoning_keywords) for path in all_paths])
    # print(f"support_reasoning: {support_reasoning}")

    # Heuristic: reasoning chunks present for simple but not for complex
    s_r = simple_stream["reasoning_chunk_count"]
    c_r = complex_stream["reasoning_chunk_count"]
    if s_r > 0 and c_r == 0:
        anomalies.append({
            "severity": "high",
            "heuristic": "reasoning_suppressed_on_complex",
            "simple_value": s_r,
            "complex_value": c_r,
            "note": (
                "Reasoning chunks present for simple question but absent for complex. "
                "Complex questions should require more reasoning, not less."
            ),
        })

    # Heuristic: reasoning completely absent on complex
    if c_r == 0 and complex_stream["chunk_count"] > 0:
        reasoning_fields = [
            f for f in complex_stream["delta_fields_seen"]
            if any(kw in f for kw in reasoning_keywords)
        ]
        
        if support_reasoning and not reasoning_fields:
            anomalies.append({
                "severity": "medium",
                "heuristic": "no_reasoning_field_in_complex",
                "simple_value": simple_stream["delta_fields_seen"],
                "complex_value": complex_stream["delta_fields_seen"],
                "note": (
                    "No reasoning/thinking delta field appeared at all in the complex question stream. "
                    "If this model supports extended thinking, check whether it is enabled."
                ),
            })

    # Heuristic: empty thinking block (Anthropic) on complex question
    if complex_stream["empty_reasoning_block"]:
        anomalies.append({
            "severity": "high",
            "heuristic": "empty_thinking_block",
            "simple_value": simple_stream["empty_reasoning_block"],
            "complex_value": True,
            "note": (
                "A thinking block opened and closed with no content on the complex question. "
                "This is a hollow reasoning shell — the model declared thinking mode but produced nothing."
            ),
        })

    # Heuristic: delta field names differ between simple and complex
    s_fields = set(simple_stream["delta_fields_seen"])
    c_fields = set(complex_stream["delta_fields_seen"])
    disappeared = s_fields - c_fields
    if disappeared:
        anomalies.append({
            "severity": "medium",
            "heuristic": "delta_field_disappeared",
            "simple_value": sorted(s_fields),
            "complex_value": sorted(c_fields),
            "note": (
                f"Delta field(s) {disappeared} present in simple stream but absent in complex. "
                "Inconsistent streaming structure may indicate different backend paths."
            ),
        })

    # 对最后一个带metadata的chunk也进行普通字段分析(待定)
    # metadata_anomalies = detect_anomalies(flatten_simple_metadata, flatten_complex_metadata)
    # anomalies.extend(metadata_anomalies)
    
    return anomalies


# ---------------------------------------------------------------------------
# Stream report printer
# ---------------------------------------------------------------------------

def print_stream_report(
    simple_stream: dict,
    complex_stream: dict,
    anomalies: list,
    model: str,
    base_url: str,
) -> None:
    print("\n" + "=" * 70)
    print("STRUCTURE PROBE REPORT  [stream mode]")
    print("=" * 70)
    print(f"  Provider : {base_url}")
    print(f"  Model    : {model}")
    print()

    for label, s in [("SIMPLE", simple_stream), ("COMPLEX", complex_stream)]:
        print(f"  [{label}]")
        print(f"    chunks total       : {s['chunk_count']}")
        print(f"    reasoning chunks   : {s['reasoning_chunk_count']}")
        print(f"    content chunks     : {s['content_chunk_count']}")
        print(f"    delta fields seen  : {s['delta_fields_seen']}")
        print(f"    empty thinking blk : {s['empty_reasoning_block']}")
        print(f"    finish_reason      : {s['finish_reason']}")
        print(f"    latency first chunk: {s['latency_first_chunk']}s")
        print(f"    latency total      : {s['latency_total']}s")
        print(f"    event sequence     : {s['event_sequence']}")
        print(f"    first {STREAM_SAMPLE_SIZE} chunks     :")
        for i, c in enumerate(s["first_chunks"]):
            print(f"      [{i}] {json.dumps(c, ensure_ascii=False)}")
        print(f"    last  {STREAM_SAMPLE_SIZE} chunks     :")
        for i, c in enumerate(s["last_chunks"]):
            print(f"      [{i}] {json.dumps(c, ensure_ascii=False)}")
        print()

    print(f"  Anomalies detected: {len(anomalies)}")
    print("  " + "-" * 64)
    if not anomalies:
        print("  ✓ No streaming anomalies detected.")
    else:
        for a in anomalies:
            label = SEVERITY_LABELS.get(a["severity"], a["severity"])
            print(f"\n  [{label}]  {a['heuristic']}")
            print(f"           simple  : {repr(a['simple_value'])[:80]}")
            print(f"           complex : {repr(a['complex_value'])[:80]}")
            print(f"           → {a['note']}")
    print()
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LLM response structure probe")
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-style", default="openai", choices=["openai", "anthropic"])
    p.add_argument("--system", default=None)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--delay", type=float, default=1.0)
    p.add_argument("--stream", action="store_true",
                   help="Use streaming mode to analyze SSE event sequence")
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

    if args.stream:
        print(f"Sending simple question (stream)...")
        simple_stream = analyze_stream(
            messages=[{"role": "user", "content": SIMPLE_PROMPT}],
            client_cfg=client_cfg,
            temperature=0.7,
        )
        print(f"  → {simple_stream['chunk_count']} chunks, "
              f"{simple_stream['latency_total']}s total")

        if args.delay > 0:
            time.sleep(args.delay)

        print(f"Sending complex question (stream)...")
        complex_stream = analyze_stream(
            messages=[{"role": "user", "content": COMPLEX_PROMPT}],
            client_cfg=client_cfg,
            temperature=0.7,
        )
        print(f"  → {complex_stream['chunk_count']} chunks, "
              f"{complex_stream['latency_total']}s total")

        anomalies = detect_stream_anomalies(simple_stream, complex_stream)
        print_stream_report(simple_stream, complex_stream, anomalies,
                            model=args.model, base_url=args.base_url)

        if args.output:
            out = {
                "mode": "stream",
                "model": args.model,
                "base_url": args.base_url,
                "simple_stream": simple_stream,
                "complex_stream": complex_stream,
                "anomalies": anomalies,
            }
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print(f"Results saved to {args.output}")

    else:
        print(f"Sending simple question...")
        simple_result = call_api_raw(
            messages=[{"role": "user", "content": SIMPLE_PROMPT}],
            client_cfg=client_cfg,
            temperature=0.7,
        )
        print(f"  → {simple_result['latency']}s, {len(simple_result['content'])} chars")

        if args.delay > 0:
            time.sleep(args.delay)

        print(f"Sending complex question...")
        complex_result = call_api_raw(
            messages=[{"role": "user", "content": COMPLEX_PROMPT}],
            client_cfg=client_cfg,
            temperature=0.7,
        )
        print(f"  → {complex_result['latency']}s, {len(complex_result['content'])} chars")

        simple_flat = flatten(simple_result["raw"])
        complex_flat = flatten(complex_result["raw"])
        anomalies = detect_anomalies(simple_flat, complex_flat)

        print_report(
            simple_result, complex_result,
            simple_flat, complex_flat,
            anomalies,
            model=args.model,
            base_url=args.base_url,
        )

        if args.output:
            out = {
                "mode": "non-stream",
                "model": args.model,
                "base_url": args.base_url,
                "simple": {
                    "latency": simple_result["latency"],
                    "content_len": len(simple_result["content"]),
                    "raw": simple_result["raw"],
                },
                "complex": {
                    "latency": complex_result["latency"],
                    "content_len": len(complex_result["content"]),
                    "raw": complex_result["raw"],
                },
                "anomalies": anomalies,
            }
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

