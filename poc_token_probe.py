"""
PoC Token Probe: use model-specific "fingerprint" prompts to detect whether a
vendor is actually serving the claimed model.

Each model entry in assets/poc_tokens.json carries one or more tests. A test
is flagged when:
  - expect_any is given and NONE of the listed keywords appear in the response
  - forbid_any is given and ANY of the listed keywords appear in the response

If the model is not registered in the database, the probe exits immediately
with no result — the database is intentionally sparse and incomplete.
"""

import json
import os
import sys
import time
import argparse
from typing import Optional

from request_llm import call_api
from model_db import lookup

_TOKENS_FILE = os.path.join(os.path.dirname(__file__), "assets", "poc_tokens.json")


def lookup_poc_entry(model_name: str) -> Optional[dict]:
    return lookup(_TOKENS_FILE, model_name)


# ---------------------------------------------------------------------------
# Per-test verdict logic
# ---------------------------------------------------------------------------

def _evaluate(response: str, flag_if_present: list, flag_if_absent: list) -> tuple[str, str]:
    """
    Returns (verdict, reason).
    verdict: "PASS" | "FAIL" | "NO_CRITERIA"

    flag_if_present: suspicious if ANY of these keywords appear in the response.
    flag_if_absent:  suspicious if NONE of these keywords appear in the response.
    """
    if not flag_if_present and not flag_if_absent:
        return "NO_CRITERIA", "no check criteria defined"

    resp_lower = response.lower()
    failures = []

    if flag_if_present:
        hits = [kw for kw in flag_if_present if kw.lower() in resp_lower]
        if hits:
            failures.append(f"suspicious keywords present: {hits}")

    if flag_if_absent:
        hits = [kw for kw in flag_if_absent if kw.lower() in resp_lower]
        if not hits:
            failures.append(f"expected keywords all absent: {flag_if_absent}")

    if failures:
        return "FAIL", "; ".join(failures)
    return "PASS", "all keyword checks passed"


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------

def run_probe(args) -> int:
    entry = lookup_poc_entry(args.model)
    if entry is None:
        print(f"No PoC tokens registered for '{args.model}'. Skipping.")
        return 0

    tests = entry.get("tests", [])
    if not tests:
        print(f"Model matched but has no tests defined. Skipping.")
        return 0

    client_cfg = {
        "base_url": args.base_url,
        "api_key":  args.api_key,
        "model":    args.model,
        "api_style": args.api_style,
        "system":   args.system,
        "max_tokens": args.max_tokens,
    }

    print(f"Model   : {args.model}")
    print(f"Tests   : {len(tests)}")
    print()

    results = []
    for i, test in enumerate(tests):
        test_id     = test.get("id", f"test-{i+1}")
        description = test.get("description", "")
        prompt      = test.get("prompt", "")
        flag_if_present = test.get("flag_if_present", [])
        flag_if_absent  = test.get("flag_if_absent", [])

        print(f"[{test_id}] {description}")

        if not prompt:
            print("  SKIPPED (no prompt defined)")
            results.append({"id": test_id, "verdict": "SKIPPED", "reason": "empty prompt"})
            continue

        try:
            response = call_api(
                [{"role": "user", "content": prompt}],
                client_cfg,
                temperature=0.0,
                timeout=args.timeout,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append({"id": test_id, "verdict": "ERROR", "reason": str(exc)})
            if i < len(tests) - 1:
                time.sleep(args.delay)
            continue

        verdict, reason = _evaluate(response, flag_if_present, flag_if_absent)

        print(f"  Response : {response[:300].strip()!r}")
        print(f"  Verdict  : {verdict}  ({reason})")

        results.append({
            "id":               test_id,
            "description":      description,
            "response_preview": response[:500],
            "flag_if_present":  flag_if_present,
            "flag_if_absent":   flag_if_absent,
            "verdict":          verdict,
            "reason":           reason,
        })

        if i < len(tests) - 1:
            time.sleep(args.delay)

    # ---- summary ----
    print()
    counts = {v: sum(1 for r in results if r["verdict"] == v)
              for v in ("PASS", "FAIL", "NO_CRITERIA", "SKIPPED", "ERROR")}
    total = len(results)
    print(f"Results : {counts['PASS']}/{total} passed  |  "
          f"{counts['FAIL']} failed  |  "
          f"{counts['NO_CRITERIA']} no-criteria  |  "
          f"{counts['ERROR']} errors")

    if counts["FAIL"] > 0:
        print(f"SUSPICIOUS: {counts['FAIL']} test(s) failed — "
              f"model may not be '{args.model}'")
    elif counts["PASS"] > 0:
        print("OK: all evaluated PoC token tests passed")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"model": args.model, "results": results}, f,
                      ensure_ascii=False, indent=2)
        print(f"\nFull results saved to {args.output}")

    return 1 if counts["FAIL"] > 0 else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="PoC token fingerprint probe for LLM vendors")
    p.add_argument("--base-url",  required=True,  help="API base URL")
    p.add_argument("--api-key",   required=True,  help="API key")
    p.add_argument("--model",     required=True,  help="Model name (will be normalised for DB lookup)")
    p.add_argument("--api-style", default="openai", choices=["openai", "anthropic"],
                   help="API style (default: openai)")
    p.add_argument("--system",    default=None,   help="System prompt (optional)")
    p.add_argument("--max-tokens", type=int, default=512,
                   help="Max output tokens per request (default: 512)")
    p.add_argument("--timeout",   type=int, default=60,
                   help="Per-request timeout in seconds (default: 60)")
    p.add_argument("--delay",     type=float, default=1.0,
                   help="Seconds between requests (default: 1.0)")
    p.add_argument("--output",    default=None,
                   help="Save full results to this JSON file")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(run_probe(parse_args()))
