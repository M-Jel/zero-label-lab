"""Report agent: Muse Spark writes a report from the measurements, a deterministic verifier
checks every number, and Spark fixes the report if needed (at most 2 corrections)."""
import argparse
import json
import re
from pathlib import Path

from zerolabel.spark_client import ask

COVERAGE_MIN = 0.8
PRICE_IN, PRICE_OUT = 1.25, 4.25  # $ per million tokens (Meta "Models" page)

SYSTEM = f"""You are a lab assistant writing a short measurement report for a scientist.
Rules:
- Use ONLY numbers that appear in the JSON measurements. Never compute new numbers, never invent any.
- Keep the decimals given in the JSON. A fraction like 0.443 may be written as 44.3%.
- If guard.estimated_coverage is below {COVERAGE_MIN}, the first line must start with 'WARNING:' and explain
  that SAM probably missed many objects, so counts are too low and size statistics may be biased
  toward the objects SAM did see.
- Sections: Summary, Counts, Sizes, Speeds, Reliability. At most 200 words. Plain text, no tables.
- Units: radius in pixels, speed in pixels per second. Do not mention model version numbers."""

NUM_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?%?")


def numbers(d):
    if isinstance(d, bool):
        return
    if isinstance(d, (int, float)):
        yield float(d)
    elif isinstance(d, dict):
        for v in d.values():
            yield from numbers(v)
    elif isinstance(d, list):
        for v in d:
            yield from numbers(v)


def check(text: str, m: dict) -> list[str]:
    values = list(numbers(m))
    issues = []
    clean = text.replace("SAM 3.1", "SAM")
    for tok in NUM_RE.findall(clean):
        pct = tok.endswith("%")
        s = tok.rstrip("%").lstrip("+")
        dec = len(s.split(".")[1]) if "." in s else 0
        v = float(s)
        tol = 0.5 * 10 ** (-dec) + 1e-9
        cands = [c * 100 for c in values] if pct else values
        if not any(abs(abs(v) - abs(c)) <= tol for c in cands):
            issues.append(f"number {tok} is not in the measurements")
    if m["guard"]["estimated_coverage"] < COVERAGE_MIN and not text.lstrip().startswith("WARNING:"):
        issues.append(f"estimated_coverage < {COVERAGE_MIN} but the report does not start with 'WARNING:'")
    return issues


def run(measures_path: str, out_dir: str, max_fix: int = 2) -> dict:
    m_full = json.load(open(measures_path, encoding="utf-8"))
    m = {k: v for k, v in m_full.items() if k not in ("sam", "cv")}
    data = json.dumps(m, indent=1)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prompt = f"Measurements (JSON):\n{data}\n\nWrite the report."
    log, tin, tout = [], 0, 0
    for attempt in range(max_fix + 1):
        res = ask(prompt, SYSTEM, effort="low", max_tokens=4000,
                  raw_path=str(out / f"spark_attempt{attempt}.jsonl"))
        u = res["usage"] or {}
        tin += u.get("input_tokens", 0)
        tout += u.get("output_tokens", 0)
        issues = check(res["text"], m)
        log.append({"attempt": attempt, "issues": issues, "complete": res["complete"]})
        print(f"attempt {attempt}: {len(issues)} issue(s)")
        if not issues:
            break
        prompt = (f"Measurements (JSON):\n{data}\n\nYour previous report:\n{res['text']}\n\n"
                  "A verifier found these problems:\n- " + "\n- ".join(issues) +
                  "\n\nRewrite the full report and fix them. Remove any number that is not in the JSON.")
    status = "VERIFIED" if not issues else "NOT VERIFIED"
    header = f"[{status}] every number checked against {Path(measures_path).name}\n\n"
    if issues:
        header += "Unverified items:\n- " + "\n- ".join(issues) + "\n\n"
    (out / "report.md").write_text(header + res["text"] + "\n", encoding="utf-8")
    summary = {"status": status, "attempts": log, "input_tokens": tin, "output_tokens": tout,
               "est_cost_usd": round((tin * PRICE_IN + tout * PRICE_OUT) / 1e6, 5)}
    (out / "verification.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measures", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = run(a.measures, a.out)
    print(f"{s['status']} | tokens {s['input_tokens']} in / {s['output_tokens']} out "
          f"| estimated cost {s['est_cost_usd']} $ -> {a.out}/report.md")


if __name__ == "__main__":
    main()
