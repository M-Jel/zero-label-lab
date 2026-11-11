"""Report agent: Muse Spark writes a report from the measurements, a deterministic verifier
checks every number, and Spark fixes the report if needed (at most 2 corrections)."""
import argparse
import json
from pathlib import Path

from zerolabel.verifier import COVERAGE_MIN, SIZE_BIAS_SENTENCE, check

PRICE_IN, PRICE_OUT = 1.25, 4.25  # $ per million tokens (Meta "Models" page)

SYSTEM = f"""You are a lab assistant writing a short measurement report for a lab scientist
who is not a programmer. The measurements come from an automatic video analysis of drops.

Meaning of the JSON fields (never copy these names, describe the quantity in plain words):
- frames, fps: length of the video and frame rate.
- count_first, count_last, count_mean: drops seen in the first frame, last frame, on average.
- tracks: number of distinct drops followed over time. frames_without_detection: frames where no drop was seen.
- radius_px (mean, p10, median, p90): drop radius in pixels; p10 and p90 mean "10% of drops are smaller
  than" and "10% of drops are larger than".
- speed_px_per_s (mean, median, p90): drop speed in pixels per second.
- guard: a classical detector is run as a cross-check. cv_unmatched_per_frame = objects it found that the
  main model did not, per frame. flagged_frames / flagged_fraction = frames with at least flag_min such
  objects. estimated_coverage = estimated share of the drops that the main model found.

Rules:
- Short, simple sentences. No tables, no lists of raw values, no field names, no underscores.
- Use ONLY numbers that appear in the JSON. Never compute a new number (no sums, no differences).
- Round every number to at most 2 decimals (0.443 -> 0.44). A fraction may be written as a percent
  (0.443 -> 44.3%).
- If estimated_coverage is below {COVERAGE_MIN}, the first line must start with 'WARNING:' and say that the
  main model probably missed many drops, so the counts are too low. The WARNING paragraph must also
  contain this exact sentence: "{SIZE_BIAS_SENTENCE}"
- If estimated_coverage is {COVERAGE_MIN} or more, do not write any WARNING.
- Sections, each with a one-word heading on its own line: Summary, Counts, Sizes, Speeds, Reliability.
- At most 180 words. Radius in pixels, speed in pixels per second. Do not mention model versions."""


def build_prompt(m: dict) -> str:
    return f"Measurements (JSON):\n{json.dumps(m, indent=1)}\n\nWrite the report."


def run(measures_path: str, out_dir: str, max_fix: int = 2) -> dict:
    from zerolabel.spark_client import ask  # paid call: imported only when really used
    m_full = json.load(open(measures_path, encoding="utf-8"))
    m = {k: v for k, v in m_full.items() if k not in ("sam", "cv")}
    data = json.dumps(m, indent=1)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(m)
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
                  "\n\nRewrite the full report and fix them. Remove any number that is not in the JSON "
                  "and follow every rule of the instructions.")
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
    ap.add_argument("--measures", required=True, nargs="+")
    ap.add_argument("--out", required=True, help="output folder; one sub-folder per measures file")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the prompt size and a cost estimate, make no API call")
    a = ap.parse_args()
    if a.dry_run:
        for mp in a.measures:
            m = {k: v for k, v in json.load(open(mp, encoding="utf-8")).items() if k not in ("sam", "cv")}
            n_in = (len(SYSTEM) + len(build_prompt(m))) // 4  # rough: 4 characters per token
            print(f"{mp}: ~{n_in} input tokens per attempt, WARNING expected: "
                  f"{m['guard']['estimated_coverage'] < COVERAGE_MIN}")
        return
    for mp in a.measures:
        out = Path(a.out) / Path(mp).stem if len(a.measures) > 1 else Path(a.out)
        s = run(mp, str(out))
        print(f"{mp}: {s['status']} | tokens {s['input_tokens']} in / {s['output_tokens']} out "
              f"| estimated cost {s['est_cost_usd']} $ -> {out}/report.md")


if __name__ == "__main__":
    main()
