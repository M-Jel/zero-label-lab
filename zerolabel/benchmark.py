"""Comparison table SAM 3.1 vs OpenCV on every difficulty level."""
import argparse
import json
from pathlib import Path

from zerolabel.evaluate import score

VISIBILITY = {1: "clear", 2: "clear", 3: "needs focus", 4: "very hard", 5: "beyond human eye"}
METHODS = {"SAM 3.1": "outputs/sam_tracks/level{l}.json",
           "OpenCV": "outputs/cv_tracks/level{l}.json"}


def fmt(v, spec):
    return "-" if v is None else format(v, spec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out-json", default="outputs/benchmark.json")
    ap.add_argument("--out-md", default="outputs/benchmark.md")
    a = ap.parse_args()

    rows = []
    for l in a.levels:
        truth = json.load(open(f"data/synth/level{l}/truth.json", encoding="utf-8"))
        for method, pattern in METHODS.items():
            p = Path(pattern.format(l=l))
            if not p.exists():
                print(f"missing: {p}")
                continue
            s = score(truth, json.load(open(p, encoding="utf-8"))["frames"])
            rows.append({"level": l, "visibility": VISIBILITY.get(l, "?"), "method": method, **s})

    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(rows, indent=1), encoding="utf-8")

    lines = ["| Level | Visibility | Method | Recall | Precision | Radius bias (px) "
             "| IDs pred / true | ID switches | Merges ok / checked |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['level']} | {r['visibility']} | {r['method']} | {r['recall']:.3f} "
                     f"| {r['precision']:.3f} | {fmt(r['radius_bias'], '+.2f')} "
                     f"| {r['pred_ids']} / {r['true_ids']} | {r['id_switches']} "
                     f"| {r['merges_ok']} / {r['merges_checked']} |")
    md = "\n".join(lines)
    Path(a.out_md).write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
