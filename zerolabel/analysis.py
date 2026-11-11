"""Evaluation analyses that use the synthetic ground truth.

- recall per true-radius bin (evidence of the size bias), SAM vs OpenCV
- sensitivity of the scores to the matching threshold (4, 8, 12 px)
- correlation between the annotation-free estimated coverage and the true recall
- failure gallery: frames where the drops missed by SAM are circled

Outputs go to outputs/analysis/ (JSON + Markdown) and outputs/figures/ (PNG).
Usage:
    python -m zerolabel.analysis
"""
import argparse
import json
import re
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

from zerolabel.evaluate import match, score  # noqa: E402

DATA = Path("data/synth")
OUT = Path("outputs")
ANALYSIS = OUT / "analysis"
FIGS = OUT / "figures"
RADIUS_BINS = [0, 4, 6, 8, 12, 16, 100]
THRESHOLDS = [4.0, 8.0, 12.0]
NAME_RE = re.compile(r"^level(\d)(?:_s(\d+))?$")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def videos() -> list[dict]:
    """Every synthetic benchmark video that has a truth file, with its level and seed."""
    out = []
    for d in sorted(DATA.iterdir()):
        m = NAME_RE.match(d.name)
        if m and (d / "truth.json").exists():
            out.append({"name": d.name, "level": int(m.group(1)), "seed": int(m.group(2) or 0)})
    return out


def tracks(method: str, name: str):
    p = OUT / f"{method}_tracks" / f"{name}.json"
    return load(p)["frames"] if p.exists() else None


# ---------------------------------------------------------------- recall per radius bin
def radius_hits(truth: dict, pred: list[list[dict]], max_dist: float = 8.0):
    """(true radius, matched?) for every true drop in every frame."""
    out = []
    for f in range(min(len(truth["frames"]), len(pred))):
        gt = truth["frames"][f]
        hit = {i for i, _, _ in match(gt, pred[f], max_dist)}
        out += [(g["r"], i in hit) for i, g in enumerate(gt)]
    return out


def recall_by_radius(vids: list[dict]) -> dict:
    res = {}
    for method in ("sam", "cv"):
        pairs = []
        for v in vids:
            pred = tracks(method, v["name"])
            if pred is not None:
                pairs += radius_hits(load(DATA / v["name"] / "truth.json"), pred)
        r = np.array([p[0] for p in pairs])
        h = np.array([p[1] for p in pairs])
        rows = []
        for lo, hi in zip(RADIUS_BINS, RADIUS_BINS[1:]):
            sel = (r >= lo) & (r < hi)
            n = int(sel.sum())
            rows.append({"r_min": lo, "r_max": hi, "n": n,
                         "recall": round(float(h[sel].mean()), 3) if n else None})
        res[method] = rows
    return res


def plot_recall_by_radius(res: dict, title: str, path: Path) -> None:
    labels = [f"{r['r_min']}–{r['r_max']}" if r["r_max"] < 100 else f"≥{r['r_min']}"
              for r in res["sam"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=110)
    for k, (method, label, color) in enumerate([("sam", "SAM 3.1", "#3b6fd4"),
                                                ("cv", "OpenCV baseline", "#d9822b")]):
        vals = [r["recall"] if r["recall"] is not None else np.nan for r in res[method]]
        ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, label=label, color=color)
    for i, r in enumerate(res["sam"]):
        ax.text(i, 1.03, f"n={r['n']}", ha="center", fontsize=7, color="#555")
    ax.set_xticks(x, labels)
    ax.set_xlabel("true drop radius (px)")
    ax.set_ylabel("recall")
    ax.set_ylim(0, 1.12)
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- matching threshold
def threshold_sensitivity(vids: list[dict]) -> list[dict]:
    rows = []
    for v in vids:
        truth = load(DATA / v["name"] / "truth.json")
        for method in ("sam", "cv"):
            pred = tracks(method, v["name"])
            if pred is None:
                continue
            for t in THRESHOLDS:
                s = score(truth, pred, t)
                rows.append({"video": v["name"], "level": v["level"], "seed": v["seed"],
                             "method": method, "max_dist": t,
                             "recall": round(s["recall"], 3), "precision": round(s["precision"], 3)})
    return rows


# ---------------------------------------------------------------- coverage vs recall
def bootstrap_ci(x, y, fn, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) == 0 or np.ptp(y[i]) == 0:
            continue
        vals.append(fn(x[i], y[i]))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def coverage_vs_recall(vids: list[dict]) -> dict:
    pts = []
    for v in vids:
        mp = OUT / "measures" / f"{v['name']}.json"
        sam = tracks("sam", v["name"])
        if sam is None or not mp.exists():
            continue
        s = score(load(DATA / v["name"] / "truth.json"), sam)
        pts.append({"video": v["name"], "level": v["level"], "seed": v["seed"],
                    "estimated_coverage": load(mp)["guard"]["estimated_coverage"],
                    "true_recall": round(s["recall"], 3)})
    res = {"n_videos": len(pts), "points": pts}
    if len(pts) >= 3:
        x = [p["estimated_coverage"] for p in pts]
        y = [p["true_recall"] for p in pts]
        pear = lambda a, b: stats.pearsonr(a, b)[0]  # noqa: E731
        spear = lambda a, b: stats.spearmanr(a, b)[0]  # noqa: E731
        res["pearson"] = {"r": round(float(pear(x, y)), 3), "ci95": bootstrap_ci(x, y, pear)}
        res["spearman"] = {"rho": round(float(spear(x, y)), 3), "ci95": bootstrap_ci(x, y, spear)}
        res["mean_abs_diff"] = round(float(np.mean(np.abs(np.array(x) - np.array(y)))), 3)
    return res


def plot_coverage(res: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=110)
    cmap = plt.get_cmap("viridis")
    for lvl in range(1, 6):
        p = [q for q in res["points"] if q["level"] == lvl]
        if p:
            ax.scatter([q["estimated_coverage"] for q in p], [q["true_recall"] for q in p],
                       color=cmap((lvl - 1) / 4), label=f"level {lvl}", s=28, zorder=3)
    ax.plot([0, 1], [0, 1], color="#999", lw=0.8, ls="--", zorder=1)
    ax.axvline(0.8, color="#c0392b", lw=0.8, ls=":")
    ax.text(0.81, 0.03, "WARNING below 0.8", color="#c0392b", fontsize=7)
    ax.set_xlabel("estimated coverage (no ground truth)")
    ax.set_ylabel("true SAM recall (ground truth)")
    if "pearson" in res:
        ax.set_title(f"{res['n_videos']} videos · Pearson r = {res['pearson']['r']}, "
                     f"Spearman ρ = {res['spearman']['rho']}", fontsize=8)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- failure gallery
def read_frame(video: Path, idx: int):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def failure_gallery(name: str, frame_idx: int, path: Path, crop=None) -> dict:
    """Green: drop found by SAM. Red: drop missed by SAM (from the ground truth)."""
    truth = load(DATA / name / "truth.json")
    sam = tracks("sam", name)
    gt = truth["frames"][frame_idx]
    pr = sam[frame_idx] if sam is not None else []
    hit = {i for i, _, _ in match(gt, pr, 8.0)}
    img = read_frame(DATA / name / "video.mp4", frame_idx)
    # stretch the contrast so the faint drops are visible to a reader
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lo, hi = np.percentile(g, 1), np.percentile(g, 99)
    g = np.clip((g - lo) / max(1.0, hi - lo) * 255, 0, 255).astype(np.uint8)
    vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    for i, d in enumerate(gt):
        color = (60, 180, 60) if i in hit else (40, 40, 230)
        cv2.circle(vis, (int(round(d["x"])), int(round(d["y"]))), int(round(d["r"] + 4)), color, 2,
                   cv2.LINE_AA)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 50), (255, 255, 255), -1)
    cv2.putText(vis, f"{name}, frame {frame_idx}: SAM found {len(hit)} of {len(gt)} drops",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(vis, "green = found by SAM   red = missed (ground truth)",
                (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis)
    missed = [round(d["r"], 2) for i, d in enumerate(gt) if i not in hit]
    found = [round(d["r"], 2) for i, d in enumerate(gt) if i in hit]
    return {"video": name, "frame": frame_idx, "found": len(found), "total": len(gt),
            "missed_radii": missed, "found_radii": found, "image": path.as_posix()}


# ---------------------------------------------------------------- driver
def md_table(rows: list[dict], cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join("-" if r[c] is None else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def run_all() -> dict:
    vids = videos()
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    result = {}

    seed0 = [v for v in vids if v["seed"] == 0]
    rb = {"all_videos": recall_by_radius(vids), "seed0": recall_by_radius(seed0),
          "per_level_seed0": {v["name"]: recall_by_radius([v]) for v in seed0}}
    result["recall_by_radius"] = rb
    plot_recall_by_radius(rb["all_videos"],
                          f"Recall by true radius ({len(vids)} synthetic videos, 8 px matching)",
                          FIGS / "recall_by_radius.png")

    th = threshold_sensitivity(seed0)
    result["threshold_sensitivity"] = th

    cov = coverage_vs_recall(vids)
    result["coverage_vs_recall"] = cov
    if cov["points"]:
        plot_coverage(cov, FIGS / "coverage_vs_recall.png")

    gal = []
    for name, f in [("level2", 75), ("level3", 75), ("level4", 75)]:
        if (DATA / name / "video.mp4").exists():
            gal.append(failure_gallery(name, f, FIGS / f"gallery_{name}_f{f}.png"))
    result["gallery"] = gal

    (ANALYSIS / "analysis.json").write_text(json.dumps(result, indent=1), encoding="utf-8")

    cols_rb = ["r_min", "r_max", "n", "recall"]
    md = ["# Analyses (ground truth used)", "",
          f"## Recall per true radius bin, all {len(vids)} videos", "",
          "SAM 3.1:", "", md_table(rb["all_videos"]["sam"], cols_rb), "",
          "OpenCV baseline:", "", md_table(rb["all_videos"]["cv"], cols_rb), "",
          "## Matching threshold sensitivity (seed 0)", "",
          md_table(th, ["video", "method", "max_dist", "recall", "precision"]), "",
          f"## Estimated coverage vs true SAM recall ({cov['n_videos']} videos)", ""]
    if "pearson" in cov:
        md += [f"- Pearson r = {cov['pearson']['r']} (bootstrap 95% CI {cov['pearson']['ci95']})",
               f"- Spearman rho = {cov['spearman']['rho']} (bootstrap 95% CI {cov['spearman']['ci95']})",
               f"- mean |estimated coverage - true recall| = {cov['mean_abs_diff']}", ""]
    md += [md_table(cov["points"], ["video", "estimated_coverage", "true_recall"]), "",
           "## Failure gallery", ""]
    for g in gal:
        md.append(f"- {g['video']} frame {g['frame']}: {g['found']}/{g['total']} found; "
                  f"missed radii {g['missed_radii']}; found radii {g['found_radii']}")
    (ANALYSIS / "analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"analyses -> {ANALYSIS / 'analysis.md'} and {FIGS}/")
    return result


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    run_all()


if __name__ == "__main__":
    main()
