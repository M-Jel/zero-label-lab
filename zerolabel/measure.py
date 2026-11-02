"""Mesures sans annotation a partir des pistes SAM, plus un garde-fou OpenCV
qui signale les gouttes probablement ratees par SAM. N'utilise jamais la verite."""
import argparse
import json
from pathlib import Path

import numpy as np

from zerolabel.evaluate import match


def tracks_of(frames):
    tr = {}
    for f, dets in enumerate(frames):
        for d in dets:
            tr.setdefault(d["id"], []).append((f, d["x"], d["y"], d["r"]))
    return tr


def q(v, p):
    return round(float(np.percentile(v, p)), 2) if len(v) else None


def measure(sam, cv, fps, max_dist=8.0, flag_min=3):
    n = len(sam)
    counts = [len(f) for f in sam]
    tr = tracks_of(sam)
    radii = [float(np.mean([p[3] for p in t])) for t in tr.values()]
    speeds = [float(np.hypot(b[1] - a[1], b[2] - a[2]) * fps)
              for t in tr.values() for a, b in zip(t, t[1:]) if b[0] == a[0] + 1]
    unmatched = [len(cv[f]) - len(match(cv[f], sam[f], max_dist)) for f in range(min(n, len(cv)))]
    flagged = [f for f, u in enumerate(unmatched) if u >= flag_min]
    total_sam, total_extra = sum(counts), sum(unmatched)
    return {
        "frames": n, "fps": fps,
        "count_first": counts[0] if n else 0, "count_last": counts[-1] if n else 0,
        "count_mean": round(float(np.mean(counts)), 2) if n else 0,
        "tracks": len(tr),
        "frames_without_detection": sum(c == 0 for c in counts),
        "radius_px": {"mean": round(float(np.mean(radii)), 2) if radii else None,
                      "p10": q(radii, 10), "median": q(radii, 50), "p90": q(radii, 90)},
        "speed_px_per_s": {"mean": round(float(np.mean(speeds)), 2) if speeds else None,
                           "median": q(speeds, 50), "p90": q(speeds, 90)},
        "guard": {"cv_unmatched_per_frame": round(total_extra / max(1, n), 2),
                  "flag_min": flag_min,
                  "flagged_frames": len(flagged),
                  "flagged_fraction": round(len(flagged) / max(1, n), 3),
                  "estimated_coverage": round(total_sam / max(1, total_sam + total_extra), 3)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam", required=True)
    ap.add_argument("--cv", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sam = json.load(open(a.sam, encoding="utf-8"))["frames"]
    cv = json.load(open(a.cv, encoding="utf-8"))["frames"]
    m = {"sam": a.sam, "cv": a.cv, **measure(sam, cv, a.fps)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(m, indent=1), encoding="utf-8")
    g = m["guard"]
    print(f"gouttes debut/fin: {m['count_first']}/{m['count_last']} | rayon median: "
          f"{m['radius_px']['median']} px | couverture estimee: {g['estimated_coverage']} "
          f"| images signalees: {g['flagged_fraction']}")


if __name__ == "__main__":
    main()
