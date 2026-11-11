"""Score detections (SAM or OpenCV) against the synthetic ground truth."""
import argparse
import json

import numpy as np
from scipy.optimize import linear_sum_assignment


def match(gt: list[dict], pr: list[dict], max_dist: float) -> list[tuple[int, int, float]]:
    if not gt or not pr:
        return []
    G = np.array([[g["x"], g["y"]] for g in gt])
    P = np.array([[p["x"], p["y"]] for p in pr])
    D = np.linalg.norm(G[:, None] - P[None], axis=2)
    gi, pi = linear_sum_assignment(D)
    return [(i, j, float(D[i, j])) for i, j in zip(gi, pi) if D[i, j] <= max_dist]


def score(truth: dict, pred_frames: list[list[dict]], max_dist: float = 8.0) -> dict:
    T, P = truth["frames"], pred_frames
    n = min(len(T), len(P))
    tp = fn = fp = 0
    dists, dr = [], []
    assign: list[dict[int, int]] = []
    for f in range(n):
        m = match(T[f], P[f], max_dist)
        tp += len(m)
        fn += len(T[f]) - len(m)
        fp += len(P[f]) - len(m)
        assign.append({T[f][i]["id"]: P[f][j]["id"] for i, j, _ in m})
        for i, j, d in m:
            dists.append(d)
            dr.append(P[f][j]["r"] - T[f][i]["r"])

    switches, last = 0, {}
    for f in range(n):
        for g, p in assign[f].items():
            if g in last and last[g] != p:
                switches += 1
            last[g] = p

    def first_pred(gid, start):
        for f in range(max(start, 0), n):
            if gid in assign[f]:
                return assign[f][gid]
        return None

    def last_pred(gid, end):
        for f in range(min(end, n - 1), -1, -1):
            if gid in assign[f]:
                return assign[f][gid]
        return None

    events = truth.get("events", [])
    ok = checked = 0
    for e in events:
        parents = {last_pred(pid, e["frame"]) for pid in e["parents"]} - {None}
        child = first_pred(e["child"], e["frame"])
        if child is None or not parents:
            continue
        checked += 1
        ok += child in parents

    return {
        "frames": n,
        "recall": tp / max(1, tp + fn),
        "precision": tp / max(1, tp + fp),
        "tp": tp, "fn": fn, "fp": fp,
        "center_err_mean": float(np.mean(dists)) if dists else None,
        "center_err_max": float(np.max(dists)) if dists else None,
        "radius_bias": float(np.mean(dr)) if dr else None,
        "radius_abs_err": float(np.mean(np.abs(dr))) if dr else None,
        "id_switches": switches,
        "pred_ids": len({d["id"] for f in P[:n] for d in f}),
        "true_ids": len({d["id"] for f in T[:n] for d in f}),
        "merges_ok": ok, "merges_checked": checked, "merges_total": len(events),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default="data/synth/demo/truth.json")
    ap.add_argument("--pred", default="outputs/sam_tracks/circle.json")
    ap.add_argument("--max-dist", type=float, default=8.0)
    a = ap.parse_args()

    truth = json.load(open(a.truth, encoding="utf-8"))
    pred = json.load(open(a.pred, encoding="utf-8"))["frames"]
    s = score(truth, pred, a.max_dist)
    print(f"frames: {s['frames']}")
    print(f"recall: {s['recall']:.3f} | precision: {s['precision']:.3f} "
          f"(TP={s['tp']}, FN={s['fn']}, FP={s['fp']})")
    if s["center_err_mean"] is not None:
        print(f"centre error: mean {s['center_err_mean']:.2f} px, max {s['center_err_max']:.2f} px")
        print(f"radius error (pred - truth): bias {s['radius_bias']:+.2f} px, "
              f"absolute error {s['radius_abs_err']:.2f} px")
    print(f"id switches: {s['id_switches']} | predicted ids: {s['pred_ids']} "
          f"| true ids: {s['true_ids']}")
    print(f"consistent merges: {s['merges_ok']}/{s['merges_checked']} (out of {s['merges_total']} merges)")


if __name__ == "__main__":
    main()
