"""Compare les detections SAM a la verite synthetique."""
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default="data/synth/demo/truth.json")
    ap.add_argument("--pred", default="outputs/sam_tracks/circle.json")
    ap.add_argument("--max-dist", type=float, default=8.0)
    a = ap.parse_args()

    truth = json.load(open(a.truth, encoding="utf-8"))
    T = truth["frames"]
    P = json.load(open(a.pred, encoding="utf-8"))["frames"]
    n = min(len(T), len(P))

    tp = fn = fp = 0
    dists, dr = [], []
    assign: list[dict[int, int]] = []
    for f in range(n):
        m = match(T[f], P[f], a.max_dist)
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

    def first_pred(gid: int, start: int):
        for f in range(max(start, 0), n):
            if gid in assign[f]:
                return assign[f][gid]
        return None

    def last_pred(gid: int, end: int):
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

    print(f"images: {n}")
    print(f"rappel: {tp / max(1, tp + fn):.3f} | precision: {tp / max(1, tp + fp):.3f} "
          f"(TP={tp}, FN={fn}, FP={fp})")
    if dists:
        print(f"erreur centre: moyenne {np.mean(dists):.2f} px, max {np.max(dists):.2f} px")
        print(f"erreur rayon (SAM - verite): biais {np.mean(dr):+.2f} px, "
              f"erreur absolue {np.mean(np.abs(dr)):.2f} px")
    print(f"changements d'identite: {switches}")
    print(f"fusions coherentes: {ok}/{checked} (sur {len(events)} fusions)")


if __name__ == "__main__":
    main()
