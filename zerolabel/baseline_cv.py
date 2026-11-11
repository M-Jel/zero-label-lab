"""Classical method (no AI): median background subtraction, threshold, blobs, simple tracking.
Writes the same format as sam_parse so that evaluate can score it."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def read_gray(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    if not frames:
        raise SystemExit(f"Cannot read {path}")
    return np.stack(frames)


def detect(video: np.ndarray, sigma: float, k: float, min_area: int) -> list[list[dict]]:
    bg = np.median(video, axis=0)
    diffs = [np.abs(cv2.GaussianBlur(f - bg, (0, 0), sigma)) for f in video]
    noise = 1.4826 * float(np.median(np.stack(diffs)))  # robust standard deviation of the noise
    thr = k * noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out = []
    for d in diffs:
        mask = cv2.morphologyEx((d > thr).astype(np.uint8), cv2.MORPH_OPEN, kernel)
        n, _, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
        dets = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < min_area:
                continue
            dets.append({"x": float(cent[i][0]), "y": float(cent[i][1]),
                         "r": float(np.sqrt(area / np.pi)),
                         "bbox": [float(x), float(y), float(x + w), float(y + h)]})
        out.append(dets)
    print(f"threshold: {thr:.2f} grey levels (estimated noise {noise:.2f})")
    return out


def track(frames: list[list[dict]], max_dist: float) -> None:
    next_id, prev = 0, []
    for cur in frames:
        if prev and cur:
            P = np.array([[d["x"], d["y"]] for d in prev])
            C = np.array([[d["x"], d["y"]] for d in cur])
            D = np.linalg.norm(P[:, None] - C[None], axis=2)
            for i, j in zip(*linear_sum_assignment(D)):
                if D[i, j] <= max_dist:
                    cur[j]["id"] = prev[i]["id"]
        for d in cur:
            if "id" not in d:
                d["id"] = next_id
                next_id += 1
        prev = cur


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma", type=float, default=1.5, help="smoothing (px)")
    ap.add_argument("--k", type=float, default=4.0, help="threshold in noise standard deviations")
    ap.add_argument("--min-area", type=int, default=6, help="minimum blob area (px)")
    ap.add_argument("--max-dist", type=float, default=10.0, help="max displacement between 2 frames (px)")
    a = ap.parse_args()

    frames = detect(read_gray(Path(a.video)), a.sigma, a.k, a.min_area)
    track(frames, a.max_dist)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source": a.video, "method": "opencv-baseline", "frames": frames}),
                   encoding="utf-8")
    ids = {d["id"] for f in frames for d in f}
    print(f"{len(frames)} frames | start: {len(frames[0])} | end: {len(frames[-1])} | distinct ids: {len(ids)}")


if __name__ == "__main__":
    main()
