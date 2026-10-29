"""Generate synthetic lab videos (droplets that drift and merge) with exact ground truth.

Usage:
    python -m zerolabel.synth --out data/synth/demo --seed 0
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

SHIFT = 4            # sub-pixel drawing: coordinates are multiplied by 2**SHIFT
SCALE = 1 << SHIFT


def background(h, w):
    """Uneven lighting, like a microscope or a phone camera."""
    y = np.linspace(0, 1, h)[:, None]
    x = np.linspace(0, 1, w)[None, :]
    base = 165 + 35 * (0.6 * x + 0.4 * y)
    return np.repeat(base[..., None], 3, axis=2).astype(np.uint8)


def draw_drop(img, x, y, r):
    """Dark rim, lighter core and a small highlight."""
    def p(px, py):
        return int(round(px * SCALE)), int(round(py * SCALE))
    cv2.circle(img, p(x, y), int(round(r * SCALE)), (70, 70, 70), -1, cv2.LINE_AA, SHIFT)
    cv2.circle(img, p(x, y), int(round(0.7 * r * SCALE)), (125, 125, 125), -1, cv2.LINE_AA, SHIFT)
    cv2.circle(img, p(x - 0.3 * r, y - 0.3 * r), int(round(max(1.0, 0.18 * r) * SCALE)),
               (235, 235, 235), -1, cv2.LINE_AA, SHIFT)


def place_drops(rng, a):
    """Random drops that do not overlap at the start."""
    drops, tries = [], 0
    while len(drops) < a.n and tries < 10_000:
        tries += 1
        r = rng.uniform(a.rmin, a.rmax)
        x, y = rng.uniform(r, a.width - r), rng.uniform(r, a.height - r)
        if all(math.hypot(x - d["x"], y - d["y"]) > r + d["r"] + 2 for d in drops):
            drops.append({"id": len(drops), "x": x, "y": y, "r": r,
                          "vx": rng.normal(0, a.speed), "vy": rng.normal(0, a.speed)})
    return drops


def step(drops, rng, a):
    """Drift with random jitter, bounce on the walls."""
    for d in drops:
        d["vx"] += rng.normal(0, a.jitter)
        d["vy"] += rng.normal(0, a.jitter)
        d["x"] += d["vx"]
        d["y"] += d["vy"]
        if not d["r"] <= d["x"] <= a.width - d["r"]:
            d["vx"] = -d["vx"]
            d["x"] = min(max(d["x"], d["r"]), a.width - d["r"])
        if not d["r"] <= d["y"] <= a.height - d["r"]:
            d["vy"] = -d["vy"]
            d["y"] = min(max(d["y"], d["r"]), a.height - d["r"])


def merge(drops, next_id, frame, events):
    """Two overlapping drops become one new drop (area and momentum are kept)."""
    while True:
        pair = next(((i, j) for i in range(len(drops)) for j in range(i + 1, len(drops))
                     if math.hypot(drops[i]["x"] - drops[j]["x"], drops[i]["y"] - drops[j]["y"])
                     < 0.9 * (drops[i]["r"] + drops[j]["r"])), None)
        if pair is None:
            return next_id
        a_, b_ = drops[pair[0]], drops[pair[1]]
        wa, wb = a_["r"] ** 2, b_["r"] ** 2
        w = wa + wb
        child = {"id": next_id, "r": math.sqrt(w),
                 **{k: (wa * a_[k] + wb * b_[k]) / w for k in ("x", "y", "vx", "vy")}}
        events.append({"frame": frame, "type": "merge",
                       "parents": [a_["id"], b_["id"]], "child": next_id})
        drops[:] = [d for k, d in enumerate(drops) if k not in pair] + [child]
        next_id += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/synth/demo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--n", type=int, default=12, help="number of drops at the start")
    ap.add_argument("--rmin", type=float, default=6.0, help="min radius (px)")
    ap.add_argument("--rmax", type=float, default=20.0, help="max radius (px)")
    ap.add_argument("--speed", type=float, default=1.5, help="initial speed spread (px/frame)")
    ap.add_argument("--jitter", type=float, default=0.3, help="random push per frame (px/frame)")
    ap.add_argument("--noise", type=float, default=8.0, help="Gaussian noise (grey levels)")
    ap.add_argument("--blur", type=float, default=1.0, help="Gaussian blur sigma (px), 0 = none")
    ap.add_argument("--no-merge", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)            # motion
    rng_noise = np.random.default_rng(a.seed + 1)  # image noise, kept apart so motion stays the same

    drops = place_drops(rng, a)
    next_id = len(drops)
    bg = background(a.height, a.width)
    writer = cv2.VideoWriter(str(out / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             a.fps, (a.width, a.height))
    if not writer.isOpened():
        raise SystemExit("Could not open the video writer")

    frames, events = [], []
    for f in range(a.frames):
        if f > 0:
            step(drops, rng, a)
            if not a.no_merge:
                next_id = merge(drops, next_id, f, events)
        img = bg.copy()
        for d in drops:
            draw_drop(img, d["x"], d["y"], d["r"])
        if a.blur > 0:
            img = cv2.GaussianBlur(img, (0, 0), a.blur)
        if a.noise > 0:
            img = np.clip(img + rng_noise.normal(0, a.noise, img.shape), 0, 255).astype(np.uint8)
        writer.write(img)
        frames.append([{"id": d["id"], "x": round(d["x"], 2), "y": round(d["y"], 2),
                        "r": round(d["r"], 2),
                        "bbox": [round(d["x"] - d["r"], 2), round(d["y"] - d["r"], 2),
                                 round(d["x"] + d["r"], 2), round(d["y"] + d["r"], 2)]}
                       for d in drops])
    writer.release()

    truth = {"params": vars(a), "units": "pixels and frames", "frames": frames, "events": events}
    (out / "truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    print(f"{a.frames} frames, {len(frames[0])} drops at start, {len(events)} merges, "
          f"{len(frames[-1])} drops at the end -> {out}")


if __name__ == "__main__":
    main()
