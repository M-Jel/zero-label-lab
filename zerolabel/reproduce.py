"""Rebuild every result from cached files. Never calls a paid API.

Steps (each one is skipped when its output already exists, unless --force):
  1. generate the synthetic videos that are missing (deterministic, bit-identical)
  2. run the OpenCV baseline on every video
  3. parse the cached SAM raw responses (outputs/sam_raw/<video>/circle.jsonl)
  4. annotation-free measurements (SAM tracks + OpenCV guard)
  5. benchmark table and analyses (figures, correlation, ...)

Usage:
    python -m zerolabel.reproduce            # seed 0 only (the original benchmark)
    python -m zerolabel.reproduce --seeds 0 1 2 3
"""
import argparse
import json
from pathlib import Path

from zerolabel import baseline_cv, synth
from zerolabel.measure import measure
from zerolabel.sam_parse import parse_file

DATA = Path("data/synth")
OUT = Path("outputs")


def video_name(level: int, seed: int) -> str:
    """Seed 0 keeps the historical names (level1 ... level5)."""
    return f"level{level}" if seed == 0 else f"level{level}_s{seed}"


def write_json(path: Path, obj, indent=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent), encoding="utf-8")


def process(name: str, level: int, seed: int, force: bool) -> dict:
    vdir = DATA / name
    status = {"video": name}
    if not (vdir / "truth.json").exists() or not (vdir / "video.mp4").exists():
        synth.make_level(vdir, level, seed, quiet=True)
        status["generated"] = True

    cv_path = OUT / "cv_tracks" / f"{name}.json"
    if force or not cv_path.exists():
        frames = baseline_cv.detect(baseline_cv.read_gray(vdir / "video.mp4"), 1.5, 4.0, 6)
        baseline_cv.track(frames, 10.0)
        write_json(cv_path, {"source": str(vdir / "video.mp4"), "method": "opencv-baseline",
                             "frames": frames})

    raw = OUT / "sam_raw" / name / "circle.jsonl"
    sam_path = OUT / "sam_tracks" / f"{name}.json"
    status["sam_cached"] = raw.exists()
    if raw.exists():
        n_frames = len(json.loads((vdir / "truth.json").read_text(encoding="utf-8"))["frames"])
        frames, _ = parse_file(raw, n_frames)
        write_json(sam_path, {"source": raw.as_posix(), "frames": frames})
        sam = frames
        cv = json.loads(cv_path.read_text(encoding="utf-8"))["frames"]
        m = {"sam": sam_path.as_posix(), "cv": cv_path.as_posix(), **measure(sam, cv, 30.0)}
        write_json(OUT / "measures" / f"{name}.json", m, indent=1)
    return status


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--force", action="store_true", help="recompute the OpenCV tracks too")
    a = ap.parse_args()

    missing = []
    for seed in a.seeds:
        for level in a.levels:
            name = video_name(level, seed)
            s = process(name, level, seed, a.force)
            print(f"{name}: {'generated, ' if s.get('generated') else ''}"
                  f"SAM cache {'found' if s['sam_cached'] else 'MISSING'}")
            if not s["sam_cached"]:
                missing.append(name)

    from zerolabel import benchmark
    benchmark.main_rows(a.levels)
    if missing:
        print(f"\nNo cached SAM response for {len(missing)} video(s): {', '.join(missing)}.")
        print("They are scored with OpenCV only. SAM calls are paid and are never made by this script.")


if __name__ == "__main__":
    main()
