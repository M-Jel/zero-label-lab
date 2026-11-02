"""Convertit la reponse brute de SAM 3.1 (jsonl) en detections par image.
SAM n'envoie rien pour les images sans detection : on les complete par des listes vides."""
import argparse
import json
import re
from pathlib import Path

FRAME_RE = re.compile(r"^<(\d+)f>")
BOX_RE = re.compile(r"(?<=[>,])(\d+)<\|box;([^|>]*)\|>")


def _det(m: re.Match) -> dict:
    kv = dict(p.split("=", 1) for p in m.group(2).split(";") if "=" in p)
    x1, y1, x2, y2 = (float(kv[k]) for k in ("x1", "y1", "x2", "y2"))
    return {
        "id": int(m.group(1)),
        "x": (x1 + x2) / 2,
        "y": (y1 + y2) / 2,
        "r": ((x2 - x1) + (y2 - y1)) / 4,
        "bbox": [x1, y1, x2, y2],
        "conf": float(kv.get("c", "nan")),
    }


def parse_file(path: Path, n_expected: int = 0) -> tuple[list[list[dict]], int]:
    frames: dict[int, list[dict]] = {}
    for line in open(path, encoding="utf-8"):
        e = json.loads(line)
        if e.get("type") != "response.output_text.delta":
            continue
        delta = e["delta"]
        m = FRAME_RE.match(delta)
        if not m:
            raise ValueError(f"delta sans numero d'image : {delta[:40]!r}")
        frames[int(m.group(1))] = [_det(b) for b in BOX_RE.finditer(delta)]
    n = max(max(frames) + 1 if frames else 0, n_expected)
    return [frames.get(i, []) for i in range(n)], len(frames)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=150, help="nombre d'images de la video")
    a = ap.parse_args()

    frames, n_received = parse_file(Path(a.raw), a.frames)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source": a.raw, "frames": frames}), encoding="utf-8")

    ids = {d["id"] for f in frames for d in f}
    confs: dict[int, set] = {}
    for f in frames:
        for d in f:
            confs.setdefault(d["id"], set()).add(round(d["conf"], 6))
    const = all(len(v) == 1 for v in confs.values())
    print(f"{len(frames)} images ({n_received} avec detections) | debut: {len(frames[0])} "
          f"| fin: {len(frames[-1])} | ids distincts: {len(ids)} | confiance constante: {const}")


if __name__ == "__main__":
    main()
