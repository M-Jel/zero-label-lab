"""Appel a SAM 3.1 via la Meta Model API. Sauvegarde la reponse brute (une ligne JSON par evenement)."""
import argparse
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

API = "https://api.meta.ai/v1"


def upload(video_path: Path, key: str) -> str:
    with open(video_path, "rb") as f:
        r = requests.post(
            f"{API}/files",
            headers={"Authorization": f"Bearer {key}"},
            data={"purpose": "user_data"},
            files={"file": (video_path.name, f, "video/mp4")},
            timeout=300,
        )
    r.raise_for_status()
    file_id = r.json().get("id")
    if not isinstance(file_id, str) or not file_id:
        raise RuntimeError("Upload termine sans id de fichier.")
    return file_id


def segment(file_id: str, prompt: str, key: str, out_path: Path) -> int:
    payload = {
        "model": "sam-3.1",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_video", "file_id": file_id},
            ],
        }],
        "stream": True,
        "metadata": {"mask_encoding": "lossless", "include_confidence": "true"},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with requests.post(
        f"{API}/responses",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        json=payload,
        stream=True,
        timeout=600,
    ) as r, open(out_path, "w", encoding="utf-8") as out:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b"data: "):
                continue
            data = line[6:].decode("utf-8", errors="replace")
            if data == "[DONE]":
                break
            out.write(data + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--prompt", default="circle")
    ap.add_argument("--out", default="outputs/sam_raw")
    args = ap.parse_args()

    load_dotenv()
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        raise SystemExit("MODEL_API_KEY absente : verifie le fichier .env")

    file_id = upload(Path(args.video), key)
    out_path = Path(args.out) / f"{args.prompt.replace(' ', '_')}.jsonl"
    n = segment(file_id, args.prompt, key, out_path)
    print(f"{n} evenements enregistres dans {out_path}")


if __name__ == "__main__":
    main()
