"""Call SAM 3.1 through the Meta Model API, retrying when the connection drops.
Saves the raw response (one JSON line per event) to <out>/<prompt>.jsonl.
Note: SAM sends no event for frames without any detection.

WARNING: every call is billed by Meta. Raw responses are cached; never call twice for the same video."""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

API = "https://api.meta.ai/v1"
NET_ERRORS = (requests.exceptions.ChunkedEncodingError,
              requests.exceptions.ConnectionError,
              requests.exceptions.Timeout)


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
        raise RuntimeError("Upload finished without a file id.")
    return file_id


def segment_once(file_id: str, prompt: str, key: str, out_path: Path,
                 mask_encoding: str) -> tuple[int, bool]:
    """One call. Returns (number of frames with detections, response complete?)."""
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
        "metadata": {"mask_encoding": mask_encoding, "include_confidence": "true"},
    }
    frames, complete = 0, False
    with open(out_path, "w", encoding="utf-8") as out:
        try:
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
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line.startswith(b"data: "):
                        continue
                    data = line[6:].decode("utf-8", errors="replace")
                    if data == "[DONE]":
                        complete = True
                        break
                    out.write(data + "\n")
                    out.flush()
                    try:
                        etype = json.loads(data).get("type")
                    except json.JSONDecodeError:
                        continue
                    if etype == "response.output_text.delta":
                        frames += 1
                    elif etype == "response.completed":
                        complete = True
        except NET_ERRORS as e:
            print(f"  connection dropped after {frames} frames ({e.__class__.__name__})")
    return frames, complete


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="video to upload (ignored when --file-id is given)")
    ap.add_argument("--file-id", help="reuse a file already uploaded")
    ap.add_argument("--prompt", default="circle")
    ap.add_argument("--mask-encoding", default="lossless")
    ap.add_argument("--retries", type=int, default=3, help="maximum number of attempts")
    ap.add_argument("--out", default="outputs/sam_raw")
    args = ap.parse_args()

    load_dotenv()
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        raise SystemExit("MODEL_API_KEY is missing: check the .env file")

    if args.file_id:
        file_id = args.file_id
    elif args.video:
        file_id = upload(Path(args.video), key)
    else:
        raise SystemExit("Give --video or --file-id")
    print(f"file_id : {file_id}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"{args.prompt.replace(' ', '_')}.jsonl"
    best_frames, best_tmp, best_complete = -1, None, False
    for attempt in range(1, args.retries + 1):
        tmp = out_dir / f"_attempt{attempt}.jsonl"
        print(f"attempt {attempt}/{args.retries}...")
        frames, complete = segment_once(file_id, args.prompt, key, tmp, args.mask_encoding)
        if frames > best_frames or (complete and frames == best_frames):
            best_frames, best_tmp, best_complete = frames, tmp, complete
        if complete:
            break
        if attempt < args.retries:
            time.sleep(5)

    shutil.copyfile(best_tmp, final)
    for t in out_dir.glob("_attempt*.jsonl"):
        t.unlink()
    status = "complete response" if best_complete else "INCOMPLETE (connection dropped)"
    print(f"{best_frames} frames with detections -> {final} ({status})")


if __name__ == "__main__":
    main()
