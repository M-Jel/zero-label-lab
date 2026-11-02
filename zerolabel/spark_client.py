"""Appel a Muse Spark via la Meta Model API (Responses, streaming)."""
import argparse
import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

API = "https://api.meta.ai/v1/responses"
MODEL = "muse-spark-1.3"


def ask(prompt: str, system: str | None = None, effort: str = "low", max_tokens: int = 4000,
        temperature: float = 0.0, model: str = MODEL,
        raw_path: str = "outputs/spark_raw/last.jsonl") -> dict:
    """Renvoie {"text", "usage", "complete"}. Garde les evenements bruts pour le debogage."""
    load_dotenv()
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        raise SystemExit("MODEL_API_KEY absente : verifie le fichier .env")
    payload = {
        "model": model,
        "input": [{"type": "message", "role": "user",
                   "content": [{"type": "input_text", "text": prompt}]}],
        "stream": True,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "reasoning": {"effort": effort},
    }
    if system:
        payload["instructions"] = system
    raw = Path(raw_path)
    raw.parent.mkdir(parents=True, exist_ok=True)
    parts, usage, complete = [], None, False
    with requests.post(API, headers={"Authorization": f"Bearer {key}",
                                     "Content-Type": "application/json",
                                     "Accept": "text/event-stream"},
                       json=payload, stream=True, timeout=600) as r, \
            open(raw, "w", encoding="utf-8") as out:
        if r.status_code >= 400:
            raise SystemExit(f"Erreur {r.status_code} : {r.text[:800]}")
        try:
            for line in r.iter_lines():
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].decode("utf-8", errors="replace")
                if data == "[DONE]":
                    break
                out.write(data + "\n")
                try:
                    e = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if e.get("type") == "response.output_text.delta":
                    parts.append(e.get("delta", ""))
                elif e.get("type") == "response.completed":
                    complete = True
                    usage = e.get("response", {}).get("usage")
        except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError) as err:
            print(f"connexion coupee ({err.__class__.__name__})")
    return {"text": "".join(parts), "usage": usage, "complete": complete}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--system")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--max-tokens", type=int, default=2000)
    a = ap.parse_args()
    res = ask(a.prompt, a.system, a.effort, a.max_tokens)
    print(res["text"])
    print("---")
    print("complet:", res["complete"], "| usage:", json.dumps(res["usage"]))


if __name__ == "__main__":
    main()
