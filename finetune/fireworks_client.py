"""Call the fine-tuned Muse Glimmer 30B deployment on Fireworks.

The deployment scales to zero after 5 idle minutes, so the first request after
a quiet period gets 503 while a replica boots. This client retries 503 (and
other transient errors) with exponential backoff until MAX_WAIT_S elapses.

Usage:
    export FIREWORKS_API_KEY=...          # in your shell, never committed
    python finetune/fireworks_client.py "Text to de-identify ..."
    python finetune/fireworks_client.py --file some_transcript.txt

Model: FIREWORKS_MODEL, or the promoted fine-tune in FIREWORKS_ACCOUNT_ID's account.
Other env overrides: FIREWORKS_SYSTEM_PROMPT, MAX_WAIT_S.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import requests

API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
FINE_TUNED_MODEL_ID = "deid-glimmer-30b-sft-synthetic"  # --output-model used when promoting
TRAIN_FILE = "data/finetune/training/sft_train.jsonl"  # system prompt source when --system is not given
RETRY_STATUSES =frozenset({429, 500, 502, 503, 504})
MAX_WAIT_S = float(os.environ.get("MAX_WAIT_S", "900"))  # cold start of a 30B model can take minutes
BASE_DELAY_S = 2.0
MAX_DELAY_S = 30.0


class FireworksError(RuntimeError):
    """Raised when the API returns a non-retryable error or retries run out."""


def _backoff_delay(attempt: int, retry_after: str | None) -> float:
    if retry_after and retry_after.isdigit():
        return min(float(retry_after), MAX_DELAY_S)
    delay = min(BASE_DELAY_S * (2 ** attempt), MAX_DELAY_S)
    return delay * random.uniform(0.8, 1.2)


def chat(
    messages: list[dict],
    *,
    model: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    max_wait_s: float = MAX_WAIT_S,
    session: requests.Session | None = None,
) -> str:
    """Send a chat completion request, retrying while the deployment wakes up."""
    key = api_key or os.environ.get("FIREWORKS_API_KEY")
    if not key:
        raise FireworksError("FIREWORKS_API_KEY is not set")
    model = model or os.environ.get("FIREWORKS_MODEL") or default_model()
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Must match the reasoning strength used in training (finetune/serverless_sft.py).
        "reasoning_effort": os.environ.get("REASONING_EFFORT", "low"),
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    http = session or requests.Session()
    deadline = time.monotonic() + max_wait_s
    attempt = 0

    while True:
        try:
            resp = http.post(API_URL, json=payload, headers=headers, timeout=300)
        except (requests.ConnectionError, requests.Timeout) as exc:
            status, detail, retry_after = None, str(exc), None
        else:
            if resp.ok:
                return resp.json()["choices"][0]["message"]["content"]
            status, detail = resp.status_code, resp.text[:500]
            retry_after = resp.headers.get("Retry-After")
            if status not in RETRY_STATUSES:
                raise FireworksError(f"HTTP {status}: {detail}")

        delay = _backoff_delay(attempt, retry_after)
        if time.monotonic() + delay > deadline:
            raise FireworksError(f"gave up after {attempt + 1} attempts; last error {status}: {detail}")
        print(f"[retry {attempt + 1}] {status or 'network error'}; waiting {delay:.1f}s "
              "(deployment may be scaling up from zero)", file=sys.stderr)
        time.sleep(delay)
        attempt += 1


def default_model() -> str:
    account = os.environ.get("FIREWORKS_ACCOUNT_ID", "").strip()
    if not account:
        raise FireworksError("set FIREWORKS_MODEL, or FIREWORKS_ACCOUNT_ID for the promoted fine-tune")
    return f"accounts/{account}/models/{FINE_TUNED_MODEL_ID}"


def training_system_prompt(path: str = TRAIN_FILE) -> str | None:
    """Reuse the exact system prompt the model was trained on (quotes included)."""
    try:
        with open(path, encoding="utf-8") as f:
            first = json.loads(f.readline())["messages"][0]
    except (OSError, ValueError, KeyError, IndexError):
        return None
    return first["content"] if first.get("role") == "system" else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", nargs="?", help="user message text")
    parser.add_argument("--file", help="read the user message from this file")
    parser.add_argument("--system", default=os.environ.get("FIREWORKS_SYSTEM_PROMPT"),
                        help="system prompt (use the same one as in training)")
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            user_text = f.read()
    elif args.text:
        user_text = args.text
    else:
        parser.error("give text or --file")

    system = args.system or training_system_prompt()
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": user_text})
    try:
        print(chat(messages))
    except FireworksError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
