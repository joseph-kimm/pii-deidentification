"""
muse_sampler.py - Call Muse Glimmer 30B through the Fireworks serverless training sampler.

Serves both models of the base-vs-fine-tuned comparison without a deployment:
  * base model:  MODEL = "accounts/fireworks/models/muse-glimmer-30b"
  * fine-tuned:  MODEL = a sampler checkpoint path "<account>/<run-id>/<checkpoint>"
                 (the path save_weights_for_sampler returned in finetune/serverless_sft.py)

Billed per token (prefill + sampled tokens); nothing runs between calls.

The sampler is not a chat endpoint: prompts are rendered locally with the same
MuseGlimmerRenderer (and reasoning strength) used in training, and the reply is
parsed back into its visible text, dropping the model's reasoning.

Needs the training SDK (`.venv`, from `fireworks-training-cookbook`) and FIREWORKS_API_KEY.

CLI (one request, for smoke tests):
    .venv/bin/python deidentify/muse_sampler.py --model <model> "Speaker 1: Hi Maria"
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from functools import lru_cache

SERVERLESS_URL = "https://api.fireworks.ai/training/v1/serverless"
TOKENIZER_MODEL = "meta-models/Muse-Glimmer-30B"  # must match finetune/serverless_sft.py
MAX_TOKENS = int(os.getenv("SAMPLER_MAX_TOKENS", "4096"))
# The SDK default (1.0). Other API models in the comparison are called without a
# temperature, i.e. at their provider's default.
TEMPERATURE = float(os.getenv("SAMPLER_TEMPERATURE", "1.0"))
MAX_ATTEMPTS = int(os.getenv("SAMPLER_MAX_ATTEMPTS", "4"))


class SamplerError(RuntimeError):
    """The sampler returned no usable answer (truncated, malformed, or empty)."""


def is_base_model(model: str) -> bool:
    """Base models are 'accounts/...' names; checkpoints are '<account>/<run-id>/<name>'."""
    return model.startswith("accounts/")


@lru_cache(maxsize=None)
def _setup(model: str, reasoning_strength: str):
    from fireworks.training.sdk import FiretitanServiceClient
    from training.renderer.muse_glimmer import MuseGlimmerOptions, MuseGlimmerRenderer
    from training.renderer.tokenizer import get_tokenizer

    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise SamplerError("FIREWORKS_API_KEY is not set")
    tokenizer = get_tokenizer(TOKENIZER_MODEL)
    renderer = MuseGlimmerRenderer(tokenizer, options=MuseGlimmerOptions(reasoning_strength=reasoning_strength))
    service = FiretitanServiceClient(api_key=api_key, base_url=SERVERLESS_URL)
    if is_base_model(model):
        sampler = service.create_sampling_client(base_model=model, tokenizer=tokenizer)
    else:
        sampler = service.create_sampling_client(model_path=model, tokenizer=tokenizer)
    return renderer, sampler


def visible_text(message) -> str:
    """The user-facing answer of a parsed assistant message (reasoning removed)."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content or [] if part.get("type") == "text")


def _sample_once(messages: list[dict], model: str, reasoning_strength: str) -> str:
    from fireworks.training.sdk import types
    from training.renderer import ParseTermination

    renderer, sampler = _setup(model, reasoning_strength)
    prompt = renderer.build_generation_prompt(messages)
    params = types.SamplingParams(
        max_tokens=MAX_TOKENS, temperature=TEMPERATURE, stop=renderer.get_stop_sequences()
    )
    seq = sampler.sample(prompt=prompt, num_samples=1, sampling_params=params).result().sequences[0]
    if seq.stop_reason == "length":
        raise SamplerError(f"hit max_tokens={MAX_TOKENS} before finishing")
    message, termination = renderer.parse_response(list(seq.tokens))
    text = visible_text(message)
    # The stop token may or may not be echoed back; a clean "stop" is enough.
    if termination is ParseTermination.MALFORMED and seq.stop_reason != "stop":
        raise SamplerError(f"malformed response (stop_reason={seq.stop_reason})")
    if not text.strip():
        raise SamplerError("empty visible answer")
    return text


def chat(messages: list[dict], model: str, reasoning_strength: str = "low") -> str:
    """One chat turn, retried with backoff on any error (network, capacity, bad parse)."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            return _sample_once(messages, model, reasoning_strength)
        except Exception as e:  # SDK raises several error types; all are retried, then re-raised
            if attempt == MAX_ATTEMPTS - 1:
                raise
            delay = 2.0 * 2 ** attempt
            print(f"[SAMPLER RETRY {attempt + 1}] {type(e).__name__}: {str(e)[:200]}; waiting {delay:.0f}s")
            time.sleep(delay)
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", help="user message")
    parser.add_argument("--model", required=True)
    parser.add_argument("--system", default="Replace person names with [REDACTED]. Return only the edited text.")
    parser.add_argument("--reasoning", default="low")
    args = parser.parse_args()
    messages = [{"role": "system", "content": args.system}, {"role": "user", "content": args.text}]
    start = time.time()
    try:
        answer = chat(messages, args.model, args.reasoning)
        print(f"OK   {args.model} ({time.time() - start:.1f}s): {answer!r}")
    except Exception as e:
        print(f"FAIL {args.model}: {type(e).__name__}: {str(e)[:300]}")
        return 1
    return 0


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    sys.exit(main())
