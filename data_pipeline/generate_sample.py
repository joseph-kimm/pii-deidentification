"""Synthetic de-identification training-pair generator (the pipeline logic).

=============================================================================
READ ME FIRST (remote desktop / Windows)
=============================================================================
What this file is:
  The step-by-step pipeline that turns ONE real chunk (original text +
  human-redacted text) into ONE fake training pair. You never run this
  file directly -- run_generation.py imports it and feeds it every chunk.
  Full design: deidentification_pipeline_spec.md.

What it depends on:
  - prompts.py (same folder): the instructions sent to the model.
  - An LLM client handed in through configure(); run_generation.py passes
    an LMStudioClient (lmstudio_client.py) pointed at LM Studio.
  - Python 3.10 or newer (uses "str | None" type hints). Standard library
    only, nothing to pip install.

Inputs (per chunk, given by run_generation.py):
  original_window  the real text of one chunk (any length).
  redacted_window  the SAME chunk from the human-redacted file, where each
                   piece of PII is the literal token [REDACTED]. Apart from
                   [REDACTED] tokens the two texts must be identical,
                   otherwise the chunk is skipped (logged in failures.jsonl).

Steps (each LLM call is one request to LM Studio):
  1. extract_pii       find what each [REDACTED] hid -> {SLOT_n}  (no LLM)
  2. summarize         generic summary of the chunk               (1 call)
  3. classify_types    PII type of each slot                      (1 call per slot)
  4. rewrite_with_slots new wording, {SLOT_n} kept in place       (1 call, retried)
  5. fill_fake_values  invent a fake value per slot               (1 call, retried)
  6. reconstruct       fake text + [REDACTED] text                (no LLM)
  Chunks with no [REDACTED] skip steps 3 and 5; their fake pair has
  input == output (teaches the model to leave clean text alone).

Model notes:
  A chunk with N redactions costs about N + 3 model calls (more on
  retries), so a slow local model x a large sample can take hours.
  Real PII values are only sent to steps 2 and 3, and only to the local
  LM Studio server -- never to the internet.
=============================================================================
"""

import difflib
import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import (  # noqa: E402
    PROMPT_A_SYSTEM,
    PROMPT_B_SYSTEM,
    PROMPT_C_SYSTEM,
    PROMPT_D_SYSTEM,
    build_prompt_a_user,
    build_prompt_b_user,
    build_prompt_c_user,
    build_prompt_d_user,
)

REDACTED_TOKEN = "[REDACTED]"

_client = None
_model = None
_default_max_retries = 3


def configure(client, model: str, max_retries: int = 3) -> None:
    """Wire up the LLM client this module's calls will use.

    client: any object with an openai-style
      client.chat.completions.create(model=..., messages=[...]) method,
      e.g. lmstudio_client.LMStudioClient().
    model: the model id to pass to client.chat.completions.create().
    max_retries: default retry count for validated LLM calls; can still be
      overridden per-call.

    Must be called once before running the pipeline.
    """
    global _client, _model, _default_max_retries
    _client = client
    _model = model
    _default_max_retries = max_retries


class LLMCallError(Exception):
    """Raised when an LLM call fails to produce valid/well-formed output
    after max_retries attempts."""


# --- LLM client helper --------------------------------------------------

def call_llm_json(system_prompt: str, user_prompt: str, max_retries: int | None = None) -> dict:
    """Call the configured LLM with a system+user prompt, expecting a
    JSON-only response, and return the parsed dict. Retries on request
    failure or JSON parse failure up to max_retries; raises LLMCallError
    on exhaustion.

    ConnectionError (server not reachable) is NOT retried: it propagates so
    the whole run stops instead of marking every remaining chunk as failed.
    """
    if _client is None:
        raise RuntimeError(
            "generate_sample.configure(client, model) must be called before "
            "running the pipeline."
        )
    retries = _default_max_retries if max_retries is None else max_retries
    last_error: Exception | None = None

    for _attempt in range(retries):
        try:
            response = _client.chat.completions.create(
                model=_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = (response.choices[0].message.content or "").strip()
            result = _parse_json_response(content)
            if not isinstance(result, dict):
                raise ValueError(f"expected a JSON object, got {type(result).__name__}")
            return result
        except ConnectionError:
            raise
        except Exception as e:  # noqa: BLE001 - defensive: any other failure is retryable
            last_error = e
            continue

    raise LLMCallError(
        f"LLM call failed after {retries} attempts: {last_error}"
    )


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _parse_json_response(content: str):
    """Parse a model reply as JSON, tolerating what local models often add:
    <think>...</think> reasoning blocks, markdown code fences, and short
    text before/after the JSON object. Raises on failure so the retry loop
    in call_llm_json can catch it."""
    text = _THINK_RE.sub("", content).strip()
    if text.startswith("```"):
        text = text.strip("`")
        # Drop an optional leading language tag (e.g. "json\n...").
        if "\n" in text:
            first_line, rest = text.split("\n", 1)
            if first_line.strip().lower() in ("json", ""):
                text = rest
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start:end + 1])


# --- Step [1]: Extract PII (no LLM) ------------------------------------------------

def extract_pii(original_window: str, redacted_window: str) -> tuple[str, dict[str, str]]:
    """Diff redacted_window against original_window to recover, for each
    [REDACTED] occurrence, the original substring it replaced.

    Returns (template, slots):
      template: original_window with each real value replaced by {SLOT_i}
      slots: slot_id -> real_value, ordered left-to-right as SLOT_1, SLOT_2, ...
    A chunk with no [REDACTED] returns (original_window, {}).
    """
    # Every [REDACTED] occurrence is its own slot, even when adjacent to
    # another (e.g. "[REDACTED] [REDACTED]" for a first + last name) — the
    # human redaction is ground truth. Step [3] assigns part-level types
    # (e.g. "first name") so the parts stay meaningful.
    #
    # Words are aligned with difflib (as build_catalog.py does), so a value
    # containing the text that follows it ("Dr. Evans." -> "[REDACTED].")
    # is still recovered whole. Each differing block is then resolved at
    # character level to separate the value from attached punctuation.
    orig_words = list(re.finditer(r"\S+", original_window))
    red_words = list(re.finditer(r"\S+", redacted_window))
    matcher = difflib.SequenceMatcher(
        a=[m.group() for m in orig_words], b=[m.group() for m in red_words], autojunk=False
    )

    spans: list[tuple[int, int]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        red_block = redacted_window[red_words[j1].start():red_words[j2 - 1].end()] if j2 > j1 else ""
        if REDACTED_TOKEN not in red_block or i2 == i1:
            raise ValueError(
                "original_window and redacted_window diverge outside of "
                "[REDACTED] spans; cannot align."
            )
        start = orig_words[i1].start()
        orig_block = original_window[start:orig_words[i2 - 1].end()]
        spans.extend((start + s, start + e) for s, e in _match_block(orig_block, red_block))

    slots: dict[str, str] = {}
    template_parts: list[str] = []
    pos = 0
    for n, (s, e) in enumerate(spans, start=1):
        slot_id = f"SLOT_{n}"
        template_parts.append(original_window[pos:s])
        template_parts.append(f"{{{slot_id}}}")
        slots[slot_id] = original_window[s:e]
        pos = e
    template_parts.append(original_window[pos:])

    template = "".join(template_parts)
    return template, slots


def _match_block(orig_block: str, red_block: str) -> list[tuple[int, int]]:
    """Character spans in orig_block hidden by each [REDACTED] in red_block.

    The text around the [REDACTED] tokens must appear in orig_block exactly;
    each value is the text in between (matched against the whole block, so
    "Dr. Evans." vs "[REDACTED]." gives "Dr. Evans").
    """
    parts = red_block.split(REDACTED_TOKEN)
    if any(not p for p in parts[1:-1]):
        # Two [REDACTED] tokens with nothing between them — cannot
        # disambiguate where one real value ends and the next begins.
        raise ValueError(
            "Ambiguous [REDACTED] span with no following literal text; "
            "cannot determine real value boundary."
        )
    pattern = "(.+?)".join(re.escape(p) for p in parts)
    match = re.fullmatch(pattern, orig_block, re.DOTALL)
    if match is None:
        raise ValueError(
            f"original_window and redacted_window diverge outside of "
            f"[REDACTED] spans; cannot align: {orig_block!r} vs {red_block!r}"
        )
    spans = [match.span(g) for g in range(1, len(parts))]
    if any(not orig_block[s:e].strip() for s, e in spans):
        raise ValueError("A [REDACTED] span maps to empty text in the original.")
    return spans


# --- Step [2]: Summarize (LLM call) -----------------------------------------

def summarize(original_window: str) -> str:
    result = call_llm_json(PROMPT_A_SYSTEM, build_prompt_a_user(original_window))
    summary = result.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise LLMCallError("summarize: response has no non-empty 'summary' string")
    return summary


# --- Step [3]: Classify type per slot (LLM call, once per slot) ------------

def mark_slot(template: str, slots: dict[str, str], target_slot_id: str) -> str:
    """Rebuild the original window from template, wrapping only the target
    slot's value in «...». Built from the template (not str.replace on the
    original) so the correct occurrence is marked even when the same value
    appears more than once."""
    marked = template
    for slot_id, real_value in slots.items():
        value = f"«{real_value}»" if slot_id == target_slot_id else real_value
        marked = marked.replace(f"{{{slot_id}}}", value)
    return marked


def classify_slot_type(template: str, slots: dict[str, str], slot_id: str) -> str:
    marked_window = mark_slot(template, slots, slot_id)
    result = call_llm_json(
        PROMPT_B_SYSTEM, build_prompt_b_user(marked_window, slots[slot_id])
    )
    pii_type = result.get("pii_type")
    if not isinstance(pii_type, str) or not pii_type.strip():
        raise LLMCallError(f"classify_slot_type: no 'pii_type' string for {slot_id}")
    return pii_type


def classify_types(template: str, slots: dict[str, str]) -> dict[str, str]:
    return {
        slot_id: classify_slot_type(template, slots, slot_id)
        for slot_id in slots
    }


# --- Step [4]: Rewrite with slots preserved (LLM call) ----------------------

# Speaker labels ("Coach:", "Teacher:", "Speaker A:", "Speaker 1:") are kept
# out of the LLM entirely: the template is split into turns at each label,
# only the turn text is rewritten, and the labels are reattached verbatim.
# The "Speaker" form matches build_catalog.py's SPEAKER_RE.
_SPEAKER_LABEL_RE = re.compile(
    r"(?<![A-Za-z])(?:Speaker\s*[A-Za-z0-9]{0,3}|Coach|Teacher)\s*:\s*",
    re.IGNORECASE,
)
_SLOT_RE = re.compile(r"\{(SLOT_\d+)\}")


def split_turns(template: str) -> tuple[list[str], list[str]]:
    """Split template into (labels, turns) so that
    join_turns(labels, turns) == template.

    Each label carries the whitespace preceding it plus the label text
    itself (e.g. " Teacher: "); turns hold the spoken text with trailing
    whitespace moved into the next label. Text before the first label is
    turn 0 with label "". A window with no labels is a single turn.
    """
    matches = list(_SPEAKER_LABEL_RE.finditer(template))
    if not matches:
        return [""], [template]

    labels: list[str] = []
    turns: list[str] = []

    lead = template[: matches[0].start()]
    pending_ws = ""
    if lead.strip():
        stripped = lead.rstrip()
        labels.append("")
        turns.append(stripped)
        pending_ws = lead[len(stripped):]
    else:
        pending_ws = lead

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(template)
        body = template[m.end():end]
        labels.append(pending_ws + m.group(0))
        if i + 1 < len(matches):
            stripped = body.rstrip()
            turns.append(stripped)
            pending_ws = body[len(stripped):]
        else:
            turns.append(body)

    return labels, turns


def join_turns(labels: list[str], turns: list[str]) -> str:
    return "".join(label + turn for label, turn in zip(labels, turns))


def rewrite_with_slots(
    summary: str,
    template: str,
    pii_types: dict[str, str],
    slot_ids: list[str],
    max_retries: int | None = None,
) -> str:
    retries = _default_max_retries if max_retries is None else max_retries
    labels, turns = split_turns(template)

    # Only non-empty turns go to the LLM (e.g. a label at the very end of the
    # window has no text after it); empty turns are passed through as-is.
    active = [i for i, t in enumerate(turns) if t.strip()]
    active_turns = [turns[i].strip() for i in active]
    expected = [sorted(_SLOT_RE.findall(t)) for t in active_turns]

    for _attempt in range(retries):
        result = call_llm_json(
            PROMPT_C_SYSTEM, build_prompt_c_user(summary, active_turns, pii_types)
        )
        rewritten = result.get("turns")
        if not isinstance(rewritten, list) or len(rewritten) != len(active_turns):
            continue
        if not all(isinstance(t, str) and t.strip() for t in rewritten):
            continue
        found = [sorted(_SLOT_RE.findall(t)) for t in rewritten]
        if found != expected:
            continue

        new_turns = list(turns)
        for i, text in zip(active, rewritten):
            new_turns[i] = text.strip()
        return join_turns(labels, new_turns)

    raise LLMCallError(
        f"rewrite_with_slots: turn count or per-turn slot ids did not match "
        f"after {retries} attempts (expected {len(active_turns)} turns with "
        f"slots {expected})"
    )


# --- Step [5]: Fill fake values (LLM call, all slots batched) --------------

def fill_fake_values(
    rewritten_text: str,
    pii_types: dict[str, str],
    slot_ids: list[str],
    max_retries: int | None = None,
) -> dict[str, str]:
    retries = _default_max_retries if max_retries is None else max_retries
    expected = set(slot_ids)

    for _attempt in range(retries):
        result = call_llm_json(
            PROMPT_D_SYSTEM, build_prompt_d_user(rewritten_text, pii_types)
        )
        if set(result.keys()) != expected:
            continue
        # Models sometimes return numbers for ages/years (e.g. 16, not "16").
        values = {
            k: str(v).strip() if isinstance(v, (str, int, float)) else ""
            for k, v in result.items()
        }
        if all(values.values()):
            return values

    raise LLMCallError(
        f"fill_fake_values: returned keys did not match after {retries} "
        f"attempts (expected {expected})"
    )


# --- Step [6]: Reconstruct (no LLM) -----------------------------------------

def reconstruct(rewritten_text: str, fake_values: dict[str, str]) -> tuple[str, str]:
    synthetic_original = rewritten_text
    synthetic_redacted = rewritten_text
    for slot_id, fake_value in fake_values.items():
        synthetic_original = synthetic_original.replace(f"{{{slot_id}}}", fake_value)
        synthetic_redacted = synthetic_redacted.replace(f"{{{slot_id}}}", REDACTED_TOKEN)
    return synthetic_original, synthetic_redacted


# --- Orchestration (one chunk) -------------------------------------------

def run_chunk(
    original_window: str,
    redacted_window: str,
    source_file: str,
    chunk_id: str,
    log: list[str] | None = None,
) -> dict:
    """Run the full pipeline on one (original, redacted) chunk and return a
    training-pair record.

    Raises ValueError if the two texts cannot be aligned (step 1) and
    LLMCallError if a model call keeps failing; the caller logs those and
    moves on. If `log` is a list, every step's output is appended to it as
    a readable section (whatever succeeded, even when a later step fails).
    """

    def note(title: str, content: str) -> None:
        if log is not None:
            log.append(f"=== {title} ===\n{content}\n")

    note("Input: original", original_window)
    note("Input: redacted", redacted_window)

    template, slots = extract_pii(original_window, redacted_window)
    slot_ids = list(slots.keys())
    note("Step 1: extract_pii -> template", template)
    note("Step 1: extract_pii -> slots", json.dumps(slots, indent=2, ensure_ascii=False))

    summary = summarize(original_window)
    note("Step 2: summarize -> summary", summary)

    pii_types = classify_types(template, slots)
    note("Step 3: classify_types -> pii_types", json.dumps(pii_types, indent=2, ensure_ascii=False))

    rewritten_text = rewrite_with_slots(summary, template, pii_types, slot_ids)
    note("Step 4: rewrite_with_slots -> rewritten_text", rewritten_text)

    fake_values = fill_fake_values(rewritten_text, pii_types, slot_ids) if slot_ids else {}
    note("Step 5: fill_fake_values -> fake_values", json.dumps(fake_values, indent=2, ensure_ascii=False))

    synthetic_original, synthetic_redacted = reconstruct(rewritten_text, fake_values)
    note("Step 6: reconstruct -> synthetic_original", synthetic_original)
    note("Step 6: reconstruct -> synthetic_redacted", synthetic_redacted)

    return {
        "input": synthetic_original,
        "output": synthetic_redacted,
        "source_file": source_file,
        "chunk_id": chunk_id,
        "num_slots": len(slots),
    }
