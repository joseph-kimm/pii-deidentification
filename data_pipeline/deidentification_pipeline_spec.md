# Synthetic De-identification Training Data — Build Spec

## Goal

Generate synthetic (original, redacted) text pairs to fine-tune an open-source LLM (e.g., Llama, GPT-OSS) for de-identification of educational transcripts, via seq2seq rewriting (input = text with PII, output = same text with PII replaced by `[REDACTED]`).

Synthetic pairs must not contain any real PII from the source transcripts, while remaining structurally/stylistically similar to real transcript language.

## Inputs

- A directory of plain-text educational transcripts (`original/*.txt`).
- A matching directory of human-redacted versions of the same files (`human_redacted/*.txt`), where every PII span is replaced with the literal token `[REDACTED]`.
- Files are paired 1:1 by filename.
- Sampling of which files / which windows to use is **out of scope for this build** (to be added later) — build the pipeline to operate on a single given "window" (see below) so sampling can be layered on top independently.

## Core unit of work: a "window"

> **Update:** windows are no longer fixed 3-sentence spans. The unit is now a *chunk* from `build_catalog.py` / `sample_chunks.py`: one speaker turn, or a piece of a long turn, of any length. Chunks without `[REDACTED]` are also run (summarize + rewrite only, input = output). Prompts A–C no longer assume a sentence count. The 3-sentence wording below is kept as the original design record. See `generate_sample_overview.txt` for the current workflow.

- A window = 3 sentences (sentence-split on `.`) taken from the same position in an (original, human_redacted) file pair.
- A window is "PII-containing" if `[REDACTED]` appears anywhere in the corresponding human-redacted window text; otherwise it's "clean."
- **This build should only implement the PII-containing window path.** Clean windows are handled separately/trivially (pass-through, input=output) and are not part of this pipeline.

## Pipeline overview

```
(original_window, redacted_window)
        │
        ▼
[1] Align (no LLM) ──► list of slots: {slot_id, real_value, char_span_in_original}
        │
        ▼
[2] Summarize (LLM call #1) ──► one-sentence generic summary of the window
        │
        ▼
[3] Classify type per slot (LLM call #2 per slot) ──► {slot_id: pii_type_description}
        │
        ▼
[4] Rewrite with slots preserved (LLM call #3) ──► rewritten_text containing {SLOT_n} placeholders
        │         │
        │         ▼
        │   structural check: slot ids in rewritten_text == slot ids from [1]/[3]
        │         │ (fail → retry [4], cap attempts, else flag for review)
        │         ▼
        ▼
[5] Fill fake values, all slots in one call (LLM call #4) ──► {slot_id: fake_value}
        │         │
        │         ▼
        │   structural check: keys returned == slot ids expected
        │         │ (fail → retry [5], cap attempts, else flag for review)
        │         ▼
        ▼
[6] Reconstruct (no LLM):
      synthetic_original = rewritten_text with {SLOT_n} → fake_value
      synthetic_redacted = rewritten_text with {SLOT_n} → "[REDACTED]"
        │
        ▼
Write both to output dataset (paired)
```

Key invariant the whole design hinges on: from step [4] onward, redaction location is **never re-detected** — it's carried forward mechanically via the `{SLOT_n}` placeholders, so `synthetic_original` and `synthetic_redacted` are guaranteed aligned by construction. No PII-leak check is needed on the output, because real PII values are never passed into calls [4] or [5].

## Step-by-step detail

### [1] Align (no LLM, pure code)

Input: `original_window: str`, `redacted_window: str` (same sentence range from paired files).

Logic: diff `redacted_window` against `original_window` to find each `[REDACTED]` occurrence's position, and recover the substring in `original_window` that it replaced. Order slots left-to-right as `SLOT_1, SLOT_2, ...`.

**Multi-word PII:** the human `[REDACTED]` tokens are ground truth. Each `[REDACTED]` occurrence is always its own slot, never merged — whether a multi-word value was redacted as one token (`Sarah Johnson` → `[REDACTED]`) or one token per word (`[REDACTED] [REDACTED]`). In the per-word case, Prompt B assigns part-level types (e.g., "student's first name", "student's last name", "school type"), Prompt C keeps parts of the same entity adjacent and in order, and Prompt D generates only the part each type names. Slots are marked for Prompt B from the template (not by string search on the original), so the correct occurrence is marked when a value repeats.

Output: `template: str` (original_window with each real value replaced by `{SLOT_i}`), `slots: dict[str, str]` mapping `slot_id -> real_value`.

### [2] Summarize (LLM call)

One call per window. See **Prompt A** below.

Input to prompt: `original_window` (real text, no masking needed — instruct model not to echo specifics).
Output: `summary: str`.

### [3] Classify type per slot (LLM call, once per slot)

One call **per slot** in the window (not batched — each needs its own marked span). See **Prompt B** below.

Input to prompt: `original_window` with the target slot's real value wrapped in `«...»`, and the real value itself. Other slots in the same window may remain as their real values (context helps disambiguation) — do not additionally mask them for this step.
Output: `pii_type: str` (short free-text phrase, no fixed taxonomy).

Run for every slot → `pii_types: dict[str, str]` mapping `slot_id -> pii_type`.

### [4] Rewrite with slots preserved (LLM call, once per window)

See **Prompt C** below.

Input to prompt: `summary`, `template` (with `{SLOT_n}` placeholders), `pii_types`.
Output: `rewritten_text: str` — new 3-sentence text, same placeholders present, no real or fake values inserted yet.

**Validation (required, not optional):** extract all `{SLOT_n}` tokens from `rewritten_text` via regex; the multiset of ids must exactly equal `slots.keys()`. On failure: retry the LLM call (up to `MAX_RETRIES`, e.g. 3), then log/flag the window for manual review and skip it — do not attempt to auto-repair the text.

### [5] Fill fake values (LLM call, once per window, all slots batched)

See **Prompt D** below.

Input to prompt: `rewritten_text` (all slots still masked as `{SLOT_n}` — never the real values), `pii_types`.
Output: JSON object `{slot_id: fake_value, ...}` covering every slot.

**Validation (required):** parsed keys must exactly equal `slots.keys()`. On failure: retry (cap attempts), else flag/skip, same policy as step [4].

### [6] Reconstruct (no LLM, pure code)

```python
synthetic_original = rewritten_text
synthetic_redacted = rewritten_text
for slot_id, fake_value in fake_values.items():
    synthetic_original = synthetic_original.replace(f"{{{slot_id}}}", fake_value)
    synthetic_redacted = synthetic_redacted.replace(f"{{{slot_id}}}", "[REDACTED]")
```

Write `(synthetic_original, synthetic_redacted)` as one training pair to the output dataset.

---

## Prompts

All LLM calls should request JSON-only output and be parsed defensively (strip code fences if present, handle parse failure as a retryable error like a structural-check failure).

### Prompt A — Summarize

**System:**
```
You will be given a short passage from an educational document (e.g., a transcript, report card, or enrollment record). Write a one-sentence, generic summary of what the passage is about — describe its topic and structure, not its specific details. Do not repeat any names, dates, ID numbers, or other specific values from the passage. Refer to people only by their role (e.g., "the student," "a parent," "a teacher").

Respond with JSON only, no other text:
{"summary": "<one sentence>"}
```

**User:**
```
Passage:
"""
{original_window}
"""
```

### Prompt B — Classify PII type (per slot)

**System:**
```
You will see a short passage with one specific piece of information marked using «double angle brackets». That marked text has already been identified as personally identifiable information (PII). Your only task is to say what type of PII it is, using the surrounding context.

Respond with a short phrase (2-4 words) naming the category, such as "student's full name," "date of birth," "parent's name," "home address," "student ID number," "phone number," "school name," "teacher's name," "email address," or another category if none of these fit.

Respond with JSON only, no other text:
{"pii_type": "<short phrase>"}
```

**User:**
```
Passage (3-sentence window):
"""
{original_window_with_target_span_marked}
"""

The marked text is: «{real_value}»

What type of PII is the marked text, in this context?
```

### Prompt C — Rewrite with slots preserved

**System:**
```
You will be given:
- A one-sentence summary describing what a passage is about.
- A template of the original passage, where every piece of personally identifiable information (PII) has been replaced with a numbered placeholder like {SLOT_1}, {SLOT_2}, etc.
- A description of what type of PII belongs in each placeholder.

Rewrite the passage as 3 new sentences that plausibly match the summary, in natural, fluent English appropriate for an educational document (e.g., a transcript, report card, or enrollment record).

Rules:
- You MUST reuse every placeholder exactly as given (e.g., {SLOT_1}), the same number of times as in the template, and nowhere else. Do not invent new placeholders. Do not remove any.
- Do NOT insert any actual names, dates, numbers, addresses, or other specific values in place of a placeholder — leave every placeholder exactly as text like {SLOT_1}.
- Each placeholder should appear in a grammatically natural position consistent with its PII type.
- Vary the sentence structure and wording from a typical template; use the summary for content, not the original wording.
- Output exactly 3 sentences.

Respond with JSON only, no other text:
{"rewritten_text": "<3 sentences with placeholders intact>"}
```

**User:**
```
Summary:
{summary}

Template (placeholders shown as {{SLOT_n}}):
"""
{template}
"""

Placeholder types:
{{"SLOT_1": "{pii_type_1}", "SLOT_2": "{pii_type_2}", ...}}
```

### Prompt D — Fill fake values (all slots)

**System:**
```
You will be given a passage with one or more placeholders like {SLOT_1}, {SLOT_2}, etc., and a description of what type of information belongs in each placeholder. Generate a realistic, completely fictional replacement value for every placeholder listed. Each value must be plausible in context but must not be a real, identifiable person, place, or record — invent all of them.

Respond with JSON only, no other text, with one entry per placeholder:
{"SLOT_1": "<value>", "SLOT_2": "<value>", ...}
```

**User:**
```
Passage (all placeholders still masked):
"""
{rewritten_text}
"""

Placeholder types:
{{"SLOT_1": "{pii_type_1}", "SLOT_2": "{pii_type_2}", ...}}

Generate a fake value for every placeholder listed above.
```

---

## Suggested code structure

```
pipeline/
  align.py        # step [1]: diff original/redacted -> template + slots
  llm_client.py    # thin wrapper around the LLM API (JSON-mode calls, retry helper)
  prompts.py       # Prompt A-D as templates/functions
  stages.py        # summarize(), classify_types(), rewrite(), fill_values()
  reconstruct.py   # step [6]
  run_window.py    # orchestrates one window end-to-end, returns (synth_original, synth_redacted) or None (flagged)
  main.py          # iterates over provided windows (sampling is external/TBD), writes output dataset
```

Suggested output format: JSONL, one training pair per line:
```json
{"input": "<synthetic_original>", "output": "<synthetic_redacted>", "source_file": "...", "window_index": ..., "num_slots": ...}
```

## Config / constants to expose

- `MAX_RETRIES` per validated LLM call (default 3)
- LLM model name/endpoint (should be swappable — dev likely wants to test against a hosted model before switching to the fine-tuning target model for these synthetic-generation calls)
- Path to flagged/failed windows log (so discarded windows are inspectable, not silently dropped)

## Explicitly out of scope for this build (deferred, do not design around)

- Sampling strategy: which files, which windows, PII vs. clean window ratio.
- Merge-vs-split logic for adjacent `[REDACTED]` spans (see note in step [1]).
- Any evaluation of the fine-tuned model.
- Fixed PII type taxonomy (type labels are free text by design, used only to condition generation).
- Leak-checking synthetic output against real values (not needed — real values never enter calls [4]/[5]).
