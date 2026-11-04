"""Prompt templates for the de-identification synthetic data pipeline.

=============================================================================
READ ME FIRST (remote desktop / Windows)
=============================================================================
What this file is:
  The four instructions ("prompts") that generate_sample.py sends to the
  model running in LM Studio. You never run this file directly.
  Prompts A-D correspond to LLM calls [2]-[5] in
  deidentification_pipeline_spec.md:
    A  summarize the chunk             -> {"summary": "..."}
    B  name the type of one PII value  -> {"pii_type": "..."}
    C  rewrite the chunk, keep {SLOT_n} -> {"turns": ["...", ...]}
    D  invent fake values for slots    -> {"SLOT_1": "...", ...}

What it depends on:
  Nothing (Python standard library only). Imported by generate_sample.py.

About the chunks:
  A chunk is one speaker turn (or one piece of a long turn) from
  sample_chunks.py, so it can be any length: a few words or many
  sentences. The prompts therefore never assume a fixed sentence count;
  the rewrite is asked to keep roughly the original length.

Model notes:
  Every prompt demands "JSON only". Small local models sometimes add text,
  code fences, or <think>...</think> blocks; generate_sample.py strips
  those and retries (MAX_RETRIES in run_generation.py) if the JSON is still
  unreadable. If a model keeps failing (see failures.jsonl), try a larger
  or instruction-tuned model in LM Studio before editing these prompts.
=============================================================================
"""

import json

PROMPT_A_SYSTEM = """You will be given a passage from an educational transcript. It may be only a few words or several sentences long. Write a brief summary of what the passage is about — describe its topic and structure, not its specific details. Use one sentence for a short passage and at most three sentences for a long one. Do not repeat any names, dates, ID numbers, or other specific values from the passage. Refer to people only by their role (e.g., "the student," "a parent," "a teacher").

Respond with JSON only, no other text:
{"summary": "<brief summary>"}"""


def build_prompt_a_user(original_window: str) -> str:
    return f'Passage:\n"""\n{original_window}\n"""'


PROMPT_B_SYSTEM = """You will see a passage with one specific piece of information marked using «double angle brackets». That marked text has already been identified as personally identifiable information (PII). Your only task is to say what type of PII it is, using the surrounding context.

The marked text may be a complete piece of PII (e.g., a full name) or only one part of a longer piece of PII whose other parts are separately marked, such as just the first name or just the last name of a person, or just the name or just the type of a school (e.g., "Riverdale" in "Riverdale High School"). Classify only the marked text itself, and if it is a part, say which part.

Rules for people: the passage may be too short to know a person's role reliably, so do NOT include a role (no "student's," "teacher's," "parent's," etc.). Use only "first name," "last name," or "full name."

Rules for schools: be as specific as the marked text or context allows about the level of the institution — e.g., "elementary school name," "middle school name," "high school name," "university name," "school district name." If the marked text is only part of a school name, say which part with the level included, e.g., "high school name prefix" for "Riverdale" in "Riverdale High School" or "high school type" for "High School". Use plain "school name" only if the level truly cannot be determined.

Respond with a short phrase (2-5 words) naming the category, such as "first name," "last name," "full name," "high school name," "elementary school name," "university name," "city," or another category if none of these fit.

Respond with JSON only, no other text:
{"pii_type": "<short phrase>"}"""


def build_prompt_b_user(original_window_with_target_span_marked: str, real_value: str) -> str:
    return (
        f'Passage:\n"""\n{original_window_with_target_span_marked}\n"""\n\n'
        f"The marked text is: «{real_value}»\n\n"
        "What type of PII is the marked text, in this context?"
    )


PROMPT_C_SYSTEM = """You will be given:
- A brief summary describing what a passage is about.
- The original passage as a JSON list of segments (one segment per speaker turn), where every piece of personally identifiable information (PII) has been replaced with a numbered placeholder like {SLOT_1}, {SLOT_2}, etc. Some passages contain no placeholders at all.
- A description of what type of PII belongs in each placeholder (empty if there are none).

Rewrite each segment with new wording that plausibly matches the summary, in natural, fluent English appropriate for an educational transcript.

Rules:
- Return exactly the same number of segments as given, in the same order. Segment i of your output is the rewrite of segment i of the input.
- Each segment MUST contain exactly the placeholders of its original segment (e.g., {SLOT_1}), the same number of times, and no others. Do not move a placeholder to a different segment. Do not invent new placeholders. Do not remove any. If a segment has no placeholders, its rewrite must have none.
- Do NOT insert any actual names, dates, numbers, addresses, or other specific values in place of a placeholder — leave every placeholder exactly as text like {SLOT_1}.
- Each placeholder should appear in a grammatically natural position consistent with its PII type.
- If placeholders are parts of the same name or entity (e.g., {SLOT_1} {SLOT_2} as a first name followed by a last name), keep them next to each other, in the same order, exactly as in the original.
- Do not add speaker labels (like "Coach:" or "Speaker 1:") — they are handled separately.
- Vary the sentence structure and wording; use the summary for content, not the original wording.
- Length: keep each segment about as long as its original, with about the same number of sentences. A short fragment stays a short fragment; a long segment stays long.

Respond with JSON only, no other text:
{"turns": ["<rewritten segment 1>", "<rewritten segment 2>", ...]}"""


def build_prompt_c_user(summary: str, turns: list[str], pii_types: dict[str, str]) -> str:
    turns_json = json.dumps(turns, ensure_ascii=False)
    types_json = json.dumps(pii_types)
    return (
        f"Summary:\n{summary}\n\n"
        f"Segments (placeholders shown as {{SLOT_n}}):\n{turns_json}\n\n"
        f"Placeholder types:\n{types_json}"
    )


PROMPT_D_SYSTEM = """You will be given a passage with one or more placeholders like {SLOT_1}, {SLOT_2}, etc., and a description of what type of information belongs in each placeholder. Generate a realistic, completely fictional replacement value for every placeholder listed. Each value must be plausible in context but must not be a real, identifiable person, place, or record — invent all of them.

Match each value to its type exactly: if the type is only part of a name or entity (e.g., "first name," "high school type"), generate only that part (e.g., "Emily," "High School"); if the type is a whole name or entity (e.g., "full name"), generate the full multi-word value.

For schools, the value must match the institution level in its type: a "high school name" must be a high school (e.g., "Maple Ridge High School"), an "elementary school name" must be an elementary school, and so on. Never turn a K-12 school into a university, college, or institute, or vice versa.

Respond with JSON only, no other text, with one entry per placeholder:
{"SLOT_1": "<value>", "SLOT_2": "<value>", ...}"""


def build_prompt_d_user(rewritten_text: str, pii_types: dict[str, str]) -> str:
    types_json = json.dumps(pii_types)
    return (
        f'Passage (all placeholders still masked):\n"""\n{rewritten_text}\n"""\n\n'
        f"Placeholder types:\n{types_json}\n\n"
        "Generate a fake value for every placeholder listed above."
    )
