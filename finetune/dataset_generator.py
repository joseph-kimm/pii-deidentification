#!/usr/bin/env python3
"""
Dataset Generator for Language Learning Conversations
Runs continuously to generate training data for AI language conversational partner.
"""

import os
import sys
import argparse
import json
import random
import logging
import time
import uuid
from datetime import datetime
from google import genai
from google.genai import types

# Custom logging filter to suppress unwanted messages
class SuppressSpecificMessages(logging.Filter):
    def filter(self, record):
        # Suppress AFC (Automatic Function Calling) messages
        if "AFC is enabled" in record.getMessage():
            return False
        return True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add filter to suppress specific messages
for handler in logging.root.handlers:
    handler.addFilter(SuppressSpecificMessages())

# Disable verbose logging from Google GenAI library and other dependencies
logging.getLogger('google').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('google.genai').setLevel(logging.WARNING)
logging.getLogger('google.auth').setLevel(logging.WARNING)

# Settings

NUM_SAMPLES_TRAIN = 100
NUM_SAMPLES_EVAL = 10

REDACTION_JUDGE_MODEL = "gemini-3.7-flash"

# Configuration constraints

NUMBER_OF_PIIS =[
    "0",
    "4-6",
    "7-9",
    "10-12"
]

FALSE_POSITIVES = [
    "common words that act as names",
    "generic role/title words",
    "brand/company/product names",
    "academic/scientific jargon",
    "vague, non-identifying geography",
]

DOMAINS = [
    "biology",
    "chemistry",
    "physics",
    "maths",
    "english_language_arts",
    "history_social_studies",
    "health",
    "foreign_language",
]


NUM_TURNS = [
    "short 6-10",
    "medium 10-15",
    "long 15-20"
]


TIME_OF_YEAR = [
    "start_of_year",
    "mid_year",
    "break_approaching",
    "exam_season",
    "end_of_year_wrap_up",
]


COACH_TEACHER_RELATIONSHIP = [
    "first_meeting_just_assigned",
    "longtime_collaborators_comfortable",
    "peer_like_informal_friendship",
    "previously_worked_together_reconnecting",
]

MODELS = ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

# Categories of personal information the redacted version must scrub.
PII_TYPES = [
    "full names and first names of people (teachers, coaches, students, other staff)",
    "ages and grade-specific age references",
    "specific locations (school names, district names, city/town names)",
    "birthdates or other calendar dates tied to a specific person",
]

TRAIN_SYS_PROMPT = """
You are a dataset generator for training a PII (personally identifiable information) deidentification agent.

Your task is to generate a realistic text transcript of a conversation between:
- An instructional coach
- A teacher they are coaching

The conversation should read like a real coaching check-in: discussing lesson plans, classroom
observations, student progress, upcoming events, or professional development, within the given
academic domain and time of year.

You must produce TWO parallel versions of the exact same conversation:
1. "messages": the natural, unredacted transcript, written the way real people actually talk
   (including realistic PII where appropriate for the scenario).
2. "redacted_messages": the identical transcript, except every span of personal information
   has been replaced with the literal string "[REDACTED]". Sentence structure, wording, and
   turn count must otherwise match "messages" exactly except for the redacted spans.

PII to redact (replace with "[REDACTED]"):
- Names and first names of people (never redact role/title words like "the teacher" or "my coach").
- Ages and specific birthdates.
- Specific locations: school names, district names, town/city names.

Do NOT redact (these must remain untouched in both versions, as hard negatives):
- Generic role or title words (e.g. "the principal", "our coach", "the students").
- Subject/domain jargon and curriculum terms.
- Brand, product, or curriculum-program names.
- Vague, non-identifying geography (e.g. "downtown", "the district", "our building").
- Common words that happen to resemble names but are not used as a person's name here.

Behavior Rules:
- Write in natural, professional but conversational English.
- Each message should be reasonably short and realistic for a text/chat exchange or transcribed conversation.
- The two speakers must be labeled with the exact generic identifiers given in the request (e.g. "Speaker 1"/"Speaker 2" or "Speaker A"/"Speaker B") in the "speaker" field — never use "coach" or "teacher" as the label itself, even though their roles as coach and teacher should be clear from what they say.
- Speaker labels must be used consistently: the same person keeps the same label for the whole conversation.
- Match the requested number of turns, time of year, relationship type, and domain.
- Include roughly the requested number of distinct PII items, chosen from the PII categories above.
- Include a few of the "do not redact" hard-negative terms so the redaction target isn't trivial.

Output Format:
- Output valid JSON only.
- Follow the exact schema provided.
- "messages" and "redacted_messages" must have the same number of entries, in the same speaker order,
  differing only in that PII spans in "redacted_messages" are replaced with "[REDACTED]".
"""

EVAL_SYS_PROMPT = """
You are a dataset generator for evaluating a PII (personally identifiable information) deidentification agent.

Your task is to generate a realistic text transcript of a conversation between:
- An instructional coach
- A teacher they are coaching

The conversation should read like a real coaching check-in: discussing lesson plans, classroom
observations, student progress, upcoming events, or professional development, within the given
academic domain and time of year.

You must produce TWO parallel versions of the exact same conversation:
1. "messages": the natural, unredacted transcript, written the way real people actually talk
   (including realistic PII where appropriate for the scenario).
2. "redacted_messages": the identical transcript, except every span of personal information
   has been replaced with the literal string "[REDACTED]". Sentence structure, wording, and
   turn count must otherwise match "messages" exactly except for the redacted spans.

PII to redact (replace with "[REDACTED]"):
- Names and first names of people (never redact role/title words like "the teacher" or "my coach").
- Ages and specific birthdates.
- Specific locations: school names, district names, town/city names.

Do NOT redact (these must remain untouched in both versions, as hard negatives):
- Generic role or title words (e.g. "the principal", "our coach", "the students").
- Subject/domain jargon and curriculum terms.
- Brand, product, or curriculum-program names.
- Vague, non-identifying geography (e.g. "downtown", "the district", "our building").
- Common words that happen to resemble names but are not used as a person's name here.

Behavior Rules:
- Write in natural, professional but conversational English.
- Each message should be reasonably short and realistic for a text/chat exchange or transcribed conversation.
- The two speakers must be labeled with the exact generic identifiers given in the request (e.g. "Speaker 1"/"Speaker 2" or "Speaker A"/"Speaker B") in the "speaker" field — never use "coach" or "teacher" as the label itself, even though their roles as coach and teacher should be clear from what they say.
- Speaker labels must be used consistently: the same person keeps the same label for the whole conversation.
- Match the requested number of turns, time of year, relationship type, and domain.
- Include roughly the requested number of distinct PII items, chosen from the PII categories above.
- Include a few of the "do not redact" hard-negative terms so the redaction target isn't trivial.

Output Format:
- Output valid JSON only.
- Follow the exact schema provided.
- "messages" and "redacted_messages" must have the same number of entries, in the same speaker order,
  differing only in that PII spans in "redacted_messages" are replaced with "[REDACTED]".
"""

REDACTION_JUDGE_SYS_PROMPT = """
You are a strict quality auditor for a PII deidentification training dataset.

You will be given two transcripts of the same coach/teacher conversation as JSON arrays of
{"speaker": ..., "content": ...} turns:
- "original": the natural, unredacted transcript.
- "redacted": the transcript with personal information replaced by the literal string "[REDACTED]".

Check ALL of the following:
1. Completeness: every span of personal information in "original" (names, ages, birthdates,
   specific locations such as school/district/town names) has been replaced with
   "[REDACTED]" in "redacted". Nothing personally identifying was missed.
2. No over-redaction: generic role/title words, subject jargon, brand/product names, and vague
   non-identifying geography were NOT redacted.
3. Alignment: "original" and "redacted" have the same number of turns, in the same speaker
   order, and each redacted turn is identical to its original turn except for the replacement
   of PII spans with "[REDACTED]" (no rewording, added, or removed content beyond that).

Output valid JSON only, matching this schema:
{
  "valid": true or false,
  "missed_pii": ["list of PII spans present in original but not redacted, empty if none"],
  "over_redacted": ["list of spans redacted that should not have been, empty if none"],
  "alignment_issues": ["list of turns where redacted diverges from original beyond redaction, empty if none"],
  "reason": "one sentence explaining the verdict"
}
"""


def validate_redaction(client, messages, redacted_messages):
    """Use a second LLM call to judge whether the redaction was done correctly.

    Checks that every PII span was redacted, that non-PII hard negatives were left alone,
    and that the two transcripts are otherwise identical turn-for-turn.

    Returns:
        tuple: (is_valid, reason) where is_valid is True if the redaction passed the audit.
    """
    usr_prompt = json.dumps({
        "original": messages,
        "redacted": redacted_messages,
    }, ensure_ascii=False)

    response = client.models.generate_content(
        model=REDACTION_JUDGE_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=REDACTION_JUDGE_SYS_PROMPT,
            response_mime_type="application/json",
            temperature=0.0
        ),
        contents=usr_prompt
    )

    verdict = json.loads(response.text)
    is_valid = bool(verdict.get("valid"))
    reason = verdict.get("reason", "")
    if not is_valid:
        reason = (
            f"{reason} | missed_pii={verdict.get('missed_pii')} "
            f"over_redacted={verdict.get('over_redacted')} "
            f"alignment_issues={verdict.get('alignment_issues')}"
        )
    return is_valid, reason

# Initialize Google GenAI client
def init_client():
    """Initialize the Google GenAI client."""
    try:
        client = genai.Client(
            vertexai=True,
            project="project-c7d39647-8a6f-46af-86a",
            location="global"
        )
        logger.info("Successfully initialized Google GenAI client")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize GenAI client: {e}")
        raise

def _validate_message_list(messages, field_name, expected_speakers):
    """Validate a single messages/redacted_messages array.

    Args:
        messages: The list of speaker turns to validate
        field_name: Name of the field being validated (for error messages)
        expected_speakers: The exact set of two speaker labels allowed (e.g. {"Speaker 1", "Speaker 2"})

    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(messages, list):
        return False, f"{field_name} is not a list"

    if len(messages) < 2:
        return False, f"{field_name} array has fewer than 2 entries (got {len(messages)})"

    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            return False, f"{field_name}[{idx}] is not a dictionary"

        if "speaker" not in message or "content" not in message:
            return False, f"{field_name}[{idx}] missing 'speaker' or 'content'"

        if message["speaker"] not in expected_speakers:
            return False, (
                f"{field_name}[{idx}] has invalid speaker: {message['speaker']!r} "
                f"(expected one of {sorted(expected_speakers)})"
            )

        if not isinstance(message["content"], str) or not message["content"].strip():
            return False, f"{field_name}[{idx}] has invalid or empty content"

    return True, None


def validate_schema(data, expected_speakers):
    """Validate that the generated data matches the expected transcript/redacted-transcript schema.

    Args:
        data: The generated data to validate
        expected_speakers: The exact set of two speaker labels the request asked for
            (e.g. {"Speaker 1", "Speaker 2"} or {"Speaker A", "Speaker B"})

    Returns:
        tuple: (is_valid, error_message) where is_valid is True if valid,
               and error_message is None if valid or contains the error description
    """
    # Check if data is a dictionary
    if not isinstance(data, dict):
        return False, "Data is not a dictionary"

    # Check for required top-level keys
    required_keys = {"messages", "redacted_messages", "metadata"}
    if not required_keys.issubset(data.keys()):
        missing = required_keys - set(data.keys())
        return False, f"Missing top-level keys: {missing}"

    # Validate messages array
    is_valid, error_msg = _validate_message_list(data.get("messages"), "messages", expected_speakers)
    if not is_valid:
        return False, error_msg

    # Validate redacted_messages array
    is_valid, error_msg = _validate_message_list(data.get("redacted_messages"), "redacted_messages", expected_speakers)
    if not is_valid:
        return False, error_msg

    # Both speaker labels must actually be used, not just one repeated
    used_speakers = {message["speaker"] for message in data["messages"]}
    if used_speakers != expected_speakers:
        return False, f"messages does not use both expected speakers: got {sorted(used_speakers)}, expected {sorted(expected_speakers)}"

    # The two transcripts must be structurally aligned turn-for-turn
    messages = data["messages"]
    redacted_messages = data["redacted_messages"]
    if len(messages) != len(redacted_messages):
        return False, (
            f"messages and redacted_messages have different lengths "
            f"({len(messages)} vs {len(redacted_messages)})"
        )

    for idx, (message, redacted_message) in enumerate(zip(messages, redacted_messages)):
        if message["speaker"] != redacted_message["speaker"]:
            return False, f"Speaker mismatch at turn {idx}: {message['speaker']} vs {redacted_message['speaker']}"

    # Validate metadata
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        return False, "metadata is not a dictionary"

    # Check for required metadata keys
    required_metadata_keys = {"domain", "num_turns", "time_of_year", "relationship", "number_of_piis"}
    if not required_metadata_keys.issubset(metadata.keys()):
        missing = required_metadata_keys - set(metadata.keys())
        return False, f"Missing metadata keys: {missing}"

    # Validate that metadata values are non-empty strings
    for key in required_metadata_keys:
        if not isinstance(metadata[key], str) or not metadata[key].strip():
            return False, f"metadata.{key} is invalid or empty"

    return True, None


SPEAKER_LABEL_SCHEMES = ["numeric", "alphabetic"]


def _pick_speaker_labels():
    """Pick a pair of generic speaker labels for one sample.

    Randomly uses either a numeric ("Speaker 1"/"Speaker 2") or alphabetic
    ("Speaker A"/"Speaker B") scheme, and randomizes which label is listed
    first so the model doesn't learn a spurious "first speaker = coach" shortcut.
    """
    scheme = random.choice(SPEAKER_LABEL_SCHEMES)
    if scheme == "numeric":
        labels = ["Speaker 1", "Speaker 2"]
    else:
        labels = ["Speaker A", "Speaker B"]
    random.shuffle(labels)
    return labels[0], labels[1]


def _build_usr_prompt(domain, num_turns, time_of_year, relationship, number_of_piis, speaker_a, speaker_b, model=None):
    """Build the user prompt shared by both the train and eval generators."""
    model_line = f'\n    "model": "{model}",' if model else ""
    return f"""
    Domain: {domain}
    Number of turns: {num_turns}
    Time of year: {time_of_year}
    Coach/teacher relationship: {relationship}
    Approximate number of PII items to include: {number_of_piis}
    Speaker labels to use (exactly these two strings, one per person): "{speaker_a}" and "{speaker_b}"

    Output schema:
    {{
      "messages": [
        {{"speaker": "{speaker_a}", "content": "..."}},
        {{"speaker": "{speaker_b}", "content": "..."}}
      ],
      "redacted_messages": [
        {{"speaker": "{speaker_a}", "content": "..."}},
        {{"speaker": "{speaker_b}", "content": "..."}}
      ],
      "metadata": {{
        "domain": "{domain}",{model_line}
        "num_turns": "{num_turns}",
        "time_of_year": "{time_of_year}",
        "relationship": "{relationship}",
        "number_of_piis": "{number_of_piis}"
      }}
    }}
    """


def generate_sample_eval(client, domain, num_turns, time_of_year, relationship, number_of_piis, speaker_a, speaker_b, model):
    """Generate a single transcript/redacted-transcript pair for evaluation."""
    usr_prompt = _build_usr_prompt(
        domain, num_turns, time_of_year, relationship, number_of_piis, speaker_a, speaker_b, model=model
    )

    response = client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(
            system_instruction=EVAL_SYS_PROMPT,
            response_mime_type="application/json",
            temperature=0.9
        ),
        contents=usr_prompt
    )

    return json.loads(response.text)

def generate_sample(client, domain, num_turns, time_of_year, relationship, number_of_piis, speaker_a, speaker_b, model):
    """Generate a single transcript/redacted-transcript pair for training."""
    usr_prompt = _build_usr_prompt(
        domain, num_turns, time_of_year, relationship, number_of_piis, speaker_a, speaker_b, model=model
    )

    response = client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(
            system_instruction=TRAIN_SYS_PROMPT,
            response_mime_type="application/json",
            temperature=0.9
        ),
        contents=usr_prompt
    )

    return json.loads(response.text)

def _humanize(value):
    """Turn a snake_case config value into title-cased, space-separated text."""
    return value.replace("_", " ").strip().title()


def _build_title(domain, time_of_year):
    """Build the title line placed at the top of each transcript file."""
    return f"Instructional Coaching Session — {_humanize(domain)}, {_humanize(time_of_year)}"


def _format_transcript(title, messages):
    """Render a title + speaker turns into the plain-text transcript format.

    Matches the reference format in data/synthetic/original + data/synthetic/human_redacted:
    "Speaker X: content" turns, separated by blank lines, with a title line up top.
    """
    turns = "\n\n".join(f"{message['speaker']}: {message['content']}" for message in messages)
    return f"{title}\n\n{turns}"


def _transcript_stats(text):
    """Compute cheap, locally-derived size stats for a rendered transcript."""
    return {
        "word_count": len(text.split()),
        "char_count": len(text),
    }


def generate_batch(choice, client, num_samples):
    """Generate a batch of coach/teacher transcript samples.

    This function guarantees exactly num_samples will be generated by retrying
    failed attempts until the target count is reached. Each sample is written as
    a matched pair of plain-text .txt files: the natural transcript in
    <base_dir>/original/<id>.txt and the PII-redacted transcript in
    <base_dir>/redacted/<id>.txt, plus one line appended to
    <base_dir>/metadata.jsonl recording the configuration used to generate it.
    """

    _here = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(_here, "..", "data", "finetune", "training" if choice == "t" else "evaluation")
    original_dir = os.path.join(base_dir, "original")
    redacted_dir = os.path.join(base_dir, "redacted")
    manifest_path = os.path.join(base_dir, "metadata.jsonl")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(redacted_dir, exist_ok=True)

    successful = 0
    failed = 0
    validation_failed = 0

    logger.info(f"Starting batch generation: {num_samples} samples")

    while successful < num_samples:
        try:
            # Randomly select parameters
            domain = random.choice(DOMAINS)
            num_turns = random.choice(NUM_TURNS)
            time_of_year = random.choice(TIME_OF_YEAR)
            relationship = random.choice(COACH_TEACHER_RELATIONSHIP)
            number_of_piis = random.choice(NUMBER_OF_PIIS)
            model = random.choice(MODELS)
            speaker_a, speaker_b = _pick_speaker_labels()

            # Generate sample
            if choice == "t":
                data_dict = generate_sample(
                    client, domain, num_turns, time_of_year,
                    relationship, number_of_piis, speaker_a, speaker_b, model
                )
            else:
                data_dict = generate_sample_eval(
                    client, domain, num_turns, time_of_year,
                    relationship, number_of_piis, speaker_a, speaker_b, model
                )

            # Validate schema
            is_valid, error_msg = validate_schema(data_dict, {speaker_a, speaker_b})
            if not is_valid:
                validation_failed += 1
                logger.warning(f"Schema validation failed (attempt {successful + failed + validation_failed}): {error_msg}")
                continue

            # Second LLM call: audit that the redaction is complete, precise, and aligned
            is_valid, error_msg = validate_redaction(
                client, data_dict["messages"], data_dict["redacted_messages"]
            )
            if not is_valid:
                validation_failed += 1
                logger.warning(f"Redaction audit failed (attempt {successful + failed + validation_failed}): {error_msg}")
                continue

            sample_id = str(uuid.uuid4())
            title = _build_title(domain, time_of_year)

            original_text = _format_transcript(title, data_dict["messages"])
            redacted_text = _format_transcript(title, data_dict["redacted_messages"])

            # Write to matched original/redacted .txt files
            with open(os.path.join(original_dir, f"{sample_id}.txt"), "w", encoding="utf-8") as f:
                f.write(original_text)
            with open(os.path.join(redacted_dir, f"{sample_id}.txt"), "w", encoding="utf-8") as f:
                f.write(redacted_text)

            original_stats = _transcript_stats(original_text)
            redacted_stats = _transcript_stats(redacted_text)

            # Append the sample's configuration to the manifest for later analysis
            manifest_record = {
                "id": sample_id,
                "domain": domain,
                "num_turns": num_turns,
                "time_of_year": time_of_year,
                "relationship": relationship,
                "number_of_piis": number_of_piis,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "model": model,
                "original_word_count": original_stats["word_count"],
                "original_char_count": original_stats["char_count"],
                "redacted_word_count": redacted_stats["word_count"],
                "redacted_char_count": redacted_stats["char_count"],
                "redacted_span_count": redacted_text.count("[REDACTED]"),
            }
            with open(manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")

            successful += 1

            # Progress logging
            if successful % 10 == 0:
                logger.info(f"Progress: {successful}/{num_samples} samples generated")

        except Exception as e:
            failed += 1
            logger.error(f"Error generating sample (attempt {successful + failed + validation_failed}): {e}")
            # Continue with next attempt instead of stopping
            continue

    logger.info(f"Batch complete: {successful} successful, {failed} API failures, {validation_failed} validation failures")
    return successful, failed


def run_generation(choice):
    """Generate the coach/teacher transcript dataset."""

    if choice == "t":
        logger.info("Starting training dataset generation")

    else:
        logger.info("Starting evaluation dataset generation")

    try:
        client = init_client()
    except Exception as e:
        logger.error("Cannot start without GenAI client. Exiting.")
        return

    start_time = time.time()

    num_samples = NUM_SAMPLES_TRAIN if choice == "t" else NUM_SAMPLES_EVAL
    total_successful, total_failed = generate_batch(choice, client, num_samples)

    elapsed_time = time.time() - start_time
    logger.info(f"=== Generation complete ===")
    logger.info(f"Total successful: {total_successful}, Total failed: {total_failed}")
    logger.info(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    try:
        # Check if argument provided, otherwise ask for input
        if len(sys.argv) > 1:
            choice = sys.argv[1]
        else:
            choice = input("Press e for evaluation and t for training: ")

        run_generation(choice)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal. Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        raise
