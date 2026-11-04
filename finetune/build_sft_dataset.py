#!/usr/bin/env python3
"""
Converts paired original/redacted .txt transcripts into a HuggingFace-ready
SFT dataset, chunked the exact same way de_identify_steplab.py chunks text at
inference time (its split_into_sentences logic), so the training distribution
matches what the model will actually be called with.

Each output line is {"messages": [system, user, assistant]}, suitable for
datasets.load_dataset("json", ...) + TRL's SFTTrainer.
"""

import argparse
import csv
import json
import logging
import os
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_SPEAKER_LINE_RE = re.compile(r'^\s*Speaker\s*[A-Za-z0-9]*\s*:')


def _is_speaker_line(line):
    """Mirrors de_identify_steplab.py's _is_speaker_line: true if the line is a turn, not a title."""
    return bool(_SPEAKER_LINE_RE.match(line))


def split_into_sentences(text, max_sentences=3):
    """Verbatim copy of de_identify_steplab.py's chunking logic, so training

    examples are chunked identically to how the harness will call the model
    at inference time.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunks.append(" ".join(sentences[i:i + max_sentences]))
    return chunks


def strip_title(text):
    """Drop the title/header line, matching how the harness separates it before calling the model."""
    lines = text.split("\n")
    if lines and not _is_speaker_line(lines[0]):
        lines = lines[1:]
    body = "\n".join(lines).strip()
    return body


def load_system_prompt(prompts_csv, prompt_label):
    """Read the exact system prompt de_identify_steplab.py would use for prompt_label."""
    with open(prompts_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["label"] == prompt_label:
                return row["prompt"]
    raise ValueError(f"No prompt labeled {prompt_label!r} found in {prompts_csv}")


def build_examples(base_dir, system_prompt, max_sentences):
    """Pair up original/redacted .txt files, chunk each, and yield chat-format examples."""
    original_dir = os.path.join(base_dir, "original")
    redacted_dir = os.path.join(base_dir, "redacted")

    filenames = sorted(f for f in os.listdir(original_dir) if f.endswith(".txt"))
    skipped_files = 0
    skipped_chunks = 0

    for filename in filenames:
        redacted_path = os.path.join(redacted_dir, filename)
        if not os.path.exists(redacted_path):
            logger.warning(f"No matching redacted file for {filename}, skipping")
            skipped_files += 1
            continue

        with open(os.path.join(original_dir, filename), "r", encoding="utf-8") as f:
            original_body = strip_title(f.read())
        with open(redacted_path, "r", encoding="utf-8") as f:
            redacted_body = strip_title(f.read())

        original_chunks = split_into_sentences(original_body, max_sentences)
        redacted_chunks = split_into_sentences(redacted_body, max_sentences)

        if len(original_chunks) != len(redacted_chunks):
            logger.warning(
                f"{filename}: chunk count mismatch ({len(original_chunks)} vs "
                f"{len(redacted_chunks)}), skipping file"
            )
            skipped_files += 1
            continue

        for original_chunk, redacted_chunk in zip(original_chunks, redacted_chunks):
            if not original_chunk.strip() or not redacted_chunk.strip():
                skipped_chunks += 1
                continue

            yield {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Text to edit: {original_chunk}"},
                    {"role": "assistant", "content": redacted_chunk},
                ]
            }

    if skipped_files:
        logger.warning(f"Skipped {skipped_files} file(s) entirely")
    if skipped_chunks:
        logger.warning(f"Skipped {skipped_chunks} empty chunk(s)")


_HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir", default=os.path.join(_HERE, "..", "data", "finetune", "training"),
        help="Directory containing original/ and redacted/ subfolders (default: data/finetune/training)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSONL path (default: <base-dir>/sft_dataset.jsonl)"
    )
    parser.add_argument(
        "--prompts-csv", default=os.path.join(_HERE, "..", "Prompts.csv"),
        help="Path to the same Prompts.csv de_identify_steplab.py reads from (default: Prompts.csv at the project root)"
    )
    parser.add_argument(
        "--prompt-label", default="prompt 1",
        help="Which Prompts.csv row to use as the system prompt, matching PROMPT_LABEL (default: 'prompt 1')"
    )
    parser.add_argument(
        "--max-sentences", type=int, default=3,
        help="Sentences per chunk, matching de_identify_steplab.py's default (default: 3)"
    )
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.base_dir, "sft_dataset.jsonl")
    system_prompt = load_system_prompt(args.prompts_csv, args.prompt_label)

    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for example in build_examples(args.base_dir, system_prompt, args.max_sentences):
            out.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1

    logger.info(f"Wrote {count} chunked examples to {output_path}")


if __name__ == "__main__":
    main()
