#!/usr/bin/env python3
"""
analyse.py - Redaction comparison for a single transcript file.

Usage:
    python analyse.py <base_name>
    python analyse.py simple
    python analyse.py example

Outputs a CSV with one row per word, categorized as:
  correctly_redacted  - AI redacted it and it should have been (TP)
  missed              - AI did NOT redact it but should have (FN)
  over_redacted       - AI redacted it but should NOT have (FP)
"""

import csv
import difflib
import re
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_DIR = PROJECT_ROOT / "data" / "synthetic" / "original"
HUMAN_DIR = PROJECT_ROOT / "data" / "synthetic" / "human_redacted"
AI_DIR = PROJECT_ROOT / "results" / "deidentify" / "new"
OUTPUT_DIR = PROJECT_ROOT / "results" / "deidentify" / "analysis"

REDACT_RE = re.compile(r"\[REDACTED\]", re.IGNORECASE)


def read_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            text = path.read_text(encoding=enc)
            text = unicodedata.normalize("NFKC", text)
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            text = text.replace(" ", " ")
            return text
        except Exception:
            continue
    raise ValueError(f"Could not read {path}")


def is_redacted(token: str) -> bool:
    return bool(REDACT_RE.search(token.strip()))


def find_redacted_positions(orig_tokens: list[str], redacted_tokens: list[str]) -> set[int]:
    """
    Return indices into orig_tokens that were replaced by [REDACTED].

    Keeps only non-[REDACTED] tokens from the redacted file, aligns them
    against the original via SequenceMatcher, and returns every original
    position not covered by a matching block.
    """
    kept = [t for t in redacted_tokens if not is_redacted(t)]
    matcher = difflib.SequenceMatcher(None, orig_tokens, kept, autojunk=False)
    matched: set[int] = set()
    for block in matcher.get_matching_blocks():
        for i in range(block.size):
            matched.add(block.a + i)
    return set(range(len(orig_tokens))) - matched


def get_context(tokens: list[str], idx: int, window: int = 4) -> str:
    start = max(0, idx - window)
    end = min(len(tokens), idx + window + 1)
    parts = []
    for i in range(start, end):
        parts.append(f"[{tokens[i]}]" if i == idx else tokens[i])
    return " ".join(parts)


def analyse(base_name: str) -> None:
    orig_path = ORIGINAL_DIR / f"{base_name}.txt"
    human_path = HUMAN_DIR / f"{base_name}.txt"

    if not orig_path.exists():
        print(f"ERROR: Original not found: {orig_path}")
        sys.exit(1)
    if not human_path.exists():
        print(f"ERROR: Human redacted not found: {human_path}")
        sys.exit(1)

    ai_files = sorted(AI_DIR.glob(f"{base_name}__*.txt"))
    if not ai_files:
        exact = AI_DIR / f"{base_name}.txt"
        if exact.exists():
            ai_files = [exact]
        else:
            print(f"ERROR: No AI files found in {AI_DIR}/ for '{base_name}'")
            sys.exit(1)

    orig_tokens = read_text(orig_path).split()
    human_tokens = read_text(human_path).split()
    human_redacted = find_redacted_positions(orig_tokens, human_tokens)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{base_name}_analysis.csv"

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["base_name", "ai_file", "word", "category", "context"])

        for ai_path in ai_files:
            ai_tokens = read_text(ai_path).split()
            ai_redacted = find_redacted_positions(orig_tokens, ai_tokens)

            tp = human_redacted & ai_redacted
            fn = human_redacted - ai_redacted
            fp = ai_redacted - human_redacted

            for idx in sorted(tp):
                writer.writerow([base_name, ai_path.name, orig_tokens[idx], "correctly_redacted", get_context(orig_tokens, idx)])
            for idx in sorted(fn):
                writer.writerow([base_name, ai_path.name, orig_tokens[idx], "missed", get_context(orig_tokens, idx)])
            for idx in sorted(fp):
                writer.writerow([base_name, ai_path.name, orig_tokens[idx], "over_redacted", get_context(orig_tokens, idx)])

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyse.py <base_name>")
        print("Example: python analyse.py simple")
        sys.exit(1)
    analyse(sys.argv[1])
