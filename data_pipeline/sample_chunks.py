"""
Step 2 of 3: randomly sample chunks from the catalog and write them, with their
text, as the input file for fake data generation.

=============================================================================
READ ME FIRST (remote desktop / Windows)
=============================================================================
Needs: catalog.csv from build_catalog.py, and the SAME three transcript
folders that build_catalog.py used (the catalog stores word positions, so
the files must not be edited in between).

Run (Command Prompt / PowerShell opened in this folder; "py" if "python"
is not found; put a path in "quotes" if it contains spaces):
    python sample_chunks.py --catalog ..\\results\\synthetic\\catalog.csv --original ..\\data\\synthetic\\original --human ..\\data\\synthetic\\human_redacted --llm ..\\data\\synthetic\\llm_redacted --out-dir ..\\results\\synthetic\\samples\\sample_001 --n 500 --seed 42
  (2_sample_chunks.bat builds these paths from settings.bat; the example uses DATA_SET=synthetic.)

  --out-dir must be a NEW folder; an existing one is never overwritten.
  --n is the number of chunks; it fails if the catalog has fewer.

Output: ..\\results\\synthetic\\samples\\sample_001\\chunks.jsonl (+ config.json). Each chunk has a
"review" field; to fix a bad human redaction by hand, put the corrected
text in review.human_text_corrected, or set review.status to "rejected"
to have run_generation.py skip that chunk.
Next: run_generation.py (set SAMPLE_DIR there to this --out-dir).
Needs Python 3.10+, standard library only.
=============================================================================

Writes into --out-dir (which must not exist yet, so a sample is never changed):
    chunks.jsonl  one sampled chunk per line: catalog fields + original/human/LLM text
    config.json   how the sample was drawn (seed, n, filters, file hashes)

Usage:
    python sample_chunks.py --catalog catalog.csv --original original/ --human human/ --llm llm/ \
        --out-dir ../results/synthetic/samples/sample_001 --n 500 --seed 42
    # --files train_files.txt     only sample from the files listed (one name per line)
    # --exclude samples/sample_000/chunks.jsonl ...   skip chunks used in earlier samples
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from datetime import date
from pathlib import Path

INT_FIELDS = [
    "turn_idx", "piece_idx", "size", "orig_start", "orig_end", "human_start", "human_end",
    "llm_start", "llm_end", "llm_redacted_span_count", "llm_redacted_word_count",
    "human_redacted_span_count", "human_redacted_word_count", "tp_count", "fp_count", "fn_count",
]
JSON_FIELDS = ["fp_details", "fn_details"]
VERSIONS = [("original", "orig"), ("human", "human"), ("llm", "llm")]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_catalog(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for k in INT_FIELDS:
            row[k] = int(row[k])
        for k in JSON_FIELDS:
            row[k] = json.loads(row[k])
    return rows


def load_names(path):
    """File names listed one per line (blank lines ignored)."""
    return {line.strip() for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()}


def load_used_ids(paths):
    used = set()
    for path in paths:
        with open(path, encoding="utf-8-sig") as f:
            used.update(json.loads(line)["chunk_id"] for line in f if line.strip())
    return used


def candidate_pool(rows, allowed_files, used_ids):
    return [r for r in rows
            if (allowed_files is None or r["file"] in allowed_files) and r["chunk_id"] not in used_ids]


class Texts:
    """Reads each transcript version once and slices chunks out of it by word position."""

    def __init__(self, folders):
        self.folders = folders  # {"original": Path, "human": Path, "llm": Path}
        self._words = {}

    def words(self, version, file):
        key = (version, file)
        if key not in self._words:
            path = self.folders[version] / file
            if not path.exists():
                raise FileNotFoundError(f"{path} is listed in the catalog but does not exist")
            self._words[key] = path.read_text(encoding="utf-8-sig").split()
        return self._words[key]

    def chunk(self, version, prefix, row):
        return " ".join(self.words(version, row["file"])[row[f"{prefix}_start"]:row[f"{prefix}_end"]])


def make_record(row, texts):
    record = dict(row)
    for version, prefix in VERSIONS:
        record[f"{version}_text"] = texts.chunk(version, prefix, row)
    record["review"] = {"status": "pending", "human_text_corrected": None, "note": None}
    return record


def sample(args):
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        sys.exit(f"{out_dir} already exists. Samples are never overwritten; choose a new --out-dir.")

    rows = load_catalog(args.catalog)
    allowed = load_names(args.files) if args.files else None
    pool = candidate_pool(rows, allowed, load_used_ids(args.exclude))
    if args.n > len(pool):
        sys.exit(f"Asked for {args.n} chunks but only {len(pool)} are available after filtering.")

    chosen = random.Random(args.seed).sample(pool, args.n)
    chosen.sort(key=lambda r: (r["file"], r["orig_start"]))

    folders = {"original": Path(args.original), "human": Path(args.human), "llm": Path(args.llm)}
    texts = Texts(folders)
    records = [make_record(r, texts) for r in chosen]

    out_dir.mkdir(parents=True)
    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    files_used = sorted({r["file"] for r in chosen})
    config = {
        "sample_id": out_dir.name, "created": date.today().isoformat(), "method": "random",
        "n": args.n, "seed": args.seed, "pool_size": len(pool),
        "catalog": str(args.catalog), "catalog_sha256": sha256(args.catalog),
        "allowed_files": args.files, "excluded_samples": args.exclude,
        "source_sha256": {
            file: {v: sha256(folders[v] / file) for v, _ in VERSIONS} for file in files_used},
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Sampled {args.n} of {len(pool)} chunks from {len(files_used)} files into {out_dir}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", required=True, help="catalog CSV from build_catalog.py")
    p.add_argument("--original", required=True, help="folder of original transcripts")
    p.add_argument("--human", required=True, help="folder of human-redacted transcripts")
    p.add_argument("--llm", required=True, help="folder of LLM-redacted transcripts")
    p.add_argument("--out-dir", required=True, help="new folder for this sample, e.g. ../results/synthetic/samples/sample_001")
    p.add_argument("--n", type=int, required=True, help="number of chunks to sample")
    p.add_argument("--seed", type=int, required=True, help="random seed, so the sample can be reproduced")
    p.add_argument("--files", default=None, help="optional text file of allowed file names, one per line")
    p.add_argument("--exclude", nargs="*", default=[], help="chunks.jsonl files of earlier samples to skip")
    return p.parse_args(argv)


if __name__ == "__main__":
    sample(parse_args())
