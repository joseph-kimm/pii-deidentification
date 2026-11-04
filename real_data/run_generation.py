"""Step 3 of 3: generate fake training pairs from a chunk sample, using LM Studio.

=============================================================================
READ ME FIRST (remote desktop / Windows)
=============================================================================
EASIEST: double-click the .bat files in this folder, in order
(1_build_catalog, 2_sample_chunks, 3_check_lmstudio, 4_generate). Folders,
chunk sizes and the sample name are set once in settings.bat; the .bat
files pass the sample folder to this script, overriding SAMPLE_DIR below.

Full workflow by hand (run each command from this folder, in a Command Prompt or
PowerShell window opened in this folder; on Windows use "py" if "python"
is not found):

  1. python build_catalog.py --original data\\original --human data\\human --llm data\\llm --out catalog.csv --min-tokens 50 --max-tokens 600 --tokenizer words
  2. python sample_chunks.py --catalog catalog.csv --original data\\original --human data\\human --llm data\\llm --out-dir samples\\sample_001 --n 500 --seed 42
  3. python run_generation.py --check     (tests LM Studio + finds the sample; no generation)
     python run_generation.py             (the real run)

Recommended folder layout (everything inside this folder):
  real_data\\
    build_catalog.py  sample_chunks.py  run_generation.py
    generate_sample.py  prompts.py  lmstudio_client.py
    data\\original\\*.txt   real transcripts
    data\\human\\*.txt      human-redacted versions (same file names)
    data\\llm\\*.txt        LLM-redacted versions   (same file names)
    catalog.csv            written by step 1
    samples\\sample_001\\    written by step 2, read and added to by step 3

What this file needs:
  - Python 3.10+ (standard library only; nothing to pip install).
  - generate_sample.py, prompts.py, lmstudio_client.py in this same folder.
  - LM Studio running with a model loaded and the local server started
    (Developer tab -> Status: Running). See lmstudio_client.py.
  - SAMPLE_DIR\\chunks.jsonl from sample_chunks.py. Each line must have
    chunk_id, file, original_text, human_text; if review.human_text_corrected
    is filled in, that corrected text is used instead of human_text.

Model choice (LLM_MODEL below):
  Leave it "" to use whatever single model LM Studio has loaded. If several
  are loaded, copy the exact id that "--check" prints (e.g.
  "qwen2.5-7b-instruct") into LLM_MODEL. Instruction-tuned models of 7B+
  follow the JSON-only rules far more reliably than small ones; a model
  that keeps failing shows up as many lines in failures.jsonl.
  Context length: set 8192+ in LM Studio's load settings so long chunks fit.

What it writes (all inside SAMPLE_DIR):
  generated.jsonl   one fake pair per line:
                    {"input": fake text, "output": same with [REDACTED],
                     "source_file", "chunk_id", "num_slots"}
  failures.jsonl    chunks that could not be generated + the reason
  step_log.txt      every step's output for every chunk (for checking by eye)

Stopping and resuming:
  Ctrl+C is safe. Re-running continues where it stopped: chunks already in
  generated.jsonl are skipped; chunks in failures.jsonl are tried again.
  To start over, delete generated.jsonl, failures.jsonl and step_log.txt.

Windows paths:
  Paths below are relative to THIS file's folder, so it doesn't matter
  where you start Python from. To use an absolute path, write it as
  Path(r"C:\\Users\\me\\data\\sample_001") -- keep the r before the quote.
=============================================================================
"""

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import generate_sample  # noqa: E402
from lmstudio_client import LMStudioClient, LMStudioError  # noqa: E402

# --- Settings: edit these ----------------------------------------------------

# Folder made by sample_chunks.py (--out-dir). Must contain chunks.jsonl.
SAMPLE_DIR = SCRIPT_DIR.parent / "results" / "real_data" / "samples" / "sample_001"

# LM Studio server address (Developer tab), with /v1 at the end.
LLM_BASE_URL = "http://localhost:1234/v1"

# Model id as LM Studio lists it. "" = use the only loaded model.
LLM_MODEL = ""

# Seconds to wait for one model reply before counting it as a failed attempt.
LLM_TIMEOUT_SECONDS = 600

# Attempts per model call before the chunk is given up (goes to failures.jsonl).
MAX_RETRIES = 3

# Also generate pairs for chunks with no [REDACTED] (input == output).
INCLUDE_CHUNKS_WITHOUT_PII = True

# Chunks whose review.status is one of these are skipped.
SKIP_REVIEW_STATUSES = {"rejected"}

# Stop after this many chunks (e.g. 5 for a quick test run). None = all.
MAX_CHUNKS = None

# Write every step's output to step_log.txt.
WRITE_STEP_LOG = True

# -----------------------------------------------------------------------------

CHUNKS_FILE = "chunks.jsonl"
OUTPUT_FILE = "generated.jsonl"
FAILURES_FILE = "failures.jsonl"
STEP_LOG_FILE = "step_log.txt"


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, skipping blank lines and warning about broken ones."""
    records = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[WARN] {path.name} line {line_no} is not valid JSON; ignored")
    return records


def pick_model(client: LMStudioClient) -> str:
    """Return LLM_MODEL, or the single listed model when LLM_MODEL is empty."""
    available = [m.id for m in client.models.list().data]
    if LLM_MODEL:
        if LLM_MODEL not in available:
            sys.exit(
                f"LLM_MODEL {LLM_MODEL!r} is not available in LM Studio.\n"
                f"Available ids: {available or 'none -- load a model first'}"
            )
        return LLM_MODEL
    if len(available) == 1:
        return available[0]
    sys.exit(
        "Set LLM_MODEL at the top of run_generation.py to one of these ids "
        f"(LM Studio lists {len(available)}): {available or 'none -- load a model first'}"
    )


def chunk_texts(record: dict) -> tuple[str, str]:
    """(original, human-redacted) text for a chunk, preferring the reviewed correction."""
    review = record.get("review") or {}
    redacted = review.get("human_text_corrected") or record["human_text"]
    return record["original_text"], redacted


def select_chunks(records: list[dict], done_ids: set[str]) -> list[dict]:
    todo = []
    for record in records:
        if record["chunk_id"] in done_ids:
            continue
        if (record.get("review") or {}).get("status") in SKIP_REVIEW_STATUSES:
            continue
        _, redacted = chunk_texts(record)
        if not INCLUDE_CHUNKS_WITHOUT_PII and generate_sample.REDACTED_TOKEN not in redacted:
            continue
        todo.append(record)
    return todo[:MAX_CHUNKS] if MAX_CHUNKS is not None else todo


def check(client: LMStudioClient, chunks_path: Path) -> None:
    """--check: confirm LM Studio answers and the sample is readable, without generating."""
    model = pick_model(client)
    print(f"[OK] LM Studio reachable at {LLM_BASE_URL}; using model {model!r}")
    generate_sample.configure(client, model, max_retries=MAX_RETRIES)
    reply = generate_sample.call_llm_json(
        'Respond with JSON only: {"ok": true}', "Say ok."
    )
    print(f"[OK] model replied with valid JSON: {reply}")
    if not chunks_path.exists():
        sys.exit(f"[MISSING] {chunks_path} -- run sample_chunks.py or fix SAMPLE_DIR")
    records = read_jsonl(chunks_path)
    missing = [k for k in ("chunk_id", "file", "original_text", "human_text")
               if records and k not in records[0]]
    if missing:
        sys.exit(f"[BAD] {chunks_path.name} records lack fields: {missing}")
    print(f"[OK] {chunks_path} has {len(records)} chunks")


def main() -> None:
    # Never crash on printing a character the Windows console can't show.
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description="Generate fake pairs with LM Studio (settings are at the top of this file).")
    parser.add_argument("--check", action="store_true", help="test LM Studio and the sample folder, then stop")
    parser.add_argument("--sample-dir", default=None, help="sample folder to use instead of SAMPLE_DIR (the .bat files pass this)")
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir) if args.sample_dir else Path(SAMPLE_DIR)
    chunks_path = sample_dir / CHUNKS_FILE
    client = LMStudioClient(base_url=LLM_BASE_URL, timeout=LLM_TIMEOUT_SECONDS)

    try:
        if args.check:
            check(client, chunks_path)
            return
        if not chunks_path.exists():
            sys.exit(f"Not found: {chunks_path}\nRun sample_chunks.py first or fix SAMPLE_DIR.")
        model = pick_model(client)
    except (LMStudioError, generate_sample.LLMCallError) as e:
        sys.exit(f"[LM Studio] {e}")
    generate_sample.configure(client, model, max_retries=MAX_RETRIES)

    output_path = sample_dir / OUTPUT_FILE
    done_ids = {r["chunk_id"] for r in read_jsonl(output_path)} if output_path.exists() else set()
    todo = select_chunks(read_jsonl(chunks_path), done_ids)
    print(f"Model {model!r} | {len(done_ids)} chunks already done | {len(todo)} to generate")

    n_ok = n_failed = 0
    started = time.time()
    with open(output_path, "a", encoding="utf-8") as out, \
         open(sample_dir / FAILURES_FILE, "a", encoding="utf-8") as failures, \
         open(sample_dir / STEP_LOG_FILE, "a", encoding="utf-8") as step_log:
        for n, record in enumerate(todo, start=1):
            chunk_id = record["chunk_id"]
            original, redacted = chunk_texts(record)
            log: list[str] | None = [] if WRITE_STEP_LOG else None
            try:
                result = generate_sample.run_chunk(original, redacted, record["file"], chunk_id, log)
            except ConnectionError as e:
                sys.exit(f"\n[STOPPED] {e}\nStart the LM Studio server and re-run; finished chunks are kept.")
            except KeyboardInterrupt:
                sys.exit("\n[STOPPED] by Ctrl+C. Re-run to continue; finished chunks are kept.")
            except (ValueError, generate_sample.LLMCallError) as e:
                result = None
                error = f"{type(e).__name__}: {e}"
                failures.write(json.dumps({"chunk_id": chunk_id, "file": record["file"], "error": error}) + "\n")
                failures.flush()
                if log is not None:
                    log.append(f"=== ERROR ===\n{error}\n")
                n_failed += 1
                print(f"[{n}/{len(todo)}] FAILED {chunk_id}: {error}")
            if result is not None:
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                n_ok += 1
                print(f"[{n}/{len(todo)}] ok {chunk_id} ({result['num_slots']} slots)")
            if log is not None:
                step_log.write(f"\n##### {chunk_id} ({record['file']}) #####\n" + "\n".join(log))
                step_log.flush()

    minutes = (time.time() - started) / 60
    print(f"\nDone in {minutes:.1f} min: {n_ok} generated, {n_failed} failed.")
    print(f"Output:   {output_path}")
    if n_failed:
        print(f"Failures: {sample_dir / FAILURES_FILE} (re-run to retry them)")


if __name__ == "__main__":
    main()
