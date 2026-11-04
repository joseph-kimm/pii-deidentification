"""
Step 1 of 3: build a chunk catalog from original / human-redacted / LLM-redacted transcripts.

=============================================================================
READ ME FIRST (remote desktop / Windows)
=============================================================================
Where the transcripts go (recommended, inside this folder):
    data\\original\\*.txt   real transcripts
    data\\human\\*.txt      human-redacted versions, SAME file names
    data\\llm\\*.txt        LLM-redacted versions,   SAME file names
  Every redaction must be the literal token [REDACTED]. Files must be
  UTF-8 text with "Speaker 1:" style labels (Notepad's "UTF-8" and
  "UTF-8 with BOM" both work). A file missing from human\\ or llm\\ is
  skipped with a [SKIP] message.

Run (Command Prompt / PowerShell opened in this folder; "py" if "python"
is not found; put a path in "quotes" if it contains spaces):
    python build_catalog.py --original data\\original --human data\\human --llm data\\llm --out catalog.csv --min-tokens 50 --max-tokens 600 --tokenizer words

  --tokenizer words   counts whitespace words: no installs, works offline.
                      Use this on the remote desktop.
  --tokenizer <HF model name> counts real model tokens, but needs
                      "pip install transformers", internet access, and for
                      gated models (meta-llama/...) a Hugging Face login.
                      This is unrelated to the model loaded in LM Studio.
  --max-tokens        longer speaker turns are split into pieces; each
                      piece becomes one chunk. This sets the largest chunk
                      the LM Studio model will have to rewrite.
  --min-tokens        shorter turns with no redactions are left out.

Output: catalog.csv (positions only, no text). Next: sample_chunks.py.
Needs Python 3.10+, standard library only (unless using an HF tokenizer).
=============================================================================

Each transcript is split into speaker turns (text before the first 'Speaker X:'
label is not a turn). A turn longer than --max-tokens is split at sentence
breaks into balanced pieces; every turn or piece is one chunk (one row). Short
turns that neither the human nor the LLM redacted anything in are left out.

Positions are 0-based word indices (whitespace split) over the whole file, end
exclusive, so `words[start:end]` gives the chunk's text in that version.

Usage:
    python build_catalog.py --original original/ --human human/ --llm llm/ \
        --out catalog.csv --min-tokens 50 --max-tokens 600 \
        --tokenizer meta-llama/Meta-Llama-3-8B-Instruct
    # --tokenizer words counts whitespace words instead of model tokens
"""
import argparse
import bisect
import csv
import difflib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

REDACTED = "[REDACTED]"
SPEAKER_RE = re.compile(r"(?<![A-Za-z])Speaker\s*[A-Za-z0-9]{0,3}\s*:")
ABBREVIATIONS = {"dr.", "mr.", "mrs.", "ms.", "prof.", "st.", "jr.", "sr.", "mt.", "ft.", "rev.", "gov.", "sen."}
_CLOSERS = "\"')]’”"

COLUMNS = [
    "chunk_id", "file", "turn_idx", "piece_idx", "size",
    "orig_start", "orig_end", "human_start", "human_end", "llm_start", "llm_end",
    "llm_redacted_span_count", "llm_redacted_word_count",
    "human_redacted_span_count", "human_redacted_word_count",
    "tp_count", "fp_count", "fn_count", "fp_details", "fn_details",
]


# -------------------------
# Words and turns
# -------------------------
def tokenize(text):
    """Whitespace-split words with their character start offsets."""
    matches = list(re.finditer(r"\S+", text))
    return [m.group() for m in matches], [m.start() for m in matches]


def find_turns(text, word_starts):
    """
    Return (turns, label_words): turns as [(start, end)] word ranges, each starting
    at a 'Speaker X:' label and running to the next one, and the set of word
    indices that belong to a label. Text before the first label is not a turn.
    """
    starts, label_words = set(), set()
    for m in SPEAKER_RE.finditer(text):
        first = max(bisect.bisect_right(word_starts, m.start()) - 1, 0)
        last = bisect.bisect_left(word_starts, m.end()) - 1
        starts.add(first)
        label_words.update(range(first, last + 1))
    bounds = sorted(starts) + [len(word_starts)]
    return list(zip(bounds[:-1], bounds[1:])), label_words


def _ends_sentence(word):
    core = word.rstrip(_CLOSERS)
    if not core or core[-1] not in ".!?":
        return False
    if core[-1] == "." and (core.lower() in ABBREVIATIONS or re.fullmatch(r"[A-Z]\.", core)):
        return False  # "Dr." / "J." are not sentence ends
    return True


def _ends_comma(word):
    core = word.rstrip(_CLOSERS)
    return bool(core) and core[-1] == ","


# -------------------------
# Alignment
# -------------------------
def _norm(word):
    word = unicodedata.normalize("NFC", word)
    return (word.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
                .replace("–", "-").replace("—", "-"))


def is_redaction(word):
    return REDACTED in word


@dataclass
class Alignment:
    redacted: list  # per original word: removed by a redaction in this version
    region: list    # per original word: id of the many-to-many region it sits in, or -1
    boundary: list  # len n+1: redacted-file position for each original word boundary


def _lines(text):
    """Non-empty lines as (first word index, normalised words)."""
    out, pos = [], 0
    for line in text.splitlines():
        words = [_norm(w) for w in line.split()]
        if words:
            out.append((pos, words))
            pos += len(words)
    return out


def align(orig_text, red_text):
    """
    Word-level alignment of a redacted version against the original.
    Lines are matched first and words are only diffed inside lines that differ,
    which keeps whole-file alignment fast.
    """
    orig, red = orig_text.split(), red_text.split()
    n, m = len(orig), len(red)
    a = Alignment([False] * n, [-1] * n, [0] * (n + 1))
    o_lines, r_lines = _lines(orig_text), _lines(red_text)
    o_bounds = [p for p, _ in o_lines] + [n]
    r_bounds = [p for p, _ in r_lines] + [m]
    line_matcher = difflib.SequenceMatcher(
        a=[" ".join(w) for _, w in o_lines], b=[" ".join(w) for _, w in r_lines], autojunk=False)
    region_ids = iter(range(sys.maxsize))
    for tag, i1, i2, j1, j2 in line_matcher.get_opcodes():
        oi, oj, ri, rj = o_bounds[i1], o_bounds[i2], r_bounds[j1], r_bounds[j2]
        if tag == "equal":
            a.boundary[oi:oj] = range(ri, rj)
            continue
        word_matcher = difflib.SequenceMatcher(
            a=[_norm(w) for w in orig[oi:oj]], b=[_norm(w) for w in red[ri:rj]], autojunk=False)
        for op in word_matcher.get_opcodes():
            _apply_word_op(a, red, next(region_ids), op, oi, ri)
    a.boundary[n] = m
    return a


def _apply_word_op(a, red, rid, op, oi, ri):
    tag, i1, i2, j1, j2 = op
    i1, i2, j1, j2 = i1 + oi, i2 + oi, j1 + ri, j2 + ri
    if tag == "equal":
        a.boundary[i1:i2] = range(j1, j2)
    elif tag == "insert":
        # Added words go to the earlier chunk: the boundary points past them.
        a.boundary[i1] = j2
    elif tag == "replace" and i2 - i1 == j2 - j1:
        for k in range(i2 - i1):
            a.boundary[i1 + k] = j1 + k
            a.redacted[i1 + k] = is_redaction(red[j1 + k])
    else:
        _align_region(a, red, rid, i1, i2, j1, j2)


def _align_region(a, red, rid, i1, i2, j1, j2):
    """Unequal replace or delete: the whole region moves as one block."""
    has_redaction = any(is_redaction(w) for w in red[j1:j2])
    for i in range(i1, i2):
        a.region[i] = rid
        a.redacted[i] = has_redaction
        # A boundary inside the block gives the block to the chunk where it starts.
        a.boundary[i] = j1 if i == i1 else j2


# -------------------------
# Splitting long turns
# -------------------------
def _can_cut(i, alignments, label_words):
    """A cut before word i must not fall inside a label, a redacted span or a merged block."""
    if i - 1 in label_words:
        return False
    for a in alignments:
        if a.redacted[i - 1] and a.redacted[i]:
            return False
        if a.region[i - 1] != -1 and a.region[i - 1] == a.region[i]:
            return False
    return True


def _make_units(s, e, size, cuttable, words, max_tokens, level=0):
    """Break [s, e) into units <= max_tokens: sentences, then commas, then words."""
    rules = [_ends_sentence, _ends_comma, lambda w: True]
    if size(s, e) <= max_tokens or level == len(rules):
        return [(s, e)]
    cuts = [i for i in range(s + 1, e) if rules[level](words[i - 1]) and cuttable(i)]
    bounds = [s] + cuts + [e]
    units = []
    for x, y in zip(bounds[:-1], bounds[1:]):
        units.extend(_make_units(x, y, size, cuttable, words, max_tokens, level + 1))
    return units


def balanced_partition(sizes, cap):
    """
    Group consecutive sizes into the fewest groups that fit under `cap`, choosing
    cuts that minimise the largest group and then maximise the smallest one.
    A single size above `cap` (uncuttable) becomes a group of its own.
    Returns the group boundaries as unit indices [0, ..., len(sizes)].
    """
    n = len(sizes)
    prefix = [0]
    for s in sizes:
        prefix.append(prefix[-1] + s)
    for k in range(max(1, -(-prefix[-1] // cap)), n + 1):
        # best[g][i] = (largest group, -smallest group) for first i units in g groups
        best = [[None] * (n + 1) for _ in range(k + 1)]
        parent = [[0] * (n + 1) for _ in range(k + 1)]
        best[0][0] = (0, float("-inf"))
        for g in range(1, k + 1):
            for i in range(g, n + 1):
                for j in range(g - 1, i):
                    total = prefix[i] - prefix[j]
                    if best[g - 1][j] is None or (total > cap and i - j > 1):
                        continue
                    prev = best[g - 1][j]
                    cand = (max(prev[0], total), max(prev[1], -total))
                    if best[g][i] is None or cand < best[g][i]:
                        best[g][i], parent[g][i] = cand, j
        if best[k][n] is not None:
            bounds, i = [n], n
            for g in range(k, 0, -1):
                i = parent[g][i]
                bounds.append(i)
            return bounds[::-1]
    return list(range(n + 1))


def split_turn(s, e, size, cuttable, words, max_tokens):
    """Return [(start, end)] pieces of the turn [s, e)."""
    if size(s, e) <= max_tokens:
        return [(s, e)]
    units = _make_units(s, e, size, cuttable, words, max_tokens)
    bounds = balanced_partition([size(x, y) for x, y in units], max_tokens)
    return [(units[b0][0], units[b1 - 1][1]) for b0, b1 in zip(bounds[:-1], bounds[1:])]


# -------------------------
# Rows
# -------------------------
def _span_count(mask):
    return sum(1 for i, v in enumerate(mask) if v and (i == 0 or not mask[i - 1]))


def _details(words, s, positions):
    return json.dumps([{"pos": s + i, "word": words[s + i]} for i in positions], ensure_ascii=False)


def chunk_row(file, turn_idx, piece_idx, s, e, words, human, llm, size):
    hm, lm = human.redacted[s:e], llm.redacted[s:e]
    return {
        "chunk_id": f"{Path(file).stem}_t{turn_idx:03d}_p{piece_idx}",
        "file": file, "turn_idx": turn_idx, "piece_idx": piece_idx, "size": size(s, e),
        "orig_start": s, "orig_end": e,
        "human_start": human.boundary[s], "human_end": human.boundary[e],
        "llm_start": llm.boundary[s], "llm_end": llm.boundary[e],
        "llm_redacted_span_count": _span_count(lm), "llm_redacted_word_count": sum(lm),
        "human_redacted_span_count": _span_count(hm), "human_redacted_word_count": sum(hm),
        "tp_count": sum(h and l for h, l in zip(hm, lm)),
        "fp_count": sum(l and not h for h, l in zip(hm, lm)),
        "fn_count": sum(h and not l for h, l in zip(hm, lm)),
        "fp_details": _details(words, s, [i for i, (h, l) in enumerate(zip(hm, lm)) if l and not h]),
        "fn_details": _details(words, s, [i for i, (h, l) in enumerate(zip(hm, lm)) if h and not l]),
    }


def catalog_file(file, orig_text, human_text, llm_text, count, min_tokens, max_tokens):
    """Return (rows, n_excluded_turns) for one transcript."""
    words, starts = tokenize(orig_text)
    human = align(orig_text, human_text)
    llm = align(orig_text, llm_text)
    turns, label_words = find_turns(orig_text, starts)

    def size(s, e):
        return count(" ".join(words[s:e]))

    def cuttable(i):
        return _can_cut(i, (human, llm), label_words)

    rows, excluded = [], 0
    for turn_idx, (s, e) in enumerate(turns):
        if size(s, e) < min_tokens and not any(human.redacted[s:e]) and not any(llm.redacted[s:e]):
            excluded += 1
            continue
        for piece_idx, (ps, pe) in enumerate(split_turn(s, e, size, cuttable, words, max_tokens)):
            rows.append(chunk_row(file, turn_idx, piece_idx, ps, pe, words, human, llm, size))
    return rows, excluded


# -------------------------
# CLI
# -------------------------
def make_counter(spec):
    if spec == "words":
        return lambda text: len(text.split())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(spec)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def build(args):
    count = make_counter(args.tokenizer)
    print(f"Settings: min={args.min_tokens} max={args.max_tokens} tokenizer={args.tokenizer}")
    rows, n_files = [], 0
    for orig_path in sorted(Path(args.original).glob("*.txt")):
        human_path, llm_path = Path(args.human) / orig_path.name, Path(args.llm) / orig_path.name
        if not human_path.exists() or not llm_path.exists():
            print(f"[SKIP] {orig_path.name}: missing human or LLM version", file=sys.stderr)
            continue
        orig_text = orig_path.read_text(encoding="utf-8-sig")
        file_rows, excluded = catalog_file(
            orig_path.name, orig_text, human_path.read_text(encoding="utf-8-sig"),
            llm_path.read_text(encoding="utf-8-sig"), count, args.min_tokens, args.max_tokens)
        if not SPEAKER_RE.search(orig_text):
            print(f"[WARN] {orig_path.name}: no 'Speaker X:' labels found", file=sys.stderr)
        rows.extend(file_rows)
        n_files += 1
        print(f"[OK] {orig_path.name}: {len(file_rows)} chunks, {excluded} short turns excluded")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} chunks from {n_files} files to {args.out}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--original", required=True, help="folder of original .txt transcripts")
    p.add_argument("--human", required=True, help="folder of human-redacted versions (same file names)")
    p.add_argument("--llm", required=True, help="folder of LLM-redacted versions (same file names)")
    p.add_argument("--out", default="catalog.csv")
    p.add_argument("--min-tokens", type=int, required=True, help="shorter turns with no redactions are excluded")
    p.add_argument("--max-tokens", type=int, required=True, help="longer turns are split into pieces")
    p.add_argument("--tokenizer", required=True, help="Hugging Face model name, or 'words' to count words")
    return p.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
