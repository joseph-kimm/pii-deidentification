"""
Step 3 of 3: statistics for chunk sizing and redaction patterns. Run census.py first.

Writes to --out-dir:
  summary.md             tables to paste back (per-file titles, turn lengths, chunk-size
                         candidates with a suggested --min-tokens / --max-tokens, over- and
                         under-redaction, multi-word spans, surprises)
  per_file.csv           one row per transcript
  turn_lengths.csv       one row per speaker turn
  overredacted_words.csv     words the LLM redacted and the human did not
  underredacted_words.csv    words the human redacted and the LLM did not
  multiword_spans.csv    each run of 2+ redacted words, per version, with its style

Definitions (all counts are on timestamp-free text):
  Turn size   = words in the turn, label token included (same as build_catalog.py).
  Over-redaction  = LLM redacted a word the human did not (LLM false positive).
  Under-redaction = human redacted a word the LLM did not (LLM false negative).
  Title       = first non-empty line before the first speaker label in the original.
  Multi-word style, per run of k redacted original words:
      per-word  = k '[REDACTED]' tokens in the redacted text
      collapsed = 1 '[REDACTED]' token
      mixed     = 2..k-1 tokens
  The style is approximate when the alignment merged the span into a larger block.

Usage:
    python analyze_transcripts.py --original original --human human --llm llm --out-dir analysis
    --min-tokens / --max-tokens-cap / --fit-pct / --clean-words-pct tune the suggestion
"""
import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import build_catalog as bc
import transcript_utils as tu

VERSIONS = ("original", "human", "llm")
MAX_CANDIDATES = (200, 300, 400, 600, 800, 1000)
MIN_CANDIDATES = (10, 25, 50, 100)
HIST_EDGES = (0, 25, 50, 100, 200, 400, 600, 800, 1000, math.inf)
PUNCT = ".,!?;:\"'()[]{}-—–"
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "had", "has", "have",
    "he", "her", "his", "i", "if", "in", "is", "it", "its", "me", "my", "not", "of", "on", "or",
    "our", "she", "so", "that", "the", "their", "them", "then", "there", "they", "this", "to", "us",
    "was", "we", "were", "what", "when", "which", "who", "will", "with", "you", "your",
    "yeah", "yes", "no", "ok", "okay", "um", "uh", "mm", "hmm",
}


def word_shape(word):
    core = word.strip(PUNCT)
    if any(c.isdigit() for c in core):
        return "contains digit"
    if core.lower() in STOPWORDS:
        return "stopword-like"
    if len(core) > 1 and core.isupper():
        return "ALLCAPS"
    if core[:1].isupper():
        return "Capitalized"
    return "lowercase/other"


def extract_title(raw, label_re):
    """First non-empty line before the first speaker label, or '' if none."""
    first = label_re.search(raw)
    preamble = raw[:first.start()] if first else ""
    return next((line.strip() for line in preamble.splitlines() if line.strip()), "")


def redaction_runs(mask):
    """Maximal runs of True in mask, as (start, end) with end exclusive."""
    runs, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def overlaps(run, mask):
    return any(mask[i] for i in range(*run))


def all_in(run, mask):
    return all(mask[i] for i in range(*run))


def span_style(side_tokens, boundary, run):
    """Style of one redacted run, from the redacted-side tokens it maps to."""
    k = run[1] - run[0]
    start, end = boundary[run[0]], boundary[run[1]]
    r = sum(1 for tok in side_tokens[start:end] if bc.REDACTED in tok)
    if r == 0:
        return "unclear (0 tokens)"
    if r >= k:
        return "per-word"
    if r == 1:
        return "collapsed"
    return "mixed"


def analyze_file(name, texts, label_re, cfg):
    """Return a dict with per-file rows and word-level findings for one transcript."""
    orig, human, llm = (tu.strip_timestamps(texts[v]) for v in VERSIONS)
    words, starts = bc.tokenize(orig)
    side_tokens = {"human": bc.tokenize(human)[0], "llm": bc.tokenize(llm)[0]}
    h, l = bc.align(orig, human), bc.align(orig, llm)
    turns, label_words = bc.find_turns(orig, starts, label_re)
    n = len(words)
    turn_of = [-1] * n
    for t, (s, e) in enumerate(turns):
        turn_of[s:e] = [t] * (e - s)

    def cuttable(i):
        return bc._can_cut(i, (h, l), label_words)

    turn_rows, piece_lengths = [], {m: [] for m in MAX_CANDIDATES}
    for t, (s, e) in enumerate(turns):
        hr, lr = sum(h.redacted[s:e]), sum(l.redacted[s:e])
        kept = (e - s) >= cfg.min_tokens or hr > 0 or lr > 0
        turn_rows.append({"file": name, "turn_idx": t, "size": e - s, "human_redacted_words": hr,
                          "llm_redacted_words": lr, "kept": kept,
                          "text": " ".join(words[s:e])[:200]})
        if kept:
            for m in MAX_CANDIDATES:
                for ps, pe in bc.split_turn(s, e, lambda a, b: b - a, cuttable, words, m):
                    piece_lengths[m].append(pe - ps)

    over = [i for i in range(n) if l.redacted[i] and not h.redacted[i]]
    under = [i for i in range(n) if h.redacted[i] and not l.redacted[i]]
    h_runs, l_runs = redaction_runs(h.redacted), redaction_runs(l.redacted)
    over_spans = [r for r in l_runs if not overlaps(r, h.redacted)]
    under_spans = [r for r in h_runs if not overlaps(r, l.redacted)]
    partial_llm = sum(1 for r in l_runs if overlaps(r, h.redacted) and not all_in(r, h.redacted))
    partial_human = sum(1 for r in h_runs if overlaps(r, l.redacted) and not all_in(r, l.redacted))

    spans, variants, glued = [], {"human": Counter(), "llm": Counter()}, {"human": Counter(), "llm": Counter()}
    crossing = Counter()
    for side, mask, runs, alignment in (("human", h.redacted, h_runs, h),
                                        ("llm", l.redacted, l_runs, l)):
        for tok in side_tokens[side]:
            if bc.REDACTED in tok and tok != bc.REDACTED:
                glued[side][tok] += 1
            elif "redact" in tok.lower() and tok != bc.REDACTED:
                variants[side][tok] += 1
        for run in runs:
            tids = {turn_of[i] for i in range(*run) if turn_of[i] >= 0}
            if len(tids) > 1:
                crossing[side] += 1
            if run[1] - run[0] >= 2:
                spans.append({"file": name, "side": side, "start_word": run[0], "length": run[1] - run[0],
                              "style": span_style(side_tokens[side], alignment.boundary, run),
                              "example": " ".join(words[run[0]:run[1]])})

    def word_row(i, side):
        lo, hi = max(0, i - 5), min(n, i + 6)
        ctx = " ".join(words[lo:i] + [f"[{words[i]}]"] + words[i + 1:hi])
        return {"file": name, "word": words[i].strip(PUNCT) or words[i], "shape": word_shape(words[i]),
                "turn_idx": turn_of[i], "context": ctx, "side": side}

    return {
        "file": name,
        "title": extract_title(texts["original"], label_re),
        "turn_rows": turn_rows,
        "piece_lengths": piece_lengths,
        "words": n,
        "turns": len(turns),
        "preamble_words": turns[0][0] if turns else n,
        "human_redacted": sum(h.redacted), "llm_redacted": sum(l.redacted),
        "human_runs": len(h_runs), "llm_runs": len(l_runs),
        "over_words": [word_row(i, "over") for i in over],
        "under_words": [word_row(i, "under") for i in under],
        "over_spans": len(over_spans), "under_spans": len(under_spans),
        "partial_llm": partial_llm, "partial_human": partial_human,
        "spans": spans, "variants": variants, "glued": glued, "crossing": crossing,
        "llm_word_ratio": len(bc.tokenize(llm)[0]) / n if n else 0,
        "human_word_ratio": len(bc.tokenize(human)[0]) / n if n else 0,
        "label_words": len(label_words),
    }


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def fmt(x, digits=1):
    return "-" if x is None else f"{x:.{digits}f}"


def turn_stats(sizes):
    vals = sorted(sizes)
    if not vals:
        return ["0", "-", "-", "-", "-", "-", "-", "-", "-"]
    return [str(len(vals)), fmt(sum(vals) / len(vals)), fmt(percentile(vals, 50)), fmt(percentile(vals, 10)),
            fmt(percentile(vals, 25)), fmt(percentile(vals, 75)), fmt(percentile(vals, 90)),
            fmt(percentile(vals, 95)), fmt(percentile(vals, 99))]


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def suggest(turns, pieces, cfg, clean_words_total):
    """Return (max_table, min_table, suggested_line) for the chunk-size candidates."""
    kept = [t for t in turns if t["kept"]]
    max_rows, best_max = [], None
    for m in MAX_CANDIDATES:
        lens = pieces[m]
        unsplit = sum(1 for t in kept if t["size"] <= m) / len(kept) * 100 if kept else 0
        max_rows.append([m, len(kept), f"{unsplit:.1f}", len(lens),
                         fmt(sum(lens) / len(lens)) if lens else "-", min(lens) if lens else "-"])
        if m <= cfg.max_tokens_cap and best_max is None and unsplit >= cfg.fit_pct:
            best_max = (m, unsplit, len(lens))
    min_rows, best_min = [], None
    for m in MIN_CANDIDATES:
        dropped = [t for t in turns if t["size"] < m and t["human_redacted_words"] == 0 and t["llm_redacted_words"] == 0]
        dropped_words = sum(t["size"] for t in dropped)
        share = dropped_words / clean_words_total * 100 if clean_words_total else 0
        min_rows.append([m, len(dropped), dropped_words, f"{share:.1f}"])
        if share <= cfg.clean_words_pct:
            best_min = (m, share)
    if best_max and best_min:
        line = (f"Suggested: --min-tokens {best_min[0]} --max-tokens {best_max[0]} "
                f"({best_max[1]:.1f}% of kept turns unsplit, {best_max[2]} chunks, "
                f"{best_min[1]:.1f}% of clean words dropped)")
    else:
        line = ("No candidate meets the thresholds. Lower --fit-pct, raise --clean-words-pct, "
                "or raise --max-tokens-cap.")
    return max_rows, min_rows, line


def word_table(rows, top=20):
    grouped = defaultdict(lambda: {"count": 0, "turns": set(), "files": set(), "shape": "", "context": ""})
    for r in rows:
        g = grouped[r["word"]]
        g["count"] += 1
        g["turns"].add((r["file"], r["turn_idx"]))
        g["files"].add(r["file"])
        g["shape"] = g["shape"] or r["shape"]
        g["context"] = g["context"] or r["context"]
    ranked = sorted(grouped.items(), key=lambda kv: (-kv[1]["count"], kv[0]))
    return ranked, [[w, g["shape"], g["count"], len(g["turns"]), len(g["files"]), g["context"]]
                    for w, g in ranked[:top]]


def build_summary(results, skipped, cfg, turn_rows, over_rows, under_rows, span_rows, suggestion):
    max_rows, min_rows, line = suggestion
    total_words = sum(r["words"] for r in results)
    total_llm = sum(r["llm_redacted"] for r in results)
    total_human = sum(r["human_redacted"] for r in results)
    total_over = len(over_rows)
    total_under = len(under_rows)
    over_spans = sum(r["over_spans"] for r in results)
    under_spans = sum(r["under_spans"] for r in results)
    partial_llm = sum(r["partial_llm"] for r in results)
    partial_human = sum(r["partial_human"] for r in results)
    precision = (total_llm - total_over) / total_llm * 100 if total_llm else 0
    recall = (total_human - total_under) / total_human * 100 if total_human else 0
    sizes_all = [t["size"] for t in turn_rows]
    sizes_red = [t["size"] for t in turn_rows if t["human_redacted_words"] or t["llm_redacted_words"]]
    sizes_clean = [t["size"] for t in turn_rows if not (t["human_redacted_words"] or t["llm_redacted_words"])]
    hist = []
    for lo, hi in zip(HIST_EDGES[:-1], HIST_EDGES[1:]):
        hist.append([f"{lo}-{'inf' if hi == math.inf else int(hi)}",
                     sum(1 for s in sizes_all if lo <= s < hi)])

    per_file_rows = []
    for r in results:
        ratio_flag = "" if 0.9 <= r["llm_word_ratio"] <= 1.1 else f"LLM words {r['llm_word_ratio']:.0%} of orig"
        per_file_rows.append([r["file"], r["title"] or "(none)", r["turns"], r["words"], r["human_redacted"],
                              r["llm_redacted"], len(r["over_words"]), len(r["under_words"]),
                              f"{r['llm_word_ratio']:.2f}", ratio_flag])

    style_counts = {side: Counter(s["style"] for s in span_rows if s["side"] == side) for side in ("human", "llm")}
    run_len = {side: Counter(min(s["length"], 5) for s in span_rows if s["side"] == side) for side in ("human", "llm")}
    odd = {side: Counter() for side in ("human", "llm")}
    glued = {side: Counter() for side in ("human", "llm")}
    crossing = {side: sum(r["crossing"][side] for r in results) for side in ("human", "llm")}
    for r in results:
        for side in ("human", "llm"):
            odd[side].update(r["variants"][side])
            glued[side].update(r["glued"][side])

    over_ranked, over_top = word_table(over_rows)
    under_ranked, under_top = word_table(under_rows)
    shapes_over = Counter(r["shape"] for r in over_rows)
    shapes_under = Counter(r["shape"] for r in under_rows)

    lines = [
        "# Transcript analysis",
        "",
        f"Files analysed: {len(results)}" + (f" (skipped, missing a version: {', '.join(skipped)})" if skipped else ""),
        f"Speaker labels: {', '.join(tu.DEFAULT_ROLES)}, Speaker X" + (f", {', '.join(cfg.labels)}" if cfg.labels else "")
        + ". Timestamps removed before counting. Words are whitespace tokens.",
        "",
        "## Per file",
        table(["file", "title (first line before first label)", "turns", "words", "human redacted words",
               "LLM redacted words", "over", "under", "LLM/orig words", "flag"], per_file_rows),
        "",
        "## Turn length (words, label included)",
        table(["group", "turns", "mean", "median", "p10", "p25", "p75", "p90", "p95", "p99"], [
            ["all turns", *turn_stats(sizes_all)],
            ["with any redaction", *turn_stats(sizes_red)],
            ["no redaction (either version)", *turn_stats(sizes_clean)],
        ]),
        "",
        table(["size range", "turns"], hist),
        "",
        "## Chunk-size candidates",
        f"Max table uses --min-tokens {cfg.min_tokens} for the kept set. Kept = size >= min, or any redaction.",
        table(["--max-tokens", "kept turns", "% unsplit", "chunks", "mean chunk", "min chunk"], max_rows),
        "",
        table(["--min-tokens", "turns dropped", "words dropped", "% of clean words dropped"], min_rows),
        "",
        f"**{line}**",
        "",
        "## Over-redaction (LLM redacted, human did not)",
        f"- Words: {total_over} of {total_llm} LLM-redacted words ({total_over / total_llm:.1%} false positives)"
        if total_llm else "- No LLM redactions.",
        f"- Spans (runs): {over_spans} wholly over-redacted, {partial_llm} partly overlapping the human span",
        f"- Shape of over-redacted words: " + ", ".join(f"{k} {v}" for k, v in shapes_over.most_common()),
        f"- LLM word-level precision: {precision:.1f}%",
        "",
        table(["word", "shape", "count", "turns", "files", "example context"], over_top),
        "",
        "## Under-redaction (human redacted, LLM did not)",
        f"- Words: {total_under} of {total_human} human-redacted words ({total_under / total_human:.1%} missed)"
        if total_human else "- No human redactions.",
        f"- Spans (runs): {under_spans} wholly missed, {partial_human} partly overlapping the LLM span",
        f"- Shape of missed words: " + ", ".join(f"{k} {v}" for k, v in shapes_under.most_common()),
        f"- LLM word-level recall: {recall:.1f}%",
        "",
        table(["word", "shape", "count", "turns", "files", "example context"], under_top),
        "",
        "## Multi-word redactions (runs of 2+ words)",
        table(["version", "runs of 2", "runs of 3", "runs of 4", "runs of 5+", "per-word", "collapsed",
               "mixed", "unclear"], [
            [side, run_len[side][2], run_len[side][3], run_len[side][4], run_len[side][5],
             style_counts[side]["per-word"], style_counts[side]["collapsed"], style_counts[side]["mixed"],
             style_counts[side]["unclear (0 tokens)"]]
            for side in ("human", "llm")]),
        "",
        "## Surprises",
        f"- Runs that cross a speaker turn: human {crossing['human']}, LLM {crossing['llm']}",
        f"- Redaction tokens that are not exactly `[REDACTED]` (human): "
        + (", ".join(f"`{t}` x{c}" for t, c in odd["human"].most_common(10)) or "none"),
        f"- Redaction tokens that are not exactly `[REDACTED]` (LLM): "
        + (", ".join(f"`{t}` x{c}" for t, c in odd["llm"].most_common(10)) or "none"),
        f"- `[REDACTED]` glued to other characters (human): "
        + (", ".join(f"`{t}` x{c}" for t, c in glued["human"].most_common(10)) or "none"),
        f"- `[REDACTED]` glued to other characters (LLM): "
        + (", ".join(f"`{t}` x{c}" for t, c in glued["llm"].most_common(10)) or "none"),
        f"- Files with a missing title (no text before first label): "
        + str(sum(1 for r in results if not r["title"])),
        f"- Files whose LLM word count is outside 90-110% of the original: "
        + (", ".join(r["file"] for r in results if not 0.9 <= r["llm_word_ratio"] <= 1.1) or "none"),
        f"- Files with no speaker turns: "
        + (", ".join(r["file"] for r in results if r["turns"] == 0) or "none"),
        f"- Words before the first label (not in any turn): {sum(r['preamble_words'] for r in results)}",
        "",
        "Detail: per_file.csv, turn_lengths.csv, overredacted_words.csv, underredacted_words.csv, multiword_spans.csv.",
        "",
    ]
    return "\n".join(lines), ranked_words_for_csv(over_ranked), ranked_words_for_csv(under_ranked)


def ranked_words_for_csv(ranked):
    return [{"word": w, "shape": g["shape"], "count": g["count"], "turns": len(g["turns"]),
             "files": len(g["files"]), "example_context": g["context"]} for w, g in ranked]


def run(args):
    label_re = tu.label_regex(args.labels)
    folders = {"original": Path(args.original), "human": Path(args.human), "llm": Path(args.llm)}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results, skipped = [], []
    for orig in sorted(folders["original"].glob("*.txt")):
        if not all((folders[v] / orig.name).exists() for v in VERSIONS[1:]):
            skipped.append(orig.name)
            print(f"[SKIP] {orig.name}: missing human or LLM version", file=sys.stderr)
            continue
        texts = {v: (folders[v] / orig.name).read_text(encoding="utf-8-sig") for v in VERSIONS}
        results.append(analyze_file(orig.name, texts, label_re, args))
        print(f"[OK] {orig.name}")

    turn_rows = [t for r in results for t in r["turn_rows"]]
    span_rows = [s for r in results for s in r["spans"]]
    over_rows = [w for r in results for w in r["over_words"]]
    under_rows = [w for r in results for w in r["under_words"]]
    pieces = {m: [x for r in results for x in r["piece_lengths"][m]] for m in MAX_CANDIDATES}
    clean_total = sum(t["size"] for t in turn_rows if not (t["human_redacted_words"] or t["llm_redacted_words"]))
    suggestion = suggest(turn_rows, pieces, args, clean_total)
    summary, over_csv, under_csv = build_summary(results, skipped, args, turn_rows, over_rows, under_rows,
                                                 span_rows, suggestion)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    write_csv(out_dir / "per_file.csv", [{"file": r["file"], "title": r["title"], "turns": r["turns"],
                                          "words": r["words"], "human_redacted": r["human_redacted"],
                                          "llm_redacted": r["llm_redacted"], "over_words": len(r["over_words"]),
                                          "under_words": len(r["under_words"]),
                                          "llm_word_ratio": round(r["llm_word_ratio"], 4)}
                                         for r in results],
              ["file", "title", "turns", "words", "human_redacted", "llm_redacted", "over_words",
               "under_words", "llm_word_ratio"])
    write_csv(out_dir / "turn_lengths.csv", turn_rows,
              ["file", "turn_idx", "size", "human_redacted_words", "llm_redacted_words", "kept", "text"])
    word_fields = ["word", "shape", "count", "turns", "files", "example_context"]
    write_csv(out_dir / "overredacted_words.csv", over_csv, word_fields)
    write_csv(out_dir / "underredacted_words.csv", under_csv, word_fields)
    write_csv(out_dir / "multiword_spans.csv", span_rows, ["file", "side", "start_word", "length", "style", "example"])
    print(summary)
    print(f"Wrote {out_dir / 'summary.md'} and CSV detail files in {out_dir}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--original", required=True)
    p.add_argument("--human", required=True)
    p.add_argument("--llm", required=True)
    p.add_argument("--out-dir", default="analysis")
    p.add_argument("--labels", type=lambda s: [x for x in s.split(",") if x.strip()], default=[])
    p.add_argument("--min-tokens", type=int, default=50, help="used for the kept-turn set in the max table")
    p.add_argument("--max-tokens-cap", type=int, default=600, help="largest max-tokens the suggestion may pick")
    p.add_argument("--fit-pct", type=float, default=90.0, help="%% of kept turns that must fit unsplit")
    p.add_argument("--clean-words-pct", type=float, default=10.0,
                   help="%% of clean (redaction-free) words the min-tokens cut may drop")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
