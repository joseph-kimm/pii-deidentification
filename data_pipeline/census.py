"""
Step 1 of 3 (run first): census of speaker labels and timestamps, to check the
format before any statistics or catalog are trusted. Writes census.txt and census.csv.

For every file and version (original / human / LLM) it reports:
  - every line-start label it sees, and whether the active label rules match it
    (labels that are NOT matched are the ones to look at: they would be missed as turns)
  - timestamp counts: total, at end of line, mid-line
  - lines containing '-->' that the timestamp rule does not recognise
  - turns found and words before the first label
Then it flags files where the three versions disagree on labels or timestamps.

Usage:
    python census.py --original original --human human --llm llm --out-dir census_out
    add --labels Student,Guest to treat extra role names as labels
"""
import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import build_catalog as bc
import transcript_utils as tu

# Any "Word:" at the start of a line, whether or not it is an active label.
GENERIC_LABEL_RE = re.compile(r"^[ \t]*([A-Za-z][A-Za-z0-9 ]{0,30}?)[ \t]*:", re.M)
END_OF_LINE_TS_RE = re.compile(rf"{tu.TIMESTAMP_RE.pattern}[ \t]*$", re.M)
VERSIONS = ("original", "human", "llm")


def census_version(raw, label_re):
    """Return one census row (dict) for one version of one file."""
    labels = Counter(m.group(1).strip() for m in GENERIC_LABEL_RE.finditer(raw))
    unmatched = {name: n for name, n in labels.items() if not label_re.match(name + ":")}
    ts_total = len(tu.TIMESTAMP_RE.findall(raw))
    ts_eol = len(END_OF_LINE_TS_RE.findall(raw))
    arrow_lines = [line for line in raw.splitlines() if "-->" in line]
    unrecognised = [line for line in arrow_lines if not tu.TIMESTAMP_RE.search(line)]
    stripped = tu.strip_timestamps(raw)
    words, starts = bc.tokenize(stripped)
    turns, _ = bc.find_turns(stripped, starts, label_re)
    first_label = next(iter(label_re.finditer(stripped)), None)
    preamble_words = len(stripped[:first_label.start()].split()) if first_label else len(words)
    return {
        "labels": dict(labels),
        "unmatched_labels": unmatched,
        "ts_total": ts_total,
        "ts_end_of_line": ts_eol,
        "ts_mid_line": ts_total - ts_eol,
        "unrecognised_arrow_lines": unrecognised,
        "turns": len(turns),
        "words": len(words),
        "preamble_words": preamble_words,
    }


def census_file(name, texts, label_re):
    """texts: {version: raw text}. Returns (rows, flags) for one file."""
    rows = {v: census_version(texts[v], label_re) for v in VERSIONS}
    flags = []
    label_sets = {v: frozenset(rows[v]["labels"]) for v in VERSIONS}
    if len(set(label_sets.values())) > 1:
        flags.append("label sets differ across versions")
    has_ts = {v: rows[v]["ts_total"] > 0 for v in VERSIONS}
    if len(set(has_ts.values())) > 1:
        flags.append("timestamps present in some versions only: "
                     + ", ".join(f"{v}={rows[v]['ts_total']}" for v in VERSIONS))
    for v in VERSIONS:
        if rows[v]["unmatched_labels"]:
            flags.append(f"{v}: unmatched labels " + ", ".join(
                f"{k}({n})" for k, n in sorted(rows[v]["unmatched_labels"].items())))
        if rows[v]["unrecognised_arrow_lines"]:
            flags.append(f"{v}: {len(rows[v]['unrecognised_arrow_lines'])} '-->' lines not recognised as timestamps")
        if rows[v]["turns"] == 0:
            flags.append(f"{v}: no turns found")
    if rows["original"]["words"] and rows["llm"]["words"]:
        ratio = rows["llm"]["words"] / rows["original"]["words"]
        if not 0.9 <= ratio <= 1.1:
            flags.append(f"LLM word count is {ratio:.0%} of original")
    return rows, flags


def load_texts(paths, name):
    return {v: paths[v].joinpath(name).read_text(encoding="utf-8-sig") for v in VERSIONS}


def run(args):
    label_re = tu.label_regex(args.labels)
    folders = {"original": Path(args.original), "human": Path(args.human), "llm": Path(args.llm)}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    global_labels, lines, csv_rows = Counter(), [], []
    for orig in sorted(folders["original"].glob("*.txt")):
        if not all((folders[v] / orig.name).exists() for v in VERSIONS[1:]):
            lines.append(f"[SKIP] {orig.name}: missing human or LLM version")
            continue
        rows, flags = census_file(orig.name, load_texts(folders, orig.name), label_re)
        global_labels.update(rows["original"]["labels"])
        for v in VERSIONS:
            r = rows[v]
            csv_rows.append({"file": orig.name, "version": v, "turns": r["turns"], "words": r["words"],
                             "preamble_words": r["preamble_words"], "ts_total": r["ts_total"],
                             "ts_end_of_line": r["ts_end_of_line"], "ts_mid_line": r["ts_mid_line"],
                             "labels": "; ".join(f"{k}={n}" for k, n in sorted(r["labels"].items())),
                             "unmatched_labels": "; ".join(f"{k}={n}" for k, n in sorted(r["unmatched_labels"].items()))})
        lines.append(f"{orig.name}: " + (" | ".join(flags) if flags else "no flags"))
    header = [f"Census of {args.original} / {args.human} / {args.llm}",
              f"Active label rules: Coach, Teacher, Speaker X (+ {', '.join(args.labels) or 'none extra'})",
              "", "Label counts (original, all line-start labels):"]
    header += [f"  {k}: {n}" for k, n in global_labels.most_common()] or ["  (none)"]
    text = "\n".join(header + ["", "Per-file flags:"] + [f"  {l}" for l in lines]) + "\n"
    (out_dir / "census.txt").write_text(text, encoding="utf-8")
    with open(out_dir / "census.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else ["file"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(text)
    print(f"Wrote {out_dir / 'census.txt'} and {out_dir / 'census.csv'}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--original", required=True)
    p.add_argument("--human", required=True)
    p.add_argument("--llm", required=True)
    p.add_argument("--out-dir", default="census_out")
    p.add_argument("--labels", type=lambda s: [x for x in s.split(",") if x.strip()], default=[])
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
