import re
import difflib
import unicodedata
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    cohen_kappa_score, confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "synthetic"
RESULTS_DIR = PROJECT_ROOT / "results" / "deidentify"
AI_FOLDER = RESULTS_DIR / "new"
HUMAN_FOLDER = DATA_DIR / "human_redacted"
# Unredacted transcripts (same filenames as HUMAN_FOLDER). Gold and AI labels are both
# projected onto these words; a file without its original falls back to positional alignment.
ORIGINAL_FOLDER = DATA_DIR / "original"
# Written by de_identify_steplab.py; supplies per-file span-expansion counts if present.
STATS_CSV = RESULTS_DIR / "processing_statistics.csv"
CSV_INPUT: Optional[Path] = None
OUTPUT_XLSX = RESULTS_DIR / "evaluation_results_llama3.3 all.xlsx"
OUTPUT_CSV  = RESULTS_DIR / "evaluation_results_llama3.3 all.csv"


NORMALIZE_FILES_ON_DISK = False
NORMALIZED_OUT_DIR = RESULTS_DIR / "normalized_utf8"  


REDACT_PATTERNS = [
    re.compile(r"\[REDACTED\]", re.I),
]

def is_punct(ch: str) -> bool:
   
    return unicodedata.category(ch).startswith('P')

def strip_punct_edges(s: str) -> str:
    
    if not s:
        return s
    i, j = 0, len(s)
    while i < j and is_punct(s[i]):
        i += 1
    while j > i and is_punct(s[j-1]):
        j -= 1
    return s[i:j].strip()

def remove_all_punct(s: str) -> str:
   
    return "".join(ch for ch in s if not is_punct(ch))

def is_redacted_token(tok: str) -> bool:
   
    orig = tok.strip()
    if not orig:
        return False
    cand1 = orig
    cand2 = strip_punct_edges(orig)
    cand3 = remove_all_punct(orig)

    for pat in REDACT_PATTERNS:
        if pat.search(cand1):
            return True

    return False

def tokenize(text: Optional[str]):
    return (text or "").split()

def get_labels(text: Optional[str]):
    
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    return [1 if is_redacted_token(tok) else 0 for tok in tokenize(text)]

def project_redaction_labels(original_tokens, other_tokens):
    """
    Per-ORIGINAL-token 0/1 labels: 1 where `other` redacted or dropped that word.

    The human gold collapses a multi-word PII span ("Lincoln High") into a single
    [REDACTED], while the harness writes one [REDACTED] per word. Labelling each text
    by its own tokens and comparing position-by-position therefore drifts out of
    alignment after the first collapsed span. Projecting both texts onto the original
    transcript with difflib puts gold and prediction on the same index space.
    """
    labels = [0] * len(original_tokens)
    sm = difflib.SequenceMatcher(a=original_tokens, b=other_tokens, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i1, i2):
                if is_redacted_token(other_tokens[j1 + (k - i1)]):
                    labels[k] = 1
        elif tag == "delete":
            for k in range(i1, i2):
                labels[k] = 1
        elif tag == "replace" and any(is_redacted_token(t) for t in other_tokens[j1:j2]):
            for k in range(i1, i2):
                labels[k] = 1
    return labels

def count_redacted_tokens(text: Optional[str]) -> float:
   
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return np.nan
    toks = tokenize(text)
    return float(sum(1 for tok in toks if is_redacted_token(tok)))


PREFERRED_READ_ORDER = ["utf-8", "utf-8-sig", "cp1252", "latin-1", "mac_roman"]

def _detect_with_charset_normalizer(raw: bytes) -> Optional[str]:
    try:
        from charset_normalizer import from_bytes  # type: ignore
    except Exception:
        return None
    try:
        res = from_bytes(raw)
        if not res:
            return None
        best = res.best()
        if best and best.encoding:
            return best.encoding
    except Exception:
        return None
    return None

def _clean_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("\u00A0", " ")
    t = "".join(ch for ch in t if (ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != 'C'))
    return t

def read_text_unified(path: Path) -> Tuple[Optional[str], Optional[str]]:
   
    try:
        raw = path.read_bytes()
    except Exception:
        return None, None

    
    for enc in PREFERRED_READ_ORDER:
        try:
            txt = raw.decode(enc)
            return _clean_text(txt), enc
        except Exception:
            continue

    
    enc = _detect_with_charset_normalizer(raw)
    if enc:
        try:
            txt = raw.decode(enc, errors="strict")
            return _clean_text(txt), enc
        except Exception:
            pass

    
    try:
        txt = raw.decode("latin-1", errors="replace")
        return _clean_text(txt), "latin-1"
    except Exception:
        return None, None

def ensure_dir_for(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def parse_model_prompt_from_stem(ai_stem: str, human_stem: str):
    prefix = human_stem + "__"
    if not ai_stem.startswith(prefix):
        return "", ""
    suffix = ai_stem[len(prefix):]
    parts = suffix.split("__", 1)
    prompt_label = parts[0] if len(parts) > 0 else ""
    model = parts[1] if len(parts) > 1 else ""
    return model, prompt_label

def paired_texts_from_folders(ai_folder: Path, human_folder: Path):
    pairs = []
    if not ai_folder.exists() or not human_folder.exists():
        return pairs
    ai_files = list(ai_folder.glob("*.txt"))
    for hfile in human_folder.glob("*.txt"):
        human_stem = hfile.stem
        matches = [p for p in ai_files if p.stem == human_stem or p.stem.startswith(human_stem + "__")]
        original_path = ORIGINAL_FOLDER / hfile.name
        original_text = read_text_unified(original_path)[0] if original_path.exists() else None
        for ai_path in matches:
            try:
                human_text, human_enc = read_text_unified(hfile)
                ai_text, ai_enc = read_text_unified(ai_path)
                model, prompt_label = parse_model_prompt_from_stem(ai_path.stem, human_stem)

                if NORMALIZE_FILES_ON_DISK:
                    out_h = NORMALIZED_OUT_DIR / "human_redacted" / hfile.name
                    out_a = NORMALIZED_OUT_DIR / "new" / ai_path.name
                    ensure_dir_for(out_h); ensure_dir_for(out_a)
                    if human_text is not None:
                        out_h.write_text(human_text, encoding="utf-8")
                    if ai_text is not None:
                        out_a.write_text(ai_text, encoding="utf-8")

                pairs.append((ai_path.name, human_text, ai_text, human_enc, ai_enc, model, prompt_label, original_text))
            except Exception:
                continue
    return pairs

def paired_texts_from_csv(csv_path: Path):
   
    raw = csv_path.read_bytes()
    enc = _detect_with_charset_normalizer(raw) or "utf-8"
    try:
        df = pd.read_csv(csv_path, encoding=enc)
    except Exception:
        
        for e in PREFERRED_READ_ORDER:
            try:
                df = pd.read_csv(csv_path, encoding=e)
                enc = e
                break
            except Exception:
                continue
        else:
            df = pd.read_csv(csv_path, encoding="latin-1", errors="ignore")
            enc = "latin-1"

    required = {"filename", "human_text", "ai_text"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {required}")

    rows = []
    for _, r in df.iterrows():
        filename = r["filename"]
        human_text = r["human_text"]
        ai_text = r["ai_text"]

        if isinstance(human_text, str):
            human_text = _clean_text(human_text)
            human_enc = enc
        else:
            human_enc = None

        if isinstance(ai_text, str):
            ai_text = _clean_text(ai_text)
            ai_enc = enc
        else:
            ai_enc = None

        # Optional column: without it the row is scored with positional alignment.
        original_text = r.get("original_text")
        original_text = _clean_text(original_text) if isinstance(original_text, str) else None

        rows.append((filename, human_text, ai_text, human_enc, ai_enc, "", "", original_text))
    return rows

def align_labels(h_labels, a_labels):
    n = min(len(h_labels), len(a_labels))
    return h_labels[:n], a_labels[:n]

def precision_recall(tp: int, fp: int, fn: int):
    """
    Precision/recall with well-defined edge cases instead of NaN.

    A transcript with no PII in the gold (human_pos == 0) is a real test case, not an
    undefined one: recall is vacuously 1.0 (nothing to miss), and precision is 1.0 if
    the model also redacted nothing, 0.0 if it redacted something that wasn't PII.
    NaN is kept only for the genuinely undefined case: PII existed but the model made
    zero predictions.
    """
    human_pos, pred_pos = tp + fn, tp + fp
    if human_pos == 0:
        return (1.0 if pred_pos == 0 else 0.0), 1.0
    prec = (tp / pred_pos) if pred_pos > 0 else np.nan
    return prec, tp / human_pos

def compute_metrics(y_true, y_pred):

    if len(y_true) == 0:
        return {
            "accuracy": None, "precision": None, "recall": None, "kappa": None,
            "tp": 0, "tn": 0, "fp": 0, "fn": 0, "total_words": 0
        }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec, rec = precision_recall(tp, fp, fn)

    # Robust kappa
    if len(set(y_true)) == 1 and len(set(y_pred)) == 1:
        kappa = 1.0 if y_true[0] == y_pred[0] else 0.0
    else:
        try:
            kappa = cohen_kappa_score(y_true, y_pred)
        except Exception:
            kappa = 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "kappa": kappa,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "total_words": int(len(y_true)),
    }

def load_span_expansion_stats(stats_csv: Path):
    """
    Map AI output filename -> (redacted_span_expansions, redacted_extra_tokens) from
    de_identify_steplab.py's processing_statistics.csv. The output filename is rebuilt
    the same way that script names it: {stem}__{safe_prompt}__{safe_model}{ext}.
    """
    if not stats_csv.exists():
        return {}
    stats = pd.read_csv(stats_csv)
    if "redacted_span_expansions" not in stats.columns:
        return {}
    lookup = {}
    for _, r in stats.iterrows():
        stem, ext = Path(str(r["filename"])).stem, Path(str(r["filename"])).suffix
        safe_prompt = re.sub(r"[^\w\-]", "_", str(r.get("prompt_label", "")))
        safe_model = re.sub(r"[^\w\-]", "_", str(r.get("model_name", "")))
        lookup[f"{stem}__{safe_prompt}__{safe_model}{ext}"] = (
            r["redacted_span_expansions"], r["redacted_extra_tokens"]
        )
    return lookup

def evaluate(pairs, span_stats=None):
    """
    Score each (gold, AI) pair. Returns the per-file DataFrame plus the concatenated
    word labels per (model, prompt_label) so corpus metrics can be computed per run.
    """
    span_stats = span_stats or {}
    rows = []
    labels_by_run = {}
    for filename, human_text, ai_text, human_enc, ai_enc, model, prompt_label, original_text in pairs:
        if original_text is not None:
            original_tokens = tokenize(original_text)
            y_true = project_redaction_labels(original_tokens, tokenize(human_text))
            y_pred = project_redaction_labels(original_tokens, tokenize(ai_text))
            alignment = "projected"
        else:
            print(f"[WARN] no original transcript for {filename}; using positional alignment")
            y_true = get_labels(human_text)
            y_pred = get_labels(ai_text)
            alignment = "positional"
        y_true, y_pred = align_labels(y_true, y_pred)

        run_true, run_pred = labels_by_run.setdefault((model, prompt_label), ([], []))
        run_true.extend(y_true)
        run_pred.extend(y_pred)

        m = compute_metrics(y_true, y_pred)
        m["filename"] = str(filename)
        m["model"] = model
        m["prompt_label"] = prompt_label
        m["alignment"] = alignment
        m["human_redacted_count"] = count_redacted_tokens(human_text)
        m["ai_redacted_count"]    = count_redacted_tokens(ai_text)
        expansions, extra_tokens = span_stats.get(str(filename), (np.nan, np.nan))
        m["ai_redacted_span_expansions"] = expansions
        m["ai_redacted_extra_tokens"] = extra_tokens
        m["human_encoding"] = human_enc
        m["ai_encoding"] = ai_enc
        rows.append(m)

    df = pd.DataFrame(rows, columns=[
        "filename","model","prompt_label","alignment","accuracy","precision","recall","kappa",
        "tp","tn","fp","fn","total_words",
        "human_redacted_count","ai_redacted_count",
        "ai_redacted_span_expansions","ai_redacted_extra_tokens",
        "human_encoding","ai_encoding"
    ])
    return df, labels_by_run

def build_summary(df: pd.DataFrame, labels_by_run: dict) -> pd.DataFrame:
    """One summary row per (model, prompt_label), each with its own corpus metrics."""
    rows = []
    for (model, prompt_label), (run_true, run_pred) in labels_by_run.items():
        run_df = df[(df["model"] == model) & (df["prompt_label"] == prompt_label)]
        corpus = compute_corpus_metrics_sklearn(np.array(run_true, dtype=int), np.array(run_pred, dtype=int))
        rows.append({
            "model": model,
            "prompt_label": prompt_label,
            "files_evaluated": len(run_df),
            "avg_accuracy": run_df["accuracy"].mean(),
            "avg_precision": run_df["precision"].mean(),
            "avg_recall": run_df["recall"].mean(),
            "avg_kappa": run_df["kappa"].mean(),
            **corpus,
            "sum_redacted_span_expansions": run_df["ai_redacted_span_expansions"].sum(min_count=1),
        })
    return pd.DataFrame(rows)

def _safe_write(ws, row, col, val, fmt=None):
    try:
        is_na = pd.isna(val)
    except Exception:
        is_na = val is None
    if is_na:
        try:
            ws.write_blank(row, col, None, fmt)
            return
        except Exception:
            ws.write(row, col, "")
            return
    if fmt is not None:
        ws.write(row, col, val, fmt)
    else:
        ws.write(row, col, val)

def compute_corpus_metrics_sklearn(all_true: np.ndarray, all_pred: np.ndarray):
    
    if all_true.size == 0:
        return {
            "corpus_accuracy": np.nan,
            "corpus_precision": np.nan,
            "corpus_recall": np.nan,
            "corpus_kappa": np.nan,
            "sum_tp": 0, "sum_tn": 0, "sum_fp": 0, "sum_fn": 0,
            "sum_total_words": 0,
        }

    acc   = accuracy_score(all_true, all_pred)
    kappa = cohen_kappa_score(all_true, all_pred)

    cm = confusion_matrix(all_true, all_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    prec, rec = precision_recall(tp, fp, fn)

    return {
        "corpus_accuracy": acc,
        "corpus_precision": prec,
        "corpus_recall": rec,
        "corpus_kappa": kappa,
        "sum_tp": int(tp), "sum_tn": int(tn), "sum_fp": int(fp), "sum_fn": int(fn),
        "sum_total_words": int(all_true.size),
    }


def write_reports(df: pd.DataFrame, summary: pd.DataFrame, xlsx_path: Path, csv_path: Path):
    append = csv_path.exists()
    df.to_csv(csv_path, mode='a', index=False, header=not append)

    if len(summary) == 0:
        summary = pd.DataFrame([{
            "model": "", "prompt_label": "",
            "files_evaluated": 0,
            "avg_accuracy": None, "avg_precision": None, "avg_recall": None, "avg_kappa": None,
            "corpus_accuracy": None, "corpus_precision": None, "corpus_recall": None, "corpus_kappa": None,
            "sum_tp": 0, "sum_tn": 0, "sum_fp": 0, "sum_fn": 0, "sum_total_words": 0,
        }])

    if xlsx_path.exists():
        existing_summary = pd.read_excel(xlsx_path, sheet_name="Summary")
        existing_perfile = pd.read_excel(xlsx_path, sheet_name="PerFile")
        summary = pd.concat([existing_summary, summary], ignore_index=True)
        df = pd.concat([existing_perfile, df], ignore_index=True)

    try:
        writer = pd.ExcelWriter(xlsx_path, engine="xlsxwriter")
    except Exception:
        writer = pd.ExcelWriter(xlsx_path, engine="openpyxl")
    with writer as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        df.to_excel(writer, sheet_name="PerFile", index=False)

        wb = getattr(writer, "book", None)
        if wb and hasattr(wb, "add_format"):
            header_fmt = wb.add_format({"bold": True, "text_wrap": True, "valign": "top", "border": 1})
            cell_fmt = wb.add_format({"border": 1})
            pct_fmt  = wb.add_format({"num_format": "0.000", "border": 1})
            int_fmt  = wb.add_format({"num_format": "0", "border": 1})
        else:
            header_fmt = cell_fmt = pct_fmt = int_fmt = None

        ws1 = writer.sheets["Summary"]
        for col, colname in enumerate(summary.columns):
            _safe_write(ws1, 0, col, colname, header_fmt)
            try:
                ws1.set_column(col, col, 22)
            except Exception:
                pass

        for r in range(1, len(summary) + 1):
            for c, colname in enumerate(summary.columns):
                val = summary.iloc[r - 1, c]
                if pct_fmt and (colname.startswith("avg_") or colname.startswith("corpus_")):
                    _safe_write(ws1, r, c, val, pct_fmt)
                elif int_fmt and (colname.startswith("sum_") or colname.startswith("files_")):
                    _safe_write(ws1, r, c, val, int_fmt)
                elif cell_fmt:
                    _safe_write(ws1, r, c, val, cell_fmt)
                else:
                    _safe_write(ws1, r, c, val)

        ws2 = writer.sheets["PerFile"]
        for col, colname in enumerate(df.columns):
            _safe_write(ws2, 0, col, colname, header_fmt if header_fmt else None)
            try:
                ws2.set_column(col, col, 22)
            except Exception:
                pass

        if len(df) > 0:
            for r in range(1, len(df) + 1):
                for c, colname in enumerate(df.columns):
                    val = df.iloc[r - 1, c]
                    if pct_fmt and colname in ("accuracy", "precision", "recall", "kappa"):
                        _safe_write(ws2, r, c, val, pct_fmt)
                    elif int_fmt and colname in ("tp", "tn", "fp", "fn", "total_words"):
                        _safe_write(ws2, r, c, val, int_fmt)
                    else:
                        _safe_write(ws2, r, c, val, cell_fmt)

def main():
    pairs = []
    if CSV_INPUT is not None and CSV_INPUT.exists():
        pairs.extend(paired_texts_from_csv(CSV_INPUT))
    pairs.extend(paired_texts_from_folders(AI_FOLDER, HUMAN_FOLDER))

    df, labels_by_run = evaluate(pairs, load_span_expansion_stats(STATS_CSV))
    summary = build_summary(df, labels_by_run)
    write_reports(df, summary, OUTPUT_XLSX, OUTPUT_CSV)
    print(f"Wrote {OUTPUT_XLSX} and {OUTPUT_CSV}. Files evaluated: {len(df)}")

if __name__ == "__main__":
    if NORMALIZE_FILES_ON_DISK:
        NORMALIZED_OUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
