import os
import re
import unicodedata
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    cohen_kappa_score, confusion_matrix
)


AI_FOLDER = Path("new")                
HUMAN_FOLDER = Path("human_redacted")  
CSV_INPUT: Optional[Path] = None       
OUTPUT_XLSX = Path("evaluation_results_llama3.3 all.xlsx")
OUTPUT_CSV  = Path("evaluation_results_llama3.3 all.csv")


NORMALIZE_FILES_ON_DISK = False
NORMALIZED_OUT_DIR = Path("normalized_utf8")  


REDACT_PATTERNS = [
    re.compile(r"^\[REDACTED\]$", re.I),
    re.compile(r"^<REDACTED>$", re.I),
    re.compile(r"^REDACTED$", re.I),
    re.compile(r"^\*{3,}$"),
    re.compile(r"^X{3,}$", re.I),
    re.compile(r"^\[.*?(name|phone|email|address|dob|ssn|id).*?\]$", re.I),
]

REDACT_HINTS = {"name", "phone", "email", "address", "dob", "ssn", "id"}

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

    for c in (cand1, cand2, cand3):
        if not c:
            continue
        for pat in REDACT_PATTERNS:
            if pat.match(c):
                return True

  
    for c in (cand2.lower(), cand3.lower()):
        if c in REDACT_HINTS:
            return True
    return False

def tokenize(text: Optional[str]):
    return (text or "").split()

def get_labels(text: Optional[str]):
    
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    return [1 if is_redacted_token(tok) else 0 for tok in tokenize(text)]

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

def paired_texts_from_folders(ai_folder: Path, human_folder: Path):
    pairs = []
    if not ai_folder.exists() or not human_folder.exists():
        return pairs
    ai_files = {p.name: p for p in ai_folder.glob("*.txt")}
    for hfile in human_folder.glob("*.txt"):
        if hfile.name in ai_files:
            try:
                human_text, human_enc = read_text_unified(hfile)
                ai_text, ai_enc = read_text_unified(ai_files[hfile.name])

                if NORMALIZE_FILES_ON_DISK:
                    out_h = NORMALIZED_OUT_DIR / "human_redacted" / hfile.name
                    out_a = NORMALIZED_OUT_DIR / "new" / hfile.name
                    ensure_dir_for(out_h); ensure_dir_for(out_a)
                    if human_text is not None:
                        out_h.write_text(human_text, encoding="utf-8")
                    if ai_text is not None:
                        out_a.write_text(ai_text, encoding="utf-8")

                pairs.append((hfile.name, human_text, ai_text, human_enc, ai_enc))
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

        rows.append((filename, human_text, ai_text, human_enc, ai_enc))
    return rows

def align_labels(h_labels, a_labels):
    n = min(len(h_labels), len(a_labels))
    return h_labels[:n], a_labels[:n]

def compute_metrics(y_true, y_pred):
   
    if len(y_true) == 0:
        return {
            "accuracy": None, "precision": None, "recall": None, "kappa": None,
            "tp": 0, "tn": 0, "fp": 0, "fn": 0, "total_words": 0
        }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)

    human_pos = tp + fn
    if human_pos == 0:
        prec = np.nan
        rec  = np.nan
    else:
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else np.nan
        rec  = tp / human_pos

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

def evaluate(pairs):
    rows = []
    all_true = []
    all_pred = []
    for filename, human_text, ai_text, human_enc, ai_enc in pairs:
        y_true = get_labels(human_text)
        y_pred = get_labels(ai_text)
        y_true, y_pred = align_labels(y_true, y_pred)

        all_true.extend(y_true)
        all_pred.extend(y_pred)

        m = compute_metrics(y_true, y_pred)
        m["filename"] = str(filename)
        m["human_redacted_count"] = count_redacted_tokens(human_text)
        m["ai_redacted_count"]    = count_redacted_tokens(ai_text)
        m["human_encoding"] = human_enc
        m["ai_encoding"] = ai_enc
        rows.append(m)

    df = pd.DataFrame(rows, columns=[
        "filename","accuracy","precision","recall","kappa",
        "tp","tn","fp","fn","total_words",
        "human_redacted_count","ai_redacted_count",
        "human_encoding","ai_encoding"
    ])
    return df, np.array(all_true, dtype=int), np.array(all_pred, dtype=int)

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

    human_pos = tp + fn
    if human_pos == 0:
        prec = np.nan
        rec  = np.nan
    else:
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else np.nan
        rec  = tp / human_pos

    return {
        "corpus_accuracy": acc,
        "corpus_precision": prec,
        "corpus_recall": rec,
        "corpus_kappa": kappa,
        "sum_tp": int(tp), "sum_tn": int(tn), "sum_fp": int(fp), "sum_fn": int(fn),
        "sum_total_words": int(all_true.size),
    }


def write_reports(df: pd.DataFrame, corpus: dict, xlsx_path: Path, csv_path: Path):
    
    df.to_csv(csv_path, index=False)

    if len(df) > 0:
        summary = pd.DataFrame([{
            "files_evaluated": len(df),
            "avg_accuracy": df["accuracy"].mean(),
            "avg_precision": df["precision"].mean(),
            "avg_recall": df["recall"].mean(),
            "avg_kappa": df["kappa"].mean(),
            "corpus_accuracy": corpus["corpus_accuracy"],
            "corpus_precision": corpus["corpus_precision"],
            "corpus_recall": corpus["corpus_recall"],
            "corpus_kappa": corpus["corpus_kappa"],
            "sum_tp": corpus["sum_tp"],
            "sum_tn": corpus["sum_tn"],
            "sum_fp": corpus["sum_fp"],
            "sum_fn": corpus["sum_fn"],
            "sum_total_words": corpus["sum_total_words"],
        }])
    else:
        summary = pd.DataFrame([{
            "files_evaluated": 0,
            "avg_accuracy": None,
            "avg_precision": None,
            "avg_recall": None,
            "avg_kappa": None,
            "corpus_accuracy": None,
            "corpus_precision": None,
            "corpus_recall": None,
            "corpus_kappa": None,
            "sum_tp": 0, "sum_tn": 0, "sum_fp": 0, "sum_fn": 0, "sum_total_words": 0,
        }])

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

    df, all_true, all_pred = evaluate(pairs)
    corpus = compute_corpus_metrics_sklearn(all_true, all_pred)
    write_reports(df, corpus, OUTPUT_XLSX, OUTPUT_CSV)
    print(f"Wrote {OUTPUT_XLSX} and {OUTPUT_CSV}. Files evaluated: {len(df)}")

if __name__ == "__main__":
    if NORMALIZE_FILES_ON_DISK:
        NORMALIZED_OUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
