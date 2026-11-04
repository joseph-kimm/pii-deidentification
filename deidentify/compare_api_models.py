"""
compare_api_models.py - Run de_identify_steplab.py on several API models and compare them.

For every model in MODELS this runs the redaction harness (in parallel, one subprocess
per model) on the transcripts in original/, scores each model's outputs against
human_redacted/ with transcript_evaluator.py, and plots the per-model results.

Usage (any working directory):
    python deidentify/compare_api_models.py --smoke          # one test request per model
    python deidentify/compare_api_models.py                  # redact + score + plot
    python deidentify/compare_api_models.py --skip-existing  # resume: only re-run unfinished models
    python deidentify/compare_api_models.py --score-only     # re-score + re-plot existing outputs

Everything is written to results/deidentify/compare_models/:
    <model>/                         redacted transcripts, processing_statistics.csv, run.log
    eval_results.csv                 per-file metrics
    eval_summary.csv                 per-model corpus metrics
    harness_stats.csv                per-model chunk/line/word failures + span expansions
    eval_summary_*.{png,svg}         plots
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openai
import pandas as pd
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import transcript_evaluator as te  # noqa: E402

load_dotenv(SCRIPT_DIR / ".env")

# -------------------------
# Config
# -------------------------
PROMPT_LABEL = "prompt 3"
PROJECT_ROOT = SCRIPT_DIR.parent
ORIGINAL_DIR = PROJECT_ROOT / "data" / "synthetic" / "original"
HUMAN_DIR = PROJECT_ROOT / "data" / "synthetic" / "human_redacted"
RESULTS_DIR = PROJECT_ROOT / "results" / "deidentify" / "compare_models"
HARNESS = SCRIPT_DIR / "de_identify_steplab.py"

# (provider, model) -- provider must be a key of PROVIDERS in de_identify_steplab.py
MODELS = [
    ("fireworks", "accounts/fireworks/models/deepseek-v4p1-flash-1017"),
    ("fireworks", "accounts/fireworks/models/minimax-m3"),
    ("fireworks", "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b"),
    ("fireworks", "accounts/fireworks/models/gpt-oss-120b"),
    ("fireworks", "accounts/fireworks/models/glm-5p3-flash"),
    ("openrouter", "meta-llama/llama-3.1-8b-instruct"),
    ("openrouter", "meta-llama/llama-3.3-70b-instruct"),
]

# Only used by --smoke; the harness itself picks its endpoint from PROVIDER.
SMOKE_ENDPOINTS = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "fireworks": ("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
}

METRIC_COLS = ["corpus_accuracy", "corpus_precision", "corpus_recall", "corpus_kappa"]
STAT_FIELDS = ["chunk_failures", "line_failures", "word_failures",
               "redacted_span_expansions", "redacted_extra_tokens"]


def safe_name(s: str) -> str:
    """Same sanitizing de_identify_steplab.py applies to model/prompt names in filenames."""
    return re.sub(r"[^\w\-]", "_", s)


def short_name(model: str) -> str:
    return model.split("/")[-1]


def model_dir(model: str) -> Path:
    return RESULTS_DIR / safe_name(model)


def eval_files():
    return sorted(p.name for p in ORIGINAL_DIR.glob("*.txt"))


# -------------------------
# Smoke test
# -------------------------
def smoke_test() -> bool:
    all_ok = True
    for provider, model in MODELS:
        base_url, key_env = SMOKE_ENDPOINTS[provider]
        start = time.time()
        try:
            response = openai.OpenAI(api_key=os.getenv(key_env), base_url=base_url).chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Replace person names with [REDACTED]. Return only the edited text."},
                    {"role": "user", "content": "Text to edit: Speaker 1: Hi Maria, did you see Josh today?"},
                ],
            )
            content = response.choices[0].message.content
            ok = content is not None
            print(f"{'OK  ' if ok else 'FAIL'} {model} ({time.time() - start:.1f}s): {content!r}")
        except Exception as e:
            ok = False
            print(f"FAIL {model}: {str(e)[:300]}")
        all_ok = all_ok and ok
    return all_ok


# -------------------------
# Redaction runs
# -------------------------
def is_complete(model: str) -> bool:
    out_dir = model_dir(model)
    expected = {f"{Path(f).stem}__{safe_name(PROMPT_LABEL)}__{safe_name(model)}.txt" for f in eval_files()}
    return (out_dir / "processing_statistics.csv").exists() and expected <= {p.name for p in out_dir.glob("*.txt")}


def run_harness(skip_existing: bool) -> None:
    procs = []
    for provider, model in MODELS:
        if skip_existing and is_complete(model):
            print(f"[skip] {model} already has all outputs")
            continue
        out_dir = model_dir(model)
        out_dir.mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            "PROVIDER": provider,
            "MODEL_NAME": model,
            "PROMPT_LABEL": PROMPT_LABEL,
            "OUTPUT_FOLDER": str(out_dir),
            "STATS_CSV": str(out_dir / "processing_statistics.csv"),
            "PYTHONUNBUFFERED": "1",
        }
        log = open(out_dir / "run.log", "w", encoding="utf-8")
        # cwd=SCRIPT_DIR: the harness reads original/ and Prompts.csv relative to cwd
        proc = subprocess.Popen([sys.executable, str(HARNESS)], cwd=SCRIPT_DIR, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
        procs.append((model, proc, log, time.time()))
        print(f"[start] {model} -> {out_dir / 'run.log'}")

    for model, proc, log, start in procs:
        code = proc.wait()
        log.close()
        status = "done" if code == 0 and is_complete(model) else f"FAILED (exit {code}, see run.log)"
        print(f"[{status}] {model} in {(time.time() - start) / 60:.1f} min")


# -------------------------
# Scoring
# -------------------------
def score():
    te.ORIGINAL_FOLDER = ORIGINAL_DIR  # evaluator resolves this relative to cwd by default
    short_by_safe = {safe_name(m): short_name(m) for _, m in MODELS}

    results, summaries, harness_rows = [], [], []
    for _, model in MODELS:
        out_dir = model_dir(model)
        stats_csv = out_dir / "processing_statistics.csv"
        if not out_dir.exists():
            print(f"[WARN] no outputs for {model}; skipping")
            continue
        pairs = te.paired_texts_from_folders(out_dir, HUMAN_DIR)
        df, labels_by_run = te.evaluate(pairs, te.load_span_expansion_stats(stats_csv))
        results.append(df)
        summaries.append(te.build_summary(df, labels_by_run))
        if stats_csv.exists():
            stats = pd.read_csv(stats_csv)
            harness_rows.append({"model": short_name(model), **stats[STAT_FIELDS].sum().to_dict()})

    results_df = pd.concat(results, ignore_index=True)
    # Report corpus metrics only; build_summary's per-file averages read like a second set of scores
    summary_df = pd.concat(summaries, ignore_index=True).drop(columns=["avg_accuracy", "avg_precision", "avg_recall", "avg_kappa"])
    for df in (results_df, summary_df):
        df["model"] = df["model"].map(lambda m: short_by_safe.get(m, m))
    harness_df = pd.DataFrame(harness_rows, columns=["model", *STAT_FIELDS])

    results_df.to_csv(RESULTS_DIR / "eval_results.csv", index=False)
    summary_df.to_csv(RESULTS_DIR / "eval_summary.csv", index=False)
    harness_df.to_csv(RESULTS_DIR / "harness_stats.csv", index=False)
    print(f"Saved eval_results.csv ({len(results_df)} rows), eval_summary.csv, harness_stats.csv to {RESULTS_DIR}")
    return summary_df, harness_df


# -------------------------
# Plots (mirrors the notebook's "Visualize results" cell)
# -------------------------
def _save_fig(fig, name: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(RESULTS_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot(summary_df: pd.DataFrame, harness_df: pd.DataFrame) -> None:
    ax = summary_df.set_index("model")[METRIC_COLS].plot(
        kind="bar", figsize=(max(8, len(summary_df)), 5), ylim=(0, 1)
    )
    ax.set_ylabel("Score")
    ax.set_title(f"Redaction quality by model ({PROMPT_LABEL})")
    ax.legend(loc="lower right")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _save_fig(ax.figure, "eval_summary_metrics")

    ax = harness_df.set_index("model")["redacted_span_expansions"].plot(
        kind="bar", figsize=(max(8, len(harness_df)), 4)
    )
    ax.set_ylabel("Multi-word spans expanded")
    ax.set_title("How often the harness fell back to one-[REDACTED]-per-word")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _save_fig(ax.figure, "eval_summary_expansions")
    print(f"Saved plots to {RESULTS_DIR}/eval_summary_metrics.* and eval_summary_expansions.*")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="send one test request per model and exit")
    parser.add_argument("--skip-existing", action="store_true", help="don't re-run models whose outputs are complete")
    parser.add_argument("--score-only", action="store_true", help="skip redaction; score and plot existing outputs")
    args = parser.parse_args()

    if args.smoke:
        raise SystemExit(0 if smoke_test() else 1)

    RESULTS_DIR.mkdir(exist_ok=True)
    if not args.score_only:
        print(f"Redacting {len(eval_files())} transcripts from {ORIGINAL_DIR} with {len(MODELS)} models")
        run_harness(args.skip_existing)

    summary_df, harness_df = score()
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary_df[["model", "files_evaluated", *METRIC_COLS, "sum_tp", "sum_fp", "sum_fn"]].to_string(index=False))
    plot(summary_df, harness_df)


if __name__ == "__main__":
    main()
