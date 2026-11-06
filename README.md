# PII De-identification of Educational Transcripts

Tools for removing personally identifiable information (PII) from teacher–coach coaching
transcripts with LLMs, scoring the results against human redactions, and fine-tuning small open
models to redact more precisely.

```
Original:  Speaker 1: Hey Sarah, how is the first week at Lincoln High going?
Redacted:  Speaker 1: Hey [REDACTED], how is the first week at [REDACTED] [REDACTED] going?
```

> **No data is included.** Transcripts, datasets and results are excluded by `.gitignore`.

Fine-tuned LoRA adapters: [jiminaa/Llama-3.1-8B-Instruct-PII-LoRA](https://huggingface.co/jiminaa/Llama-3.1-8B-Instruct-PII-LoRA)

## Structure

| Folder | What it does |
|---|---|
| `deidentify/` | Redacts transcripts with an LLM and scores the output against human redactions |
| `finetune/` | Generates synthetic transcripts and builds the fine-tuning dataset |
| `data_pipeline/` | Turns real redacted transcripts into synthetic training pairs with no real PII |
| `archive/` | The original scripts this project started from |
| `Prompts.csv` | Redaction prompts, selected by label |

- **Redaction** (`de_identify_steplab.py`): splits transcripts into three-sentence chunks, sends
  each to the model, and aligns the response with the original so the model can only replace
  words with `[REDACTED]`.
- **Evaluation** (`transcript_evaluator.py`): maps human and model redactions onto the original
  words and reports accuracy, precision, recall and Cohen's kappa per file and per corpus.
  `compare_api_models.py` runs and scores several models at once.
- **Fine-tuning data** (`finetune/`): generates synthetic transcripts with Gemini on Vertex AI and
  converts them into a chat-format SFT dataset, chunked the same way as at inference.
- **Synthetic pairs from real data** (`data_pipeline/`): replaces each redacted value with a
  placeholder, has a local LM Studio model rewrite the chunk and invent fake values, and outputs
  aligned original and redacted versions. See
  [`deidentification_pipeline_spec.md`](data_pipeline/deidentification_pipeline_spec.md).

## Setup

Python 3.10 or newer.

```bash
pip install openai requests pandas huggingface_hub python-dotenv matplotlib openpyxl
```

`data_pipeline/` uses only the standard library. `finetune/` also needs `google-genai` and
`datasets`. API keys go in a git-ignored `.env` file in `deidentify/` or `finetune/`:

```
OPENROUTER_API_KEY=...
FIREWORKS_API_KEY=...
HF_TOKEN=...
```

## Usage

Put matching `.txt` files in `data/synthetic/original/` and `data/synthetic/human_redacted/`.

```bash
# redact with one model, then score
PROVIDER=openrouter MODEL_NAME=meta-llama/llama-3.1-8b-instruct PROMPT_LABEL="prompt 3" python deidentify/de_identify_steplab.py
python deidentify/transcript_evaluator.py

# compare several API models
python deidentify/compare_api_models.py

# build the fine-tuning dataset
python finetune/build_sft_dataset.py
python finetune/split_dataset.py
```

For `data_pipeline/`, start LM Studio's local server, set the paths in `settings.bat`, and run the
numbered `.bat` files in order (or the Python scripts they call).

## Author

Joseph Kim, Penn Center for Learning Analytics, University of Pennsylvania
