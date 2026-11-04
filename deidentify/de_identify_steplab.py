import os
import re
import csv
import json
import time
import string
import difflib
import requests
import openai
import pandas as pd
import unicodedata
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

# -------------------------
# Config
# -------------------------
# Paths default to this script's folder so the harness works from any working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
original_folder = os.path.join(PROJECT_ROOT, 'data', 'synthetic', 'original')
output_folder = os.getenv('OUTPUT_FOLDER', os.path.join(PROJECT_ROOT, 'results', 'deidentify', 'new'))
stats_csv_path = os.getenv('STATS_CSV', os.path.join(PROJECT_ROOT, 'results', 'deidentify', 'processing_statistics.csv'))

# The original transcripts are the anchor transcript_evaluator.py projects gold and
# prediction onto, so they are left untouched by default. Set True to restore the old
# behaviour of rewriting each original with normalized whitespace/speaker formatting.
NORMALIZE_ORIGINALS_ON_DISK = False

# Set your OpenAI key if you switch to the OpenAI path in api_call()
openai.api_key = ''  # e.g., os.getenv("OPENAI_API_KEY")

# LM Studio (local) model + endpoint
lmstudio_model = "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q8_0.gguf"
lmstudio_url = "http://localhost:1234/v1/chat/completions"

# HuggingFace + OpenRouter (from .env)
hf_token = os.getenv('HF_TOKEN')
hf_model = os.getenv('HF_MODEL', 'meta-llama/Meta-Llama-3-8B-Instruct')
hf_client = InferenceClient(model=hf_model, token=hf_token)

openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
model_name = os.getenv('MODEL_NAME', 'meta-llama/llama-3.1-8b-instruct')

# OpenAI-compatible endpoint used by openrouter_api_call: 'openrouter' or 'fireworks'
PROVIDERS = {
    'openrouter': ("https://openrouter.ai/api/v1", openrouter_api_key),
    'fireworks': ("https://api.fireworks.ai/inference/v1", os.getenv('FIREWORKS_API_KEY')),
}
provider = os.getenv('PROVIDER', 'openrouter')
if provider not in PROVIDERS:
    print(f"Unknown PROVIDER '{provider}', expected one of {list(PROVIDERS)}")
    raise SystemExit(1)
provider_base_url, provider_api_key = PROVIDERS[provider]

# -------------------------
# Small helpers (logging/redaction)
# -------------------------
def _preview(s, n=240):
    """One-line preview for logs."""
    if s is None:
        return "None"
    s = str(s).replace("\n", "\\n")
    return (s[:n] + ("..." if len(s) > n else ""))

_PUNCT = set(string.punctuation)

def _is_redacted_with_punct(tok: str) -> bool:
    """
    Accept '[REDACTED]' with optional leading/trailing ASCII punctuation, e.g.:
    [REDACTED]?, "[REDACTED]", ([REDACTED]), [REDACTED],
    (kept for enforce_word_count_match & alignment; validation ignores punctuation anyway)

    NOTE: locates the literal "[REDACTED]" substring rather than stripping
    leading/trailing punctuation first -- "[" and "]" are themselves members of
    string.punctuation, so a naive strip-then-compare eats the token's own
    brackets and can never match (this previously made the function return
    False for every input, including the bare "[REDACTED]" token itself).
    """
    if not tok:
        return False
    idx = tok.find("[REDACTED]")
    if idx == -1:
        return False
    before = tok[:idx]
    after = tok[idx + len("[REDACTED]"):]
    # allow an attached possessive right after the token, e.g. [REDACTED]'s
    if after.startswith("'s"):
        after = after[2:]
    elif after.startswith("’s"):
        after = after[2:]
    return all(c in _PUNCT for c in before) and all(c in _PUNCT for c in after)

# -------------------------
# Read prompts CSV & prepare folders
# -------------------------
prompt_label = os.getenv('PROMPT_LABEL', '')
try:
    prompt_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'Prompts.csv'), encoding='utf-8')
    if prompt_label:
        matched = prompt_df[prompt_df['label'] == prompt_label]
        if matched.empty:
            print(f"No prompt found with label '{prompt_label}' in Prompts.csv")
            raise SystemExit(1)
        prompt_df = matched
except Exception as e:
    print("Error reading Prompts.csv:", str(e))
    raise SystemExit(1)

os.makedirs(output_folder, exist_ok=True)
original_folder_path = os.path.join(os.getcwd(), original_folder)

# -------------------------
# API wrappers
# -------------------------
def openai_api_call(post_text, prompt):
    try:
        message_content = f'{prompt}\n{post_text}'
        response_check = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message_content}
            ],
            max_tokens=1000
        )
        return response_check["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[OPENAI API ERROR] text='{_preview(post_text, 120)}' | {str(e)}")
        return "API_ERROR"

def llama3_lmstudio_call(post_text, prompt):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": lmstudio_model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Text to edit: " + post_text}
        ],
        "max_tokens": 1000
    }
    try:
        resp = requests.post(lmstudio_url, headers=headers, data=json.dumps(data), timeout=400)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        print(f"[LM STUDIO ERROR] {e}")
        return "API_ERROR"

def huggingface_call(post_text, prompt):
    try:
        response = hf_client.chat_completion(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Text to edit: " + post_text}
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[HUGGINGFACE ERROR] text='{_preview(post_text, 120)}' | {str(e)}")
        return "API_ERROR"

def openrouter_api_call(post_text, prompt):
    try:
        client = openai.OpenAI(
            api_key=provider_api_key,
            base_url=provider_base_url,
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Text to edit: " + post_text}
            ],
        )
        content = response.choices[0].message.content
        if content is None:
            finish_reason = response.choices[0].finish_reason
            print(f"[OPENROUTER ERROR] text='{_preview(post_text, 120)}' | null content, finish_reason={finish_reason}")
            return "API_ERROR"
        return content.strip()
    except Exception as e:
        print(f"[OPENROUTER ERROR] text='{_preview(post_text, 120)}' | {str(e)}")
        return "API_ERROR"

def api_call(post_text, prompt):
    try:
        return openrouter_api_call(post_text, prompt)  # default: OpenRouter
        # return huggingface_call(post_text, prompt)  # <- switch to HuggingFace
        # return openai_api_call(post_text, prompt)      # <- switch to OpenAI
        # return llama3_lmstudio_call(post_text, prompt) # <- switch to LM Studio
    except Exception as e:
        print(f"[API CALL EXC] text='{_preview(post_text, 120)}' | {str(e)}")
        return "API_ERROR"

# -------------------------
# Normalization utilities
# -------------------------
def normalize_whitespace(text):
    lines = text.split('\n')
    normalized_lines = []
    for line in lines:
        normalized_line = re.sub(r'\s+', ' ', line.strip())
        normalized_lines.append(normalized_line)
    return '\n'.join(normalized_lines)

def normalize_speaker_format(text):
    text = normalize_whitespace(text)
    speaker_pattern = r'(Speaker\s*[A-Za-z0-9]*\s*:)'
    parts = re.split(speaker_pattern, text)
    speakers_with_content = []
    current_speaker = None
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if re.match(speaker_pattern, part):
            current_speaker = part
        else:
            if current_speaker:
                speakers_with_content.append(f"{current_speaker} {part}")
                current_speaker = None
            else:
                speakers_with_content.append(part)
    return '\n\n'.join(speakers_with_content)

_SPEAKER_LINE_RE = re.compile(r'^\s*Speaker\s*[A-Za-z0-9]*\s*:')

def _is_speaker_line(line: str) -> bool:
    """True if line already starts with a 'Speaker X:' turn (no separate title/header line)."""
    return bool(_SPEAKER_LINE_RE.match(line))

def remove_empty_speaker_lines(text):
    lines = text.split('\n')
    non_empty_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r'^Speaker\s*[A-Za-z0-9]*\s*:\s*$', line):
            continue
        non_empty_lines.append(line)
    return '\n'.join(non_empty_lines)

# -------------------------
# Sentence chunking
# -------------------------
# Titles and initials end in "." but don't end a sentence. Splitting there put a
# name in a different chunk from its title ("Even Mr." | "Henderson noticed..."),
# so the model saw the name without the context that marks it as one.
ABBREVIATIONS = {"dr.", "mr.", "mrs.", "ms.", "prof.", "st.", "jr.", "sr.", "mt.", "ft.", "rev.", "gov.", "sen."}
_CLOSERS = "\"')]’”"

def _ends_sentence(word):
    """True if `word` ends a sentence: '.', '!' or '?' (before any closing quote/bracket), not a title or initial."""
    core = word.rstrip(_CLOSERS)
    if not core or core[-1] not in ".!?":
        return False
    if core[-1] == "." and (core.lower() in ABBREVIATIONS or re.fullmatch(r"[A-Z]\.", core)):
        return False  # "Dr." / "J." are not sentence ends
    return True

def split_into_sentences(text, max_sentences=3):
    sentences = []
    current = ""
    for m in re.finditer(r"\S+\s*", text):
        current += m.group()
        if _ends_sentence(m.group().rstrip()):
            # the whitespace after a sentence end is dropped, as the old re.split did
            sentences.append(current.rstrip())
            current = ""
    if current or not sentences:
        sentences.append(current.rstrip())
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunks.append(" ".join(sentences[i:i + max_sentences]))
    return chunks

def clean_llm_response(original_text, llm_response):
    """
    Normalize whitespace only; all alignment to the original is left to
    enforce_word_count_match.

    This used to also "repair" the response by comparing it to the original
    position-by-position in 5-word chunks and copying back the original word
    wherever they disagreed. That repair leaked PII: when a model collapses a
    multi-word span ("Lincoln High") into a single [REDACTED], every word after
    it is shifted, so the repair restored the original words ("[REDACTED] High",
    or the whole span) before enforce_word_count_match could expand the
    collapsed tag, and the leaked text still passed validation.
    enforce_word_count_match's difflib alignment already handles every case
    the repair covered: collapsed spans are expanded, extra text the model adds
    (e.g. "Here is the edited text:") is dropped, and other edits are reverted.
    """
    return normalize_whitespace(llm_response)

# -------------------------
# VALIDATION (punctuation-agnostic + possessive redactions)
# -------------------------
def _normalize_smart_punct(s: str) -> str:
    """Normalize common smart punctuation to ASCII-ish before tokenizing."""
    s = unicodedata.normalize("NFC", s)
    return (
        s.replace("’", "'").replace("‘", "'").replace("ʼ", "'").replace("‛", "'")
         .replace("“", '"').replace("”", '"').replace("‟", '"')
         .replace("–", "-").replace("—", "-").replace("‒", "-")
         .replace("\u00A0", " ").replace(" ", " ").replace(" ", " ")
    )

# IMPORTANT: treat [REDACTED], [REDACTED]'s, [REDACTED]’s as ONE token
# Words use \w (any Unicode letter/digit), not [A-Za-z0-9]: with the ASCII class an
# accented name like "Théo" split into "Th" + "o", so its correct one-token redaction
# always failed the word-count check and the original name was kept.
_TOKEN_RE = re.compile(r"\[REDACTED\](?:'s|’s)?|[\w']+")

def _word_tokens(s: str):
    """
    Ignore punctuation by extracting only word tokens and the literal [REDACTED].
    Normalize [REDACTED]'s / [REDACTED]’s to [REDACTED] so possessives don't add tokens.
    """
    s = _normalize_smart_punct(s)
    toks = _TOKEN_RE.findall(s)
    norm = []
    for t in toks:
        if t.startswith("[REDACTED]"):
            norm.append("[REDACTED]")  # collapse possessive redaction to plain [REDACTED]
        else:
            norm.append(t)
    return norm

def is_valid_input(str1: str, str2: str) -> bool:
    """
    Punctuation-agnostic validation:
      - Compare only words + [REDACTED] (possessive redactions normalized).
      - Word counts must match after normalization.
      - Each token must be identical to original OR exactly [REDACTED].
      - Reject common boilerplate refusals.
    """
    s2 = str2.lower()
    if "please provide the text" in s2: return False
    if "there's no actual text" in s2: return False
    if "personally identifiable information (pii)" in s2: return False

    orig = _word_tokens(str1)
    proc = _word_tokens(str2)

    if len(orig) != len(proc):
        print(f"[VALIDATE] word-count mismatch (punct ignored): orig={len(orig)} proc={len(proc)}")
        return False

    for idx, (o, p) in enumerate(zip(orig, proc)):
        if p == o or p == "[REDACTED]":
            continue
        print(f"[VALIDATE] mismatch at word idx {idx}: orig='{o}' vs proc='{p}'")
        return False

    return True

def _leading_punct(word):
    """Return word's leading punctuation run, e.g. '"Rachel' -> '"'."""
    i = 0
    while i < len(word) and word[i] in _PUNCT:
        i += 1
    return word[:i]


def _trailing_punct(word):
    """Return word's trailing punctuation run, e.g. 'Springfield.' -> '.'."""
    j = len(word)
    while j > 0 and word[j - 1] in _PUNCT:
        j -= 1
    return word[j:]


def enforce_word_count_match(original_text, processed_text, stats=None):
    """
    Keep lengths equal at the whitespace-token level. This is a safety net
    and doesn't affect the punctuation-agnostic validation.

    Uses a real alignment (difflib) between original and processed words instead
    of a naive left-to-right walk. This matters because a model will sometimes
    collapse a multi-word PII span (e.g. "Lincoln High School") into a single
    "[REDACTED]" token. The old walk had no way to "catch up" after a collapse
    and ended up reverting the extra original words back into the output
    unredacted -- i.e. it leaked PII while still looking length-correct. The
    diff-based approach instead expands a collapsed span into one "[REDACTED]"
    per original word it replaced (preserving any punctuation attached to the
    first/last original word), so PII in a collapsed span never leaks back in.

    When `stats` is given, each such expansion is counted in
    stats['redacted_span_expansions'] and the number of extra [REDACTED] tokens
    it added in stats['redacted_extra_tokens'].
    """
    original_words = original_text.split()
    processed_words = processed_text.split()
    if original_words == processed_words:
        return processed_text

    matcher = difflib.SequenceMatcher(a=original_words, b=processed_words, autojunk=False)
    corrected_words = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        orig_chunk = original_words[i1:i2]
        proc_chunk = processed_words[j1:j2]

        if tag == "equal":
            corrected_words.extend(orig_chunk)
        elif tag == "replace":
            if len(orig_chunk) == len(proc_chunk):
                # Same word count: resolve word-by-word, keeping the model's own
                # token (which may carry its own punctuation) whenever it's a
                # redaction or an exact match; otherwise revert that one word.
                for orig_word, proc_word in zip(orig_chunk, proc_chunk):
                    if _is_redacted_with_punct(proc_word) or proc_word == orig_word:
                        corrected_words.append(proc_word)
                    else:
                        corrected_words.append(orig_word)
            elif any(_is_redacted_with_punct(w) for w in proc_chunk):
                # The model intended to redact this span but collapsed it into
                # fewer tokens than the original span had words. Expand it to
                # one [REDACTED] per original word instead of leaking the rest,
                # preserving punctuation attached to the span's edges.
                leading = _leading_punct(orig_chunk[0])
                trailing = _trailing_punct(orig_chunk[-1])
                expanded = ["[REDACTED]"] * len(orig_chunk)
                expanded[0] = leading + "[REDACTED]"
                expanded[-1] = expanded[-1] + trailing
                corrected_words.extend(expanded)
                if stats is not None and len(orig_chunk) > 1:
                    stats['redacted_span_expansions'] += 1
                    stats['redacted_extra_tokens'] += len(orig_chunk) - 1
            else:
                # Not a redaction, just an unrequested wording change -- revert.
                corrected_words.extend(orig_chunk)
        elif tag == "delete":
            # Original words missing entirely from processed, with no adjacent
            # [REDACTED] evidence of intent -- keep the original words (safe default).
            corrected_words.extend(orig_chunk)
        elif tag == "insert":
            # Extra words the model added that don't exist in original -- drop them.
            continue

    return " ".join(corrected_words)

# -----------------------------
# FALLBACKS with success + failure logs
# -----------------------------
def process_word_by_word(original_text, prompt, api_call_func, stats):
    original_words = original_text.split()
    processed_words = []

    print(f"Starting word-by-word processing for: '{_preview(original_text, 80)}'")
    for i, word in enumerate(original_words):
        try:
            response = api_call_func(word, prompt)

            if response == "API_ERROR":
                print(f"  [WORD FAIL] API error | idx={i+1}/{len(original_words)} | word='{word}'")
                processed_words.append(word)
                stats['word_failures'] += 1
                continue

            cleaned_response = clean_llm_response(word, response)
            enforced = enforce_word_count_match(word, cleaned_response, stats)
            valid = is_valid_input(word, enforced)

            if not valid:
                print(
                    f"  [WORD FAIL] Validation | idx={i+1}/{len(original_words)} | word='{word}' | "
                    f"raw='{_preview(response)}' | cleaned='{_preview(cleaned_response)}' | enforced='{_preview(enforced)}'"
                )
                processed_words.append(word)
                stats['word_failures'] += 1
            else:
                response_words = enforced.split()
                out = response_words[0] if response_words else word
                if _is_redacted_with_punct(out):
                    print(f"  [WORD OK] redacted: '{word}' -> '{out}'")
                elif out != word:
                    print(f"  [WORD OK] changed: '{word}' -> '{out}'")
                else:
                    print(f"  [WORD OK] unchanged: '{word}'")
                processed_words.append(out)

        except Exception as e:
            print(f"  [WORD EXC] idx={i+1}/{len(original_words)} | word='{word}' | err={e}")
            processed_words.append(word)
            stats['word_failures'] += 1

    print(f"Word-by-word processing completed. Failures: {stats['word_failures']}")
    return " ".join(processed_words)

def process_line_with_fallback(line_text, prompt, api_call_func, stats):
    if not line_text.strip():
        return line_text

    print(f"Processing line: '{_preview(line_text, 80)}'")
    try:
        response = api_call_func(line_text.strip(), prompt)

        if response == "API_ERROR":
            print(f"  [LINE FAIL] API error -> falling back to WORD")
            stats['line_failures'] += 1
            word_result = process_word_by_word(line_text.strip(), prompt, api_call_func, stats)
            enforced = enforce_word_count_match(line_text.strip(), word_result, stats)
            valid = is_valid_input(line_text.strip(), enforced)
            if valid:
                print(f"  [LINE->WORD OK] Fallback succeeded")
                return enforced
            else:
                print(f"  [LINE->WORD FAIL] Validation after fallback | enforced='{_preview(enforced)}'")
                return line_text.strip()

        cleaned_response = clean_llm_response(line_text.strip(), response)
        enforced = enforce_word_count_match(line_text.strip(), cleaned_response, stats)
        valid = is_valid_input(line_text.strip(), enforced)

        if valid:
            print(f"  Line processing succeeded")
            return enforced

        print(
            f"  [LINE FAIL] Validation | raw='{_preview(response)}' | "
            f"cleaned='{_preview(cleaned_response)}' | enforced='{_preview(enforced)}' -> falling back to WORD"
        )
        stats['line_failures'] += 1

        word_result = process_word_by_word(line_text.strip(), prompt, api_call_func, stats)
        enforced2 = enforce_word_count_match(line_text.strip(), word_result, stats)
        valid2 = is_valid_input(line_text.strip(), enforced2)
        if valid2:
            print(f"  [LINE->WORD OK] Fallback succeeded")
            return enforced2
        else:
            print(f"  [LINE->WORD FAIL] Validation after fallback | enforced='{_preview(enforced2)}'")
            return line_text.strip()

    except Exception as e:
        print(f"  [LINE EXC] err={e} -> falling back to WORD")
        stats['line_failures'] += 1
        try:
            word_result = process_word_by_word(line_text.strip(), prompt, api_call_func, stats)
            enforced = enforce_word_count_match(line_text.strip(), word_result, stats)
            valid = is_valid_input(line_text.strip(), enforced)
            if valid:
                print(f"  [LINE->WORD OK] Fallback succeeded after exception")
                return enforced
            else:
                print(f"  [LINE->WORD FAIL] Validation after exception fallback | enforced='{_preview(enforced)}'")
                return line_text.strip()
        except Exception as e2:
            print(f"  [WORD Fallback EXC] err={e2}")
            stats['word_failures'] += len(line_text.strip().split())
            return line_text.strip()

def process_chunk_with_fallback(chunk_text, prompt, api_call_func, stats):
    print(f"Processing chunk: '{_preview(chunk_text, 80)}'")
    try:
        response = api_call_func(chunk_text, prompt)

        if response == "API_ERROR":
            print(f"  [CHUNK FAIL] API error -> falling back to LINE")
            stats['chunk_failures'] += 1
            lines = chunk_text.split('\n')
            processed_lines = []
            for line in lines:
                processed_lines.append(
                    process_line_with_fallback(line, prompt, api_call_func, stats) if line.strip() else line
                )
            return '\n'.join(processed_lines)

        cleaned_response = clean_llm_response(chunk_text, response)
        enforced = enforce_word_count_match(chunk_text, cleaned_response, stats)
        valid = is_valid_input(chunk_text, enforced)

        if valid:
            print(f"  Chunk processing succeeded")
            return enforced

        print(
            f"  [CHUNK FAIL] Validation | raw='{_preview(response)}' | "
            f"cleaned='{_preview(cleaned_response)}' | enforced='{_preview(enforced)}' -> falling back to LINE"
        )
        stats['chunk_failures'] += 1
        lines = chunk_text.split('\n')
        processed_lines = []
        for line in lines:
            processed_lines.append(
                process_line_with_fallback(line, prompt, api_call_func, stats) if line.strip() else line
            )
        return '\n'.join(processed_lines)

    except Exception as e:
        print(f"  [CHUNK EXC] err={e} -> falling back to LINE")
        stats['chunk_failures'] += 1
        lines = chunk_text.split('\n')
        processed_lines = []
        for line in lines:
            processed_lines.append(
                process_line_with_fallback(line, prompt, api_call_func, stats) if line.strip() else line
            )
        return '\n'.join(processed_lines)

def process_text_content(original_text, prompt, api_call_func, stats):
    """
    Tiered fallback:
      1) chunk
      2) line
      3) word
    """
    text_chunks = split_into_sentences(original_text)
    print(f"Split text into {len(text_chunks)} chunks")
    processed_chunks = []
    for i, chunk in enumerate(text_chunks, 1):
        print(f"\n--- Processing chunk {i}/{len(text_chunks)} ---")
        processed_chunk = process_chunk_with_fallback(chunk, prompt, api_call_func, stats)
        processed_chunks.append(processed_chunk)
    final_result = " ".join(processed_chunks)
    if is_valid_input(original_text, final_result):
        print("Final validation: PASSED")
    else:
        print("Final validation: FAILED - returning processed content anyway")
    return final_result

# -------------------------
# Main file processing
# -------------------------
try:
    original_files = os.listdir(original_folder_path)
except Exception as e:
    print(f"Error reading from directory {original_folder_path}: {str(e)}")
    raise SystemExit(1)

processing_stats = []

for original_file in original_files:
    if not original_file.endswith(".txt"):
        continue

    print(f"\n========== Processing file {original_file} ==========")

    file_stats = {
        'filename': original_file,
        'model_name': model_name,
        'prompt_label': prompt_label,
        'chunk_failures': 0,
        'line_failures': 0,
        'word_failures': 0,
        'redacted_span_expansions': 0,
        'redacted_extra_tokens': 0,
        'processing_time_minutes': 0,
        'start_time': time.time()
    }

    original_file_path = os.path.join(original_folder_path, original_file)
    safe_model = re.sub(r'[^\w\-]', '_', model_name)
    safe_prompt = re.sub(r'[^\w\-]', '_', prompt_label)
    stem, ext = os.path.splitext(original_file)
    new_filename = f"{stem}__{safe_prompt}__{safe_model}{ext}"
    new_file_path = os.path.join(output_folder, new_filename)

    try:
        with open(original_file_path, 'r') as original_file_handle, open(new_file_path, 'w') as new_file_handle:
            for _, row in prompt_df.iterrows():
                prompt = row['prompt']

                original_file_handle.seek(0)
                all_lines = original_file_handle.readlines()

                if len(all_lines) > 1:
                    has_header = not _is_speaker_line(all_lines[0])

                    if has_header:
                        first_line = normalize_whitespace(all_lines[0].rstrip('\n'))
                        remaining_content = ''.join(all_lines[1:]).strip()
                    else:
                        first_line = None
                        remaining_content = ''.join(all_lines).strip()
                        print(f"No header line detected (file starts with a speaker turn): {original_file}")

                    remaining_content = normalize_speaker_format(remaining_content)
                    remaining_content = remove_empty_speaker_lines(remaining_content)

                    if NORMALIZE_ORIGINALS_ON_DISK:
                        normalized_original = (
                            first_line + "\n\n" + remaining_content + "\n" if has_header
                            else remaining_content + "\n"
                        )
                        with open(original_file_path, 'w') as orig_write:
                            orig_write.write(normalized_original)
                        print(f"Normalized original file: {original_file}")

                    if remaining_content:
                        processed_content = process_text_content(remaining_content, prompt, api_call, file_stats)
                        processed_content = normalize_speaker_format(processed_content)
                        processed_content = remove_empty_speaker_lines(processed_content)
                        if has_header:
                            new_file_handle.write(first_line + "\n\n" + processed_content + "\n")
                            print(f"Processed file (skipped header line): {original_file}")
                        else:
                            new_file_handle.write(processed_content + "\n")
                            print(f"Processed file (no header line): {original_file}")
                    elif has_header:
                        new_file_handle.write(first_line + "\n")
                        print(f"File {original_file} has only a header line, copied as-is")

                elif len(all_lines) == 1:
                    normalized_line = normalize_speaker_format(all_lines[0])
                    normalized_line = remove_empty_speaker_lines(normalized_line)
                    if NORMALIZE_ORIGINALS_ON_DISK:
                        with open(original_file_path, 'w') as orig_write:
                            orig_write.write(normalized_line + "\n")
                        print(f"Normalized original file: {original_file}")
                    new_file_handle.write(normalized_line + "\n")
                    print(f"File {original_file} has only one line, copied as-is")
                else:
                    print(f"File {original_file} is empty, skipping...")

    except Exception as e:
        print(f"Error processing file {original_file_path}: {str(e)}")

    # timing + stats
    end_time = time.time()
    processing_time_seconds = end_time - file_stats['start_time']
    file_stats['processing_time_minutes'] = round(processing_time_seconds / 60, 2)
    del file_stats['start_time']

    processing_stats.append(file_stats)

    print(f"\n========== File {original_file} completed ==========")
    print(f"Processing time: {file_stats['processing_time_minutes']} minutes")
    print(f"Chunk failures: {file_stats['chunk_failures']}")
    print(f"Line failures: {file_stats['line_failures']}")
    print(f"Word failures: {file_stats['word_failures']}")
    print(f"Redacted span expansions: {file_stats['redacted_span_expansions']} "
          f"(+{file_stats['redacted_extra_tokens']} tokens)")

# -------------------------
# Write processing statistics
# -------------------------
try:
    with open(stats_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'model_name', 'prompt_label', 'chunk_failures', 'line_failures', 'word_failures',
                      'redacted_span_expansions', 'redacted_extra_tokens', 'processing_time_minutes']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for stats in processing_stats:
            writer.writerow(stats)

    print(f"\nProcessing statistics saved to {stats_csv_path}")

    total_files = len(processing_stats)
    total_chunk_failures = sum(stats['chunk_failures'] for stats in processing_stats)
    total_line_failures = sum(stats['line_failures'] for stats in processing_stats)
    total_word_failures = sum(stats['word_failures'] for stats in processing_stats)
    total_span_expansions = sum(stats['redacted_span_expansions'] for stats in processing_stats)
    total_time = sum(stats['processing_time_minutes'] for stats in processing_stats)

    print(f"\n=== PROCESSING SUMMARY ===")
    print(f"Total files processed: {total_files}")
    print(f"Total chunk failures: {total_chunk_failures}")
    print(f"Total line failures: {total_line_failures}")
    print(f"Total word failures: {total_word_failures}")
    print(f"Total redacted span expansions: {total_span_expansions}")
    print(f"Total processing time: {total_time:.2f} minutes")
    print(f"Average processing time per file: {total_time/total_files:.2f} minutes" if total_files > 0 else "No files processed")

except Exception as e:
    print(f"Error writing statistics CSV: {str(e)}")
