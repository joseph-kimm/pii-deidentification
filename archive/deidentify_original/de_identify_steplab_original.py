import os
import re
import csv
import json
import time
import string
import requests
import openai
import pandas as pd
import unicodedata

# -------------------------
# Config
# -------------------------
original_folder = 'original/'
output_folder = 'new/'

# Set your OpenAI key if you switch to the OpenAI path in api_call()
openai.api_key = ''  # e.g., os.getenv("OPENAI_API_KEY")

# LM Studio (local) model + endpoint
lmstudio_model = "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q8_0.gguf"
lmstudio_url = "http://localhost:1234/v1/chat/completions"

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
    """
    if not tok:
        return False
    i, j = 0, len(tok) - 1
    while i <= j and tok[i] in _PUNCT:
        i += 1
    while j >= i and tok[j] in _PUNCT:
        j -= 1
    core = tok[i:j+1]
    return core == "[REDACTED]"

# -------------------------
# Read prompts CSV & prepare folders
# -------------------------
try:
    prompt_df = pd.read_csv('Prompts.csv', encoding='utf-8')
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

def api_call(post_text, prompt):
    try:
        # return openai_api_call(post_text, prompt)  # <- switch to OpenAI if you want
        return llama3_lmstudio_call(post_text, prompt)  # default: LM Studio
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
# Alignment helpers
# -------------------------
def find_pattern_in_response(pattern, response_words):
    """
    Find where a pattern of words appears in the response.
    Allow [REDACTED] (with punctuation) as wildcard.
    """
    if len(pattern) > len(response_words):
        return -1
    for start_idx in range(len(response_words) - len(pattern) + 1):
        match = True
        for i in range(len(pattern)):
            resp_word = response_words[start_idx + i]
            pattern_word = pattern[i]
            if resp_word != pattern_word and not _is_redacted_with_punct(resp_word):
                match = False
                break
        if match:
            return start_idx
    return -1

def split_into_sentences(text, max_sentences=3):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunks.append(" ".join(sentences[i:i + max_sentences]))
    return chunks

def clean_llm_response(original_text, llm_response):
    """
    Keep [REDACTED] (with punctuation) aligned; otherwise prefer original words when mismatched.
    """
    original_normalized = normalize_whitespace(original_text)
    llm_normalized = normalize_whitespace(llm_response)
    original_words = original_normalized.split()
    llm_words = llm_normalized.split()

    if len(original_words) == len(llm_words) or len(original_words) == 0:
        return llm_normalized

    # Align start by first few words
    search_window = min(5, len(original_words))
    first_pattern = original_words[:search_window]
    start_position = find_pattern_in_response(first_pattern, llm_words)
    if start_position == -1:
        for i, w in enumerate(llm_words):
            if w == original_words[0] or _is_redacted_with_punct(w):
                start_position = i
                break
    if start_position == -1:
        return llm_normalized

    trimmed_response = llm_words[start_position:]
    original_sentences = split_into_sentences(original_normalized)
    cleaned_result = []
    resp_word_idx = 0

    for sentence in original_sentences:
        sentence_words = sentence.split()
        cleaned_sentence_words = []
        for chunk_start in range(0, len(sentence_words), 5):
            chunk_size = min(5, len(sentence_words) - chunk_start)
            orig_chunk = sentence_words[chunk_start:chunk_start + chunk_size]

            if resp_word_idx + chunk_size > len(trimmed_response):
                cleaned_sentence_words.extend(orig_chunk)
                break

            resp_chunk = trimmed_response[resp_word_idx:resp_word_idx + chunk_size]
            matches = 0
            mismatches = 0
            for i in range(chunk_size):
                if resp_chunk[i] == orig_chunk[i] or _is_redacted_with_punct(resp_chunk[i]):
                    matches += 1
                else:
                    mismatches += 1

            if mismatches == 0:
                cleaned_sentence_words.extend(resp_chunk)
                resp_word_idx += chunk_size
            elif mismatches <= 2 and (matches >= chunk_size / 2 or chunk_size <= 3):
                adjusted = [
                    resp_chunk[i] if (resp_chunk[i] == orig_chunk[i] or _is_redacted_with_punct(resp_chunk[i]))
                    else orig_chunk[i]
                    for i in range(chunk_size)
                ]
                cleaned_sentence_words.extend(adjusted)
                resp_word_idx += chunk_size
            else:
                remaining_resp = trimmed_response[resp_word_idx:]
                new_pos = find_pattern_in_response(orig_chunk, remaining_resp)
                if new_pos != -1:
                    resp_word_idx += new_pos
                    if resp_word_idx + chunk_size <= len(trimmed_response):
                        resp_chunk = trimmed_response[resp_word_idx:resp_word_idx + chunk_size]
                        cleaned_sentence_words.extend(resp_chunk)
                        resp_word_idx += chunk_size
                    else:
                        cleaned_sentence_words.extend(orig_chunk)
                else:
                    cleaned_sentence_words.extend(orig_chunk)
        cleaned_result.append(" ".join(cleaned_sentence_words))

    return " ".join(cleaned_result)

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
_TOKEN_RE = re.compile(r"\[REDACTED\](?:'s|’s)?|[A-Za-z0-9']+")

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

def enforce_word_count_match(original_text, processed_text):
    """
    Keep lengths equal at the whitespace-token level. This is a safety net
    and doesn't affect the punctuation-agnostic validation.
    """
    original_words = original_text.split()
    processed_words = processed_text.split()
    if len(original_words) == len(processed_words):
        return processed_text

    corrected_words = []
    proc_idx = 0
    for orig_word in original_words:
        if proc_idx < len(processed_words):
            proc_word = processed_words[proc_idx]
            if _is_redacted_with_punct(proc_word):
                corrected_words.append(proc_word)  # keep attached punctuation
                proc_idx += 1
            elif proc_word == orig_word:
                corrected_words.append(proc_word)
                proc_idx += 1
            else:
                corrected_words.append(orig_word)
        else:
            corrected_words.append(orig_word)
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
            enforced = enforce_word_count_match(word, cleaned_response)
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
            enforced = enforce_word_count_match(line_text.strip(), word_result)
            valid = is_valid_input(line_text.strip(), enforced)
            if valid:
                print(f"  [LINE->WORD OK] Fallback succeeded")
                return enforced
            else:
                print(f"  [LINE->WORD FAIL] Validation after fallback | enforced='{_preview(enforced)}'")
                return line_text.strip()

        cleaned_response = clean_llm_response(line_text.strip(), response)
        enforced = enforce_word_count_match(line_text.strip(), cleaned_response)
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
        enforced2 = enforce_word_count_match(line_text.strip(), word_result)
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
            enforced = enforce_word_count_match(line_text.strip(), word_result)
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
        enforced = enforce_word_count_match(chunk_text, cleaned_response)
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
        'chunk_failures': 0,
        'line_failures': 0,
        'word_failures': 0,
        'processing_time_minutes': 0,
        'start_time': time.time()
    }

    original_file_path = os.path.join(original_folder_path, original_file)
    new_file_path = os.path.join(output_folder, original_file)

    try:
        with open(original_file_path, 'r') as original_file_handle, open(new_file_path, 'w') as new_file_handle:
            for _, row in prompt_df.iterrows():
                prompt = row['prompt']

                original_file_handle.seek(0)
                all_lines = original_file_handle.readlines()

                if len(all_lines) > 1:
                    first_line = normalize_whitespace(all_lines[0].rstrip('\n'))
                    remaining_content = ''.join(all_lines[1:]).strip()
                    remaining_content = normalize_speaker_format(remaining_content)
                    remaining_content = remove_empty_speaker_lines(remaining_content)

                    # (Optional) write normalized back to original
                    with open(original_file_path, 'w') as orig_write:
                        orig_write.write(first_line + "\n\n" + remaining_content + "\n")
                    print(f"Normalized original file: {original_file}")

                    if remaining_content:
                        processed_content = process_text_content(remaining_content, prompt, api_call, file_stats)
                        processed_content = normalize_speaker_format(processed_content)
                        processed_content = remove_empty_speaker_lines(processed_content)
                        new_file_handle.write(first_line + "\n\n" + processed_content + "\n")
                        print(f"Processed file (skipped first line): {original_file}")
                    else:
                        new_file_handle.write(first_line + "\n")
                        print(f"File {original_file} has only one line, copied as-is")

                elif len(all_lines) == 1:
                    normalized_line = normalize_speaker_format(all_lines[0])
                    normalized_line = remove_empty_speaker_lines(normalized_line)
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

# -------------------------
# Write processing statistics
# -------------------------
stats_csv_path = 'processing_statistics.csv'
try:
    with open(stats_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'chunk_failures', 'line_failures', 'word_failures', 'processing_time_minutes']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for stats in processing_stats:
            writer.writerow(stats)

    print(f"\nProcessing statistics saved to {stats_csv_path}")

    total_files = len(processing_stats)
    total_chunk_failures = sum(stats['chunk_failures'] for stats in processing_stats)
    total_line_failures = sum(stats['line_failures'] for stats in processing_stats)
    total_word_failures = sum(stats['word_failures'] for stats in processing_stats)
    total_time = sum(stats['processing_time_minutes'] for stats in processing_stats)

    print(f"\n=== PROCESSING SUMMARY ===")
    print(f"Total files processed: {total_files}")
    print(f"Total chunk failures: {total_chunk_failures}")
    print(f"Total line failures: {total_line_failures}")
    print(f"Total word failures: {total_word_failures}")
    print(f"Total processing time: {total_time:.2f} minutes")
    print(f"Average processing time per file: {total_time/total_files:.2f} minutes" if total_files > 0 else "No files processed")

except Exception as e:
    print(f"Error writing statistics CSV: {str(e)}")
