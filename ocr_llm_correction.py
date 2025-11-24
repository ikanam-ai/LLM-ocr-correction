#!/usr/bin/env python3
"""Script to correct OCR outputs using an LLM (OpenAI API).

Usage examples:
- `python ocr_llm_correction.py --input train_data/ocr_results/ja-ocr_results.csv --output-dir corrected_results`
- `python ocr_llm_correction.py --input train_data/ocr_results/ --output-dir corrected_results`
"""

from __future__ import annotations
import argparse
import csv
import datetime
import json
import os
import time
import sys
from typing import Optional, List, Dict, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from ocr_prompt import PROMPT

# Configuration

LANGUAGE_MAP = {
    "ar": "Arabic",
    "ch_sim": "Chinese (Simplified)",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sw": "Swedish",
    "tr": "Turkish",
    "uk": "Ukrainian"
}

load_dotenv()

DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "")
API_KEY = os.environ.get("API_KEY", "EMPTY")
BASE_URL = os.environ.get("BASE_URL", "")

if not API_KEY:
    print("ERROR: Environment variable API_KEY is not set.")
    sys.exit(1)

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


def detect_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """Require the exact CSV schema: columns 'image', 'gt', 'ocr'.
    Return the actual column names (preserving original case) in order (image_col, ocr_col, gt_col).
    Raises ValueError if the required columns are missing.
    """
    required = ["image", "gt", "ocr"]
    cols_lower = [c.lower() for c in df.columns]
    missing = [r for r in required if r not in cols_lower]
    if missing:
        raise ValueError(
            f"Input CSV must contain columns: {', '.join(required)} (case-insensitive). Missing: {', '.join(missing)}. Found: {', '.join(list(df.columns))}"
        )
    # map to original-cased column names
    image_col = df.columns[cols_lower.index("image")]
    gt_col = df.columns[cols_lower.index("gt")]
    ocr_col = df.columns[cols_lower.index("ocr")]
    return image_col, ocr_col, gt_col


def sanitize_filename(name: str) -> str:
    """Return a filesystem-safe filename fragment from name."""
    if name is None:
        return "unknown"
    s = str(name).strip()
    if not s:
        return "unknown"
    return s


def make_prompt(original_text: str, language_hint: Optional[str] = None) -> str:
    """Construct an instruction prompt for the LLM to correct OCR text."""
    language = LANGUAGE_MAP.get(language_hint)
    prompt = PROMPT.format(language=language, text=original_text)
    return prompt


def call_chat_completion(messages: List[Dict], model: str, max_retries: int = 5, backoff_base: float = 1.0, timeout: int = 60) -> Dict:
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=2048,
                timeout=timeout,
            )
            return resp
        except Exception as e:
            last_exc = e
            wait = backoff_base * (2 ** (attempt - 1))
            print(f"API call failed on attempt {attempt}/{max_retries}: {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    raise last_exc


def process_dataframe(df: pd.DataFrame, source_csv: str, language: Optional[str], model: str, batch_size: int, out_csv_writer, output_dir: Optional[str] = None):
    # Enforce expected column names
    image_col, ocr_col, gt_col = detect_columns(df)
    total = len(df)
    processed = 0

    # iterate rows in batches to reduce number of requests for short texts
    for start in tqdm(range(0, total, batch_size)):
        end = min(total, start + batch_size)
        batch = df.iloc[start:end]
        for _, row in batch.iterrows():
            original_text = str(row[ocr_col]) if pd.notna(row[ocr_col]) else ""
            image_id = str(row[image_col]) if pd.notna(row[image_col]) else str(row.name)
            ground_truth = str(row[gt_col]) if pd.notna(row[gt_col]) else None

            # Prepare per-file path and skip if already exists
            safe_id = sanitize_filename(image_id)
            lang_folder = sanitize_filename(language) if language else "unknown"
            file_path = None
            if output_dir:
                folder_path = os.path.join(output_dir, lang_folder)
                filename = f"{safe_id}.json"
                file_path = os.path.join(folder_path, filename)
                if os.path.exists(file_path):
                    print(f"Skipping {image_id} (already processed: {file_path})")
                    continue

            # build prompt
            prompt = make_prompt(original_text, language)
            messages = [
                {"role": "system", "content": "You are a helpful assistant that corrects OCR output. Return only corrected text."},
                {"role": "user", "content": prompt}
            ]
            # call API
            resp = call_chat_completion(messages, model)
            corrected_text = resp.choices[0].message.content.strip()
            usage = resp.usage

            out_row = {
                "image_id": image_id,
                "language": language or "",
                "original_text": original_text,
                "ground_truth": ground_truth,
                "corrected_text": corrected_text,
                "model": model,
                "tokens_prompt": usage.prompt_tokens,
                "tokens_completion": usage.completion_tokens,
                "tokens_total": usage.total_tokens,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "source_csv": os.path.basename(source_csv),
            }

            json_entry = dict(out_row)

            # write per-row JSON file
            if output_dir:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as fh:
                    json.dump(json_entry, fh, ensure_ascii=False, indent=2)

            # write CSV row after successful processing
            out_csv_writer.writerow(out_row)
            processed += 1

    return processed


def process_file(input_csv: str, language: Optional[str], model: str, batch_size: int, out_csv_path: str, output_dir: Optional[str] = None):
    print(f"Processing {input_csv} (language={language})...")
    df = pd.read_csv(input_csv)
    # open outputs in append mode
    write_header = not os.path.exists(out_csv_path)
    with open(out_csv_path, "a", encoding="utf-8", newline="") as out_csv_fh:
        fieldnames = ["image_id", "language", "original_text", "ground_truth", "corrected_text", "model", "tokens_prompt", "tokens_completion", "tokens_total", "timestamp", "source_csv"]
        writer = csv.DictWriter(out_csv_fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        processed = process_dataframe(df, input_csv, language, model, batch_size, writer, output_dir)
    print(f"Finished {input_csv}: processed {processed} rows.")
    return processed


def collect_input_files(path: str) -> List[str]:
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith('.csv')]
        files.sort()
        return files
    elif os.path.isfile(path) and path.lower().endswith('.csv'):
        return [path]
    else:
        raise ValueError(f"Input {path} is neither a CSV file nor a directory with CSVs")


def main():
    parser = argparse.ArgumentParser(description="Correct OCR results using an LLM (OpenAI).")
    parser.add_argument("--input", type=str, help="Path to a CSV file or directory of CSVs with OCR results", required=True)
    parser.add_argument("--output-dir", type=str, default="corrected_results", help="Directory to write corrected CSV(s) and JSONL(s)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI model to use")
    parser.add_argument("--batch-size", type=int, default=1, help="How many rows to process per batch (default 1)")
    parser.add_argument("--language", type=str, default=None, help="Optional language hint for the LLM (e.g., ja, en)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = collect_input_files(args.input)

    summary = {"total_files": len(files), "total_processed": 0}
    out_csv_path = os.path.join(args.output_dir, "ocr_llm_corrected.csv")

    for f in files:
        # attempt to infer language from filename like 'en-ocr_results.csv'
        basename = os.path.basename(f)
        lang = args.language
        if not lang:
            if '-' in basename:
                lang = basename.split('-')[0]
        processed = process_file(f, lang, args.model, args.batch_size, out_csv_path, args.output_dir)
        summary["total_processed"] += processed

    print("Done.")
    print(summary)


if __name__ == "__main__":
    main()
