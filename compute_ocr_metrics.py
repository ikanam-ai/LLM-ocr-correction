"""
Compute OCR WER/CER metrics from JSONs with corrected results.

Usage example:
- `python compute_ocr_metrics.py --corrected-dir corrected_results/ --output ocr_aug_summary.csv`
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import Levenshtein
from tqdm import tqdm


def normalize_text(text: str) -> str:
    """Normalize whitespace so edits focus on textual content."""

    return " ".join(text.split())


@dataclass
class StageSummary:
    word_errors: int = 0
    char_errors: int = 0
    total_words: int = 0
    total_chars: int = 0

    def update(self, word_edits: int, char_edits: int, reference_words: int, reference_chars: int) -> None:
        self.word_errors += word_edits
        self.char_errors += char_edits
        self.total_words += reference_words
        self.total_chars += reference_chars

    def wer(self) -> float:
        return self.word_errors / self.total_words if self.total_words else 0.0

    def cer(self) -> float:
        return self.char_errors / self.total_chars if self.total_chars else 0.0

    def absorb(self, other: "StageSummary") -> None:
        self.word_errors += other.word_errors
        self.char_errors += other.char_errors
        self.total_words += other.total_words
        self.total_chars += other.total_chars


@dataclass
class LanguageMetrics:
    total_files: int = 0
    original: StageSummary = field(default_factory=StageSummary)
    corrected: StageSummary = field(default_factory=StageSummary)

    def update(self, orig_stats: StageSummary, corr_stats: StageSummary) -> None:
        self.total_files += 1
        self.original.update(
            orig_stats.word_errors,
            orig_stats.char_errors,
            orig_stats.total_words,
            orig_stats.total_chars,
        )
        self.corrected.update(
            corr_stats.word_errors,
            corr_stats.char_errors,
            corr_stats.total_words,
            corr_stats.total_chars,
        )

    def delta_wer(self) -> float:
        return self.original.wer() - self.corrected.wer()

    def delta_cer(self) -> float:
        return self.original.cer() - self.corrected.cer()


def iter_json_files(root: Path) -> Iterable[Tuple[str, Path]]:
    """Yield tuples of (language, json_path) for every JSON in the dataset."""

    for lang_dir in sorted(root.iterdir()):
        if not lang_dir.is_dir():
            continue
        language = lang_dir.name
        for json_path in lang_dir.rglob("*.json"):
            yield language, json_path


def collect_json_files(root: Path) -> List[Tuple[str, Path]]:
    """Return a list of all `(language, path)` pairs to enable progress tracking."""

    return list(iter_json_files(root))


def normalized_sequence_distance(seq1: Iterable[str], seq2: Iterable[str]) -> int:
    """Return Levenshtein distance between sequences by joining with a null delimiter."""

    delim = "\u0000"
    return Levenshtein.distance(delim.join(seq1), delim.join(seq2))


def compute_errors(reference: str, hypothesis: str) -> Tuple[int, int, int, int]:
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    word_edits = normalized_sequence_distance(reference_words, hypothesis_words)
    char_edits = Levenshtein.distance(reference, hypothesis)
    return word_edits, char_edits, len(reference_words), len(reference)


def compute_metrics_for_file(path: Path) -> Tuple[StageSummary, StageSummary]:
    """Return metric summaries for original OCR output and corrected text."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    reference = normalize_text(payload.get("ground_truth", ""))
    original = normalize_text(payload.get("original_text", ""))
    corrected = normalize_text(payload.get("corrected_text", ""))

    orig_stats = StageSummary()
    corr_stats = StageSummary()

    orig_edits = compute_errors(reference, original)
    corr_edits = compute_errors(reference, corrected)

    orig_stats.update(*orig_edits)
    corr_stats.update(*corr_edits)

    return orig_stats, corr_stats


def write_summary_csv(metrics: Dict[str, LanguageMetrics], output: Path) -> None:
    """Write aggregated metrics per language and overall totals."""

    fieldnames = [
        "language",
        "total_files",
        "orig_wer",
        "orig_cer",
        "corrected_wer",
        "corrected_cer",
        "delta_wer",
        "delta_cer",
        "total_words",
        "total_chars",
    ]
    overall = LanguageMetrics()
    for stats in metrics.values():
        overall.total_files += stats.total_files
        overall.original.absorb(stats.original)
        overall.corrected.absorb(stats.corrected)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for language, stats in sorted(metrics.items()):
            writer.writerow(
                {
                    "language": language,
                    "total_files": stats.total_files,
                    "orig_wer": f"{stats.original.wer():.6f}",
                    "orig_cer": f"{stats.original.cer():.6f}",
                    "corrected_wer": f"{stats.corrected.wer():.6f}",
                    "corrected_cer": f"{stats.corrected.cer():.6f}",
                    "delta_wer": f"{stats.delta_wer():.6f}",
                    "delta_cer": f"{stats.delta_cer():.6f}",
                    "total_words": stats.original.total_words,
                    "total_chars": stats.original.total_chars,
                }
            )
        writer.writerow(
            {
                "language": "all",
                "total_files": overall.total_files,
                "orig_wer": f"{overall.original.wer():.6f}",
                "orig_cer": f"{overall.original.cer():.6f}",
                "corrected_wer": f"{overall.corrected.wer():.6f}",
                "corrected_cer": f"{overall.corrected.cer():.6f}",
                "delta_wer": f"{overall.delta_wer():.6f}",
                "delta_cer": f"{overall.delta_cer():.6f}",
                "total_words": overall.original.total_words,
                "total_chars": overall.original.total_chars,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute OCR WER/CER stats from corrected JSONs.")
    parser.add_argument(
        "--corrected-dir",
        type=Path,
        default=Path("..", "corrected_results"),
        help="Path to the corrected_results directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("..", "corrected_results", "ocr_metrics_summary.csv"),
        help="Output CSV file path",
    )
    args = parser.parse_args()

    metrics: Dict[str, LanguageMetrics] = {}
    file_entries = collect_json_files(args.corrected_dir)
    for language, json_path in tqdm(file_entries, desc="Processing JSON", unit="file"):
        if language not in metrics:
            metrics[language] = LanguageMetrics()

        orig_stats, corr_stats = compute_metrics_for_file(json_path)
        metrics[language].update(orig_stats, corr_stats)

    write_summary_csv(metrics, args.output)
    print(f"Wrote summary CSV to {args.output}")


if __name__ == "__main__":
    main()