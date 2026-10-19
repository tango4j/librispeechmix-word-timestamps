# Copyright 2026 tango4j
# SPDX-License-Identifier: Apache-2.0

"""Speaker-permutation-invariant word-boundary evaluation for LibriSpeechMix."""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import math
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

DEFAULT_TOLERANCES_MS = (200, 100, 50)


def _normalized_word(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    start = 0
    end = len(text)
    while start < end and unicodedata.category(text[start])[0] not in "LMN":
        start += 1
    while end > start and unicodedata.category(text[end - 1])[0] not in "LMN":
        end -= 1
    return text[start:end]


def _word_alignment(reference: list[str], hypothesis: list[str]) -> dict[str, Any]:
    """Return deterministic Levenshtein counts and exact-word index pairs."""
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    costs = [[0] * columns for _ in range(rows)]
    operations = [[""] * columns for _ in range(rows)]
    for i in range(1, rows):
        costs[i][0] = i
        operations[i][0] = "delete"
    for j in range(1, columns):
        costs[0][j] = j
        operations[0][j] = "insert"
    priority = {"equal": 0, "substitute": 1, "delete": 2, "insert": 3}
    for i in range(1, rows):
        for j in range(1, columns):
            diagonal = "equal" if reference[i - 1] == hypothesis[j - 1] else "substitute"
            candidates = [
                (costs[i - 1][j - 1] + (diagonal != "equal"), priority[diagonal], diagonal),
                (costs[i - 1][j] + 1, priority["delete"], "delete"),
                (costs[i][j - 1] + 1, priority["insert"], "insert"),
            ]
            cost, _, operation = min(candidates)
            costs[i][j] = cost
            operations[i][j] = operation

    matches = []
    counts = defaultdict(int)
    i, j = len(reference), len(hypothesis)
    while i or j:
        operation = operations[i][j]
        if operation == "equal":
            matches.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif operation == "substitute":
            counts["substitutions"] += 1
            i -= 1
            j -= 1
        elif operation == "delete":
            counts["deletions"] += 1
            i -= 1
        elif operation == "insert":
            counts["insertions"] += 1
            j -= 1
        else:
            raise RuntimeError(f"Invalid backtrace state at ({i}, {j})")
    matches.reverse()
    return {
        "errors": costs[-1][-1],
        "substitutions": counts["substitutions"],
        "insertions": counts["insertions"],
        "deletions": counts["deletions"],
        "matches": matches,
    }


def _validate_word(item: dict[str, Any], index: int, *, reference: bool) -> dict[str, Any] | None:
    word = item.get("word", item.get("text"))
    normalized = _normalized_word(word)
    if not normalized or normalized.split() != [normalized]:
        if reference:
            raise ValueError(f"reference_words[{index}] is not one scoreable word")
        return None
    start = item.get("start", item.get("start_time"))
    end = item.get("end", item.get("end_time"))
    if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        if reference:
            raise ValueError(f"reference_words[{index}] has invalid boundaries")
        return None
    start = float(start)
    end = float(end)
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
        if reference:
            raise ValueError(f"reference_words[{index}] has invalid interval ({start}, {end})")
        return None
    return {
        "word": str(word),
        "normalized": normalized,
        "start": start,
        "end": end,
        "speaker": str(item.get("speaker", item.get("speaker_tag", "speaker0"))),
    }


def _timestamp_payload(row: dict[str, Any]) -> object:
    if "words" not in row:
        raise ValueError("Prediction row has no top-level words list")
    return row["words"]


def _reference_words(row: dict[str, Any]) -> list[dict[str, Any]]:
    items = row.get("reference_words")
    if not isinstance(items, list) or not items:
        raise ValueError("Reference row has no non-empty reference_words list")
    words = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(f"reference_words[{index}] must be an object")
        validated = _validate_word(item, index, reference=True)
        assert validated is not None
        words.append(validated)
    return words


def _prediction_words(row: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    payload = _timestamp_payload(row)
    malformed = 0
    if not isinstance(payload, list):
        raise TypeError("Prediction words must be a list")
    words = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            malformed += 1
            continue
        validated = _validate_word(item, index, reference=False)
        if validated is None:
            malformed += 1
        else:
            words.append(validated)
    return sorted(words, key=lambda item: (item["start"], item["end"], item["speaker"])), malformed


def _by_speaker(words: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for word in words:
        grouped[word["speaker"]].append(word)
    return dict(grouped)


def _speaker_assignment(
    reference: dict[str, list[dict[str, Any]]], hypothesis: dict[str, list[dict[str, Any]]]
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    ref_streams = list(reference.values())
    hyp_streams = list(hypothesis.values())
    size = max(len(ref_streams), len(hyp_streams))
    if size > 8:
        raise ValueError(f"At most 8 speakers are supported, got {size}")
    ref_streams.extend([[] for _ in range(size - len(ref_streams))])
    hyp_streams.extend([[] for _ in range(size - len(hyp_streams))])

    def cost(ref_items: list[dict[str, Any]], hyp_items: list[dict[str, Any]]) -> int:
        return int(
            _word_alignment(
                [item["normalized"] for item in ref_items],
                [item["normalized"] for item in hyp_items],
            )["errors"]
        )

    costs = [[cost(ref_items, hyp_items) for hyp_items in hyp_streams] for ref_items in ref_streams]
    permutation = min(itertools.permutations(range(size)), key=lambda p: sum(costs[i][p[i]] for i in range(size)))
    return [(ref_streams[i], hyp_streams[j]) for i, j in enumerate(permutation)]


def evaluate_sample(
    reference_row: dict[str, Any],
    prediction_row: dict[str, Any],
    tolerances_ms: Iterable[int] = DEFAULT_TOLERANCES_MS,
) -> dict[str, Any]:
    """Return additive counts for one recording."""
    tolerances = tuple(int(value) for value in tolerances_ms)
    if not tolerances or any(value < 0 for value in tolerances):
        raise ValueError("Tolerances must be non-negative milliseconds")
    reference = _reference_words(reference_row)
    hypothesis, malformed = _prediction_words(prediction_row)
    totals: dict[str, Any] = defaultdict(int)
    totals["start_error_seconds_sum"] = 0.0
    totals["end_error_seconds_sum"] = 0.0
    start_true_positives = {value: 0 for value in tolerances}
    end_true_positives = {value: 0 for value in tolerances}
    assignment = _speaker_assignment(_by_speaker(reference), _by_speaker(hypothesis))
    for ref_items, hyp_items in assignment:
        alignment = _word_alignment(
            [item["normalized"] for item in ref_items],
            [item["normalized"] for item in hyp_items],
        )
        for name in ("errors", "substitutions", "insertions", "deletions"):
            totals[name] += alignment[name]
        for ref_index, hyp_index in alignment["matches"]:
            ref_item = ref_items[ref_index]
            hyp_item = hyp_items[hyp_index]
            start_error = abs(ref_item["start"] - hyp_item["start"])
            end_error = abs(ref_item["end"] - hyp_item["end"])
            totals["matched_words"] += 1
            totals["start_error_seconds_sum"] += start_error
            totals["end_error_seconds_sum"] += end_error
            for tolerance_ms in tolerances:
                tolerance = tolerance_ms / 1000.0
                start_true_positives[tolerance_ms] += int(start_error <= tolerance + 1e-12)
                end_true_positives[tolerance_ms] += int(end_error <= tolerance + 1e-12)
    return {
        "id": row_id(reference_row),
        "reference_words": len(reference),
        "hypothesis_words": len(hypothesis),
        "errors": totals["errors"],
        "substitutions": totals["substitutions"],
        "insertions": totals["insertions"],
        "deletions": totals["deletions"],
        "matched_words": totals["matched_words"],
        "start_error_seconds_sum": totals["start_error_seconds_sum"],
        "end_error_seconds_sum": totals["end_error_seconds_sum"],
        "start_true_positives": start_true_positives,
        "end_true_positives": end_true_positives,
        "malformed_timestamp_words": malformed,
    }


def _boundary_metrics(true_positives: int, references: int, hypotheses: int) -> dict[str, float | int]:
    precision = true_positives / hypotheses if hypotheses else 0.0
    recall = true_positives / references if references else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "true_positives": true_positives,
        "reference_events": references,
        "hypothesis_events": hypotheses,
    }


def aggregate(samples: list[dict[str, Any]], tolerances_ms: Iterable[int]) -> dict[str, Any]:
    tolerances = tuple(int(value) for value in tolerances_ms)
    counts = {
        key: sum(int(sample[key]) for sample in samples)
        for key in (
            "reference_words",
            "hypothesis_words",
            "errors",
            "substitutions",
            "insertions",
            "deletions",
            "matched_words",
            "malformed_timestamp_words",
        )
    }
    if not counts["reference_words"]:
        raise ValueError("No reference words were scored")
    start_sum = sum(float(sample["start_error_seconds_sum"]) for sample in samples)
    end_sum = sum(float(sample["end_error_seconds_sum"]) for sample in samples)
    timestamp_metrics = {}
    for tolerance_ms in tolerances:
        start_tp = sum(int(sample["start_true_positives"][tolerance_ms]) for sample in samples)
        end_tp = sum(int(sample["end_true_positives"][tolerance_ms]) for sample in samples)
        timestamp_metrics[f"{tolerance_ms}_ms"] = {
            "start": _boundary_metrics(start_tp, counts["reference_words"], counts["hypothesis_words"]),
            "end": _boundary_metrics(end_tp, counts["reference_words"], counts["hypothesis_words"]),
        }
    matched = counts["matched_words"]
    return {
        "scored_records": len(samples),
        "wer": round(counts["errors"] / counts["reference_words"], 6),
        **counts,
        "start_mae_ms": round(1000.0 * start_sum / matched, 3) if matched else None,
        "end_mae_ms": round(1000.0 * end_sum / matched, 3) if matched else None,
        "timestamp_metrics": timestamp_metrics,
    }


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with _open_text(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def row_id(row: dict[str, Any]) -> str:
    for key in ("id", "sample_id", "audio_filepath", "mixed_wav"):
        if row.get(key):
            return str(row[key])
    raise ValueError("Every row needs id, sample_id, audio_filepath, or mixed_wav")


def _index_rows(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed = {}
    for row in rows:
        identifier = row_id(row)
        if identifier in indexed:
            raise ValueError(f"Duplicate {label} ID: {identifier}")
        indexed[identifier] = row
    return indexed


def evaluate_files(
    reference_path: Path,
    prediction_path: Path,
    tolerances_ms: Iterable[int] = DEFAULT_TOLERANCES_MS,
    scored_output: Path | None = None,
) -> dict[str, Any]:
    tolerances = tuple(int(value) for value in tolerances_ms)
    references = _index_rows(read_jsonl(reference_path), "reference")
    predictions = _index_rows(read_jsonl(prediction_path), "prediction")
    if set(references) != set(predictions):
        missing = sorted(set(references) - set(predictions))
        extra = sorted(set(predictions) - set(references))
        raise ValueError(f"Reference/prediction IDs differ; missing={missing[:10]}, extra={extra[:10]}")
    samples = [evaluate_sample(references[key], predictions[key], tolerances) for key in references]
    if scored_output is not None:
        scored_output.parent.mkdir(parents=True, exist_ok=True)
        with scored_output.open("wt", encoding="utf-8") as stream:
            for sample in samples:
                stream.write(json.dumps(sample, sort_keys=True) + "\n")
    return aggregate(samples, tolerances)


def _print_summary(metrics: dict[str, Any]) -> None:
    print(f"records={metrics['scored_records']}  WER={100 * metrics['wer']:.2f}%")
    print("tolerance  start-P  start-R  start-F1    end-P    end-R    end-F1")
    for label, values in metrics["timestamp_metrics"].items():
        start = values["start"]
        end = values["end"]
        print(
            f"{label:>9}  {100 * start['precision']:7.2f}  {100 * start['recall']:7.2f}  "
            f"{100 * start['f1']:8.2f}  {100 * end['precision']:7.2f}  "
            f"{100 * end['recall']:7.2f}  {100 * end['f1']:8.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Reference .jsonl or .jsonl.gz")
    parser.add_argument("--predictions", type=Path, required=True, help="Prediction .jsonl or .jsonl.gz")
    parser.add_argument("--tolerances-ms", type=int, nargs="+", default=list(DEFAULT_TOLERANCES_MS))
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--scored-output", type=Path)
    args = parser.parse_args()
    metrics = evaluate_files(args.reference, args.predictions, args.tolerances_ms, args.scored_output)
    _print_summary(metrics)
    if args.metrics_output is not None:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
