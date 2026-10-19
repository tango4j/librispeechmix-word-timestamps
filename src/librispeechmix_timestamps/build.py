# Copyright 2026 tango4j
# SPDX-License-Identifier: Apache-2.0

"""Build LibriSpeechMix word-timestamp manifests from the canonical metadata."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Iterable, TextIO

MIXES = (
    "dev-clean-1mix",
    "dev-clean-2mix",
    "dev-clean-3mix",
    "test-clean-1mix",
    "test-clean-2mix",
    "test-clean-3mix",
)
LIBRISPEECHMIX_COMMIT = "869aa26ab37b60418898549447ff62d596c73ec6"
ALIGNMENTS_COMMIT = "a34135ebf3a781e3de0b9f7fb073ca3a76311cf7"
EXPECTED_LIST_SHA256 = {
    "dev-clean-1mix": "6cd4a30e8aa1fdb41a077ae57e67f5cedbf2d552097cea2b63ecf1522cd7448e",
    "dev-clean-2mix": "27137e7a3293ddbb97f12584449e25d025f7581205e98b7c59787bb223c98abc",
    "dev-clean-3mix": "06f208aeefb7b363c49fafcbbc13374063e338a4bd0ae4fc6598c022b80e2399",
    "test-clean-1mix": "7c5a40322f7e578cc6e47eb8d68cfd6b7cb20944a980f2a6afdc7ac7db9b1076",
    "test-clean-2mix": "95a01945c855c1ec7d1dc6105beff70e6d35e3372a5e2bec40bcaa4b372d216d",
    "test-clean-3mix": "b98134a8c5e5739c53d12a0e920242ee380c3be00dce6e5b77331d7d26db695e",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def _alignment_files(root: Path, splits: set[str]) -> list[Path]:
    paths = []
    for split in sorted(splits):
        paths.extend(sorted((root / split).glob("*/*/*.alignment.txt")))
    if not paths:
        raise FileNotFoundError(f"No *.alignment.txt files found below {root}")
    return paths


def alignment_tree_sha256(root: Path, splits: set[str]) -> tuple[str, int]:
    """Hash relevant relative paths and bytes as one reproducibility fingerprint."""
    digest = hashlib.sha256()
    paths = _alignment_files(root, splits)
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest(), len(paths)


def load_alignments(root: Path, splits: set[str]) -> dict[str, list[dict]]:
    """Load the condensed CorentinJ alignment format into utterance intervals."""
    utterances: dict[str, list[dict]] = {}
    for path in _alignment_files(root, splits):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"Malformed alignment at {path}:{line_number}")
            utterance_id, token_field, end_field = fields
            tokens = token_field.strip('"').split(",")
            try:
                ends = [float(value) for value in end_field.strip('"').split(",")]
            except ValueError as error:
                raise ValueError(f"Invalid end time at {path}:{line_number}") from error
            if len(tokens) != len(ends):
                raise ValueError(f"Token/time count mismatch at {path}:{line_number}")
            words = []
            start = 0.0
            for token, end in zip(tokens, ends):
                if end < start:
                    raise ValueError(f"Non-monotonic times at {path}:{line_number}")
                if token:
                    words.append({"word": token, "start": start, "end": end})
                start = end
            if utterance_id in utterances:
                raise ValueError(f"Duplicate alignment for {utterance_id}")
            utterances[utterance_id] = words
    return utterances


def _open_reproducible_gzip(path: Path) -> TextIO:
    raw = path.open("wb")
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    return io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")


def _build_record(metadata: dict, alignments: dict[str, list[dict]], mix: str) -> dict:
    expected = ("wavs", "texts", "delays", "durations", "speakers", "genders")
    if any(not isinstance(metadata.get(key), list) for key in expected):
        raise ValueError(f"{metadata.get('id', '<unknown>')} is missing list metadata")
    lengths = {len(metadata[key]) for key in expected}
    if lengths != {int(mix[-4])}:
        raise ValueError(f"Unexpected speaker metadata lengths for {metadata.get('id')}: {lengths}")

    sources = []
    reference_words = []
    for speaker_index, (source_audio, text, delay, duration, speaker_id, gender) in enumerate(
        zip(*(metadata[key] for key in expected))
    ):
        utterance_id = Path(source_audio).stem
        if utterance_id not in alignments:
            raise KeyError(f"No condensed alignment for {utterance_id} ({metadata['id']})")
        aligned = alignments[utterance_id]
        transcript_words = text.split()
        aligned_words = [item["word"] for item in aligned]
        if aligned_words != transcript_words:
            raise ValueError(
                f"Transcript mismatch for {utterance_id}: metadata={transcript_words!r}, alignment={aligned_words!r}"
            )
        speaker = f"speaker{speaker_index}"
        delay = float(delay)
        duration = float(duration)
        sources.append(
            {
                "speaker": speaker,
                "speaker_id": str(speaker_id),
                "gender": str(gender),
                "utterance_id": utterance_id,
                "source_audio": str(source_audio),
                "delay": delay,
                "duration": duration,
                "text": str(text),
            }
        )
        for word in aligned:
            reference_words.append(
                {
                    "word": word["word"],
                    "start": round(float(word["start"]) + delay, 3),
                    "end": round(float(word["end"]) + delay, 3),
                    "speaker": speaker,
                }
            )
    reference_words.sort(key=lambda item: (item["start"], item["end"], item["speaker"]))
    return {
        "id": str(metadata["id"]),
        "mixed_wav": str(metadata["mixed_wav"]),
        "split": mix.rsplit("-", 1)[0],
        "num_speakers": len(sources),
        "duration": round(max(item["delay"] + item["duration"] for item in sources), 3),
        "sources": sources,
        "reference_words": reference_words,
    }


def build_annotations(
    mix_list_root: Path,
    alignment_root: Path,
    output_dir: Path,
    *,
    validate_pinned_lists: bool = True,
) -> dict:
    """Build all six annotations and return the generated manifest."""
    source_paths = {mix: mix_list_root / f"{mix}.jsonl" for mix in MIXES}
    for mix, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256(path)
        if validate_pinned_lists and actual_hash != EXPECTED_LIST_SHA256[mix]:
            raise ValueError(f"Upstream list hash changed for {mix}: {actual_hash}")

    splits = {"dev-clean", "test-clean"}
    alignment_hash, alignment_file_count = alignment_tree_sha256(alignment_root, splits)
    alignments = load_alignments(alignment_root, splits)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    totals = {"records": 0, "reference_words": 0}
    for mix, source_path in source_paths.items():
        output_path = output_dir / f"{mix}.jsonl.gz"
        records = words = 0
        with _open_reproducible_gzip(output_path) as stream:
            for metadata in _read_jsonl(source_path):
                record = _build_record(metadata, alignments, mix)
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                records += 1
                words += len(record["reference_words"])
        files[output_path.name] = {
            "sha256": sha256(output_path),
            "records": records,
            "reference_words": words,
            "source_list_sha256": sha256(source_path),
        }
        totals["records"] += records
        totals["reference_words"] += words

    manifest = {
        "schema_version": "1.0",
        "timestamp_unit": "seconds",
        "timestamp_precision_decimals": 3,
        "sources": {
            "librispeechmix": {
                "url": "https://github.com/NaoyukiKanda/LibriSpeechMix",
                "commit": LIBRISPEECHMIX_COMMIT,
            },
            "librispeech_alignments": {
                "url": "https://github.com/CorentinJ/librispeech-alignments",
                "commit": ALIGNMENTS_COMMIT,
                "alignment_tree_sha256": alignment_hash,
                "alignment_files": alignment_file_count,
            },
        },
        "files": files,
        "totals": totals,
    }
    manifest_path = output_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix-list-root", type=Path, required=True, help="LibriSpeechMix list/ directory")
    parser.add_argument(
        "--alignment-root",
        type=Path,
        required=True,
        help="CorentinJ condensed LibriSpeech alignment root containing dev-clean/ and test-clean/",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--allow-unpinned-lists",
        action="store_true",
        help="Build from list files whose hashes differ from the pinned upstream commit",
    )
    args = parser.parse_args()
    manifest = build_annotations(
        args.mix_list_root,
        args.alignment_root,
        args.output_dir,
        validate_pinned_lists=not args.allow_unpinned_lists,
    )
    print(json.dumps(manifest["totals"], sort_keys=True))


if __name__ == "__main__":
    main()
