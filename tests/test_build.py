from pathlib import Path

import pytest

from librispeechmix_timestamps.build import _build_record, load_alignments


def test_condensed_alignment_and_mixture_delay(tmp_path: Path):
    book = tmp_path / "dev-clean" / "1" / "2"
    book.mkdir(parents=True)
    (book / "1-2.alignment.txt").write_text(
        '1-2-0000 ",HELLO,WORLD," "0.10,0.40,0.80,1.00"\n', encoding="utf-8"
    )
    alignments = load_alignments(tmp_path, {"dev-clean"})
    metadata = {
        "id": "dev-clean-1mix/dev-clean-1mix-0000",
        "mixed_wav": "dev-clean-1mix/dev-clean-1mix-0000.wav",
        "wavs": ["dev-clean/1/2/1-2-0000.wav"],
        "texts": ["HELLO WORLD"],
        "delays": [0.25],
        "durations": [1.0],
        "speakers": ["1"],
        "genders": ["f"],
    }
    record = _build_record(metadata, alignments, "dev-clean-1mix")
    assert record["duration"] == 1.25
    assert record["reference_words"] == [
        {"word": "HELLO", "start": 0.35, "end": 0.65, "speaker": "speaker0"},
        {"word": "WORLD", "start": 0.65, "end": 1.05, "speaker": "speaker0"},
    ]


def test_missing_alignment_fails_loudly():
    metadata = {
        "id": "dev-clean-1mix/x",
        "mixed_wav": "dev-clean-1mix/x.wav",
        "wavs": ["dev-clean/1/2/missing.wav"],
        "texts": ["HELLO"],
        "delays": [0.0],
        "durations": [1.0],
        "speakers": ["1"],
        "genders": ["m"],
    }
    with pytest.raises(KeyError, match="missing"):
        _build_record(metadata, {}, "dev-clean-1mix")
