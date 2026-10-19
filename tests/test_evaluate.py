import pytest

from librispeechmix_timestamps.evaluate import _index_rows, aggregate, evaluate_sample


def reference(words):
    return {"id": "sample", "reference_words": words}


def prediction(speakers):
    words = []
    for speaker, items in speakers.items():
        words.extend({**item, "speaker": speaker} for item in items)
    return {"id": "sample", "words": words}


def test_start_and_end_f1_are_reported_separately_at_all_thresholds():
    ref = reference([{"word": "hello", "start": 0.0, "end": 0.5, "speaker": "speaker0"}])
    hyp = prediction({"hyp0": [{"word": "HELLO", "start": 0.15, "end": 0.56}]})
    metrics = aggregate([evaluate_sample(ref, hyp, (200, 100, 50))], (200, 100, 50))
    assert metrics["timestamp_metrics"]["200_ms"]["start"]["f1"] == 1.0
    assert metrics["timestamp_metrics"]["100_ms"]["start"]["f1"] == 0.0
    assert metrics["timestamp_metrics"]["100_ms"]["end"]["f1"] == 1.0
    assert metrics["timestamp_metrics"]["50_ms"]["end"]["f1"] == 0.0


def test_speaker_labels_are_permutation_invariant():
    ref = reference(
        [
            {"word": "red", "start": 0.0, "end": 0.2, "speaker": "speaker0"},
            {"word": "blue", "start": 0.1, "end": 0.3, "speaker": "speaker1"},
        ]
    )
    hyp = prediction(
        {
            "arbitrary-a": [{"word": "blue", "start": 0.1, "end": 0.3}],
            "arbitrary-b": [{"word": "red", "start": 0.0, "end": 0.2}],
        }
    )
    metrics = aggregate([evaluate_sample(ref, hyp, (50,))], (50,))
    assert metrics["wer"] == 0.0
    assert metrics["timestamp_metrics"]["50_ms"]["start"]["f1"] == 1.0
    assert metrics["timestamp_metrics"]["50_ms"]["end"]["f1"] == 1.0


def test_insertions_reduce_precision_and_deletions_reduce_recall():
    ref = reference(
        [
            {"word": "one", "start": 0.0, "end": 0.2, "speaker": "speaker0"},
            {"word": "two", "start": 0.2, "end": 0.4, "speaker": "speaker0"},
        ]
    )
    hyp = prediction(
        {
            "speaker0": [
                {"word": "one", "start": 0.0, "end": 0.2},
                {"word": "extra", "start": 0.2, "end": 0.3},
            ]
        }
    )
    metrics = aggregate([evaluate_sample(ref, hyp, (50,))], (50,))
    start = metrics["timestamp_metrics"]["50_ms"]["start"]
    assert start["precision"] == pytest.approx(0.5)
    assert start["recall"] == pytest.approx(0.5)
    assert start["f1"] == pytest.approx(0.5)


def test_duplicate_ids_fail():
    with pytest.raises(ValueError, match="Duplicate prediction ID"):
        _index_rows([{"id": "same"}, {"id": "same"}], "prediction")
