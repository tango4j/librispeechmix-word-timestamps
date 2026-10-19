# LibriSpeechMix Word Timestamps

[![CI](https://github.com/tango4j/librispeechmix-word-timestamps/actions/workflows/ci.yml/badge.svg)](https://github.com/tango4j/librispeechmix-word-timestamps/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/code%20license-Apache--2.0-blue.svg)](LICENSE)

Reproducible, speaker-aware word timestamps for the official **LibriSpeechMix dev-clean and test-clean 1-, 2-, and 3-speaker evaluation sets**.

The repository contains annotations and evaluation code only—**no audio**. Mixture definitions come from [Naoyuki Kanda's LibriSpeechMix](https://github.com/NaoyukiKanda/LibriSpeechMix), and source-utterance boundaries come from [CorentinJ's LibriSpeech alignments](https://github.com/CorentinJ/librispeech-alignments). Each source boundary is shifted by the exact LibriSpeechMix delay, preserving speaker identity and overlap in the mixed-audio clock.

## Data

| File | Mixtures | Speakers | Word intervals |
|---|---:|---:|---:|
| `data/dev-clean-1mix.jsonl.gz` | 2,703 | 1 | 54,402 |
| `data/dev-clean-2mix.jsonl.gz` | 2,703 | 2 | 108,804 |
| `data/dev-clean-3mix.jsonl.gz` | 2,703 | 3 | 163,206 |
| `data/test-clean-1mix.jsonl.gz` | 2,620 | 1 | 52,576 |
| `data/test-clean-2mix.jsonl.gz` | 2,620 | 2 | 105,152 |
| `data/test-clean-3mix.jsonl.gz` | 2,620 | 3 | 157,728 |
| **Total** | **15,969** | — | **641,868** |

Clone the repository to obtain every annotation file:

```bash
git clone https://github.com/tango4j/librispeechmix-word-timestamps.git
cd librispeechmix-word-timestamps
gzip -cd data/test-clean-2mix.jsonl.gz | head -n 1
```

The release is complete for all six canonical LibriSpeechMix lists. The condensed alignment source has known omissions elsewhere in LibriSpeech, but none intersects `dev-clean` or `test-clean` here. File counts, source revisions, input fingerprints, and output SHA-256 hashes are pinned in [`data/MANIFEST.json`](data/MANIFEST.json).

## Schema

Each JSONL row represents one mixed recording:

```json
{
  "id": "dev-clean-2mix/dev-clean-2mix-0000",
  "mixed_wav": "dev-clean-2mix/dev-clean-2mix-0000.wav",
  "split": "dev-clean",
  "num_speakers": 2,
  "duration": 14.899,
  "sources": [
    {
      "speaker": "speaker0",
      "speaker_id": "1272",
      "gender": "m",
      "utterance_id": "1272-128104-0000",
      "source_audio": "dev-clean/1272/128104/1272-128104-0000.wav",
      "delay": 0.0,
      "duration": 5.855,
      "text": "MISTER QUILTER IS ..."
    }
  ],
  "reference_words": [
    {"word": "MISTER", "start": 0.5, "end": 0.8, "speaker": "speaker0"}
  ]
}
```

Times are seconds on the **mixed-audio timeline**, rounded to one millisecond. `reference_words` is globally ordered by `(start, end, speaker)`; use the speaker label to recover each speaker stream. `sources` preserves the exact original utterance, transcript, delay, and speaker metadata.

## Recommended timestamp evaluation

This repository recommends reporting **micro-averaged precision, recall, and F1 independently for word starts and word ends at 200 ms, 100 ms, and 50 ms**. The standalone evaluator reports both boundary types at all three tolerances in one run.

The protocol is:

1. Normalize case and Unicode at the word level.
2. Choose the hypothesis-to-reference speaker permutation with minimum word edit distance.
3. Use deterministic Levenshtein alignment within each paired speaker stream.
4. Only exact lexical matches can become timestamp true positives.
5. Score start and end boundaries as independent events. A boundary is correct when `absolute_error <= tolerance`.
6. For either boundary type, use every hypothesis word as a precision event and every reference word as a recall event, then micro-average over the corpus.

This keeps recognition mistakes from receiving timestamp credit and makes speaker labels permutation-invariant. Report WER alongside boundary F1 so timestamp quality is never interpreted without transcription quality.

Install and run:

```bash
python -m pip install -e .

lsmix-ts-eval \
  --reference data/test-clean-2mix.jsonl.gz \
  --predictions /path/to/test-clean-2mix.predictions.jsonl \
  --tolerances-ms 200 100 50 \
  --metrics-output metrics.json \
  --scored-output scored.jsonl
```

Example console output:

```text
records=2620  WER=7.42%
tolerance  start-P  start-R  start-F1    end-P    end-R    end-F1
   200_ms    ...      ...       ...       ...      ...       ...
   100_ms    ...      ...       ...       ...      ...       ...
    50_ms    ...      ...       ...       ...      ...       ...
```

Predictions may be a speaker-keyed mapping:

```json
{
  "id": "dev-clean-2mix/dev-clean-2mix-0000",
  "words": [
    {"word": "mister", "start": 0.48, "end": 0.81, "speaker": "hypothesis-A"},
    {"word": "that", "start": 4.61, "end": 4.91, "speaker": "hypothesis-B"}
  ]
}
```

The speaker names do not need to match the reference names. Predictions use one top-level `words` list; every item contains `word`, `start`, `end`, and `speaker`. Reference and prediction ID sets must match exactly; missing, extra, and duplicate IDs fail rather than producing a partial score.

## Rebuild the annotations

The generator validates the six upstream list hashes pinned at LibriSpeechMix commit `869aa26` and fingerprints all 184 relevant condensed alignment files.

```bash
git clone https://github.com/NaoyukiKanda/LibriSpeechMix.git

# Download and extract the condensed archive linked by CorentinJ's repository.
# Point this variable at the extracted directory containing dev-clean/ and test-clean/.
ALIGNMENT_ROOT=/path/to/LibriSpeech-Alignments/LibriSpeech

python -m pip install -e .
lsmix-ts-build \
  --mix-list-root LibriSpeechMix/list \
  --alignment-root "$ALIGNMENT_ROOT" \
  --output-dir data
```

The gzip writer is deterministic (`mtime=0`), so identical inputs and code reproduce the published hashes.

## Provenance and limitations

- LibriSpeechMix is derived from LibriSpeech `dev-clean` and `test-clean`; each original utterance appears exactly *N* times in the *N*-speaker set.
- The timestamps are automatic Montreal Forced Aligner outputs, not human-verified boundary labels.
- In the condensed source format, word starts are the end of the preceding word or silence interval. Initial and internal silences are therefore preserved correctly when empty tokens are omitted.
- These files describe the mixtures generated by the pinned upstream metadata. They do not redistribute the source or mixed waveforms.
- Do not train or tune boundary calibration on these evaluation annotations if the same split will be reported as a held-out benchmark.

See [`DATA_LICENSE.md`](DATA_LICENSE.md) for the distinction between this repository's code license and the upstream data terms.

## Citation

Please cite LibriSpeechMix and the source corpus when using these annotations:

```bibtex
@inproceedings{kanda2020serialized,
  title={Serialized Output Training for End-to-End Overlapped Speech Recognition},
  author={Kanda, Naoyuki and Gaur, Yashesh and Wang, Xiaofei and Meng, Zhong and Yoshioka, Takuya},
  booktitle={Interspeech},
  pages={2797--2801},
  year={2020}
}

@inproceedings{panayotov2015librispeech,
  title={LibriSpeech: An ASR Corpus Based on Public Domain Audio Books},
  author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle={ICASSP},
  year={2015}
}
```

## Acknowledgements

- [NaoyukiKanda/LibriSpeechMix](https://github.com/NaoyukiKanda/LibriSpeechMix) for the reproducible mixture definitions.
- [CorentinJ/librispeech-alignments](https://github.com/CorentinJ/librispeech-alignments) for the LibriSpeech forced alignments.
