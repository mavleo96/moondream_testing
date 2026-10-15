# Moondream × BLINK Visual Correspondence

Tests Moondream (`moondream3.1-9B-A2B`) on the Visual Correspondence task of the
[BLINK benchmark](https://huggingface.co/datasets/BLINK-Benchmark/BLINK), val split (172 samples),
via the [Moondream Batch API](https://docs.moondream.ai/batch).

## Tests

| Test | Image | `reasoning` | Purpose |
|---|---|---|---|
| `actual_reasoning` | reference + query side by side | true | benchmark accuracy |
| `actual_noreasoning` | reference + query side by side | false | effect of chain-of-thought |
| `blind` | reference + black right half | true | option bias (model can't see the candidate points) |

Each sample: `image_1` (reference, REF point circled) and `image_2` (query, candidate points A–D)
are concatenated horizontally with a 10px gap. The blind variant keeps the same canvas size but
leaves the right half black. Both images are always 1024px tall.

Prompt: the dataset's `prompt` field with "first image" → "left image" and
"second image" → "right image". Settings: `temperature: 0`.

Val answer distribution (blind baseline): A=51, B=33, C=44, D=44.

## Layout

```
scripts/
  process_data.py    build concatenated PNGs from the HF dataset
  prepare_jsonl.py   build batch API input JSONLs + ground truth
  run_test.sh        upload → poll → download one batch test
data/
  images/{actual,blind}/   172 PNGs each
  batches/*.jsonl          batch inputs (query skill, base64 PNG per line)
  ground_truth.json        idx → {answer, choices}
results/                   batch outputs + batches.log
```

## Run

Requires `MOONDREAM_API_KEY` in `.env`, plus `jq`. Python deps: `datasets`, `pillow`, `tqdm`.

```bash
python3 scripts/process_data.py     # regenerate images (only needed once)
python3 scripts/prepare_jsonl.py    # regenerate JSONLs + ground truth

scripts/run_test.sh data/batches/actual_reasoning.jsonl   actual_reasoning
scripts/run_test.sh data/batches/actual_noreasoning.jsonl actual_noreasoning
scripts/run_test.sh data/batches/blind.jsonl              blind
```

`run_test.sh` uploads in 50MB parts, polls every 60s, writes `results/<name>.jsonl`
(one line per sample: `id`, `status`, `result.answer`, `result.reasoning`, `usage`).
Batch ids are logged to `results/batches.log`; results expire server-side after 7 days.
Model override: `MOONDREAM_MODEL=... scripts/run_test.sh ...`
