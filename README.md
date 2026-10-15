# Moondream × BLINK Visual Correspondence

Tests Moondream on the Visual Correspondence task of the
[BLINK benchmark](https://huggingface.co/datasets/BLINK-Benchmark/BLINK), val split (172 samples).
`moondream3.1-9B-A2B` is tested via the [Moondream Batch API](https://docs.moondream.ai/batch);
`moondream2` is tested locally via [Photon](https://docs.moondream.ai/running-locally/) (the `moondream` pip package).

## Tests

| Test | Image | `reasoning` | Purpose |
|---|---|---|---|
| `full_reasoning_1024px` | reference + query side by side, full res | true | benchmark accuracy |
| `full_noreasoning_1024px` | reference + query side by side, full res | false | effect of chain-of-thought |
| `blind_1024px` | reference + black right half, full res | true | option bias (model can't see the candidate points) |
| `full_reasoning_358px` | reference + query, downscaled to 358px tall | true | ablation: does multi-crop detail matter? |

Each sample: `image_1` (reference, REF point circled) and `image_2` (query, candidate points A–D)
are concatenated horizontally with a 10px gap, both always 1024px tall at full res. The blind
variant keeps the same canvas size but leaves the right half black. The 358px variant downscales
the full composite before encoding (see Methodology notes below for why 358px specifically).

Prompt: the dataset's `prompt` field with "first image" → "left image" and
"second image" → "right image". Settings: `temperature: 0`.

Val answer distribution (ground truth): A=51 (29.7%), B=33 (19.2%), C=44 (25.6%), D=44 (25.6%).

## Results

See [RESULTS.md](RESULTS.md) for full accuracy, answer-distribution, recall, and cross-model
agreement tables for both moondream2 and moondream3.1-9B-A2B.

## Methodology notes

- **Crop-boundary seam**: Moondream's vision encoder (`moondream/torch/image_crops.py` in the
  official repo) tiles images into overlapping 378×378 crops. Across all 27 distinct composite
  sizes in the val set, only 1–2 of 3–7 tile columns ever straddle the reference/query seam; the
  rest + a global context crop stay clean. Not a reason to change the image layout.
- **Multi-image (`images` array) is broken on the batch API**: the `query` skill's documented
  `images` array (meant for multi-image queries, as an alternative to concatenation) returns
  `image_decode_failed` for any 2+ element array in live testing — even two identical images —
  while 1-element arrays work fine. Confirmed via five real batch submissions. Concatenation
  remains the only working way to show the model two images in one request.
- **Resolution vs. tile count is a cliff, not a gradient**: tile count is governed by aspect ratio
  against a fixed `max_crops=12` budget, not by absolute resolution — it stays flat (~10-12 local
  crops) for any downscale factor down to a hard threshold at `height <= 378px`. Below that,
  every sample instantly collapses to a single squished 378×378 crop, destroying aspect ratio.
  358px (scale 0.35 on a 1024px-tall composite) sits safely below that threshold, used
  deliberately for `full_reasoning_358px` as a forced single-crop ablation.
- **moondream2 does not run via Photon on Apple Silicon (confirmed bug, not our code)**:
  `md.photon("moondream2")` fails with `ValueError: preallocated k_scale_tensor must be a device
  FP32 scalar` inside `kestrel`'s KV-cache init. Root cause confirmed directly:
  `torch.device('mps') != torch.device('mps:0')` in PyTorch, but a tensor created with
  `device='mps'` materializes as `device='mps:0'` — kestrel creates the scalar on `pool.device`
  (bare `'mps'`) then checks identity against it, which always fails. No newer package version
  fixes this (tested against latest `moondream==2.6.1`). Resolution: run on CUDA (e.g. an RTX
  3060) instead, where this device-identity quirk doesn't apply. `scripts/run_local.py` needs no
  code changes for that — `md.photon(...)` auto-selects the available accelerator.
- **moondream2 reasoning is a real, distinct mechanism, not a copied moondream3 flag**: confirmed
  from `kestrel` source (`kestrel/models/moondream/config.py`) that `DEFAULT_MOONDREAM2_CONFIG`
  defines its own `reasoning_prefix`/`post_reasoning_prefix`/token-suppression lists, different
  from the shared moondream3 template — so the planned reasoning on/off comparison is meaningful
  for moondream2 too.
- Only two cloud model ids exist for the batch API: `moondream3.1-9B-A2B` and `moondream3-preview`
  (plus finetuned checkpoints in `{base}/{id}@{step}` form). `moondream2` is open-weights/local
  only, not served by the batch API — hence the separate `run_local.py` path via Photon.

## Layout

```
scripts/
  process_data.py    build concatenated PNGs from the HF dataset (full_1024px, blind_1024px, full_358px) + ground truth
  prepare_jsonl.py    build batch API input JSONLs
  run_test.sh         upload -> poll -> download one batch test (cloud models)
  run_local.py        run one test locally via Photon (moondream2, or any model Photon supports)
data/
  images/{full_1024px,blind_1024px,full_358px}/   172 PNGs each
  batches/*.jsonl                                  batch inputs (query skill, base64 PNG per line)
  ground_truth.json                                idx -> {answer, choices}
results/                                           outputs + batches.log, named <test>_<model>.jsonl
```

## Run

### Batch API (cloud models: moondream3.1-9B-A2B, moondream3-preview)

Requires `MOONDREAM_API_KEY` in `.env`, plus `jq`. Python deps: `datasets`, `pillow`, `tqdm`.

```bash
python3 scripts/process_data.py     # regenerate images + ground truth (only needed once)
python3 scripts/prepare_jsonl.py    # regenerate JSONLs

scripts/run_test.sh data/batches/full_reasoning_1024px.jsonl   full_reasoning_1024px
scripts/run_test.sh data/batches/full_noreasoning_1024px.jsonl full_noreasoning_1024px
scripts/run_test.sh data/batches/blind_1024px.jsonl             blind_1024px
scripts/run_test.sh data/batches/full_reasoning_358px.jsonl     full_reasoning_358px
```

`run_test.sh` uploads in 50MB parts, polls every 60s, writes `results/<name>_<model>.jsonl`
(one line per sample: `id`, `status`, `result.answer`, `result.reasoning`, `usage`).
Batch ids are logged to `results/batches.log`; results expire server-side after 7 days.
Model override: `MOONDREAM_MODEL=... scripts/run_test.sh ...`

### Local (moondream2, via Photon)

Requires `pip install moondream`. Needs a CUDA GPU (Apple Silicon is currently broken — see
Methodology notes above). No API key needed for base models.

```bash
python3 scripts/run_local.py moondream2 full_reasoning_1024px --limit 3   # smoke test
python3 scripts/run_local.py moondream2 full_reasoning_1024px
python3 scripts/run_local.py moondream2 full_noreasoning_1024px
python3 scripts/run_local.py moondream2 blind_1024px
python3 scripts/run_local.py moondream2 full_reasoning_358px
```

Writes `results/<test>_<model>.jsonl` in the same schema as the batch outputs.
