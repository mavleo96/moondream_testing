# Results

172 records per file, 8 files, all `status: "ok"` (0 errors). GT distribution (172 samples):
A=51, B=33, C=44, D=44.

## Accuracy

| Test | moondream2 | moondream3.1-9B-A2B |
|---|---|---|
| `full_reasoning_1024px`   | 26.7% (46/172) | 28.5% (49/172) |
| `full_noreasoning_1024px` | 31.4% (54/172) | 33.7% (58/172) |
| `blind_1024px`            | 23.8% (41/172) | 23.8% (41/172) |
| `full_reasoning_358px`    | 20.9% (36/172) | 23.3% (40/172) |

## Answer distribution

Strict parsing: `result.answer` must equal exactly one of `A`/`B`/`C`/`D` (or parenthesized);
anything else is counted as NONSTD.

| Test | moondream2 (A/B/C/D/NONSTD) | moondream3.1-9B-A2B (A/B/C/D) |
|---|---|---|
| `full_reasoning_1024px`   | 10.5% / 41.9% / 14.0% / 33.1% / 0.6% | 16.3% / 22.7% / 48.8% / 12.2% |
| `full_noreasoning_1024px` | 4.1% / 59.9% / 24.4% / 11.6% / 0.0% | 12.2% / 9.3% / 76.2% / 2.3% |
| `blind_1024px`            | 44.2% / 31.4% / 8.7% / 15.7% / 0.0% | 9.9% / 8.1% / 51.7% / 30.2% |
| `full_reasoning_358px`    | 24.4% / 39.5% / 11.0% / 18.6% / 6.4% | 14.5% / 15.7% / 57.0% / 12.8% |

NONSTD records:
- `full_reasoning_1024px_moondream2.jsonl`: `val_Visual_Correspondence_92` → `"B, C, D"`
- `full_reasoning_358px_moondream2.jsonl`: `val_Visual_Correspondence_1` → `"B, C, D"`,
  `val_Visual_Correspondence_69` → `"B, D"`, and 9 empty-string answers
  (`39, 40, 76, 80, 122, 125, 156, 158, 171`)

No NONSTD records in any moondream3.1-9B-A2B file.

## Per-ground-truth-letter recall (correct / total for that letter)

| Test | Model | A (51) | B (33) | C (44) | D (44) |
|---|---|---|---|---|---|
| `full_reasoning_1024px`   | moondream2          | 8/51  | 16/33 | 6/44  | 16/44 |
| `full_reasoning_1024px`   | moondream3.1-9B-A2B | 12/51 | 6/33  | 23/44 | 8/44  |
| `full_noreasoning_1024px` | moondream2          | 6/51  | 24/33 | 14/44 | 10/44 |
| `full_noreasoning_1024px` | moondream3.1-9B-A2B | 14/51 | 6/33  | 37/44 | 1/44  |
| `blind_1024px`            | moondream2          | 21/51 | 8/33  | 5/44  | 7/44  |
| `blind_1024px`            | moondream3.1-9B-A2B | 6/51  | 1/33  | 21/44 | 13/44 |
| `full_reasoning_358px`    | moondream2          | 9/51  | 14/33 | 4/44  | 9/44  |
| `full_reasoning_358px`    | moondream3.1-9B-A2B | 10/51 | 8/33  | 20/44 | 2/44  |

## Reasoning on vs. off: differing final answers

| Model | Differing samples |
|---|---|
| moondream2 | 71/172 |
| moondream3.1-9B-A2B | 86/172 |

## Answer/reasoning-text contradictions

Checked for reasoning text containing "answer is X" where X differs from `result.answer`.
Found 1 total: `full_reasoning_1024px_moondream3.1-9B-A2B.jsonl`, `val_Visual_Correspondence_78`.
0 in all other 7 files.

## Cross-model agreement per test

| Test | Same answer | Both correct |
|---|---|---|
| `full_reasoning_1024px`   | 58/172 | 15/172 |
| `full_noreasoning_1024px` | 54/172 | 20/172 |
| `blind_1024px`            | 30/172 | 9/172  |
| `full_reasoning_358px`    | 37/172 | 10/172 |
