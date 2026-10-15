"""Run one of the four BLINK tests locally via Photon (no batch API, no upload).

Usage:
    python3 scripts/run_local.py <model> <test>
    test in {full_reasoning_1024px, full_noreasoning_1024px, blind_1024px, full_reasoning_358px}

Output matches the batch API's results/<name>_<model>.jsonl format:
    {"line_index": ..., "id": ..., "status": "ok"|"error", "result": {"answer": ..., "reasoning": {"text": ...}}}
"""
import argparse
import json
import os

import moondream as md
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(ROOT, "data", "images")
RESULTS_DIR = os.path.join(ROOT, "results")

TESTS = {
    "full_reasoning_1024px": {"image_dir": "full_1024px", "reasoning": True},
    "full_noreasoning_1024px": {"image_dir": "full_1024px", "reasoning": False},
    "blind_1024px": {"image_dir": "blind_1024px", "reasoning": True},
    "full_reasoning_358px": {"image_dir": "full_358px", "reasoning": True},
}


def fix_prompt(prompt):
    assert "first image" in prompt and "second image" in prompt, "unexpected prompt wording"
    return prompt.replace("first image", "left image").replace("second image", "right image")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="e.g. moondream2")
    parser.add_argument("test", choices=TESTS)
    parser.add_argument("--limit", type=int, default=None, help="only run the first N samples")
    args = parser.parse_args()

    cfg = TESTS[args.test]
    image_dir = os.path.join(IMAGES_DIR, cfg["image_dir"])
    dataset = load_dataset("BLINK-Benchmark/BLINK", "Visual_Correspondence", split="val")
    if args.limit:
        dataset = dataset.select(range(args.limit))

    print(f"loading {args.model} via Photon...")
    model = md.photon(args.model)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{args.test}_{args.model}.jsonl")

    with open(out_path, "w") as f:
        for i, item in enumerate(tqdm(dataset, desc=args.test)):
            image = Image.open(os.path.join(image_dir, f"{item['idx']}.png"))
            question = fix_prompt(item["prompt"])
            try:
                result = model.query(
                    image, question,
                    settings={"temperature": 0},
                    reasoning=cfg["reasoning"],
                )
                reasoning = result.get("reasoning")
                record = {
                    "line_index": i,
                    "id": item["idx"],
                    "status": "ok",
                    "result": {
                        "answer": result["answer"],
                        **({"reasoning": {"text": reasoning["text"]}} if reasoning else {}),
                    },
                }
            except Exception as e:
                record = {"line_index": i, "id": item["idx"], "status": "error", "error": {"message": str(e)}}
            f.write(json.dumps(record) + "\n")
            f.flush()

    model.close()
    print(f"saved {len(dataset)} results to results/{os.path.basename(out_path)}")


if __name__ == "__main__":
    main()
