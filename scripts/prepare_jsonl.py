import base64
import json
import os

from datasets import load_dataset
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(ROOT, "data", "images")
BATCHES_DIR = os.path.join(ROOT, "data", "batches")

dataset = load_dataset("BLINK-Benchmark/BLINK", "Visual_Correspondence", split="val")

MAX_LINE_BYTES = 10 * 1024 * 1024  # batch API limit per line


def b64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def fix_prompt(prompt):
    assert "first image" in prompt and "second image" in prompt, "unexpected prompt wording"
    return prompt.replace("first image", "left image").replace("second image", "right image")


def build(names, image_dir, reasonings):
    """Write one or more JSONLs sharing the same images, encoding each image once."""
    os.makedirs(BATCHES_DIR, exist_ok=True)
    out_paths = [os.path.join(BATCHES_DIR, f"{name}.jsonl") for name in names]
    files = [open(p, "w") for p in out_paths]
    max_lines = [0] * len(names)
    try:
        for item in tqdm(dataset, desc="+".join(names)):
            image_b64 = b64_image(os.path.join(image_dir, f"{item['idx']}.png"))
            question = fix_prompt(item["prompt"])
            for i, reasoning in enumerate(reasonings):
                line = json.dumps({
                    "id": item["idx"],
                    "skill": "query",
                    "image": image_b64,
                    "question": question,
                    "reasoning": reasoning,
                    "settings": {"temperature": 0},
                })
                assert len(line) <= MAX_LINE_BYTES, f"{item['idx']} exceeds 10MB line limit"
                max_lines[i] = max(max_lines[i], len(line))
                files[i].write(line + "\n")
    finally:
        for f in files:
            f.close()
    for path, max_line in zip(out_paths, max_lines):
        size_mb = os.path.getsize(path) / 1024**2
        print(f"{path}: {size_mb:.1f} MB total, largest line {max_line / 1024**2:.2f} MB")


build(
    ["full_reasoning_1024px", "full_noreasoning_1024px"],
    os.path.join(IMAGES_DIR, "full_1024px"),
    [True, False],
)
build(["blind_1024px"], os.path.join(IMAGES_DIR, "blind_1024px"), [True])
build(["full_reasoning_358px"], os.path.join(IMAGES_DIR, "full_358px"), [True])

with open(os.path.join(ROOT, "data", "ground_truth.json"), "w") as f:
    json.dump(
        {item["idx"]: {"answer": item["answer"], "choices": item["choices"]} for item in dataset},
        f,
        indent=2,
    )
print("ground_truth.json written")
