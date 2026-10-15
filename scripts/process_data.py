from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLIND_DIR = os.path.join(ROOT, "data", "images", "blind_1024px")
FULL_DIR = os.path.join(ROOT, "data", "images", "full_1024px")
FULL_358PX_DIR = os.path.join(ROOT, "data", "images", "full_358px")

# Moondream's vision encoder tiles images into overlapping crops based on aspect
# ratio, independent of resolution, until height drops under 378px (ours is always
# 1024px tall) -- then it collapses to a single squished 378x378 crop. 358px sits
# safely below that threshold, forcing that degenerate single-crop case for every
# sample, as a deliberate ablation on tile count / detail.
SMALL_HEIGHT = 358

dataset = load_dataset("BLINK-Benchmark/BLINK", "Visual_Correspondence", split="val")

os.makedirs(BLIND_DIR, exist_ok=True)
os.makedirs(FULL_DIR, exist_ok=True)
os.makedirs(FULL_358PX_DIR, exist_ok=True)

for item in tqdm(dataset):
    # image 1: reference image
    # image 2: actual image
    assert item['image_1'].size[1] == item['image_2'].size[1] == 1024

    w1, h = item['image_1'].size
    w2, h = item['image_2'].size
    img = Image.new("RGB", (w1 + w2 + 10, h))
    img.paste(item['image_1'], (0, 0))

    # blind dataset will not have image_2
    img.save(os.path.join(BLIND_DIR, f"{item['idx']}.png"))

    # full dataset will have image_2
    img.paste(item['image_2'], (w1 + 10, 0))
    img.save(os.path.join(FULL_DIR, f"{item['idx']}.png"))

    small = img.resize(
        (round(img.width * SMALL_HEIGHT / img.height), SMALL_HEIGHT), Image.LANCZOS
    )
    small.save(os.path.join(FULL_358PX_DIR, f"{item['idx']}.png"))

with open(os.path.join(ROOT, "data", "ground_truth.json"), "w") as f:
    json.dump(
        {item["idx"]: {"answer": item["answer"], "choices": item["choices"]} for item in dataset},
        f,
        indent=2,
    )
print("ground_truth.json written")
