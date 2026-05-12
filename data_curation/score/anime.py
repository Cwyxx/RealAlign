"""Anime / artwork detection on HPDv3 real images via Qwen3-VL.

HPDv3's "real_images" split is mostly photographic but contains a non-trivial
fraction of illustrations / anime / digital artwork. Those are not "real" in
the sense the paper requires (we want photo-realistic anchors for the
preference signal). This script classifies each real image as anime/artwork
or not, so the filter step can drop the positive cases.

Classification is done by zero-shot prompting Qwen3-VL-8B-Instruct with::

    "Is this image an artwork or anime style? Respond with only 'yes' or 'no'."

Reads the uid+prompt CSV produced by ``data_curation/extract/hpdv3.py`` and
opens ``<real_image_dir>/{uid}.{ext}`` for each row. Writes a CSV with columns
(uid, image_path, anime, error). ``anime`` is the raw model response ("yes" /
"no", lowercased). The filter step keeps rows where ``anime == "no"``.

Resume is automatic: rows already present in the output CSV are skipped, so
the script can be killed and restarted. Qwen3-VL inference on 20k+ images
takes hours, so this is essential.
"""

import csv
import os
from pathlib import Path

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


input_csv_path = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/HPDv3/real_images_uid_prompt.csv"
real_image_dir = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/HPDv3/real"
output_csv_path = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/HPDv3/anime/anime_classification.csv"

model_name = "Qwen/Qwen3-VL-8B-Instruct"
image_extensions = (".png", ".jpg", ".jpeg")
prompt = (
    "You are an image classification assistant. Your task is to determine if "
    "the given image is an **artwork or anime style** image. Respond with "
    "**only** 'yes' if it is an artwork or anime style, and 'no' if it is "
    "not. Do not provide any other text or explanation."
)


def find_image(uid: str) -> str | None:
    base = Path(real_image_dir) / uid
    for ext in image_extensions:
        p = base.with_suffix(ext)
        if p.exists():
            return str(p)
    return None


def read_processed(path: str) -> tuple[set[str], list[dict]]:
    if not os.path.exists(path):
        return set(), []
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return {r["uid"] for r in rows if r.get("uid")}, rows


def write_csv(rows: list[dict], path: str) -> None:
    fieldnames = ["uid", "image_path", "anime", "error"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name, dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)


def classify(image_path: str) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    generated = model.generate(**inputs, max_new_tokens=8)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip().lower()


os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
processed, results = read_processed(output_csv_path)

with open(input_csv_path, "r", encoding="utf-8") as f:
    all_uids = [row["uid"] for row in csv.DictReader(f)]

todo = [uid for uid in all_uids if uid not in processed]
print(f"{len(processed)} already classified; {len(todo)} remaining (of {len(all_uids)} total)")

for idx, uid in enumerate(todo, 1):
    image_path = find_image(uid)
    if image_path is None:
        results.append({"uid": uid, "image_path": "", "anime": "", "error": "image not found"})
    else:
        try:
            answer = classify(image_path)
            results.append({"uid": uid, "image_path": image_path, "anime": answer, "error": ""})
        except Exception as e:  # OOM, decode failures, etc. — keep going
            results.append({"uid": uid, "image_path": image_path, "anime": "", "error": str(e)})
    if idx % 50 == 0 or idx == len(todo):
        write_csv(results, output_csv_path)
        print(f"[{idx}/{len(todo)}] saved {len(results)} rows")

write_csv(results, output_csv_path)
anime_rate = sum(1 for r in results if r.get("anime") == "yes") / max(len(results), 1) * 100
print(f"Done. anime/artwork rate: {anime_rate:.2f}% of {len(results)} classified images")
