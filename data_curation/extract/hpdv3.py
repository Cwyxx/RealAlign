"""Extract (uid, prompt, path) for real images in HPDv3.

HPDv3's ``all.json`` is a list of pairwise preference entries; each entry holds
two image paths (``path1``, ``path2``) and the model that produced each one
(``model1``, ``model2``). When one of the two models is the literal string
``"real_images"``, the corresponding path points to a professional real-world
photograph rather than a generated sample. This script collects every such
real image, pairs it with the entry's prompt, and writes a CSV deduplicated by
the image filename stem (used as the uid).

The resulting CSV is the input to subsequent steps: saliency-based pair
construction (``construct_pairs/``), per-pair scoring (``score/``), and
the per-source filter (``filter/``).
"""

import json
import os

import pandas as pd
from tqdm import tqdm


hpdv3_root = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/HPDv3"
all_json_path = os.path.join(hpdv3_root, "all.json")
output_csv_path = os.path.join(hpdv3_root, "real_images_uid_prompt.csv")


print(f"Loading {all_json_path}")
with open(all_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"Total entries: {len(data)}")

real_image_records = []
for entry in tqdm(data, desc="Scanning entries"):
    model1 = entry.get("model1", "")
    model2 = entry.get("model2", "")

    if model1 == "real_images":
        real_path = entry.get("path1", "")
    elif model2 == "real_images":
        real_path = entry.get("path2", "")
    else:
        continue

    if not real_path:
        continue

    uid = os.path.splitext(os.path.basename(real_path))[0]
    real_image_records.append({
        "uid": uid,
        "prompt": entry.get("prompt", ""),
        "path": real_path,
    })

print(f"Real-image entries found: {len(real_image_records)}")

df = pd.DataFrame(real_image_records)
n_before = len(df)
n_unique = df["uid"].nunique()
df = df.drop_duplicates(subset=["uid"], keep="first")
print(f"Deduplicated by uid: {n_before} -> {len(df)} (unique uids in raw: {n_unique})")

df.to_csv(output_csv_path, index=False, encoding="utf-8")
print(f"Saved {len(df)} rows to {output_csv_path}")
