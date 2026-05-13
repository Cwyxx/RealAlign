"""Extract (uid, prompt) from Civitai-top SFW metadata.

The source dataset is the HuggingFace repository:

    wallstoneai/civitai-top-sfw-images-with-metadata

The local checkout is expected to contain an ``images/`` directory and a
``prompts.json`` file. The JSON maps image filenames to metadata records; this
script uses each filename stem as the uid and reads the text prompt from the
record's ``prompt`` field.
"""

import csv
import json
import os

import pandas as pd


civitai_root = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/civitai-top-sfw-images-with-metadata"
prompt_json_path = os.path.join(civitai_root, "prompts.json")
output_csv_path = os.path.join(civitai_root, "uid_prompt.csv")


print(f"Loading {prompt_json_path}")
with open(prompt_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"Total metadata entries: {len(data)}")

rows = []
for filename, content in data.items():
    uid = os.path.splitext(filename)[0]
    prompt = content.get("prompt", "")
    rows.append({"uid": uid, "prompt": prompt})

df = pd.DataFrame(rows)
df.to_csv(
    output_csv_path,
    index=False,
    encoding="utf-8",
    quoting=csv.QUOTE_ALL,
    escapechar="\\",
    doublequote=True,
)

print(f"Saved {len(df)} rows to {output_csv_path}")
print("First 5 rows:")
print(df.head())
