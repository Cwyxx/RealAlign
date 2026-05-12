"""Colorfulness score for (real, fake) image pairs.

Uses the Hasler & Süsstrunk (2003) metric:
    colorfulness = sqrt(std(rg)^2 + std(yb)^2) + 0.3 * sqrt(mean(rg)^2 + mean(yb)^2)
where rg = R - G and yb = 0.5 * (R + G) - B.

Reads a uid+prompt CSV (the output of ``data_curation/extract/hpdv3.py``) and
opens ``<real_image_dir>/{uid}.png`` / ``<fake_image_dir>/{uid}.png`` for each
row. Writes a CSV with columns (uid, real_image_score, fake_image_score) that
is consumed by the gap-based filter step
(``real_image_score - fake_image_score > threshold``).
"""

import os

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


input_csv_path = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/HPDv3/real_images_uid_prompt.csv"
real_image_dir = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/HPDv3/real"
fake_image_dir = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/HPDv3/fake"
output_csv_path = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/HPDv3/colorfulness/colorfulness_score.csv"


def colorfulness(image_path: str) -> float:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_rg, mean_rg = float(np.std(rg)), float(np.mean(rg))
    std_yb, mean_yb = float(np.std(yb)), float(np.mean(yb))
    return float(
        np.sqrt(std_rg ** 2 + std_yb ** 2)
        + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2)
    )


df = pd.read_csv(input_csv_path, dtype=str)
print(f"Scoring {len(df)} pairs from {input_csv_path}")

records = []
for row in tqdm(df.itertuples(index=False), total=len(df), dynamic_ncols=True):
    real_path = os.path.join(real_image_dir, f"{row.uid}.png")
    fake_path = os.path.join(fake_image_dir, f"{row.uid}.png")
    if not (os.path.exists(real_path) and os.path.exists(fake_path)):
        continue
    records.append({
        "uid": row.uid,
        "real_image_score": colorfulness(real_path),
        "fake_image_score": colorfulness(fake_path),
    })

os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
out_df = pd.DataFrame(records)
out_df.to_csv(output_csv_path, index=False)

real_win_rate = (out_df["real_image_score"] > out_df["fake_image_score"]).mean() * 100
print(f"Saved {len(out_df)} rows to {output_csv_path}")
print(f"Real-win rate (real > fake): {real_win_rate:.2f}%")
