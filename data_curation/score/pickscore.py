"""PickScore (yuvalkirstain/PickScore_v1) for (real, fake) image pairs.

PickScore is a CLIP-ViT-H/14 fine-tuned on Pick-a-Pic; for an (image, prompt)
pair the score is ``logit_scale * cos(text_emb, image_emb)``. We score the real
and inpainted-fake images with the same prompt in a single forward pass.

Reads a uid+prompt CSV (the output of ``data_curation/extract/hpdv3.py``) and
opens ``<real_image_dir>/{uid}.png`` / ``<fake_image_dir>/{uid}.png`` for each
row. Writes a CSV with columns (uid, real_image_score, fake_image_score) that
is consumed by the gap-based filter step
(``real_image_score - fake_image_score > 0.02`` in the paper's main run).
"""

import os

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


input_csv_path = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/HPDv3/real_images_uid_prompt.csv"
real_image_dir = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/HPDv3/real"
fake_image_dir = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/HPDv3/fake"
output_csv_path = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/HPDv3/pickscore/pickscore_score.csv"

processor_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_name = "yuvalkirstain/PickScore_v1"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained(processor_name)
model = AutoModel.from_pretrained(model_name).eval().to(device)


@torch.no_grad()
def score_pair(prompt: str, real_image: Image.Image, fake_image: Image.Image) -> tuple[float, float]:
    image_inputs = processor(
        images=[real_image, fake_image], return_tensors="pt"
    ).to(device)
    text_inputs = processor(
        text=prompt, padding=True, truncation=True, max_length=77, return_tensors="pt"
    ).to(device)

    image_embs = model.get_image_features(**image_inputs)
    image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)
    text_embs = model.get_text_features(**text_inputs)
    text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

    scores = (model.logit_scale.exp() * (text_embs @ image_embs.T)).flatten()
    return float(scores[0].item()), float(scores[1].item())


df = pd.read_csv(input_csv_path, dtype=str)
print(f"Scoring {len(df)} pairs from {input_csv_path}")

records = []
for row in tqdm(df.itertuples(index=False), total=len(df), dynamic_ncols=True):
    real_path = os.path.join(real_image_dir, f"{row.uid}.png")
    fake_path = os.path.join(fake_image_dir, f"{row.uid}.png")
    if not (os.path.exists(real_path) and os.path.exists(fake_path)):
        continue
    real_image = Image.open(real_path).convert("RGB")
    fake_image = Image.open(fake_path).convert("RGB")
    real_score, fake_score = score_pair(row.prompt, real_image, fake_image)
    records.append({
        "uid": row.uid,
        "real_image_score": real_score,
        "fake_image_score": fake_score,
    })

os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
out_df = pd.DataFrame(records)
out_df.to_csv(output_csv_path, index=False)

real_win_rate = (out_df["real_image_score"] > out_df["fake_image_score"]).mean() * 100
print(f"Saved {len(out_df)} rows to {output_csv_path}")
print(f"Real-win rate (real > fake): {real_win_rate:.2f}%")
