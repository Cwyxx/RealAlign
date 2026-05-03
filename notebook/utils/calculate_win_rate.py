import os
import json
import pandas as pd

model_type = "sd-v1-5" # "sd-v1-5"
base_image_dir = "/data_center/data2/dataset/chenwy/21164-data/diffusion-dpo"
seed_list = [42, 123, 456, 789, 1000]
dataset_name = "partiprompts" # "drawbench-unique"

if model_type == "sd-v1-5":
    method_dict = {
        "Diffusion-DPO": "dpo-official/ckpt-0",
        "Ours": "irl_top_512_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all_ckpt_1600-dpo_2000_top_512_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all_inpainting/ckpt-800",
    }
elif model_type == "sd-3-5-medium":
    method_dict = {
        "Diffusion-DPO": "pick-a-pic-v2-dpo_dataset_160000_pairs/ckpt-4900",
        "Ours": "irl_top_512_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all_lr_0.0002_ckpt_3200-dpo_top_512_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all/ckpt-450",
    }

# 所有指标均为 higher-is-better
metric_list = ["pickscore", "imagereward", "unifiedreward", "hpsv3", "deqa", "aesthetic"]


def load_scores(jsonl_path):
    data = {}
    with open(jsonl_path) as f:
        for line in f:
            item = json.loads(line.strip())
            data[item["sample_id"]] = item["scores"]
    return data


all_seed_records = []  # 每个 seed 的 win/tie/lose rates

for seed in seed_list:
    dpo_path = os.path.join(
        base_image_dir, model_type, f"generate_images_seed_{seed}",
        dataset_name, method_dict["Diffusion-DPO"], "evaluation_results.jsonl",
    )
    ours_path = os.path.join(
        base_image_dir, model_type, f"generate_images_seed_{seed}",
        dataset_name, method_dict["Ours"], "evaluation_results.jsonl",
    )

    dpo_scores = load_scores(dpo_path)
    ours_scores = load_scores(ours_path)

    common_ids = sorted(set(dpo_scores.keys()) & set(ours_scores.keys()))
    total = len(common_ids)

    win_counts  = {m: 0 for m in metric_list}
    tie_counts  = {m: 0 for m in metric_list}
    lose_counts = {m: 0 for m in metric_list}

    for sid in common_ids:
        for m in metric_list:
            ours_val = ours_scores[sid][m]
            dpo_val  = dpo_scores[sid][m]
            if ours_val > dpo_val:
                win_counts[m] += 1
            elif ours_val == dpo_val:
                tie_counts[m] += 1
            else:
                lose_counts[m] += 1

    record = {"seed": seed}
    for m in metric_list:
        record[f"{m}_win"]  = win_counts[m]  / total
        record[f"{m}_tie"]  = tie_counts[m]  / total
        record[f"{m}_lose"] = lose_counts[m] / total
    all_seed_records.append(record)
    print(f"Seed {seed}: {total} samples")

# ---- 汇报 ----
df_raw = pd.DataFrame(all_seed_records).set_index("seed")

for m in metric_list:
    cols = [f"{m}_win", f"{m}_tie", f"{m}_lose"]
    df_m = (df_raw[cols] * 100).round(2)
    mean_row = df_m.mean().rename("Mean")
    df_m = pd.concat([df_m, mean_row.to_frame().T])
    df_m.columns = ["Win(%)", "Tie(%)", "Lose(%)"]
    print(f"\n--- {m} ---")
    print(df_m.to_string())
