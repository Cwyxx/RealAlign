import os
import pandas as pd
import json
from tqdm import tqdm

base_parquet_dir = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/pick-a-pic-v2/data"
base_target_paired_preference_data_dir = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/pick-a-pic-v2/visualization-data"

os.makedirs(base_target_paired_preference_data_dir, exist_ok=True)

parquet_file_list = [f for f in os.listdir(base_parquet_dir) if f.startswith("train") and f.endswith(".parquet")]
parquet_file_list.sort()
print(f"parquet_file_list: {len(parquet_file_list)}")

def to_bytes(x):
    if x is None:
        return None
    if isinstance(x, bytes):
        return x
    if isinstance(x, (memoryview, bytearray)):
        return bytes(x)
    try:
        y = x.as_py()
        if isinstance(y, (bytes, bytearray, memoryview)):
            return bytes(y)
    except Exception:
        print(f"Error Processing bytes to image.")
    return None

def decide_winner(row):
    if "are_different" in row:
        if pd.isna(row["are_different"]) or not bool(row["are_different"]):
            return None
    
    try:
        l0 = float(row.get("label_0", 0.5))
        l1 = float(row.get("label_1", 0.5))
    except Exception:
        return None
    
    if l0 not in (0.0, 1.0) or l1 not in (0.0, 1.0):
        return None
    l0, l1 = int(l0), int(l1)
    
    if l0 == 1 and l1 == 0:
        return 0
    if l0 == 0 and l1 == 1:
        return 1
    
    if "best_image_uid" in row and pd.notna(row["best_image_uid"]):
        best_uid = str(row["best_image_uid"])
        uid0 = str(row["image_0_uid"]) if "image_0_uid" in row and pd.notna(row["image_0_uid"]) else None
        uid1 = str(row["image_1_uid"]) if "image_1_uid" in row and pd.notna(row["image_1_uid"]) else None
        if uid0 and best_uid == uid0 and l0 == 1 and l1 == 0:
            return 0
        if uid1 and best_uid == uid1 and l0 == 0 and l1 == 1:
            return 1
    
    return None

total_saved = 0
for parquet_idx, parquet_file in enumerate(parquet_file_list):
    parquet_path = os.path.join(base_parquet_dir, parquet_file)
    print(f"reading parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"reading done! {parquet_path}")
    
    # 为每个 parquet 文件创建一个目录
    parquet_dir_name = parquet_file.replace('.parquet', '')
    parquet_dir = os.path.join(base_target_paired_preference_data_dir, parquet_dir_name)
    if os.path.exists(parquet_dir):
        continue
    os.makedirs(parquet_dir, exist_ok=True)
    
    for row_idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  {parquet_file}", leave=False):
        winner_idx = decide_winner(row)
        if winner_idx is None:
            continue
        
        b0 = to_bytes(row.get("jpg_0"))
        b1 = to_bytes(row.get("jpg_1"))
        if b0 is None or b1 is None:
            continue
        
        uid0 = row.get("image_0_uid")
        uid1 = row.get("image_1_uid")
        if pd.isna(uid0) or pd.isna(uid1):
            continue
        uid0 = str(uid0)
        uid1 = str(uid1)
        
        prompt = row.get("caption")
        if pd.isna(prompt):
            continue
        prompt = str(prompt)
        
        write_id = f"{parquet_dir_name}-row_{row_idx}"
        write_dir = os.path.join(parquet_dir, f"row_{row_idx}")
        os.makedirs(write_dir, exist_ok=True)
        
        if winner_idx == 0:
            win_image_id, lose_image_id = uid0, uid1
            win_bytes, lose_bytes = b0, b1
        else:
            win_image_id, lose_image_id = uid1, uid0
            win_bytes, lose_bytes = b1, b0
        
        win_filename = f"win_image-{win_image_id}.png"
        lose_filename = f"lose_image-{lose_image_id}.png"
        
        with open(os.path.join(write_dir, win_filename), "wb") as f:
            f.write(win_bytes)
        with open(os.path.join(write_dir, lose_filename), "wb") as f:
            f.write(lose_bytes)
        
        caption_file = os.path.join(write_dir, "caption.txt")
        with open(caption_file, "w", encoding="utf-8") as f:
            f.write(prompt)
        
        info = {
            "id": write_id,
            "caption": prompt,
            "win_image_id": win_image_id,
            "lose_image_id": lose_image_id,
            "win_image_path": win_filename,
            "lose_image_path": lose_filename,
        }
        with open(os.path.join(write_dir, "info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=4)
        
        total_saved += 1

print(f"\n总共保存了 {total_saved} 个 paired preference 数据样本")
