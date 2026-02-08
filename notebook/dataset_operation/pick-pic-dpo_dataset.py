import os
import csv
import json
from tqdm import tqdm

base_pick_a_pic_dir = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/pick-a-pic-v2/visualization-data"
directories = os.listdir(base_pick_a_pic_dir)

# 存储所有数据的列表
data_list = []

for directory in tqdm(directories):
    directory_dir = os.path.join(base_pick_a_pic_dir, directory)
    directory_name = directory[0:len("train-00000-of-00645")]
    for row in os.listdir(directory_dir):
        row_dir = os.path.join(directory_dir, row)
        
        # 跳过非目录项
        if not os.path.isdir(row_dir):
            continue
        
        # 读取 info.json 文件
        info_json_path = os.path.join(row_dir, "info.json")
        if not os.path.exists(info_json_path):
            continue
        
        try:
            with open(info_json_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            
            # 从 info.json 中提取信息
            uid = info.get('id', f"{directory_name}-{row}")
            prompt = info.get('caption', '')
            win_image_path = info.get('win_image_path', '')
            lose_image_path = info.get('lose_image_path', '')
            
            # 如果路径是相对路径，转换为绝对路径
            if win_image_path and not os.path.isabs(win_image_path):
                win_image_path = os.path.join(row_dir, win_image_path)
                if not os.path.exists(win_image_path):
                    win_image_path = None
                
            if lose_image_path and not os.path.isabs(lose_image_path):
                lose_image_path = os.path.join(row_dir, lose_image_path)
                if not os.path.exists(lose_image_path):
                    lose_image_path = None
            
            # 如果找到了必要的信息，添加到列表
            if prompt and win_image_path and lose_image_path:
                data_list.append({
                    'uid': uid,
                    'prompt': prompt,
                    'win_image_path': win_image_path,
                    'lose_image_path': lose_image_path
                })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"读取 {info_json_path} 时出错: {e}")
            continue

# 保存到 CSV 文件
output_csv = "pick-a-pic-v2-dpo_dataset.csv"
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['uid', 'prompt', 'win_image_path', 'lose_image_path']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data_list)

print(f"已保存 {len(data_list)} 条数据到 {output_csv}")