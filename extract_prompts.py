#!/usr/bin/env python3
import csv
import os

# 输入和输出路径
csv_path = "/data_center/data2/dataset/chenwy/21164-data/all-train-demo.csv"
output_dir = "/data3/chenweiyan/notebook/fine-tune-diffusion/spo_gitee/DiffusionNFT/dataset/x_aigd"
output_file = os.path.join(output_dir, "test.txt")

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 读取 CSV 文件并提取前 1000 个 prompt
prompts = []
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 1000:
            break
        prompt = row.get('PROMPT', '').strip()
        if prompt:
            prompts.append(prompt)

# 写入 test.txt 文件
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(prompts))

print(f"Successfully extracted and wrote {len(prompts)} prompts to {output_file}")










