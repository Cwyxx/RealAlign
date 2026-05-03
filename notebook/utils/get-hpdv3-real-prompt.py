import random
import csv

csv_json_path = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/HPDv3/no_anime_all_images.csv"

# Read all prompts
with open(csv_json_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    prompts = [row["prompt"] for row in reader]

print(f"Total prompts: {len(prompts)}")

# Shuffle and split
random.seed(42)
indices = list(range(len(prompts)))
random.shuffle(indices)

n = len(prompts)
n_train = int(n * 0.8)
n_val = int(n * 0.1)

train_prompts = [prompts[i] for i in indices[:n_train]]
val_prompts   = [prompts[i] for i in indices[n_train:n_train + n_val]]
test_prompts  = [prompts[i] for i in indices[n_train + n_val:]]

print(f"Train: {len(train_prompts)}, Val: {len(val_prompts)}, Test: {len(test_prompts)}")

# Save splits
output_dir = "/data3/chenweiyan/notebook/fine-tune-diffusion/spo_gitee/DiffusionNFT/dataset/HPDv3"

for split_name, split_prompts in [("train", train_prompts), ("val", val_prompts), ("test", test_prompts)]:
    out_path = f"{output_dir}/{split_name}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for prompt in split_prompts:
            f.write(prompt.replace("\n", " ") + "\n")
    print(f"Saved {split_name} -> {out_path}")
