import lpips
import torch
import pandas as pd
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_model = lpips.LPIPS(net="alex")
lpips_model.to(device)

csv_file_path = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/paired_real_generated_dataset/high_quality_train.csv"
df = pd.read_csv(csv_file_path)
real_image_dir = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/add_noise_denoise/random_add_noise_step/real"
fake_image_dir = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/add_noise_denoise/random_add_noise_step/fake"

lpips_diff_list = []
for i in tqdm(range(len(df))):
    uid = df.iloc[i]["uid"]
    real_image_path = os.path.join(real_image_dir, f"{uid}.png")
    fake_image_path = os.path.join(fake_image_dir, f"{uid}.png")
    real_image = lpips.im2tensor(lpips.load_image(real_image_path)).to(device)
    fake_image = lpips.im2tensor(lpips.load_image(fake_image_path)).to(device)
    lpips_diff = lpips_model(real_image, fake_image)
    lpips_diff_list.append({"uid": uid, "lpips_diff": lpips_diff.item()})
pd.DataFrame(lpips_diff_list).to_csv("lpips_alex_diff.csv", index=False)