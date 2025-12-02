import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "False"
import torch
import sys
sys.path.append("/data3/chenweiyan/notebook/fine-tune-diffusion/spo_gitee/DiffusionNFT")
import pandas as pd
import cv2
import numpy as np
from flow_grpo.rewards import multi_score
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM

reward_model_name = "aesthetic"
# output_csv_path = f"/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/HPDv3/{reward_model_name}/{reward_model_name}.csv"

# csv_file_path = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/HPDv3/real_images_uid_prompt.csv"
# df = pd.read_csv(csv_file_path, dtype=str)
# real_image_dir = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/HPDv3/real"
# fake_image_dir = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/HPDv3/fake"

output_csv_path = f"/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/sam_next_inpainting/random_top_5_mask/{reward_model_name}/{reward_model_name}_score.csv"
csv_file_path = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/HPDv3/no_anime_all_images.csv"
df = pd.read_csv(csv_file_path, dtype=str)
real_image_dir = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/sam_next_inpainting/random_top_5_mask/real"
fake_image_dir = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/sam_next_inpainting/random_top_5_mask/fake"
ext_list = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]

print(f"csv file path: {csv_file_path}")
print(f"real image dir: {real_image_dir}")
print(f"fake image dir: {fake_image_dir}")
print(f"output csv path: {output_csv_path}")

device = torch.device("cuda")
if reward_model_name in ["imagereward", "pickscore", "clipscore", "code", "aesthetic" ]:
    all_reward_scorers = { reward_model_name: 1.0 }
    scoring_fn, reward_models = multi_score(device, all_reward_scorers)
    for reward_model in reward_models.values(): reward_model.to(device)
    print(f"Initializing reward models {reward_model_name} from DiffusionNFT...")
    
elif reward_model_name == "deqa":
    reward_model = AutoModelForCausalLM.from_pretrained(
        "zhiyuanyou/DeQA-Score-Mix3",
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"Initializing reward models {reward_model_name}...")
    
    def scoring_fn(images, prompts, metadata, only_strict=False):
        ### images is image_paths #### 
        images = [ Image.open(image_path) for image_path in images ]
        score_list = reward_model.score(images).tolist()
        
        score_details = { reward_model_name: score_list}
        return score_details, {}

elif reward_model_name == "hpsv3":
    from hpsv3 import HPSv3RewardInferencer
    inferencer = HPSv3RewardInferencer(device='cuda')
    # inferencer.model = inferencer.model.to(device).to(torch.float16) 
    def scoring_fn(images, prompts, metadata, only_strict=False):
        ### images is image_paths #### 
        assert type(images[0]) == str
        image_paths = images
        with torch.no_grad():
            rewards = inferencer.reward(prompts=prompts, image_paths=image_paths)
            score_list = [reward[0].item() for reward in rewards]
        
        score_details = { reward_model_name: score_list }
        return score_details, {}
    
elif reward_model_name == "brightness":
    def calculate_average_brightness(image_path):
        image = Image.open(image_path)
        gray_image = image.convert('L')
        gray_array = np.array(gray_image)
        return np.mean(gray_array)
    
    def scoring_fn(images, prompts, metadata, only_strict=False):
        #### images is image_paths ####
        image_paths = images
        score_list = [calculate_average_brightness(image_path) for image_path in image_paths]
        score_details = { reward_model_name: score_list }
        return score_details, {}
    
elif reward_model_name == "saturation":
    def calculate_saturation(image_path):
        img = Image.open(image_path).convert("HSV")
        s_channel = np.array(img)[:, :, 1]
        return s_channel.mean()
    
    def scoring_fn(images, prompts, metadata, only_strict=False):
        #### images is image_paths ####
        image_paths = images
        score_list = [calculate_saturation(image_path) for image_path in image_paths]
        score_details = { reward_model_name: score_list }
        return score_details, {}
    
elif reward_model_name == "colorfulness":
    def calculate_colorfulness(image_path):
        image = Image.open(image_path).convert("RGB")
        image = np.array(image).astype(np.float32)
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

        rg = r - g
        yb = 0.5 * (r + g) - b

        std_rg, mean_rg = np.std(rg), np.mean(rg)
        std_yb, mean_yb = np.std(yb), np.mean(yb)

        return np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
    
    def scoring_fn(images, prompts, metadata, only_strict=False):
        #### images is image_paths ####
        image_paths = images
        score_list = [calculate_colorfulness(image_path) for image_path in image_paths]
        score_details = { reward_model_name: score_list }
        return score_details, {}

real_image_score_list, fake_image_score_list = [], []
uid_list = []
real_win_count = 0
total_pairs = 0

for i in tqdm(range(len(df)), dynamic_ncols=True):
    uid = df.iloc[i]["uid"]
    prompt = df.iloc[i]["prompt"]
    real_image_path = os.path.join(real_image_dir, f"{uid}.png")
    fake_image_path = os.path.join(fake_image_dir, f"{uid}.png")
    
    if not os.path.exists(real_image_path) or not os.path.exists(fake_image_path):
        continue
    
    real_image = Image.open(real_image_path).convert("RGB")
    fake_image = Image.open(fake_image_path).convert("RGB")
    
    if reward_model_name in ["imagereward", "pickscore", "code"]:
        scores, _ = scoring_fn([real_image, fake_image], [prompt, prompt], None)
        
    elif reward_model_name in ["deqa", "hpsv3", "brightness", "saturation", "colorfulness"]:
        scores, _ = scoring_fn([real_image_path, fake_image_path], [prompt, prompt], None)
            
    elif reward_model_name in ["clipscore"]:
        images = [ np.array(real_image), np.array(fake_image) ]
        images = np.array(images)
        images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
        images = torch.tensor(images, dtype=torch.uint8) / 255.0
        scores, _ = scoring_fn(images, [prompt, prompt], None)
        
    elif reward_model_name in [ "aesthetic" ]:
        pil_images = [ real_image, fake_image ]
        pil_images = np.stack([np.array(img) for img in pil_images])
        images = pil_images.transpose(0, 3, 1, 2)
        images = torch.tensor(images, dtype=torch.uint8)
        scores, _ = scoring_fn(images, [prompt, prompt], None)
    
    if reward_model_name in ["imagereward", "pickscore", "clipscore", "code", "aesthetic" ]:
        real_score = scores[reward_model_name][0].detach().cpu().item()
        fake_score = scores[reward_model_name][1].detach().cpu().item()
    elif reward_model_name in ["deqa", "hpsv3", "brightness", "saturation", "colorfulness"]:
        real_score = scores[reward_model_name][0]
        fake_score = scores[reward_model_name][1]
    
    uid_list.append(uid)
    real_image_score_list.append(real_score)
    fake_image_score_list.append(fake_score)
    
    total_pairs += 1
    if real_score > fake_score:
        real_win_count += 1

# Calculate statistics for real_image_score_list
real_image_scores = np.array(real_image_score_list)
real_mean = np.mean(real_image_scores)
real_std = np.std(real_image_scores)
real_max = np.max(real_image_scores)
real_min = np.min(real_image_scores)

# Calculate statistics for fake_image_score_list
fake_image_scores = np.array(fake_image_score_list)
fake_mean = np.mean(fake_image_scores)
fake_std = np.std(fake_image_scores)
fake_max = np.max(fake_image_scores)
fake_min = np.min(fake_image_scores)

# Calculate win rate
real_win_rate = (real_win_count / total_pairs * 100) if total_pairs > 0 else 0

# Save detailed results to CSV
results_df = pd.DataFrame({
    'uid': uid_list,
    'real_image_score': real_image_score_list,
    'fake_image_score': fake_image_score_list,
})
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
results_df.to_csv(output_csv_path, index=False)
print(f"\nDetailed results saved to: {output_csv_path}")

# Print statistics
print("\n" + "="*60)
print(f"Reward Model Name: {reward_model_name}")
print("Real Image Score Statistics:")
print(f"  Mean: {real_mean:.4f}")
print(f"  Std:  {real_std:.4f}")
print(f"  Max:  {real_max:.4f}")
print(f"  Min:  {real_min:.4f}")
print("\n" + "="*60)
print("Fake Image Score Statistics:")
print(f"  Mean: {fake_mean:.4f}")
print(f"  Std:  {fake_std:.4f}")
print(f"  Max:  {fake_max:.4f}")
print(f"  Min:  {fake_min:.4f}")
print("\n" + "="*60)
print("Real Image Win Statistics:")
print(f"  Total Pairs: {total_pairs}")
print(f"  Real Image Wins: {real_win_count}")
print(f"  Win Rate: {real_win_rate:.2f}%")
print("="*60)
