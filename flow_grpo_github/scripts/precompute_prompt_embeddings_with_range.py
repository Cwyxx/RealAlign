import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from diffusers import StableDiffusion3Pipeline
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
from functools import partial
import tqdm
from absl import app, flags
from ml_collections import config_flags
from accelerate.logging import get_logger
from torchvision import transforms

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")
flags.DEFINE_integer("start_index", 0, "The starting index of the dataset to process.")
flags.DEFINE_integer("end_index", -1, "The ending index of the dataset to process. -1 means process to the end.")
logger = get_logger(__name__)

class Paired_Real_Fake_Dataset(Dataset):
    def __init__(self, config, image_transform, split="train", start_idx=0, end_idx=-1):
        self.image_transform = image_transform
        self.csv_file_path = config.dpo.csv_file_path[split]
        
        self.ext_list = [ ".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG" ]
        self.df = pd.read_csv(self.csv_file_path)
        
        if split == "high_quality_val": 
            self.df = self.df.head(24)
        
        # 应用 start_index 和 end_index 来限制处理的行数范围
        total_rows = len(self.df)
        if end_idx == -1:
            end_idx = total_rows
        
        # 确保索引在有效范围内
        start_idx = max(0, min(start_idx, total_rows))
        end_idx = max(start_idx, min(end_idx, total_rows))
        
        # 切片数据框
        self.df = self.df.iloc[start_idx:end_idx].reset_index(drop=True)
        
        logger.info(f"Processing dataset range: [{start_idx}:{end_idx}] (total rows in CSV: {total_rows})")
        logger.info(f"Actual processing: {len(self.df)} items")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row_data = self.df.iloc[idx]
        uid, prompt, win_image_path, lose_image_path = row_data["uid"], row_data["prompt"], row_data["win_image_path"], row_data["lose_image_path"]
        
        if win_image_path is None:
            raise FileNotFoundError(f"Missing WIN image for uid: {uid} at {win_image_path}")
        
        if lose_image_path is None:
            raise FileNotFoundError(f"Missing LOSE image for uid: {uid} at {lose_image_path}")
        
        try:
            win_pixel_values = self.image_transform(Image.open(win_image_path).convert("RGB"))
            lose_pixel_values = self.image_transform(Image.open(lose_image_path).convert("RGB"))
            pixel_values = torch.cat([win_pixel_values, lose_pixel_values], dim=0) # torch.cat [3, 512, 512] -> [6, 512, 512]
        except Exception:
            print(f"Exception uid: {uid}, Create black images")
            exit(0)
        
        return {
            "uid": uid,
            "prompt": prompt,
            "pixel_values": pixel_values
        }

    @staticmethod
    def collate_fn(examples):
        uids = [ example["uid"] for example in examples ]
        prompts = [ example["prompt"] for example in examples ]
        pixel_values = [ example["pixel_values"] for example in examples ]
        pixel_values = torch.stack(pixel_values, dim=0).to(memory_format=torch.contiguous_format).float()  # torch.stack [6, 512, 512] -> [batch_size, 6, 512, 512]
        return uids, prompts, pixel_values
    
def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 获取命令行参数中的 start_index 和 end_index
    start_idx = FLAGS.start_index
    end_idx = FLAGS.end_index
    
    logger.info(f"Starting precomputation with start_index={start_idx}, end_index={end_idx}")
    
    #### Load Scheduler, Tokenizer and Model. ####
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", 
        transfomer=None
    )
    # disable safety checker
    pipeline.safety_checker = None
    
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    
    # Move vae and text_encoder to device and cast to inference_dtype
    inference_dtype = torch.float16
    pipeline.text_encoder.to(device=device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(device=device, dtype=inference_dtype)
    pipeline.text_encoder_3.to(device=device, dtype=inference_dtype)
    
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
    
    #### image_transform, copy from dive-into-sd-3-5-medium ####
    image_transform = transforms.Compose(
        [
            transforms.Resize(config.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(config.resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    #### image_transform, copy from dive-into-sd-3-5-medium ####
    train_dataset = Paired_Real_Fake_Dataset(
        config, 
        image_transform, 
        split=config.dpo.dataset["train"],
        start_idx=start_idx,
        end_idx=end_idx
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=Paired_Real_Fake_Dataset.collate_fn,
        num_workers=16
    )
    
    precomputed_embeddings_dir = config.dpo.precomputed_embeddings_dir_dict[config.dpo.dataset["train"]]
    os.makedirs(precomputed_embeddings_dir, exist_ok=True)
    
    logger.info(f"Starting to process {len(train_dataset)} items...")
    for uids, prompts, pixel_values in tqdm(train_dataloader, desc="Processing embeddings"):
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts,
            text_encoders,
            tokenizers,
            max_sequence_length=128,
            device=device
        )
        for uid, prompt, prompt_embed, pooled_prompt_embed in zip(uids, prompts, prompt_embeds, pooled_prompt_embeds):  
            prompt_embed_path = os.path.join(precomputed_embeddings_dir, f"{uid}.pt")
            torch.save(prompt_embed, prompt_embed_path)
            pooled_prompt_embed_path = os.path.join(precomputed_embeddings_dir, f"{uid}_pooled.pt")
            torch.save(pooled_prompt_embed, pooled_prompt_embed_path)

    # 只在处理完整数据集时保存空提示的嵌入
    if start_idx == 0 and (end_idx == -1 or end_idx >= len(pd.read_csv(config.dpo.csv_file_path[config.dpo.dataset["train"]]))):
        logger.info("Saving empty prompt embeddings...")
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            [""],
            text_encoders,
            tokenizers,
            max_sequence_length=128,
            device=device
        )
        prompt_embed = prompt_embeds[0]
        pooled_prompt_embed = pooled_prompt_embeds[0]
        prompt_embed_path = os.path.join(precomputed_embeddings_dir, f"empty_prompt.pt")
        torch.save(prompt_embed, prompt_embed_path)
        pooled_prompt_embed_path = os.path.join(precomputed_embeddings_dir, f"empty_prompt_pooled.pt")
        torch.save(pooled_prompt_embed, pooled_prompt_embed_path)
    
    logger.info("Precomputation completed!")

# 使用示例:
# python scripts/precompute_prompt_embeddings_with_range.py --config config/sd3_5_medium_dpo.py:paired_real_fake_dataset_sd3 --start_index=0 --end_index=1000
# python scripts/precompute_prompt_embeddings_with_range.py --config config/sd3_5_medium_dpo.py:paired_real_fake_dataset_sd3 --start_index=0 --end_index=-1  # 处理全部
if __name__ == "__main__":
    app.run(main)
