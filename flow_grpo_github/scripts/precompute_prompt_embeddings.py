import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
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
logger = get_logger(__name__)

class Paired_Real_Fake_Dataset(Dataset):
    def __init__(self, config, image_transform, split="train"):
        self.image_transform = image_transform
        self.csv_file_path = config.irl.csv_file_path[split]
        
        self.ext_list = [ ".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG" ]
        self.df = pd.read_csv(self.csv_file_path)
        
        if split == "high_quality_val": self.df = self.df.head(24)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row_data = self.df.iloc[idx]
        uid, prompt, win_image_path, lose_image_path = row_data["uid"], row_data["prompt"], row_data["win_image_path"], row_data["lose_image_path"]
        
        if win_image_path is None:
            raise FileNotFoundError(f"Missing WIN image for uid: {uid} at {win_image_path}")
        
        if lose_image_path is None:
            raise FileNotFoundError(f"Missing LOSE image for uid: {uid} at {lose_image_path}")
        
        win_pixel_values = self.image_transform(Image.open(win_image_path).convert("RGB"))
        lose_pixel_values = self.image_transform(Image.open(lose_image_path).convert("RGB"))
        pixel_values = torch.cat([win_pixel_values, lose_pixel_values], dim=0) # torch.cat [3, 512, 512] -> [6, 512, 512]
        
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
    train_dataset = Paired_Real_Fake_Dataset(config, image_transform, split=config.irl.dataset["train"])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=Paired_Real_Fake_Dataset.collate_fn,
        num_workers=1
    )
    
    precomputed_embeddings_dir = config.irl.precomputed_embeddings_dir_dict[config.irl.dataset["train"]]
    os.makedirs(precomputed_embeddings_dir, exist_ok=True)
    for uids, prompts, pixel_values in tqdm(train_dataloader):
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
    
# python scripts/precompute_prompt_embeddings.py --config config/sd3_5_medium_irl.py:paired_real_fake_dataset_sd3
if __name__ == "__main__":
    app.run(main)