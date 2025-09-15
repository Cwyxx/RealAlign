import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.append("../../")
import torch
import torchvision
from datasets import load_dataset, Dataset
from PIL import Image
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--reward_model",
    type=str,
    default="imagereward",
    help=(
        "The reward model to use for computing rewards during training."
        "Support values include 'imagereward', 'hpsv2', 'aesthetic', 'aigi_detector'. Default is 'imagereward'."
    )
)
parser.add_argument(
    "--generated_image_dir",
    type=str,
    default=None,
    help="Directory to save the generated images."
)
parser.add_argument(
    "--val_json_data_path",
    type=str,
    default="",
    help="Path to the JSON file containing validation data. This file is used during validation to generate images to evaluate model performance.",
)
parser.add_argument(
    "--image_column",
    type=str
)
parser.add_argument(
    "--caption_column",
    type=str
)
args = parser.parse_args()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if args.reward_model == "imagereward":
    import ImageReward as RM
    reward_model = RM.load("ImageReward-v1.0")
    def imagereward_score(prompt, image_path, device):
        return reward_model.score(prompt, image_path)
    
    reward_fn = imagereward_score
    print(f"create reward model: imagereward")

elif args.reward_model == "hpsv2":
    from reward_model.hpsv2 import HPSv2
    reward_model = HPSv2()
    reward_model.requires_grad_(False)
    reward_model.to(device, torch.float32)
    
    def hpsv2_score(prompt, image_path, device):
        image = Image.open(image_path).convert("RGB")
        _transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        image = _transform(image).unsqueeze(0).to(device) 
        caption = reward_model.tokenizer(prompt).to(device)
        outputs = reward_model.model(image, caption)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits = image_features @ text_features.T
        reward = torch.diagonal(logits).item()
        return reward
    
    reward_fn = hpsv2_score
    
elif args.reward_model == "pickscore":
    from reward_model.pickscore import PickScore
    reward_model = PickScore()
    reward_model.requires_grad_(False)
    reward_model.to(device, torch.float32)
    
    def pickscore_score(prompt, image_path, device):
        image = Image.open(image_path).convert("RGB")
        _transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        
        image = _transform(image).unsqueeze(0).to(device)
        text_inputs = reward_model.processor(text=prompt, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
        
        image_embs = reward_model.model.get_image_features(image)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True) #
        text_embs = reward_model.model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True) #
        
        logits = reward_model.model.logit_scale.exp() * text_embs @ image_embs.T #
        reward = torch.diagonal(logits).item() #
        return reward
    
    reward_fn = pickscore_score
        
elif args.reward_model == "aesthetic":
    from reward_model.aesthetic import AestheticScorerDiff
    reward_model = AestheticScorerDiff()
    reward_model.requires_grad_(False)
    reward_model.to(device, torch.float32)
    
    def aesthetic_score(prompt, image_path, device):
        image = Image.open(image_path).convert("RGB")
        _transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        
        image = _transform(image).unsqueeze(0).to(device)
        reward = reward_model(image).item()

        return reward
    
    reward_fn = aesthetic_score
    
elif args.reward_model == "clipscore":
    from torchmetrics.multimodal.clip_score import CLIPScore
    import numpy as np
    reward_model = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
    
    def calculate_clip_score(prompt, image_path, device):
        pil_image = Image.open(image_path).convert("RGB")
        pil_image = pil_image.resize((224, 224))  # Resize to match model input
        image_np = np.array(pil_image)  # Convert to NumPy array (H, W, C)
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1))  # Transpose to (C, H, W); results in uint8 tensor
        image_tensor = image_tensor.to(dtype=torch.int64).to(device)
        
        score = reward_model(image_tensor, prompt)
        return score
    
    reward_fn = calculate_clip_score
    
elif args.reward_model == "clip_iqa":
    import pyiqa
    reward_model = pyiqa.create_metric("clipiqa", device=device)
    
    def calculate_clipiqa_score(prompt, image_path, device):
        score = reward_model(image_path)
        if isinstance(score, torch.Tensor):
            score = score.item()
        
        return score
    
    reward_fn = calculate_clipiqa_score
    
elif args.reward_model == "deqa":
    from transformers import AutoModelForCausalLM
    reward_model = AutoModelForCausalLM.from_pretrained(
        "zhiyuanyou/DeQA-Score-Mix3",
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    def calculate_deqa_score(prompt, image_path, device):
        pil_image = Image.open(image_path).convert("RGB")
        score = reward_model.score([pil_image])[0].item()
        return score
    
    reward_fn = calculate_deqa_score
    
elif args.reward_model == "aesthetic_v2_5":
    from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
    reward_model, preprocessor = convert_v2_5_from_siglip(
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    reward_model = reward_model.to(device)
    
    def calculate_aesthetic_v2_5_score(prompt, image_path, device):
        pil_image = Image.open(image_path).convert("RGB")
        pixel_values = (
            preprocessor(images=pil_image, return_tensors="pt")
            .pixel_values.to(device)
        )
        score = reward_model(pixel_values).logits.squeeze().float().cpu().numpy()
        
        return score
    
    reward_fn = calculate_aesthetic_v2_5_score

data_files = {}
data_files["val"] = args.val_json_data_path
dataset = load_dataset(
    os.path.splitext(args.val_json_data_path)[1][1:],
    data_files=data_files
)
val_dataset = dataset['val']
val_dataset = list(val_dataset)
val_dataset = sorted(val_dataset, key=lambda x: x[args.image_column])
val_dataset = val_dataset[:10_000]
val_dataset = Dataset.from_list(val_dataset)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    shuffle=False,
    batch_size=1
)

print_once = True
score_list = []
with torch.inference_mode():
    for batch in tqdm(val_dataloader, dynamic_ncols=True, desc=f"{args.reward_model}:{args.generated_image_dir}"):
        image_uids = batch[args.image_column]
        prompts = batch[args.caption_column]
        
        for image_uid, prompt in zip(image_uids, prompts):
            image_path = os.path.join(args.generated_image_dir, f"{image_uid}.png")
            score = reward_fn(prompt, image_path, device)
            score_list.append(score)

print(f"image num: {len(score_list)}")
print(f"average score: {sum(score_list) / len(score_list)}")