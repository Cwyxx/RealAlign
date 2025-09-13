import os
import sys
sys.path.append("/data3/chenweiyan/notebook/fine-tune-diffusion/spo_gitee/training_and_inference/spo/preference_models")
sys.path.append("/data3/chenweiyan/notebook/fine-tune-diffusion/spo_gitee/training_and_inference")
from pycocotools.coco import COCO
from models.step_aware_preference_model import StepAwarePreferenceModel
from tqdm import tqdm
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image
import torch

# HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=1 python inference.py
spm = StepAwarePreferenceModel(model_pretrained_model_name_or_path='yuvalkirstain/PickScore_v1',
        processor_pretrained_model_name_or_path='laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
        ckpt_path='/data_center/data2/dataset/chenwy/21164-data/model-ckpt/spo_step_aware_preference_model/sd-v1-5_step-aware_preference_model.bin').to("cuda")
clip_tokenizer = AutoTokenizer.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
processor = AutoProcessor.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')

real_image_dir = "/data_center/data2/dataset/detection_dataset/coco2017-val/images.cocodataset.org/val2017"
fake_image_dir = "/data_center/data2/dataset/chenwy/21164-data/drct2m/stable-diffusion-v1-5/val2017"
coco = COCO("/data_center/data2/dataset/chenwy/21164-data/coco_2017/annotations/captions_val2017.json")


def get_score(image_path, prompt):
    image = Image.open(image_path).convert("RGB")
    spm_input = processor(text=prompt, images=image, return_tensors="pt")
    score = spm.model(input_ids=spm_input['input_ids'].to("cuda"), pixel_values=spm_input["pixel_values"].to("cuda"), time=torch.LongTensor([0]).to("cuda"))[0].item()
    return score
    
total_process_image_num = 0
win_image_num = 0
for image_name in tqdm(sorted(os.listdir(real_image_dir)), dynamic_ncols=True):
    uid = os.path.splitext(image_name)[0]
    ann_ids = coco.getAnnIds(imgIds=[int(uid)])
    anns = coco.loadAnns(ann_ids)
    prompt = [a["caption"] for a in anns][0]
    
    if not os.path.exists(os.path.join(real_image_dir, image_name)) or not os.path.exists(os.path.join(fake_image_dir, f"{uid}.jpg")):
        continue
    
    real_score = get_score(os.path.join(real_image_dir, image_name), prompt)
    fake_score = get_score(os.path.join(fake_image_dir, f"{uid}.jpg"), prompt)
    
    if real_score > fake_score:
        win_image_num += 1
    total_process_image_num += 1
    
print(f"win_image_num: {win_image_num}")
print(f"total_process_image_num: {total_process_image_num}")