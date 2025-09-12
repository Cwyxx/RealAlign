import os
import argparse
import torch
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from peft import PeftModel, get_peft_model, LoraConfig

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help=(
            "Path to a pretrained Stable Diffusion model or its identifier from Hugging Face Hub. "
            "This is used to load the base model for training or inference. "
            "Default is 'CompVis/stable-diffusion-v1-4'."
        )
    )
    
    ### Diffusers to PEFT ###
    parser.add_argument(
        "--lora_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--adapter_name",
        type=str,
        default=None,
    )
    
    args = parser.parse_args()

    return args


# python analysis.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" --lora_dir="/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/spo-sdv1-5/peft/step_aware/checkpoint_1_754" --adapter_name="step_aware"
# python analysis.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" --lora_dir="/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/spo-sdv1-5/peft/code-lr_6e-05-max_gn_1.0-comp_0.0/checkpoint_0_800" --adapter_name="code"
# python analysis.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" --lora_dir="/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/spo-sdv1-5/peft/dinov2/checkpoint_0_800" --adapter_name="dinov2"
if __name__ == "__main__":
    args = get_args()
    
    # Load base model
    base_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        subfolder="unet",
    ).to("cuda")
    
    model = PeftModel.from_pretrained(
        base_unet, 
        args.lora_dir, 
        use_safetensors=True, 
        subfolder=args.adapter_name, 
        adapter_name=args.adapter_name
    )
    
    # 步骤2: 遍历模型，找到 LoRA 层并计算 Frobenius 范数
    lora_norms = {}  # 存储每个层的范数

    for name, module in model.named_modules():
        # print(f"name: {name}, module: {module}")
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            # 提取 lora_A 和 lora_B
            A = module.lora_A[args.adapter_name].weight.data  # 形状: [r, in_features]
            B = module.lora_B[args.adapter_name].weight.data  # 形状: [out_features, r]
            
            # 计算 ΔW = B @ A
            delta_W = torch.matmul(B, A)  # 形状: [out_features, in_features]
            
            # 计算 Frobenius 范数
            fro_norm = torch.norm(delta_W, p='fro').item()  # 或手动: torch.sqrt(torch.sum(delta_W**2))
            
            # 存储结果，键为层名
            lora_norms[name] = fro_norm
            
            # print(f"A.shape: {A.shape}, B.shape: {B.shape}, ΔW.shape: {delta_W.shape}, Frobenius Norm: {fro_norm}")

    # 步骤3: 输出结果
    # for layer, norm in lora_norms.items():
    #     print(f"Layer: {layer}, Frobenius Norm: {norm}")

    total_norm = sum(lora_norms.values())
    print(f"Total Frobenius Norm of all LoRA layers: {total_norm}")