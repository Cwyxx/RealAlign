import os
import argparse
from tqdm import tqdm
import torch
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from peft import PeftModel, get_peft_model, LoraConfig
from datasets import load_dataset, Dataset

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
    parser.add_argument(
        "--output_dir",
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
        "--batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
    )
    
    ### Fuse LoRA ###
    parser.add_argument(
        "--lora_dirs",
        type=str,
        nargs='+',
        default=[],
    )
    parser.add_argument(
        "--lora_weights",
        type=float,
        nargs='+',
        default=[],
    )
    parser.add_argument(
        "--adapter_names",
        type=str,
        nargs='+',
        default=[],
    )
    parser.add_argument(
        "--combination_type",
        type=str,
        default="cat",
    )
    
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output image dir: {args.output_dir}")
    base_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        subfolder="unet",
    ).to("cuda")
    
    
    if len(args.lora_dirs) > 1:
        model = PeftModel.from_pretrained(
            base_unet, 
            args.lora_dirs[0], 
            use_safetensors=True, 
            subfolder=args.adapter_names[0], 
            adapter_name=args.adapter_names[0]
        )
        print(f"Loaded LoRA from {args.lora_dirs[0]} with adapter {args.adapter_names[0]}")
        for i in range(1, len(args.lora_dirs)):
            model.load_adapter(
                args.lora_dirs[i],
                use_safetensors=True, 
                subfolder=args.adapter_names[i], 
                adapter_name=args.adapter_names[i]
            )
            print(f"Loaded LoRA from {args.lora_dirs[i]} with adapter {args.adapter_names[i]}")
            
        if args.combination_type in ["linear"]:
            model.add_weighted_adapter(
                adapters=args.adapter_names,
                weights=args.lora_weights,
                combination_type=args.combination_type,
                adapter_name="GGBond"
            )
            print(f"adapter_names: {args.adapter_names} withs weights: {args.lora_weights}")
            print(f"Added weighted adapter GGBond with combination type {args.combination_type}")
        
        elif args.combination_type in [ "magnitude_prune", "dare_linear" ]:
            model.add_weighted_adapter(
                adapters=args.adapter_names,
                weights=args.lora_weights,
                combination_type=args.combination_type,
                adapter_name="GGBond",
                density=0.7
            )
            print(f"adapter_names: {args.adapter_names} withs weights: {args.lora_weights}")
            print(f"Added weighted adapter GGBond with combination type {args.combination_type}")
            
        elif args.combination_type in [ "ties_svd" ]:
            model.add_weighted_adapter(
                adapters=args.adapter_names,
                weights=args.lora_weights,
                combination_type=args.combination_type,
                adapter_name="GGBond",
                density=0.5
            )
            print(f"adapter_names: {args.adapter_names} withs weights: {args.lora_weights}")
            print(f"Added weighted adapter GGBond with combination type {args.combination_type}")
        
        model.set_adapters("GGBond")
        model = model.to(dtype=torch.float16, device="cuda")
        
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path, 
            torch_dtype=torch.float16, 
            unet=model,
            safety_checker=None
        ).to('cuda')
        
    else:
        print("Please provide at least two LoRA directories for fusion.")
        exit(0)
    
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
        batch_size=args.batch_size
    )
    
    pipeline.set_progress_bar_config(
        leave=False,
        desc="Sampling Timestep",
        dynamic_ncols=True,
    )
    with torch.inference_mode():
        for batch in tqdm(val_dataloader):
            image_uids = batch[args.image_column]
            prompts = batch[args.caption_column]
            if args.seed is not None:
                generator = torch.Generator(device="cuda").manual_seed(args.seed)
                generate_images = pipeline(prompts, generator=[ generator ] * len(prompts)).images
            else:
                generate_images = pipeline(prompts).images
                
            for image_uid, generate_image in zip(image_uids, generate_images):
                generate_image.save(os.path.join(args.output_dir, f"{image_uid}.png"))