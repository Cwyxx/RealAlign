import os
import argparse
from tqdm import tqdm
import torch
from diffusers import StableDiffusionPipeline
from datasets import load_dataset, Dataset
from callbacks import make_callback

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
        "--method",
        type=str,
        default="composite"
    )
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
        '--switch_step', 
        default=5,
        help='number of steps to switch LoRA during denoising, applicable only in the switch method', 
        type=int)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output image dir: {args.output_dir}")
    pipeline = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, custom_pipeline="./pipelines/sd1.5_0.26.3", torch_dtype=torch.float16, safety_checker=None).to('cuda')
    
    for lora_dir, lora_weight, adapter_name in zip(args.lora_dirs, args.lora_weights, args.adapter_names):
        pipeline.load_lora_weights(lora_dir, weight_name="pytorch_lora_weights.safetensors", adapter_name=adapter_name)
        print(f"Loaded LoRA from {lora_dir} with weight {lora_weight} for adapter {adapter_name}")
    
    cur_loras = args.adapter_names
    # select the method for the composition
    if args.method == "merge":
        pipeline.set_adapters(cur_loras)
        switch_callback = None
        
    elif args.method == "switch":
        print(f"Switch")
        pipeline.set_adapters([cur_loras[0]])
        switch_callback = make_callback(switch_step=args.switch_step, loras=cur_loras)
        
    else:
        print(f"Lora_composite: {True if args.method == 'composite' else False}")
        pipeline.set_adapters(cur_loras)
        switch_callback = None
    
    print(f"activate adapters: {pipeline.get_active_adapters()}")
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
                generate_images = pipeline(prompts, generator=[ generator ] * len(prompts), callback_on_step_end=switch_callback, lora_composite=True if args.method == "composite" else False).images
            else:
                generate_images = pipeline(prompts, callback_on_step_end=switch_callback, lora_composite=True if args.method == "composite" else False).images
                
            for image_uid, generate_image in zip(image_uids, generate_images):
                generate_image.save(os.path.join(args.output_dir, f"{image_uid}.png"))