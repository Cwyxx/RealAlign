from collections import defaultdict
import contextlib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datetime
from concurrent import futures
import time
import json
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.training_utils import compute_density_for_timestep_sampling
import numpy as np
import pandas as pd
import flow_grpo.prompts
import flow_grpo.rewards
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import torch.nn.functional as F
import wandb
from functools import partial
import tqdm
import tempfile
from torchvision import transforms
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper
import swanlab

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")
logger = get_logger(__name__)

class Paired_Real_Generated_Dataset(Dataset):
    def __init__(self, config, image_transform, split="train"):
        self.dataset_dir = os.path.join(config.dpo.dataset_dir, split)
        self.generative_model_list = config.dpo.generative_model_list
        self.image_transform = image_transform
        self.csv_file_path = os.path.join(config.dpo.dataset_dir, f"{split}.csv")
        self.ext_list = [ ".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG" ]
        self.df = pd.read_csv(self.csv_file_path)
        
        if split == "high_quality_val": self.df = self.df.head(8)
        
    def __len__(self):
        return len(self.df) * len(self.generative_model_list)
    
    def __getitem__(self, idx):
        generative_model_idx = idx // len(self.df)
        idx = idx % len(self.df)
        generative_model = self.generative_model_list[generative_model_idx]
        row_data = self.df.iloc[idx]
        uid, prompt = row_data["uid"], row_data["PROMPT"]
        
        ### Load win image ###
        win_image_path = None
        for ext in self.ext_list:
            image_path = os.path.join(self.dataset_dir, "real", f"{uid}{ext}")
            if os.path.exists(image_path):
                win_image_path = image_path
                break
        ### Load win image ###
        
        ### Load lose image ###
        lose_image_path = None
        for ext in self.ext_list:
            image_path = os.path.join(self.dataset_dir, generative_model, f"{uid}{ext}")
            if os.path.exists(image_path):
                lose_image_path = image_path
                break
        ### Load lose image ###
        
        if win_image_path is None:
            raise FileNotFoundError(f"Missing WIN image for uid: {uid} at {os.path.join(self.dataset_dir, 'real')}")
        
        if lose_image_path is None:
            raise FileNotFoundError(f"Missing LOSE image for uid: {uid} (model: {generative_model}) at {os.path.join(self.dataset_dir, generative_model)}")
        
        win_pixel_values = self.image_transform(Image.open(win_image_path).convert("RGB"))
        lose_pixel_values = self.image_transform(Image.open(lose_image_path).convert("RGB"))
        pixel_values = torch.cat([win_pixel_values, lose_pixel_values], dim=0) # torch.cat [3, 512, 512] -> [6, 512, 512]
        return {
            "prompt": prompt,
            "win_model": "real",
            "lose_model": generative_model,
            "pixel_values": pixel_values
        }
        
    @staticmethod
    def collate_fn(examples):
        prompts = [ example["prompt"] for example in examples ]
        pixel_values = [ example["pixel_values"] for example in examples ]
        pixel_values = torch.stack(pixel_values, dim=0).to(memory_format=torch.contiguous_format).float()  # torch.stack [6, 512, 512] -> [batch_size, 6, 512, 512]
        return prompts, pixel_values
        
def eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters):
    pipeline.transformer.set_adapter("learner")
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    # test_dataloader = itertools.islice(test_dataloader, 2)
    for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, pixel_values = test_batch
        del pixel_values
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, 
            text_encoders, 
            tokenizers, 
            max_sequence_length=128, 
            device=accelerator.device
        )
        # The last batch may not be full batch_size
        if len(prompt_embeds)<len(sample_neg_prompt_embeds):
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[:len(prompt_embeds)]
            sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:len(prompt_embeds)]
        with autocast():
            with torch.no_grad():
                images, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution, 
                    noise_level=0,
                    generator=torch.Generator(device=accelerator.device).manual_seed(42),
                )
    
    last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy()
    last_batch_prompt_ids = tokenizers[0](
        prompts,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )

    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(8, len(last_batch_images_gather))
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            swanlab.log(
                {
                    "eval_images": [
                        swanlab.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | ",
                        )
                        for idx, prompt in enumerate(sampled_prompts)
                    ],
                },
                step=global_step,
            )
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)

def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


def copy_learner_to_ref(transformer):
    for name, param in transformer.named_parameters():
        if "learner" in name:
            ref_name = name.replace("learner", "ref")
            ref_param = dict(transformer.named_parameters())[ref_name]
            ref_param.data.copy_(param.data)

def get_sigmas(noise_scheduler, timesteps, accelerator, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
    timesteps = timesteps.to(accelerator.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config
    
    #### Construct Accelerator ####
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )
    if accelerator.is_main_process:
        swanlab.init(project=config.dpo.project_name,
                     config=config.to_dict(),
                     name=config.run_name)
        os.makedirs(config.save_dir, exist_ok=True)
        with open(os.path.join(config.save_dir, "exp_config.txt"), "w") as f:
            json.dump(config.to_dict(), f, indent=4)    
    logger.info(f"\n{config}")
    #### Construct Accelerator ####
    
    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)
    
    #### Load Scheduler, Tokenizer and Model. ####
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model, 
        text_encoder_3=None, 
        tokenizer_3=None
    )
    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    
    # freeze parameters of models to save more memory
    pipeline.transformer.requires_grad_(not config.use_lora)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    if pipeline.text_encoder_3 is not None: pipeline.text_encoder_3.requires_grad_(False)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    #### Load Scheduler, Tokenizer and Model. ####
    
    #### Mixed Precision Training ####
    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to inference_dtype
    pipeline.transformer.to(accelerator.device)
    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    if pipeline.text_encoder_3 is not None: pipeline.text_encoder_3.to(accelerator.device, dtype=inference_dtype)
    #### Mixed Precision Training ####
    
    #### Use LoRA to fine-tune SD-3-5-Medium ####
    if config.use_lora:
        # Set correct lora layers
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            # After loading with PeftModel.from_pretrained, all parameters have requires_grad set to False. You need to call set_adapter to enable gradients for the adapter parameters.
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path, adapter_name="learner", is_trainable=True)
            pipeline.transformer.load_adapter(config.train.lora_path, adapter_name="ref", is_trainable=False)# append an exist adapter to pipeline.transformer.
            logger.info(f"Loading lora form {config.train.lora_path}")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config, adapter_name="learner")
            pipeline.transformer.add_adapter("ref", transformer_lora_config)
            
        pipeline.transformer.set_adapter("learner")
    #### Use LoRA to fine-tune SD-3-5-Medium ####
        
    transformer = pipeline.transformer
    pipeline.transformer.set_adapter("learner")
    for name, param in transformer.named_parameters():
        if "learner" in name:
            assert param.requires_grad == True
    
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    pipeline.transformer.set_adapter("ref")
    ref_transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    transformer.set_adapter("learner")
    for src_param, tgt_param in zip(
        transformer_trainable_parameters, ref_transformer_trainable_parameters, strict=True
    ):
        assert src_param is not tgt_param
    
    logger.info(f"trainable_parameters_num: {len(transformer_trainable_parameters)}")
    
    # This ema setting affects the previous 20 × 8 = 160 steps on average.
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)
    
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )
    
    #### image_transform, copy from dive-into-sd-3-5-medium ####
    image_transform = transforms.Compose(
        [
            transforms.Resize(config.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(config.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    #### image_transform, copy from dive-into-sd-3-5-medium ####
    
    if config.prompt_fn == "paired_real_generated_dataset":
        train_dataset = Paired_Real_Generated_Dataset(config, image_transform, split="high_quality_train")
        val_dataset = Paired_Real_Generated_Dataset(config, image_transform, split="high_quality_val")
        
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.dpo.batch_size,
            shuffle=True,
            collate_fn=Paired_Real_Generated_Dataset.collate_fn,
            num_workers=config.dpo.batch_size
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.dpo.batch_size,
            shuffle=False,
            collate_fn=Paired_Real_Generated_Dataset.collate_fn,
            num_workers=config.dpo.batch_size
        )
    
    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    
    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, val_dataloader = accelerator.prepare(transformer, optimizer, train_dataloader, val_dataloader)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config.pretrained.model, subfolder="scheduler")
    total_batch_size = config.dpo.batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps
    global_step = 0
        
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info(f"  Max_Train_Steps = {config.dpo.max_train_steps}")

    progress_bar = tqdm(range(global_step, config.dpo.max_train_steps), position=0, disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    eval_and_save_ckpt = True
    info = defaultdict(list)
    while True:
        for step, (prompts, pixel_values) in enumerate(train_dataloader):
            #################### EVAL ####################
            if global_step>0 and global_step%config.train.ref_update_step==0:
                copy_learner_to_ref(transformer)
                
            pipeline.transformer.eval()
            if eval_and_save_ckpt:
                eval_and_save_ckpt = False
                eval(pipeline, val_dataloader, text_encoders, tokenizers, config, accelerator, global_step, None, None, autocast, None, ema, transformer_trainable_parameters)
                if accelerator.is_main_process:
                    save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)
            
            #################### TRAINING ####################
            pipeline.transformer.set_adapter("learner")
            pipeline.transformer.train()
            with torch.no_grad():
                # y_w and y_l were concatenated along channel dimension
                feed_pixel_values = torch.cat(pixel_values.chunk(2, dim=1)).to(device=accelerator.device, dtype=pipeline.vae.dtype) # pixel [batch_size, 6, H, W] -> [ batch_size, 3, H, W]*2 -> [batch_size*2, 3, H, W]
                model_input = pipeline.vae.encode(feed_pixel_values).latent_dist.sample()
                model_input = (model_input - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor

                prompt_embeds, pooled_prompt_embds = compute_text_embeddings(
                    prompts,
                    text_encoders,
                    tokenizers,
                    max_sequence_length=128,
                    device=accelerator.device
                )
                prompt_embeds = prompt_embeds.repeat(2, 1, 1)
                pooled_prompt_embds = pooled_prompt_embds.repeat(2, 1)
                
                
            with accelerator.accumulate(transformer):
                bsz = model_input.shape[0] // 2
                # chosen and rejected using same noise as in Diffusion-DPO
                noise = torch.randn_like(model_input)
                noise = torch.cat([noise[:bsz], noise[:bsz]], dim=0)
                
                # Sample a random timestep for each image for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme='logit_normal',
                    batch_size=bsz,
                    logit_mean=0,
                    logit_std=1,
                    mode_scale=1.29,
                )
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[indices].to(device=model_input.device)
                timesteps = torch.cat([timesteps, timesteps], dim=0)
                
                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(noise_scheduler, timesteps, accelerator, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
                
                with autocast():
                    pipeline.transformer.set_adapter("learner")
                    model_pred = transformer(
                        hidden_states=noisy_model_input,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embds,
                        return_dict=False,
                    )[0]
                    
                    with torch.no_grad():
                        pipeline.transformer.set_adapter("ref")
                        model_pred_ref = transformer(
                            hidden_states=noisy_model_input,
                            timestep=timesteps,
                            encoder_hidden_states=prompt_embeds,
                            pooled_projections=pooled_prompt_embds,
                            return_dict=False,
                        )[0]
                        model_pred_ref = model_pred_ref.detach()
                        pipeline.transformer.set_adapter("learner")
                
                target = noise - model_input
                theta_mse = ((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1).mean(dim=1)
                ref_mse = ((model_pred_ref.float() - target.float()) ** 2).reshape(target.shape[0], -1).mean(dim=1)
                model_w_err = theta_mse[:bsz]
                model_l_err = theta_mse[bsz:]
                ref_w_err = ref_mse[:bsz]
                ref_l_err = ref_mse[bsz:]
                w_diff = model_w_err - ref_w_err
                l_diff = model_l_err - ref_l_err
                w_l_diff = w_diff - l_diff
                inside_term = -0.5 * config.train.beta * w_l_diff
                loss = -F.logsigmoid(inside_term)
                
                loss = torch.mean(loss)
                info["loss"].append(loss)
                info["model_w_err"].append(torch.mean(model_w_err))
                info["model_l_err"].append(torch.mean(model_l_err))
                info["ref_w_err"].append(torch.mean(ref_w_err))
                info["ref_l_err"].append(torch.mean(ref_l_err))
                info["w_diff"].append(torch.mean(w_diff))
                info["l_diff"].append(torch.mean(l_diff))
                info["w_l_diff"].append(torch.mean(w_l_diff))
                info["inside_term"].append(torch.mean(inside_term))
                implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
                info["implicit_acc"].append(torch.mean(implicit_acc))
                # backward pass
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        transformer.parameters(), config.train.max_grad_norm
                    )
                optimizer.step()
                optimizer.zero_grad()


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                info = accelerator.reduce(info, reduction="mean")
                if accelerator.is_main_process: 
                    swanlab.log(info, step=global_step)
                info = defaultdict(list)
                progress_bar.update(1)
                global_step += 1
                if global_step % config.save_freq == 0:
                    eval_and_save_ckpt = True
                    
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)
                    
            if global_step >= config.dpo.max_train_steps:
                break
        
        if global_step >= config.dpo.max_train_steps:
                break

if __name__ == "__main__":
    app.run(main)