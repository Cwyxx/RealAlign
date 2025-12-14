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
from diffusers import FlowMatchEulerDiscreteScheduler
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
from scripts.irl_tools import ReplayBuffer, EasyDict
from diffusers import StableDiffusion3Pipeline

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")
logger = get_logger(__name__)

class Paired_Real_Fake_Dataset(Dataset):
    def __init__(self, config, image_transform, split="train", random_drop_prompt_probability=0.2):
        self.precomputed_embeddings_dir = config.irl.precomputed_embeddings_dir_dict[split]
        self.image_transform = image_transform
        self.csv_file_path = config.irl.csv_file_path[split]
        
        self.ext_list = [ ".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG" ]
        self.df = pd.read_csv(self.csv_file_path)
        
        ### Inverse Reinforcement Learning ###
        self.split = split
        self.random_drop_prompt_probability = random_drop_prompt_probability
        ### Inverse Reinforcement Learning ###
        
        if split == "high_quality_val": self.df = self.df.head(24)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row_data = self.df.iloc[idx]
        uid, prompt, win_image_path, lose_image_path = row_data["uid"], row_data["prompt"], row_data["win_image_path"], row_data["lose_image_path"]
        
        if random.random() < self.random_drop_prompt_probability and self.split != "high_quality_val":
            prompt = ""
            uid = "empty_prompt"
        
        win_pixel_values = self.image_transform(Image.open(win_image_path).convert("RGB"))
        lose_pixel_values = self.image_transform(Image.open(lose_image_path).convert("RGB"))
        pixel_values = torch.cat([win_pixel_values, lose_pixel_values], dim=0) # torch.cat [3, 512, 512] -> [6, 512, 512]
        prompt_embed_path = os.path.join(self.precomputed_embeddings_dir, f"{uid}.pt")
        pooled_prompt_embed_path = os.path.join(self.precomputed_embeddings_dir, f"{uid}_pooled.pt")
        
        prompt_embed = torch.load(prompt_embed_path, map_location="cpu")
        pooled_prompt_embed = torch.load(pooled_prompt_embed_path, map_location="cpu")
        return {
            "prompt": prompt,
            "prompt_embed": prompt_embed,
            "pooled_prompt_embed": pooled_prompt_embed,
            "pixel_values": pixel_values
        }
        
    @staticmethod
    def collate_fn(examples):
        prompts = [ example["prompt"] for example in examples ]
        pixel_values = [ example["pixel_values"] for example in examples ]
        pixel_values = torch.stack(pixel_values, dim=0).to(memory_format=torch.contiguous_format).float()  # torch.stack [6, 512, 512] -> [batch_size, 6, 512, 512]
        prompt_embeds = [ example["prompt_embed"] for example in examples ]
        pooled_prompt_embeds = [ example["pooled_prompt_embed"] for example in examples ]
        prompt_embeds = torch.stack(prompt_embeds, dim=0).to(memory_format=torch.contiguous_format).float()  # torch.stack [205, 4096] -> [batch_size, 205, 4096]
        pooled_prompt_embeds = torch.stack(pooled_prompt_embeds, dim=0).to(memory_format=torch.contiguous_format).float()  # torch.stack [2048] -> [batch_size, 2048]
        return prompts, pixel_values, prompt_embeds, pooled_prompt_embeds
        
def eval(pipeline, test_dataloader, config, accelerator, global_step, reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters):
    pipeline.transformer.set_adapter("learner")
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    # test_dataloader = itertools.islice(test_dataloader, 2)
    for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, pixel_values, prompt_embeds, pooled_prompt_embeds = test_batch

        with autocast():
            with torch.no_grad():
                images, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution, 
                    noise_level=0,
                    generator=torch.Generator(device=accelerator.device).manual_seed(42),
                )
    
    last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy()
    last_batch_prompts_gather = accelerator.gather_for_metrics(prompts)
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

def save_ckpt(save_dir, pipeline, global_step, accelerator, ema, transformer_trainable_parameters, config):
    pipeline.transformer.set_adapter("learner")
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    logger.info(f"Saving LoRA to {save_root_lora}")
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        pipeline.transformer.save_pretrained(save_root_lora, selected_adapters=["learner"])
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
        swanlab.init(project=config.irl.project_name,
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
    ## Disable all text_encoder and tokenizer, only use transformer. ##
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model, 
        text_encoder=None,
        tokenizer=None,
        text_encoder_2=None,
        tokenizer_2=None,
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
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path, adapter_name="learner", is_trainable=True)
            pipeline.transformer.load_adapter(config.train.lora_path, adapter_name="ref", is_trainable=False)
            # After loading with PeftModel.from_pretrained, all parameters have requires_grad set to False. You need to call set_adapter to enable gradients for the adapter parameters.
            pipeline.transformer.set_adapter("learner")
            logger.info(f"Loaded LoRA from {config.train.lora_path}")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config, adapter_name="learner")
            pipeline.transformer.add_adapter("ref", transformer_lora_config)
            pipeline.transformer.set_adapter("learner")
        
        logger.info(f"type(pipeline.transformer) {type(pipeline.transformer)}")
        logger.info(f"LoRA adapter names {pipeline.transformer.peft_config.keys()}")
    #### Use LoRA to fine-tune SD-3-5-Medium ####
    
    ### set requires_grad for LoRA adapter parameters ###
    for name, param in pipeline.transformer.named_parameters():
        if "learner" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    transformer = pipeline.transformer
    transformer_trainable_parameters = []
    for name, param in transformer.named_parameters():
        if "learner" in name:
            assert param.requires_grad == True
            transformer_trainable_parameters.append(param)
            
    logger.info(f"trainable_parameters_num: {len(transformer_trainable_parameters)}")
    copy_learner_to_ref(transformer)
    
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
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    #### image_transform, copy from dive-into-sd-3-5-medium ####
    
    train_dataset = Paired_Real_Fake_Dataset(config, image_transform, split=config.irl.dataset["train"], random_drop_prompt_probability=0.2)
    val_dataset = Paired_Real_Fake_Dataset(config, image_transform, split=config.irl.dataset["val"], random_drop_prompt_probability=0.2)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.irl.batch_size,
        shuffle=True,
        collate_fn=Paired_Real_Fake_Dataset.collate_fn,
        num_workers=config.irl.batch_size
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.irl.batch_size,
        shuffle=False,
        collate_fn=Paired_Real_Fake_Dataset.collate_fn,
        num_workers=config.irl.batch_size
    )
    
    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    
    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, val_dataloader = accelerator.prepare(transformer, optimizer, train_dataloader, val_dataloader)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config.pretrained.model, subfolder="scheduler")
    total_batch_size = config.irl.batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps
    global_step = 0
        
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info(f"  Max_Train_Steps = {config.irl.max_train_steps}")
    logger.info(f"  IRL Margin = {config.irl.margin}")
    logger.info(f"  Learning Rate = {config.train.learning_rate}")

    progress_bar = tqdm(range(global_step, config.irl.max_train_steps), position=0, disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    def infinite_loop(loader):
        while True:
            for batch in loader:
                yield batch
    train_dataloader = infinite_loop(train_dataloader)
    
    step_loss = EasyDict()
    step_loss.loss = 0
    step_loss.diff = 0
    step_loss.expert = 0
    step_loss.policy = 0
    step_loss.expert0 = 0
    step_loss.policy0 = 0
    step_loss.expert_diff = 0
    step_loss.policy_diff = 0

    eval(pipeline, val_dataloader, config, accelerator, global_step, None, None, autocast, None, ema, transformer_trainable_parameters)
    if accelerator.is_main_process:
        save_ckpt(config.save_dir, pipeline, global_step, accelerator, ema, transformer_trainable_parameters, config)
    
    accelerator.wait_for_everyone()
    precomputed_embeddings_dir = config.irl.precomputed_embeddings_dir_dict[config.irl.dataset["train"]]
    negative_prompt_embeds = torch.load(os.path.join(precomputed_embeddings_dir, "empty_prompt.pt"), map_location="cpu")
    negative_pooled_prompt_embeds = torch.load(os.path.join(precomputed_embeddings_dir, "empty_prompt_pooled.pt"), map_location="cpu")
    negative_prompt_embeds = negative_prompt_embeds.repeat(config.irl.batch_size, 1, 1).to(accelerator.device, dtype=inference_dtype)
    negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(config.irl.batch_size, 1).to(accelerator.device, dtype=inference_dtype)
    
    while global_step <= config.irl.max_train_steps:
        batch = next(train_dataloader)
        prompts, pixel_values, prompt_embeds, pooled_prompt_embeds = batch
        bsz = config.irl.batch_size
        with torch.no_grad():
            pixel_values = pixel_values.chunk(2, dim=1)[0] # [batch_size, 6, 512, 512] -> [batch_size, 3, 512, 512]
            x0 = pipeline.vae.encode(pixel_values).latent_dist.sample()
            x0_expert = (x0 - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor # [batch_size, 4, 64, 64]
            # accelerator.print(f"x0_expert: {x0_expert.shape}")
            
        with autocast():
            with torch.no_grad():
                pipeline.transformer.set_adapter("learner")
                x0_policy = pipeline(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    num_inference_steps=config.irl.buffer_num_inference_steps,
                    guidance_scale=config.irl.buffer_guidance_scale,
                    output_type="latent",
                    height=config.resolution,
                    width=config.resolution, 
                    return_dict=False,
                )[0] # [batch_size, 4, 64, 64]
                
        with torch.no_grad():
            prompt_embeds = prompt_embeds.repeat(2, 1, 1)
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(2, 1)
        
        model_input = torch.cat([x0_expert, x0_policy], dim=0) #[batch_size * 2, 4, 64, 64]
        noise = torch.randn_like(model_input)
        noise = torch.cat([noise[:bsz], noise[:bsz]], dim=0) #[batch_size * 2]
        
        # Sample a random timestep for each image for weighting schemes where we sample timesteps non-uniformly
        u = compute_density_for_timestep_sampling(
            weighting_scheme='logit_normal',
            batch_size=config.irl.batch_size,
            logit_mean=0,
            logit_std=1,
            mode_scale=1.29,
        )
        indices = (u * noise_scheduler.config.num_train_timesteps).long()
        timesteps = noise_scheduler.timesteps[indices].to(device=model_input.device)
        timesteps = torch.cat([timesteps, timesteps], dim=0) #[batch_size * 2]
        
        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        sigmas = get_sigmas(noise_scheduler, timesteps, accelerator, n_dim=model_input.ndim, dtype=model_input.dtype)
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
        target = noise - model_input
        
        eps_expert = target[:bsz]
        xt_policy = noisy_model_input[bsz:]
        
        with accelerator.accumulate(transformer):
            # Variable          Paper
            # -----------------------------
            # t              -> $t$
            # c              -> $c$
            # x0             -> $x_0$
            # eps_policy     -> $\epsilon_\theta(x_t, c)$
            # eps_expert     -> $\bar{\epsilon}$
            # xt_policy      -> $x_t$
            # xt_expert      -> $\bar{x}_t$
            # unet           -> $p_{\phi}$
            # unet_ref       -> $p_{\theta_{ref}}$
            # unet_policy    -> $p_{\theta}$

            
            with autocast():
                with torch.no_grad():
                    pipeline.transformer.set_adapter("learner")
                    pred_eps_policy = transformer(
                        hidden_states=xt_policy.to(dtype=inference_dtype),
                        timestep=timesteps[bsz:],
                        encoder_hidden_states=prompt_embeds[bsz:],
                        pooled_projections=pooled_prompt_embeds[bsz:],
                        return_dict=False,
                    )[0]
                    eps_policy = pred_eps_policy.detach()
            
            eps = torch.cat([eps_expert, eps_policy], dim=0)
            with autocast():
                pipeline.transformer.set_adapter("learner")
                pred_eps = transformer(
                    hidden_states=noisy_model_input.to(dtype=inference_dtype),
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
                
                
                with torch.no_grad():
                    pipeline.transformer.set_adapter("ref")
                    pred_eps_ref = transformer(
                        hidden_states=noisy_model_input.to(dtype=inference_dtype),
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        return_dict=False,
                    )[0]
                    
                    pred_eps_ref = pred_eps_ref.detach()
                    pipeline.transformer.set_adapter("learner")
                    
            losses0 = (pred_eps_ref - eps).square().mean(dim=[1, 2, 3])
            loss_expert0, loss_policy0 = losses0.chunk(2)
            
            losses = (pred_eps - eps).square().mean(dim=[1, 2, 3])
            loss_expert, loss_policy = losses.chunk(2)
            loss_expert_diff = loss_expert - loss_expert0
            loss_policy_diff = loss_policy - loss_policy0

            diff = loss_expert_diff - loss_policy_diff
            loss = torch.maximum(diff + config.irl.margin, torch.zeros_like(diff)).mean()
            accelerator.backward(loss)
            
            step_loss.loss += accelerator.gather(loss.unsqueeze(0).detach()).cpu().mean()
            step_loss.diff += accelerator.gather(diff.detach()).cpu().mean()
            step_loss.expert += accelerator.gather(loss_expert.detach()).cpu().mean()
            step_loss.policy += accelerator.gather(loss_policy.detach()).cpu().mean()
            step_loss.expert0 += accelerator.gather(loss_expert0.detach()).cpu().mean()
            step_loss.policy0 += accelerator.gather(loss_policy0.detach()).cpu().mean()
            step_loss.expert_diff += accelerator.gather(loss_expert_diff.detach()).cpu().mean()
            step_loss.policy_diff += accelerator.gather(loss_policy_diff.detach()).cpu().mean()

            if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        transformer.parameters(), config.train.max_grad_norm
                    )
            optimizer.step()
            optimizer.zero_grad()
            
            progress_bar.set_postfix(timesteps=timesteps.cpu().tolist(), loss=loss.item())
        
        # Log the training loss
        if accelerator.sync_gradients:
            if accelerator.is_main_process:
                for tag in step_loss.keys():
                    step_loss[tag] /= config.train.gradient_accumulation_steps
                
                swanlab.log(step_loss, step=global_step)            
                for tag in step_loss.keys():
                    step_loss[tag] = 0
            
            progress_bar.update(1)
            global_step += 1
            if config.train.ema:
                ema.step(transformer_trainable_parameters, global_step)
                
            if global_step % config.save_freq == 0:
                eval(pipeline, val_dataloader, config, accelerator, global_step, None, None, autocast, None, ema, transformer_trainable_parameters)
                if accelerator.is_main_process:
                    save_ckpt(config.save_dir, pipeline, global_step, accelerator, ema, transformer_trainable_parameters, config)
        
        # Wait for main processes to save the state
        accelerator.wait_for_everyone()
    
    # Destroy process group
    accelerator.end_training()        

if __name__ == "__main__":
    app.run(main)