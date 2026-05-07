import json
import os
from functools import partial
from typing import Callable

import accelerate
import click
import diffusers
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from transformers import (
    AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection)

from misc.buffer import ReplayBuffer
from misc.dataset import TrainingDataset, PromptDataset, ScoreDataset
from misc.scores import get_score
from misc.patch import (
    DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler,
    StableDiffusionPipeline, StableDiffusionXLPipeline)
from misc.utils import CommandAwareConfig, EasyDict
import swanlab
from peft import LoraConfig
from diffusers.utils import convert_unet_state_dict_to_peft
import copy
from peft.utils import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

@click.command(cls=CommandAwareConfig, context_settings={'show_default': True})
@click.option(
    '--config', default=None, type=str, metavar="FILE",
    help=(
        "Path to the config file. The command line arguments will overwrite "
        "the config file."
    )
)
@click.option(
    "--logdir", default="./logs/sd15_diffusion-dro", type=str, metavar="DIR",
    help=(
        "The output directory where the model predictions and checkpoints "
        "will be written."
    )
)
@click.option(
    "--seed", default=0, type=int,
    help="A seed for reproducible training."
)
@click.option(
    "--pretrained_model_name_or_path",
    default="runwayml/stable-diffusion-v1-5", type=str,
    help="Path to pretrained model or model identifier.",
)
@click.option(
    "--variant", default=None, type=str,
    help=(
        "Variant of the model files of the pretrained model identifier "
        "from huggingface.co/models, 'e.g.' fp16"
    )
)
@click.option(
    "--sdxl/--no-sdxl", default=False,
    help="Whether the model is a Stable Diffusion XL model."
)
@click.option(
    "--train_dataset", default="./data/pickapicv2_hpsv2_500", type=str,
    metavar="DIR",
    help=(
        "The root directory of the training dataset. See the "
        "`TrainingDataset` class for more information on the dataset format."
    ),
)
@click.option(
    "--resolution", default=512, type=int,
    help=(
        "The resolution for input images, all the images in the "
        "train/validation dataset will be resized to this resolution."
    ),
)
@click.option(
    "--random_crop/--no-random_crop", default=False,
    help=(
        "Whether to random crop the input images to the resolution. If "
        "not set, the images will be center cropped. The images will be "
        "resized to the resolution first before cropping."
    ),
)
@click.option(
    "--random_flip/--no-random_flip", default=True,
    help="whether to randomly flip images horizontally",
)
@click.option(
    "--validation_dataset", default="./data/pickapicv2_test", type=str,
    metavar="DIR",
    help=(
        "The root directory of the validation dataset. See the "
        "`TrainingDataset` class for the dataset format."
    ),
)
@click.option(
    "--validation_scheduler", default="DDPM",
    type=click.Choice(["DDPM", "DDIM", "DPMSolver++"]),
    help="The scheduler to use for the validation.",
)
@click.option(
    "--validation_num_inference_steps", default=50, type=int,
    help=(
        "Number of steps for the inference. The RL training only updates "
        "these many steps."
    )
)
@click.option(
    "--validation_guidance_scale", default=7.5, type=float,
    help="Guidance scale for the validation."
)
@click.option(
    "--score", default=None, type=click.Choice([
        "pickscore", "hpsv2", "aestheticv1", "aestheticv2", "clip",
        "imagereward"
    ]),
    help=(
        "The score to compute for full validation set. If not set, the score "
        "will not be computed."
    )
)
@click.option(
    "--score_batch_size", default=4, type=int,
)
@click.option(
    "--score_num_images_per_prompt", default=1, type=int,
)
@click.option(
    "--batch_size", default=4, type=int,
    help="Batch size (per device) for each forward and backward step."
)
@click.option(
    "--num_steps", default=25600, type=int,
    help="Number of forward and backward steps to take."
)
@click.option(
    "--gradient_accumulation_steps", default=16, type=int,
    help="Number of gradient accumulations steps per update."
)
@click.option(
    "--gradient_checkpointing/--no-gradient_checkpoint", default=False,
    help=(
        "Whether or not to use gradient checkpointing to save memory at "
        "the expense of slower backward pass."
    )
)
@click.option(
    "--learning_rate", default=1e-4, type=float,
    help="Initial learning rate (after the potential warmup period).",
)
@click.option(
    "--scale_lr/--no-scale_lr", default=False,
    help=(
        "Scale the learning rate by the gradient accumulation steps, batch "
        "size and number of GPUs."
    )
)
@click.option(
    "--lr_scheduler", default="constant",
    type=click.Choice([
        "linear", "cosine", "cosine_with_restarts", "polynomial",
        "constant", "constant_with_warmup"
    ]),
    help='The scheduler type to use.',
)
@click.option(
    "--lr_warmup_steps", default=0, type=int,
    help="Number of steps for the warmup in the lr scheduler."
)
@click.option(
    "--max_grad_norm", default=1.0, type=float,
    help="Max gradient norm. Set to 0 to disable gradient clipping."
)
@click.option(
    "--allow_tf32/--no-allow_tf32", default=False,
    help=(
        "Whether or not to allow TF32 on Ampere GPUs. Can be used to "
        "speed up training."
    ),
)
@click.option(
    "--dataloader_num_workers", default=8, type=int,
    help=(
        "Number of subprocesses to use for data loading. 0 means that "
        "the data will be loaded in the main process."
    ),
)
@click.option(
    "--mixed_precision", default='fp16',
    type=click.Choice(["no", "fp16", "bf16"]),
    help="Whether to use mixed precision training."
)
@click.option(
    "--use_ema/--no-use_ema", default=True,
    help="Whether to use exponential moving average for the policy network."
)
@click.option(
    "--offload_ema/--no-offload_ema", default=True,
    help="Whether to offload the EMA model to CPU."
)
@click.option(
    "--checkpointing_steps", default=50, type=int,
    help=(
        "Save a checkpoint of the training state every these many steps."
        "These checkpoints are only suitable for resuming training using "
        "`--resume`."
    ),
)
@click.option(
    "--resume/--no-resume", default=False,
    help="Whether to resume training from the state."
)
@click.option(
    "--unet_init", default="runwayml/stable-diffusion-v1-5", type=str,
)
@click.option(
    "--pretrained_lora_path", default=None, type=str
)
@click.option(
    "--run_name", required=True, type=str,
    help="The name of the run."
)
def main(**kwargs):
    """Fine-tune a Stable Diffusion model with Diffusion-DRO."""


    args = EasyDict(kwargs)
    gradient_accumulation_plugin = accelerate.utils.GradientAccumulationPlugin(
        num_steps=args.gradient_accumulation_steps,
        sync_with_dataloader=False,
    )
    accelerator = accelerate.Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_plugin=gradient_accumulation_plugin,
    )
    device = accelerator.device
    if accelerator.is_main_process:
        swanlab.init(project="diffusion-sd-v1-5-sft", name=args.run_name, config=dict(args))

    # Set the random seed for reproducibility
    accelerate.utils.set_seed(args.seed, device_specific=True)

    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Load noise scheduler, tokenizer and models.
    scheduler_classes = {
        "DDIM": DDIMScheduler,
        "DDPM": DDPMScheduler,
        "DPMSolver++": DPMSolverMultistepScheduler,
    }

    # Load training scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")

    # Load schedulers for validation and buffer sampling
    validation_scheduler_cls = scheduler_classes[args.validation_scheduler]
    validation_scheduler: DDPMScheduler = validation_scheduler_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    
    # Load the tokenizer and text encoder
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False)
    text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        variant=args.variant,
        device_map={"": str(device)},
        torch_dtype=dtype)
    text_encoder.requires_grad_(False)

    if args.sdxl:
        # Load the second tokenizer and text encoder
        tokenizer_2 = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            use_fast=False,
        )
        text_encoder_2: CLIPTextModelWithProjection = CLIPTextModelWithProjection.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            variant=args.variant,
            device_map={"": str(device)},
            torch_dtype=dtype)
        text_encoder_2.requires_grad_(False)

    # Load the VAE model
    if args.sdxl:
        vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path="madebyollin/sdxl-vae-fp16-fix",
            variant=args.variant,
            device_map={"": str(device)},
            torch_dtype=dtype)
    else:
        vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            variant=args.variant,
            subfolder="vae",
            device_map={"": str(device)},
            torch_dtype=dtype)
    vae.requires_grad_(False)

    accelerator.print(f"Loading unet from {args.unet_init}")
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
        args.unet_init,
        variant=args.variant,
        subfolder="unet",
        device_map={"": str(device)},
        torch_dtype=dtype)
    unet.requires_grad_(False)
    # Build and attach LoRA adapters (PEFT) like train_spo.py
    unet_lora_config = LoraConfig(
        r=4,
        lora_alpha=4,
        init_lora_weights="gaussian", # initialize the LoRA weights to a Gaussian distribution to ensure the ref == sd-v1-5
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)
    # Load the UNet models
    if args.pretrained_lora_path is not None:
        lora_state_dict, network_alphas = StableDiffusionPipeline.lora_state_dict(
            args.pretrained_lora_path,          
            weight_name="pytorch_lora_weights.safetensors"
        )
        unet_state_dict = {
            k.replace("unet.", ""): v
            for k, v in lora_state_dict.items()
            if k.startswith("unet.")
        }
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet, unet_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                accelerator.print(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )
        accelerator.print(f"Reuse lora weights from {args.pretrained_lora_path}")
        

    unet_trainable_parameters = list(filter(lambda p: p.requires_grad, unet.parameters()))
    # Cast the trainable parameters to the desired dtype
    if dtype == torch.float16: diffusers.training_utils.cast_training_params(unet, dtype=torch.float32)
    accelerator.print(f"trainable_para_num: {sum([ 1 for p in unet.parameters() if p.requires_grad ])}")
    

    if args.use_ema:
        unet_ema = diffusers.training_utils.EMAModel(
            unet_trainable_parameters, foreach=True)

        if args.offload_ema:
            unet_ema.to('cpu')
            unet_ema.pin_memory()
        else:
            unet_ema.to(device)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere and later CUDA devices.
    # cf https://huggingface.co/docs/diffusers/optimization/fp16#tensorfloat-32
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.batch_size * accelerator.num_processes
        )

    optimizer = torch.optim.AdamW(unet_trainable_parameters, lr=args.learning_rate)

    lr_scheduler = diffusers.optimization.get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.num_steps * accelerator.num_processes,
    )

    if args.sdxl:
        validation_pipeline = StableDiffusionXLPipeline(
            vae,
            text_encoder,
            text_encoder_2,
            tokenizer,
            tokenizer_2,
            unet,
            validation_scheduler,
        )

        # buffer_pipeline = StableDiffusionXLPipeline(
        #     vae,
        #     text_encoder,
        #     text_encoder_2,
        #     tokenizer,
        #     tokenizer_2,
        #     unet_policy,
        #     buffer_scheduler,
        # )
    else:
        validation_pipeline = StableDiffusionPipeline(
            vae,
            text_encoder,
            tokenizer,
            unet,
            validation_scheduler,
            safety_checker=None,            # Disable NSFW checker
            feature_extractor=None,         # Disable NSFW checker
            requires_safety_checker=False,  # Disable NSFW checker
        )
        # buffer_pipeline = StableDiffusionPipeline(
        #     vae,
        #     text_encoder,
        #     tokenizer,
        #     unet_policy,
        #     buffer_scheduler,
        #     safety_checker=None,            # Disable NSFW checker
        #     feature_extractor=None,         # Disable NSFW checker
        #     requires_safety_checker=False,  # Disable NSFW checker
        # )
    validation_pipeline.set_progress_bar_config(disable=True)
    # buffer_pipeline.set_progress_bar_config(disable=True)

    def encode_prompt(prompts, tokenizers, text_encoders):
        prompt_embeds_list = []

        captions = []
        for caption in prompts:
            captions.append(caption)

        with torch.no_grad():
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                text_inputs = tokenizer(
                    captions,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(text_encoder.device)

                if args.sdxl:
                    prompt_embeds = text_encoder(
                        text_input_ids, output_hidden_states=True, return_dict=False)
                    # We are only ALWAYS interested in the pooled output of the final text encoder
                    pooled_prompt_embeds = prompt_embeds[0]
                    prompt_embeds = prompt_embeds[-1][-2]
                    bs_embed, seq_len, _ = prompt_embeds.shape
                    prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                    prompt_embeds_list.append(prompt_embeds)
                else:
                    prompt_embeds = text_encoder(text_input_ids, return_dict=False)[0]

        if args.sdxl:
            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
            pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
            return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}
        else:
            return {"prompt_embeds": prompt_embeds}

    if args.sdxl:
        tokenizers = [tokenizer, tokenizer_2]
        text_encoders = [text_encoder, text_encoder_2]
    else:
        tokenizers = [tokenizer]
        text_encoders = [text_encoder]
        
    encode_prompt_fn = partial(
        encode_prompt,
        tokenizers=tokenizers,
        text_encoders=text_encoders)

    train_dataset = TrainingDataset(
        args.train_dataset, args.resolution, args.random_flip, args.random_crop)

    def collate_fn(batch_list):
        batch = dict()
        for key in batch_list[0].keys():
            batch[key] = [batch[key] for batch in batch_list]
            if isinstance(batch[key][0], torch.Tensor):
                batch[key] = torch.stack(batch[key], dim=0)
        return batch

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
    )

    unet, optimizer, lr_scheduler, train_loader = \
        accelerator.prepare(unet, optimizer, lr_scheduler, train_loader)

    def infinite_loop(loader):
        while True:
            for batch in loader:
                yield batch

    train_loader = infinite_loop(train_loader)

    # Create the log directory and save the arguments
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=args.logdir)
    else:
        writer = None

    if args.resume:
        state_path = os.path.join(args.logdir, "state")
        accelerator.load_state(state_path)
        training_state = torch.load(os.path.join(state_path, "training_state.pt"))
        init_step = training_state["step"] + 1
        resume_path = state_path
        # replay_buffer = ReplayBuffer(args.buffer_size, time_key="t")
    else:
        if accelerator.is_main_process:
            writer.add_text(
                "training_config.json", f"```\n{json.dumps(args, indent=2)}\n```")
            with open(os.path.join(args.logdir, "training_config.json"), "w") as f:
                json.dump(args, f, indent=4)
        init_step = 0
        resume_path = None
        # replay_buffer = ReplayBuffer(args.buffer_size, time_key="t")
    # unet_policy.load_state_dict(accelerator.unwrap_model(unet).state_dict())

    effective_batch_size = args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes
    accelerator.print(
        f"Training Configurations\n"
        f"- Num GPUs                : {accelerator.num_processes}\n"
        f"- Batch Size per Device   : {args.batch_size}\n"
        f"- Gradient Accumulation   : {args.gradient_accumulation_steps}\n"
        f"- Effective Batch Size    : {effective_batch_size}\n"
        f"- Dataset Size            : {len(train_dataset)}\n"
        f"- Total Optimization Steps: {args.num_steps}\n"
        f"- Resuming from states    : {resume_path}\n"
        f"- Mixed Precision         : {accelerator.mixed_precision}\n"
        f"- Training Configurations : {os.path.join(args.logdir, 'training_config.json')}\n"
        f"- Learning Rate           : {args.learning_rate}\n"
        f"- Warmup Steps            : {args.lr_warmup_steps}\n"
        f"- Scheduler               : {args.lr_scheduler}\n"
        f"- Max Grad Norm           : {args.max_grad_norm}\n"
        f"- Use EMA                 : {args.use_ema}\n"
        f"- Scale LR                : {args.scale_lr}\n"
    )
    
    step_loss = EasyDict()
    step_loss.loss = 0

    progress_bar = tqdm(
        range(init_step + 1, args.num_steps + 1),
        total=args.num_steps,
        initial=init_step,
        ncols=0,
        desc="Steps",
        disable=not accelerator.is_main_process,
    )
    evaluation_indicator = True
    step = init_step
    unet.train()
    while step < args.num_steps:
        with accelerator.accumulate(unet):
            batch = next(train_loader)
            # preprocessing. (# get prompt embedding | encode pixels --> latents)
            with torch.no_grad():
                prompt_embeds_dict = encode_prompt_fn(batch["prompt"])
                encoder_hidden_states = prompt_embeds_dict["prompt_embeds"]
                latents = vae.encode(batch["image"].to(dtype=dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            target = noise
            # Make the prediction from the model we're learning
            model_batch_args = (noisy_latents, timesteps, encoder_hidden_states)
            added_cond_kwargs = None

            model_pred = unet(*model_batch_args,
                            added_cond_kwargs=added_cond_kwargs
                            ).sample
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            
            accelerator.backward(loss)
            step_loss.loss += accelerator.gather(loss.unsqueeze(0).detach()).cpu().mean()
            if accelerator.sync_gradients and args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(unet_trainable_parameters, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            evaluation_indicator = True
            step += 1
            progress_bar.update(1)
            
        # Log the training loss
        if accelerator.sync_gradients and accelerator.is_main_process:
            for tag in step_loss.keys():
                step_loss[tag] /= args.gradient_accumulation_steps
                writer.add_scalar(f"loss/{tag}", step_loss[tag].float(), step)
            writer.add_scalar("params/lr", lr_scheduler.get_last_lr()[0], step)
            progress_bar.set_postfix_str(
                f"loss: {step_loss['loss']: .3E}, lr: {lr_scheduler.get_last_lr()[0]: .3E}")
            swanlab.log(step_loss, step=step)
            for tag in step_loss.keys():
                step_loss[tag] = 0

        # Update the EMA model
        if accelerator.sync_gradients and args.use_ema:
            if args.offload_ema:
                unet_ema.to(device, non_blocking=True)
            unet_ema.step(unet_trainable_parameters)
            if args.offload_ema:
                unet_ema.to("cpu", non_blocking=True)

        # Save the training state and a checkpoint of the model
        if step % args.checkpointing_steps == 0 and evaluation_indicator:
            evaluation_indicator = False
            os.makedirs(args.logdir, exist_ok=True)
            # State path
            state_path = os.path.join(args.logdir, "state")
            # Save ReplayBuffer
            # replay_buffer.save(state_path, accelerator)
            # Checkpoint path
            ckpt_path = os.path.join(args.logdir, "checkpoints", f"checkpoint-{step}")
            if accelerator.is_main_process:
                os.makedirs(ckpt_path, exist_ok=True)
                # Save unet weights
                unwrapped_unet: UNet2DConditionModel = accelerator.unwrap_model(unet)
                # Save the EMA model
                if args.use_ema:
                    unet_ema.store(unet_trainable_parameters)
                    unet_ema.copy_to(unet_trainable_parameters)
                    unet_lora_state_dict = get_peft_model_state_dict(unwrapped_unet)
                    StableDiffusionPipeline.save_lora_weights(
                        save_directory=ckpt_path,
                        unet_lora_layers=unet_lora_state_dict,
                    )
                    unet_ema.restore(unet_trainable_parameters)
                    progress_bar.write(f"Saved EMA weights to {ckpt_path}")
                else:
                    unet_lora_state_dict = get_peft_model_state_dict(unwrapped_unet)
                    StableDiffusionPipeline.save_lora_weights(
                        save_directory=ckpt_path,
                        unet_lora_layers=unet_lora_state_dict,
                    )
                    progress_bar.write(f"Saved weights to {ckpt_path}")

            if args.score is not None:
                if args.use_ema:
                    unet_ema.store(unet_trainable_parameters)
                    unet_ema.copy_to(unet_trainable_parameters)
                output_dir = os.path.join(args.logdir, "images", f"checkpoint-{step}")
                validation_pipeline.unet = accelerator.unwrap_model(unet)
                score = log_score(
                    accelerator=accelerator,
                    writer=writer,
                    dataset=PromptDataset(args.validation_dataset),
                    pipeline=validation_pipeline,
                    encode_prompt_fn=encode_prompt_fn,
                    num_inference_steps=args.validation_num_inference_steps,
                    guidance_scale=args.validation_guidance_scale,
                    batch_size=args.score_batch_size,
                    num_images_per_prompt=args.score_num_images_per_prompt,
                    score_name=args.score,
                    output_dir=output_dir,
                    step=step,
                    root_seed=args.seed,
                    sdxl=args.sdxl,
                )
                if args.use_ema:
                    unet_ema.restore(unet_trainable_parameters)
                if accelerator.is_main_process:
                    progress_bar.write(f"Step: {step:5d}, {args.score}: {score:.6f}")

        # Wait for main processes to save the state
        accelerator.wait_for_everyone()
    # Destroy process group
    accelerator.end_training()


def log_score(
    accelerator: accelerate.Accelerator,
    writer: SummaryWriter,
    dataset: PromptDataset,
    pipeline: diffusers.DiffusionPipeline,
    encode_prompt_fn: Callable,
    num_inference_steps: int,
    guidance_scale: float,
    batch_size: int,
    num_images_per_prompt: int,
    score_name: str,
    output_dir: str,
    step: int,
    root_seed: int,
    sdxl: bool,
):
    device = accelerator.device

    loader = DataLoader(
        dataset,
        batch_size=max(batch_size // num_images_per_prompt, 1),
        num_workers=4,
    )
    total_prompts = len(dataset)
    num_digits = len(str(total_prompts - 1))

    loader = accelerator.prepare(loader)

    total_images = total_prompts * num_images_per_prompt
    done_images = 0
    with tqdm(
        loader,
        ncols=0,
        leave=False,
        desc=f"Evaluating {score_name} 1/2",
        disable=not accelerator.is_main_process,
    ) as pbar:
        for batch_index, batch in enumerate(pbar):
            prompts = batch['prompt']
            B = len(prompts)

            # Base seed for each prompt
            seeds = torch.arange(
                root_seed + batch_index * (B * accelerator.num_processes) + (B * accelerator.process_index),
                root_seed + batch_index * (B * accelerator.num_processes) + (B * accelerator.process_index) + B,
                device=device)
            # Shift base seeds for images in the same prompt
            seeds = [seeds + i * total_prompts
                     for i in range(num_images_per_prompt)]
            seeds = torch.stack(seeds, dim=1).view(-1)

            # Get prompt embedding manually to supress the warning of long text
            embeds = encode_prompt_fn(prompts)
            prompt_embeds = embeds["prompt_embeds"]
            _, S, D = prompt_embeds.shape
            prompt_embeds = prompt_embeds.unsqueeze(1).expand(B, num_images_per_prompt, S, D)
            prompt_embeds = prompt_embeds.reshape(-1, S, D)

            if sdxl:
                pooled_prompt_embeds = embeds["pooled_prompt_embeds"]
                _, D = pooled_prompt_embeds.shape
                pooled_prompt_embeds = pooled_prompt_embeds.unsqueeze(1).expand(B, num_images_per_prompt, D)
                pooled_prompt_embeds = pooled_prompt_embeds.reshape(-1, D)
            else:
                pooled_prompt_embeds = torch.empty(B * num_images_per_prompt)

            # Split the prompt_embeds and seeds into batches to avoid OOM
            for prompt_embeds_batch, pooled_prompt_embeds_batch, seeds_batch in zip(
                prompt_embeds.split(batch_size),
                pooled_prompt_embeds.split(batch_size),
                seeds.split(batch_size),
            ):
                if sdxl:
                    pipeline_kwargs = {"pooled_prompt_embeds": pooled_prompt_embeds_batch}
                else:
                    pipeline_kwargs = {}
                # Generate the images
                generator = [
                    torch.Generator(device=device).manual_seed(seed.item())
                    for seed in seeds_batch]
                with accelerator.autocast():
                    images_batch = pipeline(
                        prompt_embeds=prompt_embeds_batch,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        output_type='pt',
                        **pipeline_kwargs,
                    ).images
                images_batch = images_batch.float()

                # Save the images
                for seed, image in zip(seeds_batch, images_batch):
                    # The index in the dataset
                    dataset_index = (seed - root_seed) % total_prompts
                    # The n-th images corresponding to the prompt
                    image_index_in_prompt = (seed - root_seed) // total_prompts
                    # Skip padding (DDP sampler duplicates)
                    if image_index_in_prompt >= num_images_per_prompt:
                        continue
                    # The index of the prompt in this batch
                    prompt_index = (dataset_index % (B * accelerator.num_processes)) % B
                    # The directory to save the images
                    dir_path = os.path.join(
                        output_dir, f"{dataset_index:0{num_digits}d}")
                    os.makedirs(dir_path, exist_ok=True)
                    # The path to the image
                    image_path = os.path.join(dir_path, f"{seed.item()}.png")
                    # Save the image
                    save_image(image, image_path)
                    # The path to the prompt file
                    prompt_path = os.path.join(dir_path, "caption.txt")
                    # Save the prompt
                    with open(prompt_path, "w") as f:
                        f.write(prompts[prompt_index])
                        
                    if accelerator.is_main_process:
                        swanlab.log({"image": swanlab.Image(image, caption=prompts[prompt_index])}, step=step)
                        
                done_images = min(
                    done_images + len(seeds_batch) * accelerator.num_processes,
                    total_images)
                pbar.set_postfix_str(f"Generated {done_images}/{total_images} images")

            accelerator.wait_for_everyone()

    compute_score, transform = get_score(score_name, device)

    # Load the dataset
    dataset = ScoreDataset(root=output_dir, transform=transform)
    loader = DataLoader(dataset, batch_size, num_workers=4)

    # accelerator will handle the duplicates of last batch
    loader = accelerator.prepare(loader)

    # Load the scores from the cache
    cache_path = os.path.join(output_dir, f"{score_name}.json")
    path2score = {}

    # Compute the scores for the images
    total_images = len(dataset)
    done_images = 0
    with tqdm(
        loader,
        ncols=0,
        leave=False,
        desc=f"Evaluating {score_name} 2/2",
        disable=not accelerator.is_main_process
    ) as pbar:
        for batch in pbar:
            images = batch['image']
            prompts = batch['prompt']
            paths = batch['path']

            scores = compute_score(images, prompts)
            paths = accelerator.gather_for_metrics(paths, use_gather_object=True)
            scores = accelerator.gather_for_metrics(scores).cpu()

            for path, score in zip(paths, scores):
                path2score[path] = score.item()

            done_images += len(scores)
            pbar.set_postfix_str(f"Processed {done_images}/{total_images} images")

    # Sanity check
    assert len(path2score) == total_images, f"{len(path2score)} != {total_images}"

    average_score = sum(path2score.values()) / len(path2score)
    if accelerator.is_main_process:
        # Log the average score
        writer.add_scalar(f"metrics/{score_name}", average_score, step)
        # Save the scores to the cache file
        with open(cache_path, "w") as f:
            json.dump(path2score, f)

    return average_score


if __name__ == "__main__":
    main()