#!/usr/bin/env python
# coding=utf-8
"""
Generate real-fake image pairs using noise addition and denoising
"""
import inspect
import argparse
import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Generate real-fake image pairs")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing real images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save real-fake pairs",
    )
    parser.add_argument(
        "--add_noise_step",
        type=int,
        default=10,
        help="Number of noise steps to add (default: 10)",
    )
    parser.add_argument(
        "--total_inference_step",
        type=int,
        default=50,
        help="Number of denoising steps (default: 20)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for denoising (default: 7.5)",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="CSV file stores the uid and corresponding prompt.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image resolution",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    return parser.parse_args()

#### Copy from diffusers/pipeline_stable_diffusion.py ####
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    r"""
    Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure. Based on
    Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
    Flawed](https://huggingface.co/papers/2305.08891).

    Args:
        noise_cfg (`torch.Tensor`):
            The predicted noise tensor for the guided diffusion process.
        noise_pred_text (`torch.Tensor`):
            The predicted noise tensor for the text-guided diffusion process.
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            A rescale factor applied to the noise predictions.

    Returns:
        noise_cfg (`torch.Tensor`): The rescaled noise prediction tensor.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

def prepare_extra_step_kwargs(self, generator, eta):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://huggingface.co/papers/2010.02502
    # and should be between [0, 1]

    accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs

#### Copy from diffusers/pipeline_stable_diffusion.py ####


class RealFakePairGenerator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        # Load models
        print("Loading models...")
        self.noise_scheduler = DDIMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="scheduler"
        )
        print(f"noise_scheduler.config.num_train_timesteps: {self.noise_scheduler.config.num_train_timesteps}")
        self.noise_scheduler.set_timesteps(args.total_inference_step, device=self.device) # use 50 timesteps to generate image from a clean noise.
        self.inference_timesteps = self.noise_scheduler.timesteps
        print(f"self.inference_timesteps ({len(self.inference_timesteps)}): {self.inference_timesteps}")
                
        self.pipeline = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.float32).to(self.device)
        self.vae = self.pipeline.vae
        self.unet = self.pipeline.unet
        
        # Set to eval mode
        self.vae.eval()
        self.unet.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        self.guidance_scale = args.guidance_scale
        self.guidance_rescale = 0.0
        self.do_classifier_free_guidance = self.guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None
        self.df = pd.read_csv(args.prompt_file)        
        
    
    def encode_image(self, image):
        """Encode image to latent space"""
        with torch.no_grad():
            latent = self.vae.encode(image).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
        return latent
    
    def decode_latent(self, latent):
        """Decode latent to image"""
        latent = latent / self.vae.config.scaling_factor
        with torch.no_grad():
            image = self.vae.decode(latent).sample
        return image
    
    def add_noise(self, latent, noise_steps):
        """Add noise to latent for specified steps"""
        noise = torch.randn_like(latent)
        timesteps = torch.tensor([noise_steps], device=self.device)
        noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)
        return noisy_latent
    
    def denoise(self, noisy_latent, prompt=None):
        """Denoise the noisy latent"""
        
        # Encode prompt
        with torch.no_grad():
            prompt_embeds, negative_prompt_embeds = self.pipeline.encode_prompt(prompt, self.device, 1, self.do_classifier_free_guidance)
            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        
        timesteps = self.inference_timesteps[self.args.total_inference_step - self.args.add_noise_step:]
        print(f"len timesteps: {len(timesteps)}")
        print(f"timesteps: {timesteps}")
        
        latents = noisy_latent
        
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = prepare_extra_step_kwargs(self.pipeline, generator=None, eta=0.0)
        
        # Denoising loop
        with torch.no_grad():
            for t in timesteps:
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                
                if hasattr(self.noise_scheduler, "scale_model_input"):
                    latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)
                
                # Predict noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=None,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=None,
                    return_dict=False,
                )[0]
                
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://huggingface.co/papers/2305.08891
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)
                
                
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
        
        return latents
    
    def process_image(self, image_path, prompt=None):
        """Process a single image to generate real-fake pair"""
        # Load and preprocess image
        real_image = Image.open(image_path).convert("RGB")
        real_tensor = self.transform(real_image).unsqueeze(0).to(self.device) # [1, 3, 512, 512]
        # print(f"real_tensor.shape: {real_tensor.shape}")
        
        # Encode to latent
        latent = self.encode_image(real_tensor)
        # print(f"encode_image_latent.shape: {latent.shape}")
        
        # Add noise
        noisy_latent = self.add_noise(latent, noise_steps=self.inference_timesteps[self.args.total_inference_step - self.args.add_noise_step])
        
        # Denoise
        fake_latent = self.denoise(noisy_latent, prompt)
        
        # Decode to image
        fake_tensor = self.decode_latent(fake_latent)
        
        # Convert to PIL
        fake_tensor = (fake_tensor / 2 + 0.5).clamp(0, 1)
        fake_image = transforms.ToPILImage()(fake_tensor.squeeze(0).cpu())
        
        real_preprocess = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512)
        ])
        real_image = real_preprocess(real_image)
        return real_image, fake_image
    
    def generate_pairs(self):
        os.makedirs(self.args.output_dir, exist_ok=True)
        real_image_output_dir = os.path.join(self.args.output_dir, "real")
        fake_image_output_dir = os.path.join(self.args.output_dir, "fake")
        os.makedirs(os.path.join(self.args.output_dir, "real"), exist_ok=True)
        os.makedirs(os.path.join(self.args.output_dir, "fake"), exist_ok=True)
        
        image_extensions = {'.jpg', '.jpeg', '.png'}
        image_files = [ f for f in os.listdir(self.args.input_dir) if os.path.splitext(f)[1].lower() in image_extensions ]
        
        print(f"Processing {len(image_files)} images...")
        
        # Set seed for reproducibility
        torch.manual_seed(self.args.seed)
        
        for img_file in tqdm(image_files):
            uid, _ = os.path.splitext(img_file)
            prompt = self.df[self.df["uid"] == uid ].iloc[0]['PROMPT']
                        
            real_image, fake_image = self.process_image(os.path.join(self.args.input_dir, img_file), prompt)
            
            real_image_output_path = os.path.join(real_image_output_dir, f"{uid}.png")
            fake_image_output_path = os.path.join(fake_image_output_dir, f"{uid}.png")
            real_image.save(real_image_output_path)
            fake_image.save(fake_image_output_path)
            


def main():
    args = parse_args()
    generator = RealFakePairGenerator(args)
    generator.generate_pairs()


if __name__ == "__main__":
    main()