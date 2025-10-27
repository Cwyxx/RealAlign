import os
import torch.nn as nn
from qwen_vl_utils import process_vision_info
from transformers import  AutoProcessor, AutoModelForCausalLM
import math
from PIL import Image
import numpy as np
import torch

class ImageDoctor(nn.Module):
    def __init__(self):
        checkpoint = "GYX97/ImageDoctor"
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", trust_remote_code=True)
        
    
    def build_messages(image, task_prompt):
      return [{
          "role": "user",
          "content": [
              {"type": "image", "image": image},
              {
                      "type": "text",
                      "text": f"Given a caption and an image generated based on this caption, please analyze the provided image in detail. Evaluate it on various dimensions including Semantic Alignment (How well the image content corresponds to the caption), Aesthetics (composition, color usage, and overall artistic quality), Plausibility (realism and attention to detail), and Overall Impression (General subjective assessment of the image's quality). For each evaluation dimension, provide a score between 0-1 and provide a concise rationale for the score. Use a chain-of-thought process to detail your reasoning steps, and enclose all potential important areas and detailed reasoning within <think> and </think> tags. The important areas are represented in following format: \” I need to focus on the bounding box area. Proposed regions (xyxy): ..., which is an enumerated list in the exact format:1.[x1,y1,x2,y2];\n2.[x1,y1,x2,y2];\n3.[x1,y1,x2,y2]… Here, x1,y1 is the top-left corner, and x2,y2 is the bottom-right corner. Then, within the <answer> and </answer> tags, summarize your assessment in the following format: \"Semantic Alignment score: ... \nMisalignment Locations: ...\nAesthetic score: ...\nPlausibility score: ... nArtifact Locations: ...\nOverall Impression score: ...\". No additional text is allowed in the answer section.\n\n Your actual evaluation should be based on the quality of the provided image.**\n\nYour task is provided as follows:\nText Caption: [{task_prompt}]"
                  }
          ]
      }]

    def __call__(self, image_path, prompt):
        img = Image.open(image_path).convert("RGB")
        r = math.sqrt(512*512 / (img.width * img.height))
        new_size = (max(1, int(img.width * r)), max(1, int(img.height * r)))
        img = img.resize(new_size, resample=Image.BICUBIC)
        
        messages = self.build_messages(img, prompt)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)    
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        gen_kwargs = dict(
            max_new_tokens=20000,
            use_cache=True,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
        outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode assistant output (strip prompt tokens)
        generated_ids = outputs.sequences
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        has_tokens = all(hasattr(self.model.config, a) for a in ["image_token_id", "misalignment_token_id", "artifact_token_id"])
        has_heads = all(hasattr(self.model, a) for a in ["text_hidden_fcs", "image_hidden_fcs", "prompt_encoder", "heatmap", "sigmoid"])
        
        if has_tokens and has_heads and outputs.hidden_states is not None:
            true_generated = generated_ids[:, inputs.input_ids.shape[1]:]

            # Find special tokens
            misalignment_mask = (true_generated[:, 1:] == self.model.config.misalignment_token_id)
            artifact_mask     = (true_generated[:, 1:] == self.model.config.artifact_token_id)

            if misalignment_mask.any() or artifact_mask.any():
                # Gather final-layer hidden states across decoding steps
                step_states = []
                for step in outputs.hidden_states[1:]:  # skip encoder states at index 0
                    step_states.append(step[-1])        # last layer [B, 1, H]
                all_gen_h = torch.cat(step_states, dim=1)  # [B, T, H]

                # Map text hidden → special token embeddings
                last_hidden_state = self.model.text_hidden_fcs[0](all_gen_h)  # [B, T, H’]
                # Index by masks (flatten batch/time)
                mis_tokens = last_hidden_state[misalignment_mask].unsqueeze(1) if misalignment_mask.any() else None
                art_tokens = last_hidden_state[artifact_mask].unsqueeze(1) if artifact_mask.any() else None

                # Visual embeddings (grid features)
                image_embeds = self.model.visual(
                    inputs["pixel_values"].to(self.model.device),
                    grid_thw=inputs["image_grid_thw"].to(self.model.device)
                )
                img_hidden = self.model.image_hidden_fcs[0](image_embeds.unsqueeze(0))  # [1, L, C]
                # reshape to low-res feature map (18x18 matches your original; change if your head differs)
                img_hidden = img_hidden.transpose(1, 2).view(1, -1, 18, 18)

                def run_heatmap(text_tokens):
                    sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                        points=None, boxes=None, masks=None, text_embeds=text_tokens
                    )
                    low_res = self.model.heatmap(
                        image_embeddings=img_hidden,
                        image_pe=self.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings.to(img_hidden.dtype),
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False
                    )
                    return self.model.sigmoid(low_res)  # [N, 1, H, W]

                artifact_np_path = None
                misalign_np_path = None

                if self.output_dir:
                    os.makedirs( self.output_dir, exist_ok=True)

                if mis_tokens is not None and art_tokens is not None:
                    fused = torch.cat([mis_tokens, art_tokens], dim=0)
                    pred = run_heatmap(fused)
                    mis_pred = pred[0:1, 0]  # [1,H,W] pick first
                    art_pred = pred[1:2, 0]  # [1,H,W] pick second

                    mis_np = mis_pred[0].detach().cpu().float().numpy()
                    art_np = art_pred[0].detach().cpu().float().numpy()

                    if  self.output_dir:
                        misalign_np_path = os.path.join( self.output_dir, f"misalignment.npy")
                        artifact_np_path = os.path.join( self.output_dir, f"artifact.npy")
                        np.save(misalign_np_path, mis_np)
                        np.save(artifact_np_path, art_np)

                elif art_tokens is not None:
                    pred = run_heatmap(art_tokens[:1])
                    art_np = pred[0, 0].detach().cpu().float().numpy()
                    if  self.output_dir:
                        artifact_np_path = os.path.join( self.output_dir, f"artifact.npy")
                        np.save(artifact_np_path, art_np)

                elif mis_tokens is not None:
                    pred = run_heatmap(mis_tokens[:1])
                    mis_np = pred[0, 0].detach().cpu().float().numpy()
                    if  self.output_dir:
                        misalign_np_path = os.path.join( self.output_dir, f"misalignment.npy")
                        np.save(misalign_np_path, mis_np)