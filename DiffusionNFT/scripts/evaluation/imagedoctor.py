import os
import torch.nn as nn
from qwen_vl_utils import process_vision_info
from transformers import  AutoProcessor, AutoModelForCausalLM
import math
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import cv2

class ImageDoctor(nn.Module):
    def __init__(self, image_dir):
        super().__init__()
        checkpoint = "GYX97/ImageDoctor"
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", trust_remote_code=True)
        method = os.path.basename(os.path.dirname(image_dir))
        dataset = os.path.basename(os.path.dirname(os.path.dirname(image_dir)))
        self.output_dir = f"/data_center/data2/dataset/chenwy/21164-data/Artifact_Segmentor_Tmp/visualization/ImageDoctor/{dataset}/{method}"
        os.makedirs( self.output_dir, exist_ok=True)
    
    def build_messages(self, image, task_prompt):
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
      
    def overlay_heatmap(self, img, heatmap, output_path, alpha=0.4):
        img_np = np.array(img)
        
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap_colored, alpha, 0)
        
        Image.fromarray(overlay).save(output_path)

    def __call__(self, image_path, prompt):
        uid, _ = os.path.splitext((os.path.basename(image_path)))
        img = Image.open(image_path).convert("RGB")
        original_img = img.copy()
        r = math.sqrt(512*512 / (img.width * img.height))
        new_size = (max(1, int(img.width * r)), max(1, int(img.height * r)))
        img = img.resize(new_size, resample=Image.BICUBIC)
        # print(f"image_size: {img.size}")
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

                if art_tokens is not None:
                    pred = run_heatmap(art_tokens[:1])
                    art_pred = pred[0, 0]  # [H, W]
                    art_pred = art_pred.unsqueeze(0).unsqueeze(0)
                    target_size = (original_img.height, original_img.width)
                    art_pred = F.interpolate(
                        art_pred, 
                        size=target_size, 
                        mode='bilinear', 
                        align_corners=False
                    )
                    art_np = art_pred[0, 0].detach().cpu().float().numpy()
                    
                    overlay_path = os.path.join(self.output_dir, f"{uid}.png")
                    self.overlay_heatmap(original_img, art_np, overlay_path, alpha=0.4)
                    
                    # artifact_np_path = os.path.join(self.output_dir, f"{uid}.npy")
                    # np.save(artifact_np_path, art_np)

        art_problem_pixels = np.sum(art_np >= 0.5)
        total_pixel_nums = art_np.size
        perceptual_artifact_ratio = art_problem_pixels / total_pixel_nums
        return perceptual_artifact_ratio