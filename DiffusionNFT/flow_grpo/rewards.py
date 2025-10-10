from PIL import Image
import io
import numpy as np
import torch
import torchvision
from collections import defaultdict


def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew / 500, meta

    return _fn


def aesthetic_score(device):
    from flow_grpo.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn, scorer


def clip_score(device):
    from flow_grpo.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device)

    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8) / 255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn, scorer


def hpsv2_score(device):
    from flow_grpo.hpsv2_scorer import HPSv2Scorer

    scorer = HPSv2Scorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8) / 255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn, scorer


def pickscore_score(device):
    from flow_grpo.pickscore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn, scorer


def imagereward_score(device):
    from flow_grpo.imagereward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn, scorer


def geneval_score(device):
    from flow_grpo.gen_eval import load_geneval

    batch_size = 64
    compute_geneval = load_geneval(device)

    def _fn(images, prompts, metadatas, only_strict):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadatas_batched = np.array_split(metadatas, np.ceil(len(metadatas) / batch_size))
        all_scores = []
        all_rewards = []
        all_strict_rewards = []
        all_group_strict_rewards = []
        all_group_rewards = []
        for image_batch, metadata_batched in zip(images_batched, metadatas_batched):
            pil_images = [Image.fromarray(image) for image in image_batch]

            data = {
                "images": pil_images,
                "metadatas": list(metadata_batched),
                "only_strict": only_strict,
            }
            scores, rewards, strict_rewards, group_rewards, group_strict_rewards = compute_geneval(**data)

            all_scores += scores
            all_rewards += rewards
            all_strict_rewards += strict_rewards
            all_group_strict_rewards.append(group_strict_rewards)
            all_group_rewards.append(group_rewards)
        all_group_strict_rewards_dict = defaultdict(list)
        all_group_rewards_dict = defaultdict(list)
        for current_dict in all_group_strict_rewards:
            for key, value in current_dict.items():
                all_group_strict_rewards_dict[key].extend(value)
        all_group_strict_rewards_dict = dict(all_group_strict_rewards_dict)

        for current_dict in all_group_rewards:
            for key, value in current_dict.items():
                all_group_rewards_dict[key].extend(value)
        all_group_rewards_dict = dict(all_group_rewards_dict)

        return all_scores, all_rewards, all_strict_rewards, all_group_rewards_dict, all_group_strict_rewards_dict

    return _fn


def ocr_score(device):
    from flow_grpo.ocr import OcrScorer

    scorer = OcrScorer()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn


def unifiedreward_score_sglang(device):
    import asyncio
    from openai import AsyncOpenAI
    import base64
    from io import BytesIO
    import re

    def pil_image_to_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        return base64_qwen

    def _extract_scores(text_outputs):
        scores = []
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        for text in text_outputs:
            match = re.search(pattern, text)
            if match:
                try:
                    scores.append(float(match.group(1)))
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores

    client = AsyncOpenAI(base_url="http://127.0.0.1:17140/v1", api_key="flowgrpo")

    async def evaluate_image(prompt, image):
        question = f"<image>\nYou are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nBased on the above criteria, assign a score from 1 to 5 after 'Final Score:'.\nYour task is provided as follows:\nText Caption: [{prompt}]"
        images_base64 = pil_image_to_base64(image)
        response = await client.chat.completions.create(
            model="UnifiedReward-7b-v1.5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": images_base64},
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    async def evaluate_batch_image(images, prompts):
        tasks = [evaluate_image(prompt, img) for prompt, img in zip(prompts, images)]
        results = await asyncio.gather(*tasks)
        return results

    def _fn(images, prompts, metadata):
        # 处理Tensor类型转换
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        # 转换为PIL Image并调整尺寸
        images = [Image.fromarray(image).resize((512, 512)) for image in images]

        # 执行异步批量评估
        text_outputs = asyncio.run(evaluate_batch_image(images, prompts))
        score = _extract_scores(text_outputs)
        score = [sc / 5.0 for sc in score]
        return score, {}

    return _fn

def unifiedreward_score(device):
    import re
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    # print(f"Loading from /data3/chenweiyan/model-ckpt/UnifiedReward-7b-v1.5")
    # model_path = "/data3/chenweiyan/model-ckpt/UnifiedReward-7b-v1.5"
    model_path="CodeGoat24/UnifiedReward-qwen-7b"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map={"": 'cuda:0'})
    processor = AutoProcessor.from_pretrained(model_path)
    print(f"Loading Done!")
    
    def _extract_scores(text_outputs):
        scores = []
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        for text in text_outputs:
            match = re.search(pattern, text)
            if match:
                try:
                    scores.append(float(match.group(1)))
                except ValueError:
                    scores.append(0.0)
            else:
                print(f"error: dont match")
                scores.append(0.0)
        return scores
    
    def _fn(images, prompts, metadata):
        # 处理Tensor类型转换
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image).resize((512, 512)) for image in images]
            
        text_outputs = []
        for image, prompt in zip(images, prompt):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": f'You are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nExtract key elements from the provided text caption, evaluate their presence in the generated image using the format: \'element (type): value\' (where value=0 means not generated, and value=1 means generated), and assign a score from 1 to 5 after \'Final Score:\'.\nYour task is provided as follows:\nText Caption: [{prompt}]'
                        },
                    ],
                }
            ]    
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [ out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids) ]
            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print(f'prompt: You are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nExtract key elements from the provided text caption, evaluate their presence in the generated image using the format: \'element (type): value\' (where value=0 means not generated, and value=1 means generated), and assign a score from 1 to 5 after \'Final Score:\'.\nYour task is provided as follows:\nText Caption: [{prompt}]')
            print(f"output_text: {output_text}")
            exit(0)
            text_outputs.append(output_text)
        scores = _extract_scores(text_outputs)
        return scores, {}

    return _fn, model
            

def code(device):
    from flow_grpo.code_models.models import VITContrastiveHF as CoDE_Model
    classification_type = "linear"
    aigi_detector_path = "/data_center/data2/dataset/chenwy/21164-data/detection-method-ckpt/CoDE"
    aigi_detector = CoDE_Model(classification_type, aigi_detector_path)
    aigi_detector = aigi_detector.to(dtype=torch.float32, device=device)
    aigi_detector.eval()
    
    _transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(224),               # Center crop to 224x224
        torchvision.transforms.ToTensor(),                     # Convert PIL image to tensor
        torchvision.transforms.Normalize(                      # Normalize with mean and std
            mean=[0.485, 0.456, 0.406],            # ImageNet mean values
            std=[0.229, 0.224, 0.225]              # ImageNet std values
        )
    ])
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        
        transformed_images = [_transform(image) for image in images]
        image_tensor = torch.stack(transformed_images).to(device)
        with torch.no_grad():
            logits = aigi_detector(image_tensor)
            outputs = logits[:, 1].reshape(-1, 1)
            scores = 1 - outputs
            scores = scores.squeeze()
            
        return scores, {}
    
    return _fn, aigi_detector

def b_free(device):
    import os, yaml
    from flow_grpo.b_free_models.networks import get_network, load_weights

    def get_config(model_name, weights_dir='/data_center/data2/dataset/chenwy/21164-data/model-ckpt/B-Free'):
        with open(os.path.join(weights_dir, model_name, 'config.yaml')) as fid:
            data = yaml.load(fid, Loader=yaml.FullLoader)
        model_path = os.path.join(weights_dir, model_name, data['weights_file'])
        return data['model_name'], model_path, data['arch'], data['norm_type']
    
    # defining model and transforms
    _, model_path, arch, norm_type = get_config("BFREE_dino2reg4")
    model = load_weights(get_network(arch), model_path)
    model = model.to(device).eval()
    
    if norm_type == "resnet":
        _transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),                     # Convert PIL image to tensor
                torchvision.transforms.Normalize(                      # Normalize with mean and std
                    mean=[0.485, 0.456, 0.406],            # ImageNet mean values
                    std=[0.229, 0.224, 0.225]              # ImageNet std values
                )
        ])
    
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        
        transformed_images = [_transform(image) for image in images]
        image_tensor = torch.stack(transformed_images).to(device)
        with torch.no_grad():
            outputs = torch.sigmoid(model(image_tensor))
            outputs = outputs.reshape(-1, 1)
            scores = 1 - outputs
            scores = scores.squeeze()
        return scores, {}
    
    return _fn, model


def multi_score(device, score_dict):
    score_functions = {
        "ocr": ocr_score,
        "imagereward": imagereward_score,
        "pickscore": pickscore_score,
        "aesthetic": aesthetic_score,
        "jpeg_compressibility": jpeg_compressibility,
        "unifiedreward": unifiedreward_score,
        "geneval": geneval_score,
        "clipscore": clip_score,
        "hpsv2": hpsv2_score,
        "code": code,
        "b_free": b_free
    }
    score_fns, score_models = {}, {}
    for score_name, weight in score_dict.items():
        score_fn, score_model = score_functions[score_name](device) if "device" in score_functions[score_name].__code__.co_varnames else score_functions[score_name]()
        score_fns[score_name] = score_fn
        score_models[score_name] = score_model
        score_model.to(torch.device("cpu"))
        
    offload_reward_model_list = ["pickscore", "clipscore", "hpsv2", "code", "b_free"]
    # only_strict is only for geneval. During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(images, prompts, metadata, only_strict=True, offload_to_cpu=False):
        total_scores = []
        score_details = {}
                
        for score_name, weight in score_dict.items():
            if score_name in offload_reward_model_list and offload_to_cpu:
                score_models[score_name].to(device)
                
            if score_name == "geneval":
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = score_fns[score_name](
                    images, prompts, metadata, only_strict
                )
                score_details["accuracy"] = rewards
                score_details["strict_accuracy"] = strict_rewards
                for key, value in group_strict_rewards.items():
                    score_details[f"{key}_strict_accuracy"] = value
                for key, value in group_rewards.items():
                    score_details[f"{key}_accuracy"] = value
            else:
                scores, rewards = score_fns[score_name](images, prompts, metadata)
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]

            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]
                
            ### offLoad_to_cpu ###
            if score_name in offload_reward_model_list and offload_to_cpu:
                score_models[score_name].to(torch.device("cpu"))

        score_details["avg"] = total_scores
        return score_details, {}

    return _fn, score_models


def main():
    import torchvision.transforms as transforms

    image_paths = [
        "test_cases/nasa.jpg",
    ]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert to tensor
        ]
    )

    images = torch.stack([transform(Image.open(image_path).convert("RGB")) for image_path in image_paths])
    prompts = [
        'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    metadata = {}  # Example metadata
    score_dict = {"unifiedreward": 1.0}
    # Initialize the multi_score function with a device and score_dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    # Get the scores
    scores, _ = scoring_fn(images, prompts, metadata)
    # Print the scores
    print("Scores:", scores)


if __name__ == "__main__":
    main()
