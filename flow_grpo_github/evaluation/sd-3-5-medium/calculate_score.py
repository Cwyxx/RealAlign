"""Score images previously rendered by generate_image.py.

Six metrics supported: PickScore, ImageReward, Aesthetic, HPSv3, DeQA,
UnifiedReward. All routed through `flow_grpo.rewards.multi_score`.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

# Make `flow_grpo` importable: this script lives in
# flow_grpo_github/evaluation/sd-3-5-medium/, package is two levels up.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from flow_grpo.rewards import multi_score


SUPPORTED_METRICS = {
    "pickscore", "imagereward", "aesthetic",
    "hpsv3", "deqa", "unifiedreward",
}
SUPPORTED_DATASETS = {"pick_a_pic_v2", "partiprompts", "drawbench"}


class TextPromptDataset(Dataset):
    def __init__(self, dataset_path, split="test"):
        file_path = os.path.join(dataset_path, f"{split}.txt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found at {file_path}")
        with open(file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}, "original_index": idx}


def collate_fn(examples):
    prompts = [ex["prompt"] for ex in examples]
    metadatas = [ex["metadata"] for ex in examples]
    indices = [ex["original_index"] for ex in examples]
    return prompts, metadatas, indices


def prepare_input(reward_model, image_paths):
    """Translate file paths into the input format each scorer expects."""
    if reward_model in {"pickscore", "imagereward", "unifiedreward"}:
        return [Image.open(p).convert("RGB") for p in image_paths]
    if reward_model == "aesthetic":
        return np.stack([np.array(Image.open(p).convert("RGB")) for p in image_paths])
    if reward_model in {"hpsv3", "deqa"}:
        return image_paths
    raise ValueError(f"Unsupported reward model: {reward_model}")


def main(args):
    device = torch.device("cuda")
    base_image_dir = os.path.join(args.output_dir, "images")
    results_filepath = os.path.join(args.output_dir, "evaluation_results.jsonl")

    dataset_path = f"../dataset/{args.dataset}"
    print(f"Loading dataset from: {dataset_path}")
    dataset = TextPromptDataset(dataset_path, split="test")

    eval_batch_size = 1 if args.reward_model in {"hpsv3", "unifiedreward"} else 2
    dataloader = DataLoader(
        dataset, batch_size=eval_batch_size, collate_fn=collate_fn, shuffle=False,
    )

    print(f"Initializing reward model: {args.reward_model}")
    scoring_fn = multi_score(device, {args.reward_model: 1.0})

    result_this_rank = []
    with open(results_filepath, "r") as f:
        for line in f:
            if line.strip():
                result_this_rank.append(json.loads(line))
    result_this_rank.sort(key=lambda x: x["sample_id"])

    for batch in tqdm(dataloader, desc="Evaluating"):
        prompts, metadata, indices = batch
        image_paths = [
            os.path.join(base_image_dir, f"{idx:05d}.png") for idx in indices
        ]
        images = prepare_input(args.reward_model, image_paths)
        all_scores, _ = scoring_fn(images, prompts, metadata)

        for i, idx in enumerate(indices):
            result_item = result_this_rank[idx]
            assert result_item["sample_id"] == idx, (
                f'result_item[sample_id]: {result_item["sample_id"]} / sample_idx: {idx}'
            )
            for score_name, score_values in all_scores.items():
                if isinstance(score_values, torch.Tensor):
                    result_item["scores"][score_name] = score_values[i].detach().cpu().item()
                else:
                    result_item["scores"][score_name] = float(score_values[i])

        torch.cuda.empty_cache()

    result_this_rank.sort(key=lambda x: x["sample_id"])
    with open(results_filepath, "w") as f_out:
        for result_item in result_this_rank:
            f_out.write(json.dumps(result_item) + "\n")
    print(f"Evaluation finished. {len(result_this_rank)} results saved to {results_filepath}")

    all_scores_agg = defaultdict(list)
    for result in result_this_rank:
        for score_name, score_value in result["scores"].items():
            if isinstance(score_value, (int, float)):
                all_scores_agg[score_name].append(score_value)

    average_scores = {
        name: float(np.mean([s for s in scores if s != -10.0]))
        for name, scores in all_scores_agg.items()
    }

    print("\n--- Average Scores ---")
    for name, avg in sorted(average_scores.items()):
        print(f"{name:<20}: {avg:.10f}")
    print("----------------------")

    avg_scores_filepath = os.path.join(args.output_dir, "average_scores.json")
    with open(avg_scores_filepath, "w") as f_avg:
        json.dump(average_scores, f_avg, indent=4)
    print(f"Average scores saved to {avg_scores_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score generated images against one of six supported reward models."
    )
    parser.add_argument(
        "--reward_model", type=str, required=True, choices=sorted(SUPPORTED_METRICS),
    )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=sorted(SUPPORTED_DATASETS),
    )
    parser.add_argument(
        "--output_dir", type=str, default="./evaluation_output",
        help="Directory containing 'images/' and 'evaluation_results.jsonl' from generate_image.py.",
    )
    args = parser.parse_args()
    main(args)
