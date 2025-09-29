# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example script for running VILA model."""
import argparse

from absl import app
from absl import flags
from absl import logging
import torch
import jax
import jax.numpy as jnp
from lingvo import compat as tf
from lingvo.core import tokenizers as lingvo_tokenizers
from paxml import checkpoints
from paxml import learners
from paxml import tasks_lib
from paxml import train_states
from praxis import base_layer
from praxis import optimizers
from praxis import pax_fiddle
from praxis import py_utils
from praxis import schedules

import os
from tqdm import tqdm
from collections import defaultdict
import numpy as np

from vila import coca_vila
from vila import coca_vila_configs


NestedMap = py_utils.NestedMap

# _CKPT_DIR = flags.DEFINE_string('ckpt_dir', '', 'Path to checkpoint.')
# _SPM_MODEL_PATH = flags.DEFINE_string(
#     'spm_model_path', '', 'Path to sentence piece tokenizer model.'
# )
# _IMAGE_DIR = flags.DEFINE_string('image_dir', '', 'Path to input image dir.')

_PRE_CROP_SIZE = 272
_IMAGE_SIZE = 224
_MAX_TEXT_LEN = 64
_TEXT_VOCAB_SIZE = 64000

# ### MODIFIED: Copied Dataset classes from the second script ###
import torch
from torch.utils.data import DataLoader, Dataset
import json

class TextPromptDataset(Dataset):
    def __init__(self, dataset_path, split="test"):
        self.file_path = os.path.join(dataset_path, f"{split}.txt")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset file not found at {self.file_path}")
        with open(self.file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}, "original_index": idx}

class GenevalPromptDataset(Dataset):
    def __init__(self, dataset_path, split="test"):
        self.file_path = os.path.join(dataset_path, f"{split}_metadata.jsonl")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset file not found at {self.file_path}")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx], "original_index": idx}

def collate_fn(examples):
    prompts = [example["prompt"] for example in examples]
    metadatas = [example["metadata"] for example in examples]
    indices = [example["original_index"] for example in examples]
    return prompts, metadatas, indices
# ### END MODIFIED SECTION ###

_ZSL_QUALITY_PROMPTS = [
    ['good image', 'bad image'],
    ['good lighting', 'bad lighting'],
    ['good content', 'bad content'],
    ['good background', 'bad background'],
    ['good foreground', 'bad foreground'],
    ['good composition', 'bad composition'],
]


def load_vila_model(
    ckpt_dir,
):
  """Loads the VILA model from checkpoint directory.

  Args:
    ckpt_dir: The path to checkpoint directory

  Returns:
    VILA model, VILA model states
  """
  coca_config = coca_vila_configs.CocaVilaConfig()
  coca_config.model_type = coca_vila.CoCaVilaRankBasedFinetune
  coca_config.decoding_max_len = _MAX_TEXT_LEN
  coca_config.text_vocab_size = _TEXT_VOCAB_SIZE
  model_p = coca_vila_configs.build_coca_vila_model(coca_config)
  model_p.model_dims = coca_config.model_dims
  model = model_p.Instantiate()

  dummy_batch_size = 4  # For initialization only
  text_shape = (dummy_batch_size, 1, _MAX_TEXT_LEN)
  image_shape = (dummy_batch_size, _IMAGE_SIZE, _IMAGE_SIZE, 3)
  input_specs = NestedMap(
      ids=jax.ShapeDtypeStruct(shape=text_shape, dtype=jnp.int32),
      image=jax.ShapeDtypeStruct(shape=image_shape, dtype=jnp.float32),
      paddings=jax.ShapeDtypeStruct(shape=text_shape, dtype=jnp.float32),
      # For initialization only
      labels=jax.ShapeDtypeStruct(shape=text_shape, dtype=jnp.float32),
      regression_labels=jax.ShapeDtypeStruct(
          shape=(dummy_batch_size, 10), dtype=jnp.float32
      ),
  )
  prng_key = jax.random.PRNGKey(123)
  prng_key, _ = jax.random.split(prng_key)
  vars_weight_params = model.abstract_init_with_metadata(input_specs)

  # `learner` is only used for initialization.
  learner_p = pax_fiddle.Config(learners.Learner)
  learner_p.name = 'learner'
  learner_p.optimizer = pax_fiddle.Config(
      optimizers.ShardedAdafactor,
      decay_method='adam',
      lr_schedule=pax_fiddle.Config(schedules.Constant),
  )
  learner = learner_p.Instantiate()

  train_state_global_shapes = tasks_lib.create_state_unpadded_shapes(
      vars_weight_params, discard_opt_states=False, learners=[learner]
  )
  model_states = checkpoints.restore_checkpoint(
      train_state_global_shapes, ckpt_dir
  )
  return model, model_states


def preprocess_image(
    image_path, pre_crop_size, image_size
):
  """Image preprocessing."""
  with tf.compat.v1.gfile.FastGFile(image_path, 'rb') as f:
    image_bytes = f.read()
  image = tf.io.decode_image(image_bytes, 3, expand_animations=False)
  image = tf.image.resize_bilinear(
      tf.expand_dims(image, 0), [pre_crop_size, pre_crop_size]
  )
  image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)
  image = tf.cast(image, tf.float32)
  image = image / 255.0
  image = tf.clip_by_value(image, 0.0, 1.0)
  return image.numpy()


def main(_):
    logging.set_verbosity(logging.ERROR)
    model, model_states = load_vila_model(args.ckpt_dir)
    model_params = model_states.mdl_vars['params']
    print("JAX detected devices:", jax.devices())
    
    # --- 1. Define a JIT-compiled prediction function ---
    # We use partial to "bake in" the model and its parameters.
    # The function will only take the input_batch as its argument.
    @jax.jit
    def predict_step(params, batch):
        return model.apply(
            {'params': params},
            batch,
            method=model.compute_predictions,
        )

    # ### MODIFIED: Create a scoring_fn wrapper for VILA ###
    def scoring_fn(image_paths, prompts, metadata):
        """
        A wrapper function to make the VILA model conform to the scoring_fn
        interface from the second script.
        """
        # Preprocess all images in the batch
        batch_images = jnp.concatenate([
            preprocess_image(f, _PRE_CROP_SIZE, _IMAGE_SIZE) for f in image_paths
        ])
        current_batch_size = batch_images.shape[0]

        # Create the input batch for the VILA model
        input_batch = NestedMap(
            image=batch_images,
            ids=jnp.zeros((current_batch_size, 1, _MAX_TEXT_LEN), dtype=jnp.int32),
            paddings=jnp.zeros((current_batch_size, 1, _MAX_TEXT_LEN), dtype=jnp.int32),
        )

        # Run prediction
        predictions = predict_step(model_params, input_batch)
        
        # Extract scores and format them as required
        quality_scores = predictions['quality_scores'][:, 0].tolist()
        score_details = {"vila_score": quality_scores}
        
        return score_details, {}
    
    # ### MODIFIED: Logic ported from calculate_score.py ###
    # --- 2. Setup Dataset and DataLoader ---
    print(f"Loading dataset: {args.dataset}")
    dataset_path = f"dataset/{args.dataset}"
    if args.dataset == "geneval":
        dataset = GenevalPromptDataset(dataset_path, split="test")
    else: # Handles ocr, pickscore, drawbench
        dataset = TextPromptDataset(dataset_path, split="test")
    
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        collate_fn=collate_fn,
        shuffle=False,
    )
    
    # --- 3. Load existing results to update them ---
    results_filepath = os.path.join(args.output_dir, "evaluation_results.jsonl")
    result_this_rank = []
    if os.path.exists(results_filepath):
        with open(results_filepath, 'r') as f:
            for line in f:
                if line.strip(): result_this_rank.append(json.loads(line))
        result_this_rank.sort(key=lambda x: x["sample_id"])
        print(f"Loaded {len(result_this_rank)} existing results from {results_filepath}")
    else:
        print("No existing results file found. This script will only work if another script has created it.")
        return

    # --- 4. Evaluation Loop ---
    for batch in tqdm(dataloader, desc="Evaluating with VILA"):
        prompts, metadata, indices = batch
        
        # Construct image paths based on the output directory and sample index
        image_paths = [os.path.join(args.output_dir, "images", f"{sample_idx:05d}.png") for sample_idx in indices]
        
        # Calculate scores using the wrapped VILA scoring function
        all_scores, _ = scoring_fn(image_paths, prompts, metadata)

        # Update the results list with the new scores
        for i, sample_idx in enumerate(indices):
            # Find the corresponding result item to update
            # This assumes the result_this_rank is sorted by sample_id
            result_item = result_this_rank[sample_idx]
            assert result_item["sample_id"] == sample_idx, f"Mismatched sample_id at index {sample_idx}"

            for score_name, score_values in all_scores.items():
                result_item["scores"][score_name] = float(score_values[i])
    
    # --- 5. Save updated results ---
    with open(results_filepath, "w") as f_out:
        for result_item in result_this_rank:
            f_out.write(json.dumps(result_item) + "\n")
    print(f"\nEvaluation finished. All {len(result_this_rank)} results updated in {results_filepath}")

    # --- 6. Calculate and display average scores ---
    all_scores_agg = defaultdict(list)
    for result in result_this_rank:
        for score_name, score_value in result.get("scores", {}).items():
            if isinstance(score_value, (int, float)):
                all_scores_agg[score_name].append(score_value)

    average_scores = {
        name: np.mean(scores) for name, scores in all_scores_agg.items() if scores
    }
    
    print("\n--- Average Scores ---")
    if not average_scores:
        print("No scores were found to average.")
    else:
        for name, avg_score in sorted(average_scores.items()):
            print(f"{name:<20}: {avg_score:.10f}")
    print("----------------------")

    avg_scores_filepath = os.path.join(args.output_dir, "average_scores.json")
    with open(avg_scores_filepath, "w") as f_avg:
        json.dump(average_scores, f_avg, indent=4)
    print(f"Average scores also saved to {avg_scores_filepath}")



if __name__ == '__main__':
    # ### MODIFIED: Replaced absl.flags with argparse ###
    parser = argparse.ArgumentParser(description="Evaluate pre-generated images using the VILA model.")
    parser.add_argument(
        "--ckpt_dir", type=str, required=True, help="Path to the VILA model checkpoint directory."
    )
    parser.add_argument(
        "--spm_model_path", type=str, required=True, help="Path to sentence piece tokenizer model."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["geneval", "ocr", "pickscore", "drawbench"], help="Dataset used to find prompts and image names."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./evaluation_output", help="Directory where 'images/' and 'evaluation_results.jsonl' are located."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for VILA model inference."
    )
    
    args = parser.parse_args()
    main(args)