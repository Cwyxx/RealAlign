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
import pandas as pd

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
import random

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
    dataset_path = f"/data3/chenweiyan/notebook/fine-tune-diffusion/spo_gitee/DiffusionNFT/dataset/{args.dataset}"
    if args.dataset == "geneval":
        dataset = GenevalPromptDataset(dataset_path, split="test")
        
    elif args.dataset in [ "drawbench", "pick_a_pic_spo", "pickscore-analysis" ]: # Handles ocr, pickscore, drawbench
        dataset = TextPromptDataset(dataset_path, split="test")
        
    elif args.dataset == "drawbench-analysis":
        dataset_path = f"/data3/chenweiyan/notebook/fine-tune-diffusion/spo_gitee/DiffusionNFT/dataset/drawbench"
        dataset = TextPromptDataset(dataset_path, split="test")
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=False,
    )
    
    # --- 3. Load existing results to update them ---
    prediction_logs = []
    mini_group_size = 4
    group_size = 24

    # --- 4. Evaluation Loop ---
    for batch in tqdm(dataloader, desc="Evaluating with VILA"):
        prompts, metadata, indices = batch
        
        original_prompts, original_metadata, indices = batch
        current_batch_size = len(original_prompts)
        
        for group_idx in range(group_size // mini_group_size):
            start_idx = group_idx*mini_group_size
            end_idx = (group_idx+1)*mini_group_size
            prompts = [ original_prompts[sample_idx] for sample_idx in range(current_batch_size) for tmp_group_idx in range(start_idx, end_idx) ]
            metadata = [ original_metadata[sample_idx] for sample_idx in range(current_batch_size) for tmp_group_idx in range(start_idx, end_idx) ]
            image_paths = [ os.path.join(args.output_dir, "images", f"{indices[sample_idx]:05d}_{tmp_group_idx}.png") for sample_idx in range(current_batch_size) for tmp_group_idx in range(start_idx, end_idx) ]

            assert len(prompts) == len(image_paths)
        
            # Calculate scores using the wrapped VILA scoring function
            all_scores, _ = scoring_fn(image_paths, prompts, metadata)

            for sample_idx in range(current_batch_size):
                for _ in range(mini_group_size):
                    tmp_idx = sample_idx * mini_group_size + _
                    result_item = {
                        "prompt": prompts[tmp_idx],
                        "image_name": os.path.splitext(os.path.basename(image_paths[tmp_idx]))[0],
                        args.reward_model: all_scores[args.reward_model][tmp_idx] if not isinstance(all_scores[args.reward_model][tmp_idx], torch.Tensor) else all_scores[args.reward_model][tmp_idx].detach().cpu().item()
                    }
                    prediction_logs.append(result_item)
                    
    pd.DataFrame(prediction_logs).to_csv(
        os.path.join("/data_center/data2/dataset/chenwy/21164-data/diffusionnft/generate_images/sd3_textencoder_3_none_cfg_1.0/pickscore-analysis/SD3.5M-DiffusionNFT-MultiReward/ckpt-0/reward_score", f"{args.reward_model}.csv"),
        index=False)

if __name__ == '__main__':
    # ### MODIFIED: Replaced absl.flags with argparse ###
    parser = argparse.ArgumentParser(description="Evaluate pre-generated images using the VILA model.")
    parser.add_argument(
        "--reward_model",
        type=str,
        default="vila_score",
        help="Reward Model.",
    )
    parser.add_argument(
        "--ckpt_dir", type=str, required=True, help="Path to the VILA model checkpoint directory."
    )
    parser.add_argument(
        "--spm_model_path", type=str, required=True, help="Path to sentence piece tokenizer model."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["geneval", "ocr", "pickscore", "drawbench", "pick_a_pic_spo", "drawbench-analysis", "pickscore-analysis"], help="Dataset used to find prompts and image names."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./evaluation_output", help="Directory where 'images/' and 'evaluation_results.jsonl' are located."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for VILA model inference."
    )
    
    args = parser.parse_args()
    main(args)