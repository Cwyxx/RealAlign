#!/bin/bash

./evaluation_script_command_multi_seed.sh 1 irl_top_512_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all_lr_0.0002_ckpt_3200-dpo_top_512_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all 450 diffusion-dpo drawbench-unique
./evaluation_script_command_multi_seed.sh 1 FlowGRPO-PickScore-next-irl_top_512_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all_lr_0.0002_ckpt_3200-dpo_top_512_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all 350 diffusion-dpo drawbench-unique
./evaluation_script_command_multi_seed.sh 1 sd-3-5-medium 0 diffusion-dpo drawbench-unique
./evaluation_script_command_multi_seed.sh 1 FlowGRPO-PickScore 0 diffusion-dpo drawbench-unique
./evaluation_script_command_multi_seed.sh 1 GRPO-Guard 0 diffusion-dpo drawbench-unique
./evaluation_script_command_multi_seed.sh 1 DiffusionNFT 0 diffusion-dpo drawbench-unique