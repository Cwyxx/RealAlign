#!/bin/bash

./evaluation_script_command_multi_seed.sh 4 irl-top_512_images_pickscore_002-civitai_top_sfw_images-uids_lr_0.0002_ckpt_3200-dpo_top_512_images_pickscore_002-civitai_top_sfw_images-uids 400 diffusion-dpo drawbench-unique
./evaluation_script_command_multi_seed.sh 4 sd-3-5-medium 0 diffusion-dpo drawbench-unique
./evaluation_script_command_multi_seed.sh 4 FlowGRPO-PickScore 0 diffusion-dpo drawbench-unique