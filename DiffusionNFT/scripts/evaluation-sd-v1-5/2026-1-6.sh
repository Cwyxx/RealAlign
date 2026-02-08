#!/bin/bash

./script_command_multi_seed.sh 2 sd-v1-5 0 diffusion-dpo partiprompts
./script_command_multi_seed.sh 3 dpo-official 0 diffusion-dpo partiprompts   
# ./script_command_multi_seed.sh 2 spo-official 0 diffusion-dpo pick_a_pic_v2
# ./script_command_multi_seed.sh 2 kto-official 0 diffusion-dpo pick_a_pic_v2
# ./script_command_multi_seed.sh 2 inpo-official 0 diffusion-dpo pick_a_pic_v2
# ./script_command_multi_seed.sh 2 gradspo-official 0 diffusion-dpo pick_a_pic_v2
# ./script_command_multi_seed.sh 2 dpo-official 0 diffusion-dpo pick_a_pic_v2
./script_command_multi_seed.sh 4 irl_top_512_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all_ckpt_1600-dpo_2000_top_512_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all_inpainting 800 diffusion-dpo partiprompts