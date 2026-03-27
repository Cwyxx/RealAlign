# #!/bin/bash

# ./evaluation_script_command_multi_seed.sh 1 irl_top_512_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all_lr_0.0002_ckpt_3200-dpo_top_512_images_no_anime_colorfulness_pickscore_0.02-hpdv3_all 450 diffusion-dpo partiprompts
# ./evaluation_script_command_multi_seed.sh 1 sd-3-5-medium 0 diffusion-dpo partiprompts
# ./evaluation_script_command_multi_seed.sh 1 FlowGRPO-PickScore 0 diffusion-dpo partiprompts
# ./evaluation_script_command_multi_seed.sh 1 pick-a-pic-v2-dpo_dataset_160000_pairs 4900 diffusion-dpo partiprompts
./evaluation_script_command.sh 0 pick-a-pic-v2-dpo_dataset_160000_pairs 4900 diffusion-dpo HPDv2-anime 42
./evaluation_script_command.sh 0 pick-a-pic-v2-dpo_dataset_160000_pairs 4900 diffusion-dpo HPDv2-concept-art 42
./evaluation_script_command.sh 0 pick-a-pic-v2-dpo_dataset_160000_pairs 4900 diffusion-dpo HPDv2-paintings 42
./evaluation_script_command.sh 0 pick-a-pic-v2-dpo_dataset_160000_pairs 4900 diffusion-dpo HPDv2-photo 42