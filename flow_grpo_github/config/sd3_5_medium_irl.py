import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def compressibility():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config


def paired_real_fake_dataset_sd3():
    config = compressibility()

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 40
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 24
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = 1
    config.sample.test_batch_size = 1 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.algorithm = 'irl'
    # Change ref_update_step to a small number, e.g., 40, to switch to OnlineDPO.
    config.train.ref_update_step=False # True for OnlineDPO, False for OfflineDPO
    
    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = 1
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 100
    config.sample.global_std=True
    config.train.ema=True
    config.save_freq = 50 # epoch
    config.eval_freq = 50
    config.save_dir = 'logs/pickscore/sd3.5-M-dpo'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    #### IRL parameters ####
    config.train.learning_rate = 1e-4
    config.irl = ml_collections.ConfigDict()
    config.irl.project_name = "diffusion-irl-sd-3-5-medium"
    config.irl.batch_size = 1
    config.irl.buffer_size = 1
    config.irl.buffer_batch_size = 1
    config.irl.buffer_num_inference_steps = 10
    config.irl.buffer_perturb_timesteps = True
    config.irl.buffer_sample_steps = 1
    config.irl.max_train_steps = 1600
    config.irl.margin=0.001
    config.train.beta = 100
    config.train.ref_update_step = False # True for OnlineDPO, False for OfflineDPO

    config.irl.csv_file_path = {
        "top_500_pickscore_images_hpdv3_all": "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/u2net_next_inpainting/HPDv3/top_500_pickscore_images_hpdv3_all.csv",
        "high_quality_val": "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/paired_real_generated_dataset/high_quality_val/high_quality_val.csv"
    }
    config.irl.precomputed_embeddings_dir_dict = {
        "top_500_pickscore_images_hpdv3_all": "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/precompute_prompt_embeddings/HPDv3/top_500_pickscore_images_hpdv3_all",
        "high_quality_val": "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/precompute_prompt_embeddings/general_1/high_quality_val/",
        }
    
    config.irl.dataset = {
        "train": "top_500_pickscore_images_hpdv3_all",
        "val": "high_quality_val"
    }
    
    config.run_name = f"irl-top_500_pickscore_images_hpdv3_all"
    # ### DiffusionNFT parameters ###
    # config.sample.guidance_scale = 1.0
    # config.train.lora_path = "/data_center/data2/dataset/chenwy/21164-data/diffusion-dpo/sd-3-5-medium/model-ckpt/DiffusionNFT/checkpoints/checkpoint-0/lora/learner"
    # config.run_name = f"DiffusionNFT-next-random_9984_images_no_anime_pickscore_002-hpdv3_all-inpainting"
    # ### DiffusionNFT parameters ###
    
    # #### Flow-GRPO parameters ####
    # config.train.lora_path = "/data_center/data2/dataset/chenwy/21164-data/diffusion-dpo/sd-3-5-medium/model-ckpt/FlowGRPO-PickScore/checkpoints/checkpoint-0/lora/learner"
    # config.run_name = f"FlowGRPO-PickScore-next-irl-top_500_pickscore_images_hpdv3_all"
    # #### Flow-GRPO parameters ####
    
    # #### GRPO-Guard Parameters ####
    # config.train.lora_path = "/data_center/data2/dataset/chenwy/21164-data/diffusion-dpo/sd-3-5-medium/model-ckpt/GRPO-Guard/checkpoints/checkpoint-0/lora/learner"
    # config.run_name = f"GRPO-Guard-next-top_512_images_no_anime_colorfulness_pickscore_002-hpdv3_all-inpainting-w_sft"
    
    config.save_dir = f"/data_center/data2/dataset/chenwy/21164-data/diffusion-dro/sd-3-5-medium/model-ckpt/{config.run_name}"
    
    # ### Resume from DiffusionNFT ###
    # config.train.lora_path = "/data_center/data2/dataset/chenwy/21164-data/diffusionnft/model-ckpt/sd3/SD3.5M-DiffusionNFT-MultiReward/checkpoints/checkpoint-0/lora"
    # config.sample.guidance_scale = 1.0
    # ### Resume from DiffusionNFT ###
    #### DPO parameters ####
    

    config.per_prompt_stat_tracking = True
    return config



def get_config(name):
    return globals()[name]()
