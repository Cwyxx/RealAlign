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


def paired_real_generated_dataset_sd3():
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

    config.train.algorithm = 'dpo'
    # Change ref_update_step to a small number, e.g., 40, to switch to OnlineDPO.
    config.train.ref_update_step=10000000
    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = 1
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 100
    config.sample.global_std=True
    config.train.ema=True
    config.save_freq = 40 # epoch
    config.eval_freq = 40
    config.save_dir = 'logs/geneval/sd3.5-M-dpo'
    config.reward_fn = {
        "geneval": 1.0,
    }
    
    #### DPO parameters ####
    config.dpo = ml_collections.ConfigDict()
    config.dpo.project_name = "online-dpo"
    config.dpo.batch_size = 1
    config.dpo.max_train_steps = 2000
    config.dpo.dataset_dir = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/paired_real_generated_dataset"
    config.dpo.dataset = {
        "train": "train_real_better_3_5",
        "val": "high_quality_val"
    }
    ### ToDo ####
    config.dpo.csv_file_path = {
        "train_real_better_3_5": "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/add_noise_denoise/random_add_noise_step/train_real_better_3.5.csv",
        "high_quality_val": "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/paired_real_generated_dataset/high_quality_val/high_quality_val.csv"
    }
    ### ToDo ####
    config.prompt_fn = "paired_real_generated_dataset"
    config.run_name = f"{config.prompt_fn}/add_noise-denoise-random-real_better_3_5-multi_step"
    config.save_dir = f"/data_center/data2/dataset/chenwy/21164-data/online-dpo/model-ckpt/{config.run_name}"
    config.train.gradient_accumulation_steps = 6
    
    # ### Resume from DiffusionNFT ###
    # config.train.lora_path = "/data_center/data2/dataset/chenwy/21164-data/diffusionnft/model-ckpt/sd3/SD3.5M-DiffusionNFT-MultiReward/checkpoints/checkpoint-0/lora"
    # config.sample.guidance_scale = 1.0
    # ### Resume from DiffusionNFT ###
    #### DPO parameters ####
    

    config.per_prompt_stat_tracking = True
    return config



def get_config(name):
    return globals()[name]()
