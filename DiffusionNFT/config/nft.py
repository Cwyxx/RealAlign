import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def get_config(name):
    return globals()[name]()


def _get_config(base_model="sd3", n_gpus=1, gradient_step_per_epoch=1, dataset="pickscore", reward_fn={}, name=""):
    config = base.get_config()
    assert base_model in ["sd3"]
    assert dataset in ["pickscore", "ocr", "geneval"]

    config.base_model = base_model
    config.dataset = os.path.join(os.getcwd(), f"dataset/{dataset}")
    if base_model == "sd3":
        config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
        config.sample.num_steps = 10
        config.sample.eval_num_steps = 40
        config.sample.guidance_scale = 4.5
        config.resolution = 512
        config.train.beta = 0.0001
        config.sample.noise_level = 0.7
        bsz = 7

    config.sample.num_image_per_prompt = 24 # 24
    num_groups = 48 # 48

    while True:
        if bsz < 1:
            assert False, "Cannot find a proper batch size."
        if (
            num_groups * config.sample.num_image_per_prompt % (n_gpus * bsz) == 0
            and bsz * n_gpus % config.sample.num_image_per_prompt == 0
        ):
            n_batch_per_epoch = num_groups * config.sample.num_image_per_prompt // (n_gpus * bsz)
            if n_batch_per_epoch % gradient_step_per_epoch == 0:
                config.sample.train_batch_size = bsz
                config.sample.num_batches_per_epoch = n_batch_per_epoch
                config.train.batch_size = 2
                config.train.gradient_accumulation_steps = (
                    config.sample.num_batches_per_epoch * (config.sample.train_batch_size // config.train.batch_size) // gradient_step_per_epoch
                )
                break
        bsz -= 1

    # special design, the test set has a total of 1018/2212/2048 for ocr/geneval/pickscore, to make gpu_num*bs*n as close as possible to it, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    config.sample.test_batch_size = 5 if dataset == "geneval" else 5
    if n_gpus > 32:
        config.sample.test_batch_size = config.sample.test_batch_size // 2

    config.prompt_fn = "geneval" if dataset == "geneval" else "general_ocr"

    config.run_name = f"/data_center/data2/dataset/chenwy/21164-data/diffusionnft/logs/nft_{base_model}_{name}"
    config.save_dir = f"/data_center/data2/dataset/chenwy/21164-data/diffusionnft/model-ckpt/{base_model}/{name}"
    config.reward_fn = reward_fn

    config.decay_type = 1
    config.beta = 1.0
    config.train.adv_mode = "all"

    config.sample.guidance_scale = 1.0
    config.sample.deterministic = True
    config.sample.solver = "dpm2"
    return config


def sd3_ocr():
    reward_fn = {
        "ocr": 1.0,
    }
    config = _get_config(
        base_model="sd3", n_gpus=8, gradient_step_per_epoch=2, dataset="ocr", reward_fn=reward_fn, name="ocr"
    )
    config.beta = 0.1
    config.decay_type = 2
    return config


def sd3_geneval():
    reward_fn = {
        "geneval": 1.0,
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="geneval",
        reward_fn=reward_fn,
        name="geneval",
    )
    return config


def sd3_pickscore():
    reward_fn = {
        "pickscore": 1.0,
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=6,
        gradient_step_per_epoch=1,
        dataset="pickscore",
        reward_fn=reward_fn,
        name="pickscore",
    )
    config.sample.num_steps=10
    config.beta = 1.0
    return config


def sd3_hpsv2():
    reward_fn = {
        "hpsv2": 1.0,
    }
    config = _get_config(
        base_model="sd3", n_gpus=6, gradient_step_per_epoch=1, dataset="pickscore", reward_fn=reward_fn, name="hpsv2"
    )
    config.sample.num_steps=10
    config.beta = 1.0
    return config

def sd3_code():
    reward_fn = {
        "code": 1.0,
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=6,
        gradient_step_per_epoch=1,
        dataset="pickscore",
        reward_fn=reward_fn,
        name="sd3.5m-diffusionnft-multireward-next-code-beta_1.0-cfg_1.0",
    )
    config.sample.num_steps=10
    config.sample.mini_sample_size = 2
    
    #### for faster reward increase ####
    config.save_freq = 1 # 30
    config.eval_freq = 1
    #### for faster reward increase ####
    
    #### config.resume_lora_path ####
    config.resume_lora = "/data_center/data2/dataset/chenwy/21164-data/diffusionnft/model-ckpt/sd3/SD3.5M-DiffusionNFT-MultiReward"
    return config

def sd3_b_free():
    reward_fn = {
        "b_free": 1.0,
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=6,
        gradient_step_per_epoch=1,
        dataset="pickscore",
        reward_fn=reward_fn,
        name="b_free-beta_1.0-cfg_4.5",
    )
    config.sample.num_steps=10
    config.sample.guidance_scale = 4.5
    config.sample.mini_sample_size = 2
    #### for faster reward increase ####
    # config.beta = 0.1
    # config.decay_type = 2
    config.beta = 1.0
    config.save_freq = 1 # 30
    config.eval_freq = 1
    #### for faster reward increase ####
    return config

def sd3_multi_reward():
    reward_fn = {
        "pickscore": 1.0,
        "hpsv2": 1.0,
        "clipscore": 1.0,
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=6,
        gradient_step_per_epoch=1,
        dataset="pickscore",
        reward_fn=reward_fn,
        name="multi_reward",
    )
    config.sample.num_steps = 25
    config.beta = 0.1
    return config

def sd3_pickscore_b_free():
    reward_fn = {
        "pickscore": 1.0,
        "b_free": 1.0,
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=6,
        gradient_step_per_epoch=1,
        dataset="pickscore",
        reward_fn=reward_fn,
        name="pickscore_b_free",
    )
    config.sample.num_steps=10
    #### for faster reward increase ####
    config.beta = 0.1
    config.decay_type = 2
    config.save_freq = 1 # 30
    config.eval_freq = 1
    #### for faster reward increase ####
    return config

