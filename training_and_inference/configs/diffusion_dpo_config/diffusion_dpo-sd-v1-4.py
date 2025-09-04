import ml_collections
from training_and_inference.configs.spo_config.basic_config import basic_config

def get_config():
    return exp_config()

def exp_config():
    config = basic_config()
    
    ###### Diffudion-DPO-Paramer
    config.diffusion_dpo = diffusion_dpo = ml_collections.ConfigDict()
    diffusion_dpo.beta_dpo = 5000
    diffusion_dpo.num_inference_steps = 50
    diffusion_dpo.guidance_scale = 7.5
    diffusion_dpo.eta = 1.0
    diffusion_dpo.num_images_per_prompt = 4
    
    ###### Preference Model ######
    preference_model = "hpsv2"
    config.preference_model_func_cfg = dict(
        type=f"{preference_model}_preference_model_func"
    )
    # aigi_detector = "dinov2"
    # config.preference_model_func_cfg = dict(
    #     type="aigi_detector_preference_model_func",
    #     aigi_detector=f"{aigi_detector}",
    #     aigi_detector_path=f"/data_center/data2/dataset/chenwy/21164-data/model-ckpt/{aigi_detector}/genimage/best_model/model.safetensors"
    # )
    
    ###### Training ######
    config.train.train_batch_size = 2
    config.train.gradient_accumulation_steps = 2
    config.train.early_stop_threshold = None
    config.train.early_stop_warmup_step = 300
    config.train.max_train_steps = 1000
    config.num_epochs = 10
    
    #### logging ####
    # total_train_batch_size = 4 * 1 * 4 = 16
    config.eval_global_step_interval = 100
    config.save_global_step_interval = config.eval_global_step_interval
    config.wandb_project_name = "diffusion_dpo"
    config.logdir = "/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_4/spo_4k/diffusion_dpo"
    config.run_name = f"{preference_model}"# config.run_name = "spo_sd-v1-5_4k-prompts_num-sam-4_10ep_bs10"
    config.resume_from = None
    config.validation_prompts = [ 'a cat.', 'a dog', 'a horse.', 'A bus stopped on the side of the road while people board it.', 'A woman holding a plate of cake in her hand.']
    config.num_validation_images = 1

    return config
