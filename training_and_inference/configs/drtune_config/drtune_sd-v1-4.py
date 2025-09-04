import ml_collections
from training_and_inference.configs.spo_config.basic_config import basic_config

def get_config():
    return exp_config()

def exp_config():
    config = basic_config()
    
    ##### dataset #####
    config.dataset_cfg = dict(
        type="PromptDataset",
        meta_json_path='assets/prompts/spo_4k.json',
        pretrained_tokenzier_path='laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
    )
    
    ###### Preference Model ######
    # preference_model = "imagereward"
    # config.preference_model_func_cfg = dict(
    #     type=f"{preference_model}_preference_model_func"
    # ) 
    config.train.early_stop_threshold = 0.4
    preference_model = "dinov2"
    config.preference_model_func_cfg = dict(
        type=f"aigi_detector_preference_model_jpeg_func",
        aigi_detector=f"{preference_model}",
        aigi_detector_path=f"/data_center/data2/dataset/chenwy/21164-data/model-ckpt/{preference_model}/genimage/best_model/model.safetensors"
    ) 
    
    ###### logging ######
    # total_batch_size: 2 * 2 * 4
    config.wandb_project_name = "drtune_cfg"
    config.logdir = f"/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_4/spo_4k"
    config.run_name = f"{config.wandb_project_name}-{preference_model}" # experiment name under a project (wandb_project_name) in swanlab.
    
    ###### Training ######
    config.train.train_batch_size = 2
    config.train.gradient_accumulation_steps = 2
    config.train.eta = 1.0
    config.train.guidance_scale = 7.5
    config.resume_from = None
    config.num_epochs = 5
    config.pipeline_num_inference_steps = 50
    
    ###### drtune ######
    config.drtune = drtune = ml_collections.ConfigDict()
    drtune.K = 5 # number of training time steps (denoisign time steps)
    drtune.T = 50 # Total number of denoising steps used during finetuning.
    drtune.M = 20 # Maximal early stop step m.
    
    #### validation ####
    config.validation_prompts = [ 'a cat.', 'a dog', 'a horse.', 'A bus stopped on the side of the road while people board it.', 'A woman holding a plate of cake in her hand.']
    config.num_validation_images = 1

    return config