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
    config.bool_spo_reward_aigi_detector_func = True
    preference_model = "hpsv2"
    aigi_detector = "code"
    return_label = False
    config.preference_model_func_cfg = dict(
        type="spo_reward_aigi_detector_func",
        reward_model_func_cfg=dict(
            type=f"{preference_model}_preference_model_func",
        ),
        aigi_detector_func_cfg=dict(
            type=f"aigi_detector_preference_model_func", 
            aigi_detector=f"{aigi_detector}",
            aigi_detector_path=config.aigi_detector_path_dict[aigi_detector],
            return_label=return_label
        )
    )
    
    ###### Compare Function ######
    aigi_detector_weight = 0.5
    compare_func_threshold = 0
    config.compare_func_cfg = dict(
        type="spo_reward_aigi_detector_compare",
        threshold=compare_func_threshold,
        aigi_detector_weight=aigi_detector_weight
    )
    
    
    ###### Training ######
    config.sample.num_sample_each_step = 4
    config.train.train_batch_size = 4
    config.train.gradient_accumulation_steps = 1 # total_train_batch_size = 4 * 1 * 4 = 16
    config.num_epochs = 3
    
    #### logging ####
    config.train.early_stop_threshold = None
    config.train.save_and_eval_batch_interval = 250
    config.wandb_project_name = 'spo-label' if return_label else 'spo'
    config.logdir = f"/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_4/spo_4k/{config.wandb_project_name}"
    config.run_name = f"{config.wandb_project_name}-{preference_model}_{1-aigi_detector_weight}-{aigi_detector}_{aigi_detector_weight}-comp_func_threshold_{compare_func_threshold}"
    config.validation_prompts = [ 'a cat.', 'a dog', 'a horse.', 'A bus stopped on the side of the road while people board it.', 'A woman holding a plate of cake in her hand.']
    config.num_validation_images = 1
    
    return config
