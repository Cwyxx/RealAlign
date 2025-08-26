from configs.basic_config import basic_config

def get_config():
    return exp_config()

def exp_config():
    config = basic_config()
    
    ##### SPO-training config #####
    ###### General ######
    # random seed for reproducibility.
    config.seed = 42
    config.pretrained.model = "runwayml/stable-diffusion-v1-5"
    config.use_lora = True
    config.lora_rank = 4
    ###### Training ######
    config.train.learning_rate = 6e-5
    config.sample.guidance_scale = 5.0
    config.sample_batch_size = 2
    config.train.learning_rate = 6e-5
    config.train.adam_weight_decay = 1e-4
    config.train.gradient_accumulation_steps = 1
    config.train.max_grad_norm = 1.0
    config.sample.num_sample_each_step = 2
    
    ##### dataset #####
    config.dataset_cfg = dict(
        type="PromptDataset",
        meta_json_path='assets/prompts/spo_4k.json',
        pretrained_tokenzier_path='laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
    )
    
    ###### Preference Model ######
    config.bool_spo_reward_aigi_detector_func = True
    preference_model = "step_aware"
    aigi_detector = "dinov2"
    return_label = False
    config.preference_model_func_cfg = dict(
        type="spo_reward_aigi_detector_func",
        reward_model_func_cfg=dict(
            type=f"{preference_model}_preference_model_func",
            model_pretrained_model_name_or_path='yuvalkirstain/PickScore_v1',
            processor_pretrained_model_name_or_path='laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
            ckpt_path='/data_center/data2/dataset/chenwy/21164-data/model-ckpt/spo_step_aware_preference_model/sd-v1-5_step-aware_preference_model.bin'
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
    compare_func_threshold = 0.3
    config.compare_func_cfg = dict(
        type="aggregate_rewards_by_rank_compare",
        threshold=compare_func_threshold,
        aigi_detector_weight=aigi_detector_weight
    )
    
    
    ###### Training ######
    config.sample.sample_batch_size = 2
    config.sample.num_sample_each_step = 4
    config.train.train_batch_size = 8
    config.train.gradient_accumulation_steps = 1 # total_train_batch_size = 4 * 1 * 8 = 32
    config.num_epochs = 10
    
    #### logging ####
    config.train.early_stop_threshold = None
    config.train.save_and_eval_batch_interval = 250
    config.wandb_project_name = "spo-sdv1-5"
    config.logdir = f"/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/{config.wandb_project_name}"
    config.run_name = f"{config.wandb_project_name}-{preference_model}_{1-aigi_detector_weight}-{aigi_detector}_{aigi_detector_weight}-comp_func_threshold_{compare_func_threshold}"
    config.validation_prompts = [ 'a cat.', 'a dog', 'a horse.', 'A bus stopped on the side of the road while people board it.', 'A woman holding a plate of cake in her hand.']
    config.num_validation_images = 1
    
    return config
