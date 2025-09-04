from training_and_inference.configs.spo_config.basic_config import basic_config

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
    config.sample.guidance_scale = 5.0
    config.sample_batch_size = 4
    config.train.learning_rate = 6e-5# custom: 1e-5 # 6e-5
    config.train.adam_weight_decay = 1e-4
    config.train.gradient_accumulation_steps = 1
    config.train.max_grad_norm = 1.0 # custom: 0.1 # 1.0
    config.sample.num_sample_each_step = 4
    
    ###### Preference Model ######
    # config.preference_model_func_cfg = dict(
    #     type="hpsv2_preference_model_func"
    # )
    aigi_detector = "dinov2"
    return_label = False
    config.preference_model_func_cfg = dict(
        type="aigi_detector_preference_model_func",
        aigi_detector=f"{aigi_detector}",
        aigi_detector_path=config.aigi_detector_path_dict[aigi_detector],
        return_label = return_label
    )
    ###### Compare Function ######
    compare_func_threshold=0.0
    config.compare_func_cfg = dict(
        type="preference_score_compare",
        threshold=compare_func_threshold,
    )
    ###### Training ######
    config.sample.sample_batch_size = 4
    config.sample.num_sample_each_step = 4
    config.train.train_batch_size = 4
    config.train.gradient_accumulation_steps = 1 # total_train_batch_size = 4 * 1 * 4 = 16
    config.num_epochs = 4
    
    
    #### Resume LoRA ####
    config.pretrained.lora_path = "/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/spo-sdv1-5/step_aware/checkpoint_1_754"
    
    #### logging ####
    config.train.early_stop_threshold = None
    config.train.save_and_eval_batch_interval = 25
    config.wandb_project_name = "spo-sdv1-5"
    config.logdir = f"/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/{config.wandb_project_name}"
    config.run_name = f"reuse_lora-step_aware_checkpoint_1_754-{aigi_detector}-lr_{config.train.learning_rate}-max_gn_{config.train.max_grad_norm}-comp_{compare_func_threshold}"
    config.validation_prompts = [ 'a cat.', 'a dog', 'a horse.', 'A bus stopped on the side of the road while people board it.', 'A woman holding a plate of cake in her hand.']
    config.num_validation_images = 1

    return config
