from training_and_inference.configs.grpo_config.base_config import basic_config

def get_config():
    return exp_config()

def exp_config():
    config = basic_config()
    
    config.train.gradient_accumulation_steps = 8
    
    ###### Sampling ######
    config.sample.batch_size = 4 # # Number of prompts to load from the dataloader to form a single batch.
    config.num_generations = 16 # number of images to generate for a prompt.
    
    
    ###### Train ######
    config.train.batch_size = 4
    config.train.save_and_eval_batch_interval = 25
    
    ###### Preference Model ######
    aigi_detector = "dinov2"
    config.preference_model_func_cfg = dict(
        type="aigi_detector_preference_model_func",
        aigi_detector=f"{aigi_detector}",
        aigi_detector_path=config.aigi_detector_path_dict[aigi_detector],
    )
    
    ###### Logging ######
    config.wandb_project_name = "grpo-sdv1-5"
    config.logdir = f"/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_5/spo_4k/{config.wandb_project_name}"
    config.run_name = f"{aigi_detector}-lr_{config.train.learning_rate}-max_gn_{config.train.max_grad_norm}"
    config.validation_prompts = [ 'a cat.', 'a dog', 'a horse.', 'A bus stopped on the side of the road while people board it.', 'A woman holding a plate of cake in her hand.']
    config.num_validation_images = 1
    
    return config