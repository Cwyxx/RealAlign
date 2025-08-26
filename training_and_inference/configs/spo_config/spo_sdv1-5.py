from configs.basic_config import basic_config

def get_config():
    return exp_config()

def exp_config():
    config = basic_config()
    
    ###### Preference Model ######
    # config.preference_model_func_cfg = dict(
    #     type="hpsv2_preference_model_func"
    # )
    aigi_detector = "dinov2"
    return_label = True
    config.preference_model_func_cfg = dict(
        type="aigi_detector_preference_model_func",
        aigi_detector=f"{aigi_detector}",
        aigi_detector_path=config.aigi_detector_path_dict[aigi_detector],
        return_label = return_label
    )
    
    ###### Training ######
    config.sample.num_sample_each_step = 4
    config.train.train_batch_size = 4
    config.train.gradient_accumulation_steps = 1 # total_train_batch_size = 4 * 1 * 4 = 16
    config.num_epochs = 10
    
    #### logging ####
    config.train.early_stop_threshold = None
    config.train.save_and_eval_batch_interval = 250
    config.wandb_project_name = 'spo-label' if return_label else 'spo'
    config.logdir = f"/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_4/spo_4k/{config.wandb_project_name}"
    config.run_name = f"{aigi_detector}"
    config.validation_prompts = [ 'a cat.', 'a dog', 'a horse.', 'A bus stopped on the side of the road while people board it.', 'A woman holding a plate of cake in her hand.']
    config.num_validation_images = 1

    return config
