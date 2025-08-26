import ml_collections

def get_config():
    return basic_config()

def basic_config():
    config = ml_collections.ConfigDict()
    
    ###### General ######
    # random seed for reproducibility.
    config.seed = 42 # 1
    # number of checkpoints to keep before overwriting old ones.
    config.num_checkpoint_limit = None
    # allow tf32 on Ampere GPUs, which can speed up training.
    config.allow_tf32 = True
    # whether or not to use xFormers to reduce memory usage.
    config.use_xformers = False
    # enable activation checkpointing or not. 
    # this reduces memory usage at the cost of some additional compute.
    config.use_checkpointing = True
    
    ###### Model Setting ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
    pretrained.model = "runwayml/stable-diffusion-v1-5" # "CompVis/stable-diffusion-v1-4"# default "runwayml/stable-diffusion-v1-5"
    config.use_lora = True
    config.lora_rank = 4 # 128 # default 4
    
    ###### Preference Model ######
    config.preference_model_func_cfg = dict(
        type="step_aware_preference_model_func",
        model_pretrained_model_name_or_path='yuvalkirstain/PickScore_v1',
        processor_pretrained_model_name_or_path='laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
        ckpt_path='model_ckpts/sd-v1-5_step-aware_preference_model.bin',
    )
    
    ###### Compare Function ######
    config.compare_func_cfg = dict(
        type="preference_score_compare",
        threshold=0.3,
    )
    
    ##### dataset #####
    config.dataset_cfg = dict(
        type="PromptDataset",
        meta_json_path='assets/prompts/spo_4k.json',
        pretrained_tokenzier_path='laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
    )
    
    ##### dataloader ####
    config.dataloader_num_workers = 16
    config.dataloader_shuffle = True
    config.dataloader_pin_memory = True
    config.dataloader_drop_last = False

    ###### Training ######
    config.num_epochs = 10
    # resume training from a checkpoint. either an exact checkpoint directory (e.g. checkpoint_50), or a directory
    # containing checkpoints, in which case the latest one will be used. `config.use_lora` must be set to the same value
    # as the run that generated the saved checkpoint.
    config.resume_from = ""
    
    config.sample = sample = ml_collections.ConfigDict()
    # number of sampler inference steps.
    sample.num_steps = 20 # origin. 20
    # eta parameter for the DDIM sampler. this controls the amount of noise injected into the sampling process, with 0.0
    # being fully deterministic and 1.0 being equivalent to the DDPM sampler.
    sample.eta = 1.0
    # classifier-free guidance weight. 1.0 is no guidance.
    sample.guidance_scale = 5.0 # 7.5 # origin 5.0
    sample.sample_batch_size = 2 # 1 # origin 10
    # number of x_{t-1} sampled at each timestep.
    sample.num_sample_each_step = 2 # 4 # origin 2

    config.train = train = ml_collections.ConfigDict()
    # early stop 
    train.early_stop_threshold = None
    # batch size (per GPU!) to use for training.
    train.train_batch_size = 1 # origin 10
    # whether to use the 8bit Adam optimizer from bitsandbytes.
    train.use_8bit_adam = False
    # learning rate.
    train.learning_rate = 6e-5 # 1e-05 # origin 6e-5
    # Adam beta1.
    train.adam_beta1 = 0.9
    # Adam beta2.
    train.adam_beta2 = 0.999
    # Adam weight decay.
    train.adam_weight_decay = 1e-4 # 1e-2 # origin. 1e-4
    # Adam epsilon.
    train.adam_epsilon = 1e-8
    # number of gradient accumulation steps. the effective batch size is `batch_size * num_gpus *
    # gradient_accumulation_steps`.
    train.gradient_accumulation_steps = 1 # origin. 1
    # maximum gradient norm for gradient clipping.
    train.max_grad_norm = 1.0 # 0.1 # origin. 1.0
    # whether or not to use classifier-free guidance during training. if enabled, the same guidance scale used during
    # sampling will be used during training.
    train.cfg = True

    train.divert_start_step = 4
    # coefficient of the KL divergence
    train.beta = 10.0
    # The coefficient constraining the probability ratio.
    train.eps = 0.1

    #### validation ####
    config.validation_prompts = ['A woman holding a plate of cake in her hand.']
    config.num_validation_images = 2
    config.eval_interval = 1
    
    #### logging ####
    # run name for wandb logging and checkpoint saving.
    config.run_name = "hpsv2"
    config.wandb_project_name = 'spo'
    config.wandb_entity_name = None
    # top-level logging directory for checkpoint saving.
    config.logdir = "work_dirs"
    config.save_interval = 1
    
    #### Aigi Detector Path ####
    config.aigi_detector_path_dict = {
        "fatformer" : "/data_center/data2/dataset/chenwy/21164-data/detection-method-ckpt/FatFormer/fatformer_4class_ckpt.pth",
        "dinov2": "/data_center/data2/dataset/chenwy/21164-data/model-ckpt/dinov2/genimage/best_model/model.safetensors",
        "dinov2-full_train": "/data_center/data2/dataset/chenwy/21164-data/model-ckpt/dinov2-full_train/genimage/best_model/model.safetensors",
        "drct_convb-sdv14": "/data_center/data2/dataset/chenwy/21164-data/detection-method-ckpt/DRCT/sdv14/convnext_base_in22k_224_drct_amp_crop/14_acc0.9996.pth",
        "drct_clip-sdv14": "/data_center/data2/dataset/chenwy/21164-data/detection-method-ckpt/DRCT/sdv14/clip-ViT-L-14_224_drct_amp_crop/13_acc0.9664.pth",
        "code": "/data_center/data2/dataset/chenwy/21164-data/detection-method-ckpt/CoDE"
    }
    return config
