import torch
import torchvision

from .builder import PREFERENCE_MODEL_FUNC_BUILDERS, get_preference_model_func

@PREFERENCE_MODEL_FUNC_BUILDERS.register_module(name='step_aware_preference_model_func')
def step_aware_preference_model_func_builder(cfg):
    from .models.step_aware_preference_model import StepAwarePreferenceModel
    step_aware_preference_model = StepAwarePreferenceModel(
        model_pretrained_model_name_or_path=cfg.model_pretrained_model_name_or_path,
        processor_pretrained_model_name_or_path=cfg.processor_pretrained_model_name_or_path,
        ckpt_path=cfg.ckpt_path,
    ).eval().to(cfg.device)
    step_aware_preference_model.requires_grad_(False)
    
    @torch.no_grad()
    def preference_fn(img, extra_info):
        # b
        scores = step_aware_preference_model.get_preference_score(
            img, 
            extra_info['input_ids'],
            extra_info['timesteps'],
        )
        return scores
    
    return preference_fn

@PREFERENCE_MODEL_FUNC_BUILDERS.register_module(name='hpsv2_preference_model_func')
def hpsv2_preference_model_func_builder(cfg):
    from .models.hpsv2 import HPSv2
    hpsv2 = HPSv2()
    hpsv2.eval().to(cfg.device)
    hpsv2.requires_grad_(False)
    _transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    def preference_fn(img, extra_info):
        img = (img / 2 + 0.5).clamp(0, 1).float()
        img = _transform(img)
        caption = hpsv2.tokenizer(extra_info["prompts"]).to(cfg.device)
        outputs = hpsv2.model(img, caption)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits = image_features @ text_features.T
        scores = torch.diagonal(logits) # reward
        return scores

    return preference_fn

@PREFERENCE_MODEL_FUNC_BUILDERS.register_module(name="imagereward_preference_model_func")
def imagereward_preference_model_func_builder(cfg):
    import ImageReward as RM
    imagereward = RM.load("ImageReward-v1.0", device=cfg.device).eval().requires_grad_(False)
    _transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    def preference_fn(img, extra_info):
        img = (img / 2 + 0.5).clamp(0, 1).float()
        img = _transform(img)
        
        rm_input_ids, rm_attention_masks = [], []
        for prompt in extra_info['prompts']:
            tokenizer_dict = imagereward.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt")
            rm_input_ids.append(tokenizer_dict.input_ids)
            rm_attention_masks.append(tokenizer_dict.attention_mask)
            
        rm_input_ids = torch.stack(rm_input_ids)
        rm_attention_masks = torch.stack(rm_attention_masks)
        rm_input_ids = rm_input_ids.view(-1, rm_input_ids.shape[-1]).to(cfg.device)
        rm_attention_masks = rm_attention_masks.view(-1, rm_attention_masks.shape[-1]).to(cfg.device)
        
        scores = imagereward.score_gard(rm_input_ids, rm_attention_masks, img) # reward
        return scores
    
    return preference_fn

@PREFERENCE_MODEL_FUNC_BUILDERS.register_module(name="pickscore_preference_model_func")
def pickscore_preference_model_func_builder(cfg):
    from transformers import AutoProcessor, AutoModel
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"
    pickscore_processor = AutoProcessor.from_pretrained(processor_name_or_path)
    pickscore = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(cfg.device).requires_grad_(False)
    _transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    def preference_fn(img, extra_info):
        img = (img / 2 + 0.5).clamp(0, 1).float()
        img = _transform(img)
        
        text_inputs = pickscore_processor(text=extra_info["prompts"], padding=True, truncation=True, max_length=77, return_tensors="pt").to(cfg.device)
        image_embs = pickscore.get_image_features(img)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True) # [B, embedding_dim]
        text_embs = pickscore.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True) # [B, embedding_dim]
    
        logits = pickscore.logit_scale.exp() * text_embs @ image_embs.T # [B, B]
        scores = torch.diagonal(logits) # [B]
        
        return scores
    
    return preference_fn

@PREFERENCE_MODEL_FUNC_BUILDERS.register_module(name="aigi_detector_preference_model_jpeg_func")
def aigi_detector_preference_model_jpeg_func_builder(cfg):
    import safetensors
    from diff_jpeg import diff_jpeg_coding
    if cfg.aigi_detector == "univfd":
        from aigi_detector_training.univfd import UnivFD
        aigi_detector = UnivFD("openai/clip-vit-large-patch14")
        _transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    elif cfg.aigi_detector == "dinov2":
        from aigi_detector_training.dinov2 import Dinov2
        aigi_detector = Dinov2("facebook/dinov2-base")
        _transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
    ckpt = safetensors.torch.load_file(cfg.aigi_detector_path)
    aigi_detector.load_state_dict(ckpt)
    aigi_detector.eval().to(cfg.device).requires_grad_(False)
    
    def preference_fn(img, extra_info):
        img = (img / 2 + 0.5).clamp(0, 1).float()*255.0 # [B, C, H, W], [0, 255]
        batch_size, _, _, _ = img.shape
        assert img.requires_grad == True
        
        jpeg_quality = torch.randint(low=60, high=100, size=(batch_size,), device=cfg.device)
        img_jpeg = diff_jpeg_coding(image_rgb=img, jpeg_quality=jpeg_quality) 
        img_jpeg = img_jpeg / 255.0 # [0, 1]
        assert img_jpeg.requires_grad == True
        
        img_transformed = _transform(img_jpeg)
        assert img_transformed.requires_grad == True
        
        logits = aigi_detector(img_transformed)
        outputs = torch.sigmoid(logits) # 0 -> real, 1 -> fake
        scores = 1 - outputs
        
        return scores
    
    return preference_fn

@PREFERENCE_MODEL_FUNC_BUILDERS.register_module(name="aigi_detector_preference_model_func")
def aigi_detector_preference_model_func_builder(cfg):
    import safetensors
    if cfg.aigi_detector == "univfd":
        from aigi_detector_training.univfd import UnivFD
        aigi_detector = UnivFD("openai/clip-vit-large-patch14")
        ckpt = safetensors.torch.load_file(cfg.aigi_detector_path)
        aigi_detector.load_state_dict(ckpt)
        
        _transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    elif cfg.aigi_detector.startswith("dinov2"):
        from aigi_detector_training.dinov2 import Dinov2
        aigi_detector = Dinov2("facebook/dinov2-base")
        ckpt = safetensors.torch.load_file(cfg.aigi_detector_path)
        aigi_detector.load_state_dict(ckpt)
        
        _transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
    elif cfg.aigi_detector == "fatformer":
        from .fatformer_models import build_model as fatformer_build_model
        cfg.backbone = "CLIP:ViT-L/14"
        cfg.num_classes = 2
        cfg.num_vit_adapter = 8
        cfg.num_context_embedding = 8
        cfg.init_context_embedding = ""
        cfg.hidden_dim = 768
        cfg.clip_vision_width = 1024
        cfg.frequency_encoder_layer = 2
        cfg.decoder_layer = 4
        cfg.num_heads = 12
        aigi_detector = fatformer_build_model(cfg)
        ckpt = torch.load(cfg.aigi_detector_path, map_location='cpu')
        aigi_detector.load_state_dict(ckpt['model'], strict=False)
        
        _transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
    elif cfg.aigi_detector in [ "drct_convb-sdv14", "drct_clip-sdv14", "drct_convb-sdv21", "drct_clip-sdv21"]:
        from .drct_models.models import get_models as drct_get_models
        model_name_dict = { "drct_convb-sdv14": "convnext_base_in22k", "drct_convb-sdv21": "convnext_base_in22k", "drct_clip-sdv14": "clip-ViT-L-14", "drct_clip-sdv21": "clip-ViT-L-14" }
        cfg.is_train = False
        cfg.num_classes = 2
        cfg.freeze_extractor = False
        cfg.embedding_size = 1024
        aigi_detector = drct_get_models(model_name=model_name_dict[cfg.aigi_detector], pretrained=cfg.is_train, num_classes=cfg.num_classes, freeze_extractor=cfg.freeze_extractor, embedding_size=cfg.embedding_size)
        aigi_detector.load_state_dict(torch.load(cfg.aigi_detector_path, map_location='cpu'), strict=False)
        
        _transform = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
    # elif cfg.aigi_detector == "rine": # -> wait for check.
    #     from .rine_models.utils import get_our_trained_model as rine_get_models
    #     cfg.ncls = "ldm"
    #     cfg.ckpt_path = cfg.aigi_detector_path
    #     aigi_detector = rine_get_models(ncls=cfg.ncls, device=cfg.device, opt=cfg)
        
    #     _transform = torchvision.transforms.Compose([
    #         torchvision.transforms.CenterCrop(224),
    #         torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    #     ])
        
    elif cfg.aigi_detector == "code":
        from .code_models.models import VITContrastiveHF as CoDE_Model
        cfg.classification_type = "linear"
        aigi_detector = CoDE_Model(cfg.classification_type, cfg.aigi_detector_path)
        
        _transform = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
    aigi_detector.eval().to(cfg.device).requires_grad_(False)

    def preference_fn(img, extra_info):
        img = (img / 2 + 0.5).clamp(0, 1).float()# [B, C, H, W]
        img_transformed = _transform(img)
        
        logits = aigi_detector(img_transformed)
        if cfg.aigi_detector in [ "fatformer", "drct_convb-sdv14", "drct_clip-sdv14", "drct_convb-sdv21", "drct_clip-sdv21" ] :
            outputs = logits.softmax(dim=1)[:, 1].reshape(-1, 1) # [B, 1], 0 -> real, 1 -> fake
            
        elif cfg.aigi_detector == "code":
            outputs = logits[:, 1].reshape(-1, 1) # [B, 1], 0 -> real, 1 -> fake
        
        elif cfg.aigi_detector in [ "univfd", "dinov2", "dinov2-full_train"]:
            outputs = torch.sigmoid(logits) # 0 -> real, 1 -> fake
            
        if cfg.return_label:
            outputs = ( outputs > 0.5 ).long()
            
        scores = 1 - outputs
        return scores
    
    return preference_fn

# @PREFERENCE_MODEL_FUNC_BUILDERS.register_module(name="aigi_detector_preference_model_bceloss_func")
# def aigi_detector_preference_model_func_builder(cfg):
#     import safetensors
#     if cfg.aigi_detector == "univfd":
#         from aigi_detector_training.univfd import UnivFD
#         aigi_detector = UnivFD("openai/clip-vit-large-patch14")
#         _transform = torchvision.transforms.Compose([
#             torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
#             torchvision.transforms.CenterCrop(224),
#             torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
#         ])
#     elif cfg.aigi_detector == "dinov2":
#         from aigi_detector_training.dinov2 import Dinov2
#         aigi_detector = Dinov2("facebook/dinov2-base")
#         _transform = torchvision.transforms.Compose([
#             torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
#             torchvision.transforms.CenterCrop(224),
#             torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#         ])
        
#     ckpt = safetensors.torch.load_file(cfg.aigi_detector_path)
#     aigi_detector.load_state_dict(ckpt)
#     aigi_detector.eval().to(cfg.device).requires_grad_(False)
#     bceloss = torch.nn.BCEWithLogitsLoss(reduction="none")

#     def preference_fn(img, extra_info):
#         img = (img / 2 + 0.5).clamp(0, 1).float()# [B, C, H, W]
#         assert img.requires_grad == True
        
#         img_transformed = _transform(img)
#         assert img_transformed.requires_grad == True
        
#         logits = aigi_detector(img_transformed)
#         zero_label = torch.zeros_like(logits) # zero means real.
#         loss = bceloss(logits, zero_label) # lower is better.
#         scores = -1 * loss
        
#         return scores
    
#     return preference_fn

@PREFERENCE_MODEL_FUNC_BUILDERS.register_module(name="spo_reward_aigi_detector_func")
def spo_reward_aigi_detector_func_builder(cfg):
    reward_model_func_cfg = cfg.pop("reward_model_func_cfg")    
    aigi_detector_func_cfg = cfg.pop("aigi_detector_func_cfg")
    
    reward_model_func = get_preference_model_func(reward_model_func_cfg, cfg.device)
    aigi_detector_func = get_preference_model_func(aigi_detector_func_cfg, cfg.device)
    
    def preference_fn(img, extra_info):
        reward_model_score = reward_model_func(img, extra_info)
        aigi_detector_score = aigi_detector_func(img, extra_info)
        
        batch_size, _, _ , _ = img.shape
        reward_model_score = reward_model_score.reshape(batch_size, 1)
        aigi_detector_score = aigi_detector_score.reshape(batch_size, 1)
        
        return (reward_model_score, aigi_detector_score)
    
    return preference_fn