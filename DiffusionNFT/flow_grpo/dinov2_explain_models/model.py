import torch
import torch.nn as nn

dino_backbones = {
    'dinov2_s':{
        'name':'dinov2_vits14',
        'embedding_dim':384,
        'patch_size':14
    },
    'dinov2_b':{
        'name':'dinov2_vitb14',
        'embedding_dim':768,
        'patch_size':14
    },
    'dinov2_l':{
        'name':'dinov2_vitl14',
        'embedding_dim':1024,
        'patch_size':14
    },
    'dinov2_g':{
        'name':'dinov2_vitg14',
        'embedding_dim':1536,
        'patch_size':14
    },
}

# detection head
class linear_head(nn.Module):
    def __init__(self, embedding_size = 384, num_classes = 5):
        super(linear_head, self).__init__()
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        # print(f"detection head input shape: {x.shape}")
        return self.fc(x)


    
class Baseline_Model_v2(nn.Module):
    def __init__(self, backbone_name, target_size=518, num_classes=1):
        super().__init__() 
        self.backbone = torch.hub.load(repo_or_dir="/data3/chenweiyan/notebook/fine-tune-diffusion/spo_gitee/DiffusionNFT/flow_grpo/dinov2_explain_models/dinov2", model=dino_backbones[backbone_name]['name'], source="local")
        self.embedding_dim = dino_backbones[backbone_name]['embedding_dim']
        self.patch_size = dino_backbones[backbone_name]['patch_size']
        self.detection_head = linear_head(embedding_size=self.embedding_dim, num_classes=num_classes)

    def forward(self, x):
        height, width = x.shape[2], x.shape[3]
        patch_num = (height // self.patch_size, width // self.patch_size)
        
        x = self.backbone.forward_features(x)
        detection_embedding = x["x_norm_clstoken"]
        
        # detection task
        # [B, embedding_dim] -> [B, 1]
        predict_detection = self.detection_head(detection_embedding)
        return predict_detection
