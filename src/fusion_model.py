# fusion_model.py
import torch
import torch.nn as nn
from model_rgb import EmbeddingModel
import torchvision.models as models

class DepthEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple small CNN for single-channel depth input
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 112x112
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 56x56
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.out_dim = 32

    def forward(self, x):
        return self.net(x)

class FusionModel(nn.Module):
    def __init__(self, rgb_backbone_out_dim=512, depth_emb_dim=32, lbp_dim=59, hidden=256, pretrained_rgb=True):
        super().__init__()
        self.rgb_emb_model = EmbeddingModel(pretrained=pretrained_rgb)
        self.depth_model = DepthEmbedding()
        self.lbp_dim = lbp_dim

        total_dim = rgb_backbone_out_dim + depth_emb_dim + lbp_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, 2)
        )

    def forward(self, rgb, depth, lbp_hist):
        # rgb: batch x 3 x H x W
        # depth: batch x 1 x H x W
        # lbp_hist: batch x lbp_dim
        rgb_feat = self.rgb_emb_model(rgb)  # batch x rgb_dim
        depth_feat = self.depth_model(depth)  # batch x depth_emb_dim
        x = torch.cat([rgb_feat, depth_feat, lbp_hist], dim=1)
        logits = self.fusion(x)
        return logits
