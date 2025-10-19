# model_rgb.py
import torch
import torch.nn as nn
import torchvision.models as models

class FeatureBackbone(nn.Module):
    def __init__(self, backbone_name="resnet18", pretrained=True):
        super().__init__()
        if backbone_name == "resnet18":
            backbone = models.resnet18(pretrained=pretrained)
            self.out_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
        else:
            # you can add efficientnet/mobile net etc.
            backbone = models.resnet18(pretrained=pretrained)
            self.out_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)  # returns feature vector (batch x out_dim)

class Stage1Model(nn.Module):
    def __init__(self, backbone_name="resnet18", pretrained=True, num_classes=2):
        super().__init__()
        self.feature = FeatureBackbone(backbone_name, pretrained)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature.out_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feat = self.feature(x)
        logits = self.classifier(feat)
        return logits

# helper to get embedding-only model
class EmbeddingModel(nn.Module):
    def __init__(self, backbone_name="resnet18", pretrained=True):
        super().__init__()
        self.feature = FeatureBackbone(backbone_name, pretrained)

    def forward(self, x):
        return self.feature(x)
