import torch
import torch.nn as nn
import torchvision.models as models
from dataclasses import dataclass

@dataclass
class Stage1Output:
    score: float
    embedding: torch.Tensor
    cascade_reject: bool

class IRAntiSpoofNet(nn.Module):
    """
    Simulated IR-based Unimodal Model using MobileNetV3-Large.
    """
    def __init__(self, embedding_dim=512, spoof_threshold=0.85):
        super(IRAntiSpoofNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.spoof_threshold = spoof_threshold
        
        # Load MobileNetV3 Large
        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        
        # Modify first layer for 1-channel input
        old_conv = mobilenet.features[0][0]
        new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size, 
                             stride=old_conv.stride, padding=old_conv.padding, bias=False)
        with torch.no_grad():
            new_conv.weight = nn.Parameter(old_conv.weight.sum(dim=1, keepdim=True))
            
        mobilenet.features[0][0] = new_conv
        
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # MobileNet features output 960 dim for large
        self.projection_head = nn.Sequential(
            nn.Linear(960, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.classifier_head = nn.Sequential(
            nn.Linear(self.embedding_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Input: (B, 1, 224, 224)
        Returns: spoof_score, feature_embedding
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        embedding = self.projection_head(x)
        spoof_score = self.classifier_head(embedding)
        return spoof_score.squeeze(1), embedding

    def forward_with_cascade(self, x):
        scores, embeddings = self.forward(x)
        rejects = scores > self.spoof_threshold
        
        outputs = []
        for i in range(x.size(0)):
            outputs.append(Stage1Output(
                score=scores[i].item(),
                embedding=embeddings[i],
                cascade_reject=rejects[i].item()
            ))
        return outputs if x.size(0) > 1 else outputs[0]

    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, path):
        torch.save({
            'state_dict': self.state_dict(),
            'embedding_dim': self.embedding_dim,
            'spoof_threshold': self.spoof_threshold
        }, path)

    @classmethod
    def load_checkpoint(cls, path, device='cpu'):
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            embedding_dim=checkpoint.get('embedding_dim', 512),
            spoof_threshold=checkpoint.get('spoof_threshold', 0.85)
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        return model
