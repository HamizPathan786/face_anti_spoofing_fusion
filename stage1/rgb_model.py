import torch
import torch.nn as nn
import timm
from dataclasses import dataclass

@dataclass
class Stage1Output:
    score: float
    embedding: torch.Tensor
    cascade_reject: bool

class RGBAntiSpoofNet(nn.Module):
    """
    RGB-based Unimodal Model using EfficientNet-B4.
    """
    def __init__(self, embedding_dim=512, spoof_threshold=0.85):
        super(RGBAntiSpoofNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.spoof_threshold = spoof_threshold
        
        # Load backbone
        self.backbone = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0) # num_classes=0 for GAP output
        
        # Accessing blocks in timm's efficientnet is specific:
        # We will freeze early layers and fine-tune later ones.
        # efficientnet in timm has 'blocks' sequential.
        num_blocks = len(self.backbone.blocks)
        freeze_blocks = 3  # Based on prompt specification
        
        for i, block in enumerate(self.backbone.blocks):
            if i < freeze_blocks:
                for param in block.parameters():
                    param.requires_grad = False
                    
        # The efficientnet_b4 outputs 1792 features after GAP
        self.feature_dim = self.backbone.num_features 
        
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.embedding_dim),
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
        Input: (B, 3, 224, 224)
        Returns: spoof_score, feature_embedding
        """
        features = self.backbone(x)
        embedding = self.projection_head(features)
        spoof_score = self.classifier_head(embedding)
        return spoof_score.squeeze(1), embedding

    def forward_with_cascade(self, x):
        """
        Executes forward pass and determines if cascade early rejection should occur.
        """
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
