import torch
import torch.nn as nn
import torchvision.models as models
from dataclasses import dataclass

@dataclass
class Stage1Output:
    score: float
    embedding: torch.Tensor
    cascade_reject: bool

class DepthAntiSpoofNet(nn.Module):
    """
    Depth-based Unimodal Model using ResNet-50.
    """
    def __init__(self, embedding_dim=512, spoof_threshold=0.85):
        super(DepthAntiSpoofNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.spoof_threshold = spoof_threshold
        
        # Load pretrained ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Modify the first conv layer to accept 1 channel instead of 3
        old_conv = resnet.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Initialize the new conv layer with the average of the old conv layer's weights
        with torch.no_grad():
            self.conv1.weight = nn.Parameter(old_conv.weight.sum(dim=1, keepdim=True))
            
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.projection_head = nn.Sequential(
            nn.Linear(2048, self.embedding_dim),
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

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
