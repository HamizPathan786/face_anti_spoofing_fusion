import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class Stage1Output:
    score: float
    embedding: torch.Tensor
    cascade_reject: bool

class rPPGAntiSpoofNet(nn.Module):
    """
    1D Temporal CNN for rPPG signal analysis.
    Key insight: Real faces show periodic pulse (0.7-3.5 Hz).
    """
    def __init__(self, embedding_dim=512, spoof_threshold=0.85):
        super(rPPGAntiSpoofNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.spoof_threshold = spoof_threshold
        
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.projection_head = nn.Sequential(
            nn.Linear(256, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(inplace=True)
            # Intentionally omitting dropout as 1D features are smaller and less prone to mass co-adapt
        )
        
        self.classifier_head = nn.Sequential(
            nn.Linear(self.embedding_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Input: (B, 1, 30) - rPPG signal tensor
        Returns: spoof_score, feature_embedding
        """
        x = self.features(x)
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
