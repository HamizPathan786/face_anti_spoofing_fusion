import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class FusionOutput:
    final_score: float
    attention_weights: torch.Tensor
    confidence: float

class MultiModalFusionNetwork(nn.Module):
    """
    Cross-Modal Attention Fusion for all 5 branches.
    """
    def __init__(self, embedding_dim=512, attention_dim=256, num_heads=8, dropout=0.3):
        super(MultiModalFusionNetwork, self).__init__()
        
        # Step 1: Per-modality projection
        # We project each 512-dim to specified attention_dim (256)
        self.proj_rgb = nn.Linear(embedding_dim, attention_dim)
        self.proj_depth = nn.Linear(embedding_dim, attention_dim)
        self.proj_ir = nn.Linear(embedding_dim, attention_dim)
        self.proj_rppg = nn.Linear(embedding_dim, attention_dim)
        self.proj_deepfake = nn.Linear(embedding_dim, attention_dim)
        
        self.ln1 = nn.LayerNorm(attention_dim)
        self.gelu = nn.GELU()
        
        # Step 2: Multi-Head Self-Attention
        self.mha = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, dropout=0.1)
        
        # Step 3: Attended feature aggregation
        self.attention_scorer = nn.Sequential(
            nn.Linear(attention_dim, 1, bias=False)
        )
        
        # Step 4: Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(attention_dim * 2, attention_dim),
            nn.BatchNorm1d(attention_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(attention_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, depth_feat, ir_feat, rppg_feat, deepfake_feat):
        """
        Inputs: 5 Tensors of shape (B, 512)
        """
        B = rgb_feat.size(0)
        
        # Project
        f_rgb = self.gelu(self.ln1(self.proj_rgb(rgb_feat)))
        f_depth = self.gelu(self.ln1(self.proj_depth(depth_feat)))
        f_ir = self.gelu(self.ln1(self.proj_ir(ir_feat)))
        f_rppg = self.gelu(self.ln1(self.proj_rppg(rppg_feat)))
        f_df = self.gelu(self.ln1(self.proj_deepfake(deepfake_feat)))
        
        # Stack as sequence: (B, 5, 256)
        seq = torch.stack([f_rgb, f_depth, f_ir, f_rppg, f_df], dim=1)
        
        # Original mean pooled
        mean_pooled = torch.mean(seq, dim=1) # (B, 256)
        
        # Multi-Head Attention expects (L, B, E)
        seq_t = seq.transpose(0, 1)
        attn_out_t, _ = self.mha(seq_t, seq_t, seq_t)
        attn_out = attn_out_t.transpose(0, 1) # (B, 5, 256)
        
        # Weighted sum across modality dimension
        # Calculate dynamic weights across the 5 modalities
        scores = self.attention_scorer(attn_out).squeeze(-1) # (B, 5)
        attn_weights = F.softmax(scores, dim=1)              # (B, 5)
        
        # Weighted sum: (B, 5, 256) * (B, 5, 1) -> sum(1) -> (B, 256)
        weighted_sum = torch.sum(attn_out * attn_weights.unsqueeze(-1), dim=1) # (B, 256)
        
        # Concatenate attended and mean pooled
        concat_feat = torch.cat([weighted_sum, mean_pooled], dim=1) # (B, 512)
        
        # Final classification
        final_score = self.classifier(concat_feat).squeeze(1)
        
        # Calculate entropy-based confidence
        # Using binary entropy mapping to confidence [0, 1]
        eps = 1e-7
        p = final_score.clamp(eps, 1 - eps)
        entropy = - (p * torch.log(p) + (1 - p) * torch.log(1 - p))
        max_entropy = 0.693147  # -ln(0.5)
        confidence = 1.0 - (entropy / max_entropy)
        
        return FusionOutput(
            final_score=final_score,
            attention_weights=attn_weights,
            confidence=confidence
        )

    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, path):
        torch.save({'state_dict': self.state_dict()}, path)

    @classmethod
    def load_checkpoint(cls, path, device='cpu'):
        checkpoint = torch.load(path, map_location=device)
        model = cls()
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        return model
