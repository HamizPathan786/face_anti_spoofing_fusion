import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingBCE(nn.Module):
    """
    Binary Cross Entropy with Label Smoothing.
    Target values are smoothed:
      0 -> eps
      1 -> 1.0 - eps
    """
    def __init__(self, eps=0.1):
        super(LabelSmoothingBCE, self).__init__()
        self.eps = eps
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, pred, target):
        # Target shape must match pred shape
        smoothed_target = target * (1.0 - self.eps) + (1.0 - target) * self.eps
        loss = self.bce(pred, smoothed_target)
        return loss.mean()

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance, typical in anti-spoofing
    where spoof samples might overwhelm real samples or vice versa.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        pred: (B,) - probabilities after Sigmoid
        target: (B,) - 0 or 1
        """
        # Clamp predictions to avoid log(0)
        pred = torch.clamp(pred, min=1e-7, max=1-1e-7)
        
        # Cross entropy terms
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        
        # pt is the probability of the true class
        pt = target * pred + (1 - target) * (1 - pred)
        
        # Focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weighting
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        
        # Final loss
        loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
