import torch
import time
from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class VerificationResult:
    decision: str
    stage_rejected: int
    final_score: float = 0.0
    confidence: float = 0.0
    attention_weights: List[float] = None
    per_modality_scores: Dict[str, float] = None
    explanation: str = ""

def generate_explanation(fusion_out, per_modality_scores):
    """Generates a human-readable explanation of the result based on fusion output."""
    weights = fusion_out.attention_weights.tolist() if fusion_out.attention_weights is not None else [0]*5
    modalities = ['RGB', 'Depth', 'IR', 'rPPG', 'Deepfake']
    
    # Identify the most heavily weighted modality
    max_idx = weights.index(max(weights))
    top_modality = modalities[max_idx]
    
    decision_type = "spoof" if fusion_out.final_score > 0.5 else "real"
    
    explanation = f"Network concludes the presentation is {decision_type} " \
                  f"(score: {fusion_out.final_score:.2f}). " \
                  f"The most influential modality was {top_modality} " \
                  f"(attention weight: {weights[max_idx]:.2f})."
    
    # Add flags for extreme scores
    for mod, score in per_modality_scores.items():
        if score > 0.8:
            explanation += f" Warning: {mod} showed high spoof confidence ({score:.2f})."
    
    return explanation

class CascadeController:
    """
    Implements the two-stage cascade logic bridging unimodal classifiers
    and the stage 2 fusion/deepfake network.
    """
    def __init__(self, models_dict, config):
        """
        models_dict should contain initialized models:
        'rgb', 'depth', 'ir', 'rppg', 'deepfake', 'fusion'
        """
        self.models = models_dict
        self.config = config
        self.timeout = 8.0 # seconds

    @torch.inference_mode()
    def verify(self, rgb, depth, ir, rppg, rgb_frames=None) -> VerificationResult:
        start_time = time.time()
        
        # Batch size fallback check for OOM
        # Assuming batch size 1 for verification. For batch processing, logic is similar.
        
        # Stage 1 execution
        # Forward pass returning Stage1Output(score, embedding, cascade_reject)
        rgb_out = self.models['rgb'].forward_with_cascade(rgb)
        depth_out = self.models['depth'].forward_with_cascade(depth)
        ir_out = self.models['ir'].forward_with_cascade(ir)
        rppg_out = self.models['rppg'].forward_with_cascade(rppg)
        
        # Collect modality scores
        scores = {
            'RGB': rgb_out.score,
            'Depth': depth_out.score,
            'IR': ir_out.score,
            'rPPG': rppg_out.score
        }
        
        # 1. Early cascade rejection check
        # If any modality returns confident spoof, reject immediately.
        if rgb_out.cascade_reject or depth_out.cascade_reject or \
           ir_out.cascade_reject or rppg_out.cascade_reject:
            
            suspect_modalities = [k for k, v in [
                ("RGB", rgb_out.cascade_reject),
                ("Depth", depth_out.cascade_reject),
                ("IR", ir_out.cascade_reject),
                ("rPPG", rppg_out.cascade_reject)
            ] if v]
            
            return VerificationResult(
                decision="SPOOF",
                stage_rejected=1,
                final_score=max(scores.values()),
                confidence=1.0,
                per_modality_scores=scores,
                explanation=f"Stage 1 early rejection: {', '.join(suspect_modalities)} detected high-confidence spoof attack."
            )
            
        # 2. Timeout Check
        if time.time() - start_time > self.timeout:
            return VerificationResult(decision="MANUAL_REVIEW", stage_rejected=1, explanation="Timeout exceeded at Stage 1")

        # 3. Stage 2 Execution
        # Deepfake expects standard RGB video frames, here assumed to be derived from the 'rgb' input or explicitly provided.
        # If rgb_frames is None, we use the middle frame from the batch or the first spatial frame.
        if rgb_frames is None:
            # Assumes rgb is shape (B, 3, 224, 224), deepfake requires (B, 3, 299, 299)
            # We resize rgb to match deepfake input needs
            b, c, h, w = rgb.shape
            rgb_frames = torch.nn.functional.interpolate(rgb, size=(299, 299), mode='bilinear', align_corners=False)
            
        deepfake_score, df_embedding = self.models['deepfake'](rgb_frames)
        scores['Deepfake'] = deepfake_score.squeeze().item() if deepfake_score.numel() == 1 else deepfake_score.mean().item()
        
        # Fusion network
        fusion_out = self.models['fusion'](
            rgb_out.embedding.unsqueeze(0) if rgb_out.embedding.dim() == 1 else rgb_out.embedding,
            depth_out.embedding.unsqueeze(0) if depth_out.embedding.dim() == 1 else depth_out.embedding,
            ir_out.embedding.unsqueeze(0) if ir_out.embedding.dim() == 1 else ir_out.embedding,
            rppg_out.embedding.unsqueeze(0) if rppg_out.embedding.dim() == 1 else rppg_out.embedding,
            df_embedding.unsqueeze(0) if df_embedding.dim() == 1 else df_embedding
        )

        final_score = fusion_out.final_score.squeeze().item()
        confidence = fusion_out.confidence.squeeze().item()
        attn_weights = fusion_out.attention_weights.squeeze()
        
        # 4. Final Decision logic
        threshold_real = self.config['stage2']['fusion']['final_threshold_real']
        threshold_spoof = self.config['stage2']['fusion']['final_threshold_spoof']
        
        if final_score < threshold_real:
            decision = "REAL"
        elif final_score > threshold_spoof:
            decision = "SPOOF"
        else:
            decision = "MANUAL_REVIEW"
            
        return VerificationResult(
            decision=decision,
            stage_rejected=2,
            final_score=final_score,
            confidence=confidence,
            attention_weights=attn_weights.tolist() if attn_weights.dim() else [attn_weights.item()],
            per_modality_scores=scores,
            explanation=generate_explanation(fusion_out, scores)
        )
