import torch
import torch.nn.functional as F
import yaml
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from data.dataset import get_dataloader
from models.stage1.rgb_model import RGBAntiSpoofNet
from models.stage1.depth_model import DepthAntiSpoofNet
from models.stage1.ir_model import IRAntiSpoofNet
from models.stage1.rppg_model import rPPGAntiSpoofNet
from models.stage2.deepfake_detector import DeepfakeDetector
from models.stage2.fusion_network import MultiModalFusionNetwork
from models.stage2.cascade_controller import CascadeController
from utils.metrics import evaluate_metrics, print_cross_dataset_protocol
from utils.visualization import plot_roc_curve, plot_confusion_matrix, plot_attention_weights

def prepare_models(device):
    checkpoints_dir_s1 = Path("checkpoints/stage1")
    checkpoints_dir_s2 = Path("checkpoints/stage2")
    
    models = {
        'rgb': RGBAntiSpoofNet().to(device),
        'depth': DepthAntiSpoofNet().to(device),
        'ir': IRAntiSpoofNet().to(device),
        'rppg': rPPGAntiSpoofNet().to(device),
        'deepfake': DeepfakeDetector().to(device),
        'fusion': MultiModalFusionNetwork().to(device)
    }
    
    # Load weights if available, otherwise initialized randomly
    try:
        if (checkpoints_dir_s1 / "rgb_best.pth").exists():
            models['rgb'] = RGBAntiSpoofNet.load_checkpoint(checkpoints_dir_s1 / "rgb_best.pth", device)
            models['depth'] = DepthAntiSpoofNet.load_checkpoint(checkpoints_dir_s1 / "depth_best.pth", device)
            models['ir'] = IRAntiSpoofNet.load_checkpoint(checkpoints_dir_s1 / "ir_best.pth", device)
            models['rppg'] = rPPGAntiSpoofNet.load_checkpoint(checkpoints_dir_s1 / "rppg_best.pth", device)
            
            models['deepfake'] = DeepfakeDetector.load_checkpoint(checkpoints_dir_s2 / "deepfake_best.pth", device)
            models['fusion'] = MultiModalFusionNetwork.load_checkpoint(checkpoints_dir_s2 / "fusion_best.pth", device)
    except Exception as e:
        print(f"Warning: Could not load some checkpoints. Ensure models are trained. Error: {e}")
        
    for m in models.values():
        m.eval()
        
    return models

def evaluate():
    print_cross_dataset_protocol()
    print("Starting full evaluation suite...")
    
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    
    test_loader = get_dataloader("data/processed/test", config['training']['batch_size'], config['training']['num_workers'], augment=False, balance=False)
    
    models = prepare_models(device)
    cascade = CascadeController(models, config)
    
    all_targets = []
    all_final_scores = []
    attention_weights_all = []
    attack_breakdown = {}
    
    stage1_rejects = 0
    total_samples = 0
    
    # We will simulate the cascade controller functionality natively here to log everything explicitly
    # But using the models rather than the controller directly for deep analysis
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Test Set"):
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            ir = batch['ir'].to(device)
            rppg = batch['rppg'].to(device)
            targets = batch['label'].to(device)
            attack_types = batch['attack_type']
            
            for i in range(len(rgb)):
                # Evaluate per sample
                target = targets[i].item()
                att_type = attack_types[i]
                
                rgb_s = rgb[i:i+1]
                depth_s = depth[i:i+1]
                ir_s = ir[i:i+1]
                rppg_s = rppg[i:i+1]
                
                res = cascade.verify(rgb_s, depth_s, ir_s, rppg_s)
                
                total_samples += 1
                if res.stage_rejected == 1:
                    stage1_rejects += 1
                    
                all_targets.append(target)
                all_final_scores.append(res.final_score)
                
                if res.attention_weights:
                    attention_weights_all.append(res.attention_weights)
                    
                if att_type not in attack_breakdown:
                    attack_breakdown[att_type] = {'targets': [], 'scores': []}
                attack_breakdown[att_type]['targets'].append(target)
                attack_breakdown[att_type]['scores'].append(res.final_score)
                
    # Calculate Overall Metrics
    metrics = evaluate_metrics(all_targets, all_final_scores, threshold=0.5)
    
    cascade_efficiency = (stage1_rejects / total_samples) * 100 if total_samples > 0 else 0
    
    print("\n--- Final Evaluation Results ---")
    print(f"HTER:  {metrics['hter']:.4f}")
    print(f"EER:   {metrics['eer']:.4f} (Thresh: {metrics['eer_threshold']:.4f})")
    print(f"AUC:   {metrics['auc']:.4f}")
    print(f"ACER:  {metrics['acer']:.4f}")
    print(f"APCER: {metrics['apcer']:.4f}")
    print(f"BPCER: {metrics['bpcer']:.4f}")
    print(f"Stage 1 Cascade Rejection Rate: {cascade_efficiency:.2f}%")
    
    # Per-attack breakdown
    print("\n--- Per-Attack Modality Breakdown ---")
    attack_metrics = {}
    for att_type, data in attack_breakdown.items():
        if att_type == 'none': continue # Real face doesn't have an attack type that makes sense to EER against reals normally
        # For attack breakdown, we create a binary classification problem: Reals vs this specific Attack
        # So we combine Reals (all) and this specific attack
        real_targets = [t for t in all_targets if t == 0]
        real_scores = [all_final_scores[i] for i, t in enumerate(all_targets) if t == 0]
        
        subset_targets = real_targets + data['targets']
        subset_scores = real_scores + data['scores']
        
        m = evaluate_metrics(subset_targets, subset_scores)
        attack_metrics[att_type] = {'hter': m['hter'], 'eer': m['eer']}
        print(f"Attack: {att_type.upper():<15} | HTER: {m['hter']:.4f} | EER: {m['eer']:.4f}")
        
    # Save Report
    report = {
        'overall_metrics': metrics,
        'cascade_efficiency_percent': cascade_efficiency,
        'per_attack_metrics': attack_metrics
    }
    
    out_dir = Path("evaluation/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eval_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    # Plots
    plot_roc_curve(all_targets, all_final_scores, "evaluation/plots/roc_curve.png")
    plot_confusion_matrix(metrics['cm'], "evaluation/plots/confusion_matrix.png")
    if attention_weights_all:
        plot_attention_weights(attention_weights_all, "evaluation/plots/attention_weights.png")
        
    print("\nEvaluation report and plots saved to evaluation/ directory.")

if __name__ == "__main__":
    evaluate()
