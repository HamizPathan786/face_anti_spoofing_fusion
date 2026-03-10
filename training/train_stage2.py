import os
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import logging
import numpy as np

from data.dataset import get_dataloader
from models.stage1.rgb_model import RGBAntiSpoofNet
from models.stage1.depth_model import DepthAntiSpoofNet
from models.stage1.ir_model import IRAntiSpoofNet
from models.stage1.rppg_model import rPPGAntiSpoofNet
from models.stage2.deepfake_detector import DeepfakeDetector
from models.stage2.fusion_network import MultiModalFusionNetwork
from training.losses import FocalLoss
from utils.metrics import evaluate_metrics
from utils.visualization import plot_training_curves, plot_attention_weights

def setup_logger():
    Path("training/logs").mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('stage2')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler("training/logs/train_stage2.log")
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

def get_stage1_models(device):
    models = {}
    checkpoints_dir = Path("checkpoints/stage1")
    
    # Initialize and load
    models['rgb'] = RGBAntiSpoofNet.load_checkpoint(checkpoints_dir / "rgb_best.pth", device) if (checkpoints_dir / "rgb_best.pth").exists() else RGBAntiSpoofNet().to(device)
    models['depth'] = DepthAntiSpoofNet.load_checkpoint(checkpoints_dir / "depth_best.pth", device) if (checkpoints_dir / "depth_best.pth").exists() else DepthAntiSpoofNet().to(device)
    models['ir'] = IRAntiSpoofNet.load_checkpoint(checkpoints_dir / "ir_best.pth", device) if (checkpoints_dir / "ir_best.pth").exists() else IRAntiSpoofNet().to(device)
    models['rppg'] = rPPGAntiSpoofNet.load_checkpoint(checkpoints_dir / "rppg_best.pth", device) if (checkpoints_dir / "rppg_best.pth").exists() else rPPGAntiSpoofNet().to(device)
    
    # Freeze
    for m in models.values():
        for param in m.parameters():
            param.requires_grad = False
        m.eval()
        
    return models

def train_stage2():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    logger = setup_logger()
    logger.info("--- Starting Training for STAGE 2 (Fusion + Deepfake) ---")
    
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    
    train_loader = get_dataloader("data/processed/train", config['training']['batch_size'], config['training']['num_workers'], augment=True, balance=True)
    val_loader = get_dataloader("data/processed/val", config['training']['batch_size'], config['training']['num_workers'], augment=False, balance=False)
    
    s1_models = get_stage1_models(device)
    logger.info("Stage 1 backbones loaded and frozen.")
    
    deepfake_det = DeepfakeDetector(embedding_dim=512).to(device)
    fusion_net = MultiModalFusionNetwork(embedding_dim=512, attention_dim=256, num_heads=8).to(device)
    
    # Trainable parameters are only from deepfake and fusion
    trainable_params = list(deepfake_det.parameters()) + list(fusion_net.parameters())
    optimizer = AdamW(trainable_params, lr=5e-5)
    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    
    use_amp = config['training']['mixed_precision'] and device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    
    epochs = 30 # From requirement
    best_val_auc = 0.0
    
    train_losses, val_losses, train_aucs, val_aucs = [], [], [], []
    attention_weights_history = []
    
    checkpoint_dir = Path("checkpoints/stage2")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # For deepfake we need 299x299 RGB. The dataset gives 224x224 RGB.
    # We will interpolate dynamically.
    def prep_df_input(rgb_tensor):
        return F.interpolate(rgb_tensor, size=(299, 299), mode='bilinear', align_corners=False)
    
    for epoch in range(epochs):
        deepfake_det.train()
        fusion_net.train()
        
        running_loss = 0.0
        train_preds, train_targets = [], []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            # inputs
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            ir = batch['ir'].to(device)
            rppg = batch['rppg'].to(device)
            targets = batch['label'].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=use_amp):
                # 1. Get frozen stage 1 embeddings
                with torch.no_grad():
                    _, e_rgb = s1_models['rgb'](rgb)
                    _, e_depth = s1_models['depth'](depth)
                    _, e_ir = s1_models['ir'](ir)
                    _, e_rppg = s1_models['rppg'](rppg)
                
                # 2. Get deepfake embedding
                df_input = prep_df_input(rgb)
                _, e_df = deepfake_det(df_input)
                
                # 3. Fusion network
                fusion_out = fusion_net(e_rgb, e_depth, e_ir, e_rppg, e_df)
                
                # 4. Loss
                loss = criterion(fusion_out.final_score, targets)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, config['training']['gradient_clip'])
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * targets.size(0)
            train_preds.extend(fusion_out.final_score.detach().cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_metrics = evaluate_metrics(train_targets, train_preds)
        
        # Validation
        deepfake_det.eval()
        fusion_net.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        epoch_attention = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                rgb = batch['rgb'].to(device)
                depth = batch['depth'].to(device)
                ir = batch['ir'].to(device)
                rppg = batch['rppg'].to(device)
                targets = batch['label'].to(device)
                
                with autocast(enabled=use_amp):
                    _, e_rgb = s1_models['rgb'](rgb)
                    _, e_depth = s1_models['depth'](depth)
                    _, e_ir = s1_models['ir'](ir)
                    _, e_rppg = s1_models['rppg'](rppg)
                    
                    df_input = prep_df_input(rgb)
                    _, e_df = deepfake_det(df_input)
                    
                    fusion_out = fusion_net(e_rgb, e_depth, e_ir, e_rppg, e_df)
                    loss = criterion(fusion_out.final_score, targets)
                    
                val_loss += loss.item() * targets.size(0)
                val_preds.extend(fusion_out.final_score.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
                epoch_attention.extend(fusion_out.attention_weights.cpu().numpy())
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_metrics = evaluate_metrics(val_targets, val_preds)
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_aucs.append(train_metrics['auc'])
        val_aucs.append(val_metrics['auc'])
        
        logger.info(f"Epoch {epoch+1}/{epochs}")
        logger.info(f"Train - Loss: {epoch_train_loss:.4f}, AUC: {train_metrics['auc']:.4f}, EER: {train_metrics['eer']:.4f}")
        logger.info(f"Val   - Loss: {epoch_val_loss:.4f}, AUC: {val_metrics['auc']:.4f}, EER: {val_metrics['eer']:.4f}")
        
        # Save Best
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            deepfake_det.save_checkpoint(checkpoint_dir / "deepfake_best.pth")
            fusion_net.save_checkpoint(checkpoint_dir / "fusion_best.pth")
            
        plot_attention_weights(epoch_attention, f"training/plots/attention_epoch_{epoch+1}.png")
        
    plot_training_curves(train_losses, val_losses, train_aucs, val_aucs, 'AUC', "training/plots/stage2_curves.png")
    logger.info("Stage 2 Training Completed.")

if __name__ == "__main__":
    train_stage2()
