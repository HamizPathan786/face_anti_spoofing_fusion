import os
import yaml
import torch
from pathlib import Path
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
import logging

from data.dataset import get_dataloader
from models.stage1.rgb_model import RGBAntiSpoofNet
from models.stage1.depth_model import DepthAntiSpoofNet
from models.stage1.ir_model import IRAntiSpoofNet
from models.stage1.rppg_model import rPPGAntiSpoofNet
from training.losses import LabelSmoothingBCE
from utils.metrics import evaluate_metrics
from utils.visualization import plot_training_curves

def setup_logger(modality):
    Path("training/logs").mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(modality)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(f"training/logs/train_stage1_{modality}.log")
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

def get_model(modality, config):
    if modality == 'rgb': return RGBAntiSpoofNet()
    elif modality == 'depth': return DepthAntiSpoofNet()
    elif modality == 'ir': return IRAntiSpoofNet()
    elif modality == 'rppg': return rPPGAntiSpoofNet()
    else: raise ValueError("Unknown modality")

def train_modality(modality, config):
    logger = setup_logger(modality)
    logger.info(f"--- Starting Training for {modality.upper()} ---")
    
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    
    # Dataloaders
    processed_val = "data/processed/val"
    if not Path(processed_val).exists():
         logger.error("Validation data not found. Please run preprocessing first.")
         return
    
    train_loader = get_dataloader("data/processed/train", config['training']['batch_size'], 
                                  config['training']['num_workers'], augment=True, balance=True)
    val_loader = get_dataloader(processed_val, config['training']['batch_size'], 
                                config['training']['num_workers'], augment=False, balance=False)
                                
    # Model
    model = get_model(modality, config).to(device)
    logger.info(f"Model parameters: {model.get_parameter_count():,}")
    
    # Optimizer & Scheduler & Loss
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)
    criterion = LabelSmoothingBCE(eps=0.1)
    
    # Mixed precision
    use_amp = config['training']['mixed_precision'] and device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    
    epochs = config['stage1'][modality].get('epochs', 50)
    patience = config['training']['early_stopping_patience']
    
    best_val_auc = 0.0
    patience_counter = 0
    
    # Trackers
    train_losses, val_losses, train_aucs, val_aucs = [], [], [], []
    
    checkpoint_dir = Path("checkpoints/stage1")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{modality}_best.pth"
    
    # Resume
    start_epoch = 0
    if checkpoint_path.exists():
        logger.info(f"Resuming from checkpoint {checkpoint_path}")
        model = type(model).load_checkpoint(checkpoint_path, device)
        # Note: robust resume would save opt state too.
        
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []
        
        for batch in tqdm(train_loader, desc=f"{modality.upper()} Epoch {epoch+1}/{epochs} [Train]"):
            inputs = batch[modality].to(device)
            targets = batch['label'].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=use_amp):
                scores, _ = model(inputs)
                loss = criterion(scores, targets)
                
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
            
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            train_preds.extend(scores.detach().cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
            
        scheduler.step()
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_metrics = evaluate_metrics(train_targets, train_preds)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"{modality.upper()} Epoch {epoch+1}/{epochs} [Val]"):
                inputs = batch[modality].to(device)
                targets = batch['label'].to(device)
                
                with autocast(enabled=use_amp):
                    scores, _ = model(inputs)
                    loss = criterion(scores, targets)
                    
                val_loss += loss.item() * inputs.size(0)
                val_preds.extend(scores.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
                
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
            patience_counter = 0
            model.save_checkpoint(checkpoint_path)
            logger.info(f"--> Saved new best model with AUC: {best_val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
    # Plotting
    plot_path = f"training/plots/stage1_{modality}_curves.png"
    plot_training_curves(train_losses, val_losses, train_aucs, val_aucs, 'AUC', plot_path)
    logger.info(f"Training completed for {modality.upper()}.")

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    modalities = ['rgb', 'depth', 'ir', 'rppg']
    for mod in modalities:
        train_modality(mod, config)

if __name__ == "__main__":
    main()
