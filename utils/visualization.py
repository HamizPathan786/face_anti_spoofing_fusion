import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc

def plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, metric_name='AUC', save_path=None):
    """
    Plots training and validation loss & metrics.
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Metric plot
    if train_metrics and val_metrics:
        ax2.plot(epochs, train_metrics, 'b-', label=f'Training {metric_name}')
        ax2.plot(epochs, val_metrics, 'r-', label=f'Validation {metric_name}')
        ax2.set_title(f'Training and Validation {metric_name}')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel(metric_name)
        ax2.legend()
        ax2.grid(True)
        
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_roc_curve(y_true, y_score, save_path=None):
    """
    Plots ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(cm_dict, save_path=None):
    """
    Plots confusion matrix from dict cm_dict = {'tn', 'fp', 'fn', 'tp'}
    """
    cm = np.array([
        [cm_dict['tn'], cm_dict['fp']],
        [cm_dict['fn'], cm_dict['tp']]
    ])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Spoof'], 
                yticklabels=['Real', 'Spoof'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_attention_weights(weights_list, save_path=None):
    """
    Plots the average attention weights for each modality over an epoch or testing run.
    weights_list: List of shape (B, 5) or similar.
    """
    avg_weights = np.mean(weights_list, axis=0) if len(weights_list) > 0 else np.zeros(5)
    modalities = ['RGB', 'Depth', 'IR', 'rPPG', 'Deepfake']
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(modalities, avg_weights, color=sns.color_palette("muted"))
    plt.title('Average Cross-Modal Attention Weights per Modality')
    plt.ylabel('Attention Weight')
    plt.ylim(0, 1)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom')
        
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
