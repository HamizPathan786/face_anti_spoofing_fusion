import numpy as np
import scipy.interpolate
from sklearn.metrics import roc_curve, auc, confusion_matrix

def compute_eer(y_true, y_score):
    """
    Compute Equal Error Rate (EER).
    Returns EER and the corresponding threshold.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    
    # Ideally EER is where FPR == FNR. 
    # Since thresholds are discrete, we find the index where they cross.
    eer_index = np.nanargmin(np.absolute((fnr - fpr)))
    
    # Precise EER using interpolation for continuous theoretical curve
    if len(fpr) > 1 and len(fnr) > 1:
        eer = scipy.interpolate.interp1d(fpr - fnr, fpr)(0.0)
    else:
        eer = fpr[eer_index]
        
    threshold = thresholds[eer_index]
    return float(eer), float(threshold)

def compute_hter(fpr, fnr):
    """
    Half Total Error Rate.
    """
    return (fpr + fnr) / 2.0

def evaluate_metrics(y_true, y_score, threshold=0.5):
    """
    Computes all standard anti-spoofing metrics:
    APCER: Attack Presentation Classification Error Rate (False Acceptance Rate for Fakes)
    BPCER: Bona Fide Presentation Classification Error Rate (False Rejection Rate for Reals)
    ACER: Average Classification Error Rate
    EER: Equal Error Rate
    HTER: Half Total Error Rate evaluated at threshold
    AUC: Area Under ROC Curve
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = (y_score >= threshold).astype(int)
    
    # Handle edge case where only one class is present in batch
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
        roc_auc = auc(fpr, tpr)
        eer, eer_thresh = compute_eer(y_true, y_score)
    else:
        roc_auc = 0.5
        eer, eer_thresh = 0.5, 0.5
    
    # Confusion Matrix
    # cm: [[TN, FP], [FN, TP]] -> index 0=real, 1=spoof
    # Wait, usually y=1 is spoof. Positives=Spoof, Negatives=Real.
    # TN: Real classified as Real (Bona Fide accepted)
    # FP: Real classified as Spoof (Bona Fide rejected) -> BPCER (False Rejection of Real)
    # FN: Spoof classified as Real (Attack accepted) -> APCER (False Acceptance of Spoof)
    # TP: Spoof classified as Spoof (Attack rejected)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Failsafe if only one class exists
        tn, fp, fn, tp = 0, 0, 0, 0
        if y_true[0] == 0:
            if y_pred[0] == 0: tn = len(y_true)
            else: fp = len(y_true)
        else:
            if y_pred[0] == 1: tp = len(y_true)
            else: fn = len(y_true)
        
    # Attack Presentation Classification Error Rate
    total_spoof = tp + fn
    apcer = fn / total_spoof if total_spoof > 0 else 0.0
    
    # Bona Fide Presentation Classification Error Rate
    total_real = tn + fp
    bpcer = fp / total_real if total_real > 0 else 0.0
    
    # Average Classification Error Rate
    acer = (apcer + bpcer) / 2.0
    
    # HTER at evaluation threshold (different from EER which finds optimal threshold)
    hter = compute_hter(fp / total_real if total_real > 0 else 0.0, fn / total_spoof if total_spoof > 0 else 0.0)
    
    acc = (tp + tn) / (total_spoof + total_real)
    
    return {
        'acc': acc,
        'apcer': apcer,
        'bpcer': bpcer,
        'acer': acer,
        'hter': hter,
        'eer': eer,
        'eer_threshold': eer_thresh,
        'auc': roc_auc,
        'cm': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
    }

def print_cross_dataset_protocol():
    """
    Prints a standard note on cross-dataset testing protocol.
    """
    print("="*60)
    print("CROSS-DATASET EVALUATION PROTOCOL NOTE:")
    print("For robust anti-spoofing validation, models must be evaluated")
    print("on datasets unseen during training (e.g., train on CASIA-FASD,")
    print("test on Replay-Attack or OULU-NPU). Avoid optimizing hyper-parameters")
    print("based on the target cross-dataset to prevent indirect leakage.")
    print("="*60)
