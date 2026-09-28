"""
Evaluation metrics for link prediction.
"""
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

def compute_metrics(y_true, y_pred_prob, k=10):
    """
    Computes standard classification and ranking metrics.
    """
    y_pred_binary = (y_pred_prob >= 0.5).astype(int)
    
    auc = roc_auc_score(y_true, y_pred_prob)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    # Precision@K
    if len(y_pred_prob) >= k:
        top_k_idx = np.argsort(y_pred_prob)[-k:]
        prec_at_k = np.mean(y_true[top_k_idx])
    else:
        prec_at_k = 0.0
        
    return {
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        f"Precision@{k}": prec_at_k
    }
