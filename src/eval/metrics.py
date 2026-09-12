import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, brier_score_loss

def compute_classification_metrics(y_true, y_pred, y_probs=None):
    """
    Computes accuracy, macro F1, weighted F1, and Brier score.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    results = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm.tolist()
    }
    
    if y_probs is not None:
        # Multi-class Brier score: 1/N \sum_i \sum_k (p_ik - y_ik)^2
        num_classes = y_probs.shape[1]
        y_true_onehot = np.eye(num_classes)[y_true]
        brier = np.mean(np.sum((y_probs - y_true_onehot) ** 2, axis=1))
        results["brier_score"] = float(brier)
        
    return results

def compute_ece(probs, labels, n_bins=15):
    """
    Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    probs: (N, num_classes) numpy array or torch tensor of predicted probabilities.
    labels: (N,) ground truth class indices.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
        
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    
    bin_stats = []
    
    for bin_idx in range(n_bins):
        bin_lower = bin_boundaries[bin_idx]
        bin_upper = bin_boundaries[bin_idx + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            abs_diff = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += abs_diff * prop_in_bin
            mce = max(mce, abs_diff)
            
            bin_stats.append({
                "bin_idx": bin_idx,
                "bin_range": [float(bin_lower), float(bin_upper)],
                "accuracy": float(accuracy_in_bin),
                "confidence": float(avg_confidence_in_bin),
                "prop": float(prop_in_bin),
                "count": int(np.sum(in_bin))
            })
        else:
            bin_stats.append({
                "bin_idx": bin_idx,
                "bin_range": [float(bin_lower), float(bin_upper)],
                "accuracy": 0.0,
                "confidence": 0.0,
                "prop": 0.0,
                "count": 0
            })
            
    return {
        "ece": float(ece),
        "mce": float(mce),
        "bin_stats": bin_stats
    }

def compute_entropy(probs, eps=1e-12):
    """
    Computes Shannon predictive entropy: H(p) = - \sum p * log(p).
    probs: (N, C)
    returns: (N,)
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    probs = np.clip(probs, eps, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    return entropy
