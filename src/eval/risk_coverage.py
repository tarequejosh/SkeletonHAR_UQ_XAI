import numpy as np
import matplotlib.pyplot as plt
import os

def compute_risk_coverage_curve(confidences, predictions, targets, num_thresholds=100):
    """
    Computes selective prediction risk-coverage trade-off.
    confidences: (N,) confidence scores used for deferral (higher = more confident)
    predictions: (N,) predicted class labels
    targets: (N,) true class labels
    returns:
      coverages: (K,) array of coverage fractions
      risks: (K,) array of selective risks (error rates)
      accuracies: (K,) array of selective accuracies
      aurc: float, Area Under Risk-Coverage Curve
    """
    N = len(targets)
    # Sort samples in descending order of confidence
    sorted_order = np.argsort(-confidences)
    sorted_preds = predictions[sorted_order]
    sorted_targets = targets[sorted_order]
    
    # Correctness indicator (1 if correct, 0 if error)
    is_correct = (sorted_preds == sorted_targets).astype(float)
    is_error = 1.0 - is_correct
    
    # Cumulative sums as coverage increases from 1 to N samples
    k_samples = np.arange(1, N + 1)
    coverages = k_samples / float(N)
    
    cum_errors = np.cumsum(is_error)
    risks = cum_errors / k_samples
    accuracies = 1.0 - risks
    
    # Area Under Risk-Coverage Curve (AURC)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    aurc = trapz_fn(risks, coverages)
    
    return {
        "coverages": coverages.tolist(),
        "risks": risks.tolist(),
        "accuracies": accuracies.tolist(),
        "aurc": float(aurc)
    }

def plot_risk_coverage_comparison(
    curve_dict, 
    save_path="outputs/figures/risk_coverage_curve.png"
):
    """
    Plots comparative Risk-Coverage curves for multiple confidence methods.
    curve_dict: dict mapping method_name -> result_dict from compute_risk_coverage_curve
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a"]
    for idx, (method_name, res) in enumerate(curve_dict.items()):
        cov = np.array(res["coverages"])
        risk = np.array(res["risks"])
        aurc = res["aurc"]
        color = colors[idx % len(colors)]
        ax.plot(cov * 100, risk * 100, label=f"{method_name} (AURC: {aurc*100:.2f}%)", lw=2.2, color=color)
        
    ax.set_xlabel("Coverage (%) [Fraction of Retained Predictions]", fontsize=12, fontweight='bold')
    ax.set_ylabel("Selective Risk (%) [Error on Retained Subset]", fontsize=12, fontweight='bold')
    ax.set_title("Selective Prediction: Risk-Coverage Trade-off", fontsize=14, pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    ax.set_xlim(10, 100)
    ax.set_ylim(0, max([max(res["risks"][:int(len(res["risks"])*0.9)]) * 100 for res in curve_dict.values()] + [25]))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Risk-Coverage curve plot to: {save_path}")
