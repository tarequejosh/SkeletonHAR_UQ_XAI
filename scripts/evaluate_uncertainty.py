import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy import stats

from src.data.utd_dataset import get_utd_loaders
from src.models.stgcn import STGCN
from src.uncertainty.mc_dropout import MCDropoutInference
from src.eval.metrics import compute_classification_metrics, compute_ece, compute_entropy

def evaluate_uncertainty(
    checkpoint_path="outputs/checkpoints/stgcn_utd_baseline.pt",
    config_path="configs/utd_baseline.yaml",
    num_passes=25,
    dropout_rate=0.3
):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Instantiate model with dropout enabled for MC-Dropout
    model = STGCN(
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["num_classes"],
        num_nodes=config["model"]["num_nodes"],
        dropout=dropout_rate,
        channel_plan=config["model"]["channel_plan"]
    ).to(device)
    
    # Load weights (flexible for dropout parameter compatibility)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    
    _, test_loader = get_utd_loaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=2,
        augment=False
    )
    
    print("\n--- 1. Deterministic Point Prediction (Standard Softmax) ---")
    all_det_logits = []
    all_targets = []
    
    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.to(device)
            logits = model(x)
            all_det_logits.append(logits.cpu())
            all_targets.append(y)
            
    det_logits = torch.cat(all_det_logits, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    
    det_probs = torch.softmax(torch.from_numpy(det_logits), dim=1).numpy()
    det_preds = np.argmax(det_probs, axis=1)
    det_confs = np.max(det_probs, axis=1)
    det_entropy = compute_entropy(det_probs)
    
    det_metrics = compute_classification_metrics(targets, det_preds, det_probs)
    det_ece = compute_ece(det_probs, targets)
    det_metrics["ece"] = det_ece["ece"]
    print(f"Deterministic: Accuracy = {det_metrics['accuracy']*100:.2f}%, "
          f"ECE = {det_metrics['ece']*100:.2f}%, Brier = {det_metrics['brier_score']:.4f}")

    print(f"\n--- 2. MC-Dropout Uncertainty Inference (N={num_passes} passes, p={dropout_rate}) ---")
    mc_engine = MCDropoutInference(model, num_passes=num_passes, device=device)
    mc_results = mc_engine.predict_loader(test_loader)
    
    mc_mean_probs = mc_results["mean_probs"]
    mc_preds = np.argmax(mc_mean_probs, axis=1)
    mc_confs = np.max(mc_mean_probs, axis=1)
    mc_entropy = mc_results["pred_entropy"]
    mc_mi = mc_results["mutual_info"]
    mc_var = mc_results["total_variance"]
    
    mc_metrics = compute_classification_metrics(targets, mc_preds, mc_mean_probs)
    mc_ece = compute_ece(mc_mean_probs, targets)
    mc_metrics["ece"] = mc_ece["ece"]
    print(f"MC-Dropout: Accuracy = {mc_metrics['accuracy']*100:.2f}%, "
          f"ECE = {mc_metrics['ece']*100:.2f}%, Brier = {mc_metrics['brier_score']:.4f}")
    
    # 3. Uncertainty vs Error Correlation Analysis
    is_correct = (mc_preds == targets)
    correct_entropy = mc_entropy[is_correct]
    error_entropy = mc_entropy[~is_correct]
    
    ttest_stat, ttest_pval = stats.ttest_ind(error_entropy, correct_entropy, equal_var=False)
    mw_stat, mw_pval = stats.mannwhitneyu(error_entropy, correct_entropy, alternative='greater')
    
    print("\n--- 3. Predictive Entropy Analysis (Correct vs Misclassified) ---")
    print(f"Correct Predictions (n={len(correct_entropy)}): "
          f"Mean Entropy = {np.mean(correct_entropy):.4f} +/- {np.std(correct_entropy):.4f}")
    print(f"Misclassified Predictions (n={len(error_entropy)}): "
          f"Mean Entropy = {np.mean(error_entropy):.4f} +/- {np.std(error_entropy):.4f}")
    print(f"Mann-Whitney U test p-value: {mw_pval:.4e} (stat={mw_stat:.1f})")
    
    # Save box plot figure
    figures_dir = "outputs/figures"
    os.makedirs(figures_dir, exist_ok=True)
    box_path = os.path.join(figures_dir, "uncertainty_entropy_boxplot.png")
    
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    box_data = [correct_entropy, error_entropy]
    box_plot = ax.boxplot(box_data, patch_artist=True, labels=["Correct Predictions", "Misclassifications"],
                          medianprops=dict(color="black", lw=1.8),
                          boxprops=dict(facecolor="skyblue", color="darkblue", alpha=0.7),
                          flierprops=dict(marker='o', markersize=4, alpha=0.5))
    
    # Color error box differently
    box_plot['boxes'][1].set_facecolor("salmon")
    box_plot['boxes'][1].set_edgecolor("darkred")
    
    ax.set_ylabel("Predictive Entropy H(p)", fontsize=12, fontweight='bold')
    ax.set_title(f"Uncertainty Separation: Correct vs. Error (p={mw_pval:.2e})", fontsize=13, pad=12)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(box_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved entropy boxplot to: {box_path}")
    
    # Save results json
    results_dir = "outputs/results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "phase3_mc_dropout.json")
    
    summary = {
        "num_passes": num_passes,
        "dropout_rate": dropout_rate,
        "deterministic_metrics": {
            "accuracy": det_metrics["accuracy"],
            "macro_f1": det_metrics["macro_f1"],
            "weighted_f1": det_metrics["weighted_f1"],
            "ece": det_metrics["ece"],
            "brier_score": det_metrics["brier_score"]
        },
        "mc_dropout_metrics": {
            "accuracy": mc_metrics["accuracy"],
            "macro_f1": mc_metrics["macro_f1"],
            "weighted_f1": mc_metrics["weighted_f1"],
            "ece": mc_metrics["ece"],
            "brier_score": mc_metrics["brier_score"]
        },
        "entropy_separation": {
            "correct_mean": float(np.mean(correct_entropy)),
            "correct_std": float(np.std(correct_entropy)),
            "error_mean": float(np.mean(error_entropy)),
            "error_std": float(np.std(error_entropy)),
            "mann_whitney_p_value": float(mw_pval)
        },
        "predictions": {
            "targets": targets.tolist(),
            "det_preds": det_preds.tolist(),
            "det_confs": det_confs.tolist(),
            "det_entropy": det_entropy.tolist(),
            "mc_preds": mc_preds.tolist(),
            "mc_confs": mc_confs.tolist(),
            "mc_entropy": mc_entropy.tolist(),
            "mc_mutual_info": mc_mi.tolist(),
            "mc_total_variance": mc_var.tolist()
        }
    }
    
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved Phase 3 MC-Dropout results to {results_file}")
    
    return summary

if __name__ == "__main__":
    evaluate_uncertainty()
