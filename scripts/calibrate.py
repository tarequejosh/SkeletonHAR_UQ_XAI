import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.data.utd_dataset import UTDMHADDataset
from torch.utils.data import DataLoader
from src.models.stgcn import STGCN
from src.uncertainty.temperature_scaling import ModelWithTemperature
from src.uncertainty.conformal import AdaptivePredictionSets
from src.eval.metrics import compute_ece, compute_classification_metrics

def calibrate_and_evaluate(
    checkpoint_path="outputs/checkpoints/stgcn_utd_baseline.pt",
    config_path="configs/utd_baseline.yaml",
    cal_subject=7, # Held out from train subjects {1, 3, 5, 7}
    target_coverage=0.90
):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    model = STGCN(
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["num_classes"],
        num_nodes=config["model"]["num_nodes"],
        dropout=0.0,
        channel_plan=config["model"]["channel_plan"]
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    # Partition training data into Sub-Train (subjects 1, 3, 5) and Calibration (subject 7)
    train_dataset = UTDMHADDataset(config["data"]["data_dir"], os.path.join(config["data"]["data_dir"], "metadata.json"), split="train")
    test_dataset = UTDMHADDataset(config["data"]["data_dir"], os.path.join(config["data"]["data_dir"], "metadata.json"), split="test")
    
    cal_indices = [i for i, m in enumerate(train_dataset.meta) if m.get("subject") == cal_subject]
    print(f"Calibration split: Subject {cal_subject} ({len(cal_indices)} samples)")
    
    # Extract logits for calibration set
    cal_sub_dataset = torch.utils.data.Subset(train_dataset, cal_indices)
    cal_loader = DataLoader(cal_sub_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    def get_logits_and_labels(loader):
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for x, y, _ in loader:
                x = x.to(device)
                logits = model(x)
                all_logits.append(logits.cpu())
                all_labels.append(y)
        return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)

    cal_logits, cal_labels = get_logits_and_labels(cal_loader)
    test_logits, test_labels = get_logits_and_labels(test_loader)
    
    # --- 1. Temperature Scaling ---
    print("\n--- 1. Fitting Temperature Scaling ---")
    ts_model = ModelWithTemperature()
    optimal_temp, before_nll, after_nll = ts_model.fit(cal_logits, cal_labels)
    
    # Evaluate calibration before vs after on Test set
    test_uncal_probs = F.softmax(test_logits, dim=1).numpy()
    test_cal_probs = ts_model.predict_probs(test_logits)
    test_labels_np = test_labels.numpy()
    
    ece_before = compute_ece(test_uncal_probs, test_labels_np)
    ece_after = compute_ece(test_cal_probs, test_labels_np)
    
    print(f"Test ECE Before Scaling: {ece_before['ece']*100:.2f}% | After Scaling (T={optimal_temp:.3f}): {ece_after['ece']*100:.2f}%")
    print(f"Test MCE Before Scaling: {ece_before['mce']*100:.2f}% | After Scaling: {ece_after['mce']*100:.2f}%")
    
    # Plot side-by-side reliability diagrams
    figures_dir = "outputs/figures"
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = os.path.join(figures_dir, "reliability_diagram_pre_post.png")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Before scaling
    bins_b = ece_before["bin_stats"]
    centers_b = [(b["bin_range"][0] + b["bin_range"][1]) / 2.0 for b in bins_b if b["count"] > 0]
    widths_b = [b["bin_range"][1] - b["bin_range"][0] for b in bins_b if b["count"] > 0]
    accs_b = [b["accuracy"] for b in bins_b if b["count"] > 0]
    confs_b = [b["confidence"] for b in bins_b if b["count"] > 0]
    
    ax1.plot([0, 1], [0, 1], "--", color="gray", label="Optimal")
    ax1.bar(centers_b, accs_b, width=widths_b, alpha=0.7, color="royalblue", edgecolor="black", label="Accuracy")
    for c, a, w, conf in zip(centers_b, accs_b, widths_b, confs_b):
        if conf > a:
            ax1.bar(c, conf - a, bottom=a, width=w, alpha=0.3, color="crimson", edgecolor="red", hatch="//", label="Overconfident Gap" if c == centers_b[0] else "")
    ax1.set_title(f"Uncalibrated (ECE = {ece_before['ece']*100:.2f}%)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Confidence", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # After scaling
    bins_a = ece_after["bin_stats"]
    centers_a = [(b["bin_range"][0] + b["bin_range"][1]) / 2.0 for b in bins_a if b["count"] > 0]
    widths_a = [b["bin_range"][1] - b["bin_range"][0] for b in bins_a if b["count"] > 0]
    accs_a = [b["accuracy"] for b in bins_a if b["count"] > 0]
    confs_a = [b["confidence"] for b in bins_a if b["count"] > 0]
    
    ax2.plot([0, 1], [0, 1], "--", color="gray", label="Optimal")
    ax2.bar(centers_a, accs_a, width=widths_a, alpha=0.7, color="forestgreen", edgecolor="black", label="Accuracy")
    for c, a, w, conf in zip(centers_a, accs_a, widths_a, confs_a):
        if conf > a:
            ax2.bar(c, conf - a, bottom=a, width=w, alpha=0.3, color="crimson", edgecolor="red", hatch="//")
    ax2.set_title(f"Temperature Scaled T={optimal_temp:.2f} (ECE = {ece_after['ece']*100:.2f}%)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Confidence", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.suptitle("Reliability Diagrams: Before vs. After Calibration", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved pre/post reliability diagrams to: {fig_path}")

    # --- 2. Conformal Prediction (APS) ---
    print(f"\n--- 2. Conformal Prediction (APS, Target Coverage={target_coverage*100:.0f}%) ---")
    cal_probs = ts_model.predict_probs(cal_logits)
    aps = AdaptivePredictionSets(alpha=1.0 - target_coverage)
    q_hat = aps.calibrate(cal_probs, cal_labels.numpy())
    
    conformal_results = aps.evaluate_coverage(test_cal_probs, test_labels_np)
    print(f"Target Coverage: {conformal_results['target_coverage']*100:.1f}%")
    print(f"Empirical Test Coverage: {conformal_results['empirical_coverage']*100:.2f}%")
    print(f"Average Prediction Set Size: {conformal_results['avg_set_size']:.2f}")
    print(f"Singleton Proportion: {conformal_results['singleton_proportion']*100:.2f}%")
    
    # Save results json
    results_dir = "outputs/results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "phase4_calibration.json")
    
    output_data = {
        "temperature_scaling": {
            "optimal_temperature": float(optimal_temp),
            "cal_nll_before": float(before_nll),
            "cal_nll_after": float(after_nll),
            "test_ece_before": float(ece_before["ece"]),
            "test_ece_after": float(ece_after["ece"]),
            "test_mce_before": float(ece_before["mce"]),
            "test_mce_after": float(ece_after["mce"])
        },
        "conformal_prediction": {
            "target_coverage": float(conformal_results["target_coverage"]),
            "empirical_coverage": float(conformal_results["empirical_coverage"]),
            "avg_set_size": float(conformal_results["avg_set_size"]),
            "singleton_proportion": float(conformal_results["singleton_proportion"]),
            "conformal_quantile_q_hat": float(q_hat)
        }
    }
    
    with open(results_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved Phase 4 calibration results to {results_file}")
    return output_data

if __name__ == "__main__":
    calibrate_and_evaluate()
