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
from src.eval.metrics import compute_ece

def evaluate_calibration_pipeline(model, cal_loader, test_loader, device, target_coverage=0.90, randomized=True):
    model.eval()
    
    def get_logits_and_labels(loader):
        all_logits, all_labels = [], []
        with torch.no_grad():
            for x, y, _ in loader:
                x = x.to(device)
                logits = model(x)
                all_logits.append(logits.cpu())
                all_labels.append(y)
        return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)

    cal_logits, cal_labels = get_logits_and_labels(cal_loader)
    test_logits, test_labels = get_logits_and_labels(test_loader)
    
    # 1. Temperature Scaling
    ts_model = ModelWithTemperature()
    optimal_temp, before_nll, after_nll = ts_model.fit(cal_logits, cal_labels)
    
    test_uncal_probs = F.softmax(test_logits, dim=1).numpy()
    test_cal_probs = ts_model.predict_probs(test_logits)
    cal_cal_probs = ts_model.predict_probs(cal_logits)
    test_labels_np = test_labels.numpy()
    cal_labels_np = cal_labels.numpy()
    
    ece_before = compute_ece(test_uncal_probs, test_labels_np)
    ece_after = compute_ece(test_cal_probs, test_labels_np)
    
    # 2. Conformal Prediction (Adaptive Prediction Sets / APS)
    aps = AdaptivePredictionSets(alpha=1.0 - target_coverage, randomized=randomized, seed=42)
    q_hat = aps.calibrate(cal_cal_probs, cal_labels_np)
    cov_dict = aps.evaluate_coverage(test_cal_probs, test_labels_np)
    
    return {
        "cal_samples_n": int(len(cal_labels_np)),
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
            "target_coverage": float(cov_dict["target_coverage"]),
            "empirical_coverage": float(cov_dict["empirical_coverage"]),
            "avg_set_size_overall": float(cov_dict["avg_set_size"]),
            "avg_set_size_correct": float(cov_dict["avg_set_size_correct"]),
            "avg_set_size_misclass": float(cov_dict["avg_set_size_misclass"]),
            "singleton_proportion": float(cov_dict["singleton_proportion"]),
            "conformal_quantile_q_hat": float(q_hat),
            "randomized": randomized
        },
        "ece_before_obj": ece_before,
        "ece_after_obj": ece_after
    }

def run_calibration_study(config_path="configs/utd_baseline.yaml", target_coverage=0.90):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = config["data"]["data_dir"]
    meta_path = os.path.join(data_dir, "metadata.json")
    
    test_dataset = UTDMHADDataset(data_dir, meta_path, split="test")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # -------------------------------------------------------------
    # 1. Strategy A: Original Single-Subject Holdout (Subject 7)
    # Model: stgcn_utd_baseline.pt
    # -------------------------------------------------------------
    print("\n[Strategy A] Evaluating Single-Subject Holdout (Subject 7, n=108)...")
    ckpt_base = torch.load("outputs/checkpoints/stgcn_utd_baseline.pt", map_location=device)
    model_base = STGCN(
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["num_classes"],
        num_nodes=config["model"]["num_nodes"],
        dropout=0.0,
        channel_plan=config["model"]["channel_plan"]
    ).to(device)
    model_base.load_state_dict(ckpt_base["model_state_dict"])
    
    train_full_dataset = UTDMHADDataset(data_dir, meta_path, split="train")
    cal_subj7_indices = [i for i, m in enumerate(train_full_dataset.meta) if m.get("subject") == 7]
    cal_subj7_loader = DataLoader(torch.utils.data.Subset(train_full_dataset, cal_subj7_indices), batch_size=32, shuffle=False)
    
    res_a_det = evaluate_calibration_pipeline(model_base, cal_subj7_loader, test_loader, device, target_coverage=target_coverage, randomized=False)
    res_a_rand = evaluate_calibration_pipeline(model_base, cal_subj7_loader, test_loader, device, target_coverage=target_coverage, randomized=True)
    
    # -------------------------------------------------------------
    # 2. Strategy B: Stratified Multi-Subject Slice (Subjects 1,3,5,7)
    # Model: stgcn_utd_subtrain.pt (Zero leakage)
    # -------------------------------------------------------------
    print("\n[Strategy B] Evaluating Stratified Multi-Subject Slice (Subjects 1,3,5,7, n=108)...")
    ckpt_sub_path = "outputs/checkpoints/stgcn_utd_subtrain.pt"
    if not os.path.exists(ckpt_sub_path):
        from scripts.train_subtrain_baseline import train_subtrain
        train_subtrain(config_path)
        
    ckpt_sub = torch.load(ckpt_sub_path, map_location=device)
    model_sub = STGCN(
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["num_classes"],
        num_nodes=config["model"]["num_nodes"],
        dropout=0.0,
        channel_plan=config["model"]["channel_plan"]
    ).to(device)
    model_sub.load_state_dict(ckpt_sub["model_state_dict"])
    
    cal_strat_dataset = UTDMHADDataset(data_dir, meta_path, split="cal_stratified")
    cal_strat_loader = DataLoader(cal_strat_dataset, batch_size=32, shuffle=False)
    
    res_b_det = evaluate_calibration_pipeline(model_sub, cal_strat_loader, test_loader, device, target_coverage=target_coverage, randomized=False)
    res_b_rand = evaluate_calibration_pipeline(model_sub, cal_strat_loader, test_loader, device, target_coverage=target_coverage, randomized=True)
    
    # Side-by-side comparison table
    print("\n" + "=" * 90)
    print("               CONFORMAL PREDICTION CALIBRATION STRATEGY COMPARISON")
    print("=" * 90)
    print(f"{'Metric':<32} | {'Strategy A (Single-Subj)':<25} | {'Strategy B (Stratified)':<25}")
    print("-" * 90)
    print(f"{'Calibration Set Size (n)':<32} | {res_a_rand['cal_samples_n']:<25} | {res_b_rand['cal_samples_n']:<25}")
    print(f"{'Data Leakage Risk':<32} | {'High (In Baseline Train)':<25} | {'Zero (Held out)':<25}")
    print(f"{'Optimal Temperature (T)':<32} | {res_a_rand['temperature_scaling']['optimal_temperature']:<25.4f} | {res_b_rand['temperature_scaling']['optimal_temperature']:<25.4f}")
    print(f"{'Test ECE Before Scaling':<32} | {res_a_rand['temperature_scaling']['test_ece_before']*100:<24.2f}% | {res_b_rand['temperature_scaling']['test_ece_before']*100:<24.2f}%")
    print(f"{'Test ECE After Scaling':<32} | {res_a_rand['temperature_scaling']['test_ece_after']*100:<24.2f}% | {res_b_rand['temperature_scaling']['test_ece_after']*100:<24.2f}%")
    print(f"{'Conformal Quantile (q_hat)':<32} | {res_a_rand['conformal_prediction']['conformal_quantile_q_hat']:<25.4f} | {res_b_rand['conformal_prediction']['conformal_quantile_q_hat']:<25.4f}")
    print(f"{'Target Nominal Coverage':<32} | {res_b_rand['conformal_prediction']['target_coverage']*100:<24.1f}% | {res_b_rand['conformal_prediction']['target_coverage']*100:<24.1f}%")
    print(f"{'Empirical Coverage (Randomized)':<32} | {res_a_rand['conformal_prediction']['empirical_coverage']*100:<24.2f}% | {res_b_rand['conformal_prediction']['empirical_coverage']*100:<24.2f}%")
    print(f"{'Empirical Coverage (Deterministic)':<32} | {res_a_det['conformal_prediction']['empirical_coverage']*100:<24.2f}% | {res_b_det['conformal_prediction']['empirical_coverage']*100:<24.2f}%")
    print(f"{'Average Set Size (Overall)':<32} | {res_a_rand['conformal_prediction']['avg_set_size_overall']:<25.2f} | {res_b_rand['conformal_prediction']['avg_set_size_overall']:<25.2f}")
    print(f"{'Avg Set Size (Correct)':<32} | {res_a_rand['conformal_prediction']['avg_set_size_correct']:<25.2f} | {res_b_rand['conformal_prediction']['avg_set_size_correct']:<25.2f}")
    print(f"{'Avg Set Size (Misclassified)':<32} | {res_a_rand['conformal_prediction']['avg_set_size_misclass']:<25.2f} | {res_b_rand['conformal_prediction']['avg_set_size_misclass']:<25.2f}")
    print("=" * 90)
    
    # Save to phase4_calibration.json
    results_dir = "outputs/results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "phase4_calibration.json")
    
    output_data = {
        "strategy_a_single_subject_holdout": {
            "description": "Original calibration strategy: Subject 7 held out from training pool, evaluated on baseline model.",
            "cal_samples_n": res_a_rand["cal_samples_n"],
            "temperature_scaling": res_a_rand["temperature_scaling"],
            "conformal_prediction_randomized": res_a_rand["conformal_prediction"],
            "conformal_prediction_deterministic": res_a_det["conformal_prediction"]
        },
        "strategy_b_stratified_multisubject": {
            "description": "Corrected calibration strategy: Stratified 25% slice across subjects 1, 3, 5, 7 held out completely from backbone training.",
            "cal_samples_n": res_b_rand["cal_samples_n"],
            "temperature_scaling": res_b_rand["temperature_scaling"],
            "conformal_prediction_randomized": res_b_rand["conformal_prediction"],
            "conformal_prediction_deterministic": res_b_det["conformal_prediction"]
        },
        "side_by_side_comparison": {
            "target_coverage": target_coverage,
            "strategy_a_empirical_coverage": res_a_rand["conformal_prediction"]["empirical_coverage"],
            "strategy_b_empirical_coverage": res_b_rand["conformal_prediction"]["empirical_coverage"],
            "strategy_a_avg_set_size": res_a_rand["conformal_prediction"]["avg_set_size_overall"],
            "strategy_b_avg_set_size": res_b_rand["conformal_prediction"]["avg_set_size_overall"],
            "strategy_b_set_size_correct": res_b_rand["conformal_prediction"]["avg_set_size_correct"],
            "strategy_b_set_size_misclass": res_b_rand["conformal_prediction"]["avg_set_size_misclass"]
        },
        # Primary reference keys
        "temperature_scaling": res_b_rand["temperature_scaling"],
        "conformal_prediction": res_b_rand["conformal_prediction"]
    }
    
    with open(results_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved updated Phase 4 calibration results to {results_file}")
    
    # Save updated reliability diagram
    figures_dir = "outputs/figures"
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = os.path.join(figures_dir, "reliability_diagram_pre_post.png")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ece_b = res_b_rand["ece_before_obj"]
    ece_a = res_b_rand["ece_after_obj"]
    
    bins_b = ece_b["bin_stats"]
    centers_b = [(b["bin_range"][0] + b["bin_range"][1]) / 2.0 for b in bins_b if b["count"] > 0]
    widths_b = [b["bin_range"][1] - b["bin_range"][0] for b in bins_b if b["count"] > 0]
    accs_b = [b["accuracy"] for b in bins_b if b["count"] > 0]
    confs_b = [b["confidence"] for b in bins_b if b["count"] > 0]
    
    ax1.plot([0, 1], [0, 1], "--", color="gray", label="Optimal")
    ax1.bar(centers_b, accs_b, width=widths_b, alpha=0.7, color="royalblue", edgecolor="black", label="Accuracy")
    for c, a, w, conf in zip(centers_b, accs_b, widths_b, confs_b):
        if conf > a:
            ax1.bar(c, conf - a, bottom=a, width=w, alpha=0.3, color="crimson", edgecolor="red", hatch="//", label="Calibration Gap" if c == centers_b[0] else "")
    ax1.set_title(f"Uncalibrated (ECE = {ece_b['ece']*100:.2f}%)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Confidence", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    bins_a = ece_a["bin_stats"]
    centers_a = [(b["bin_range"][0] + b["bin_range"][1]) / 2.0 for b in bins_a if b["count"] > 0]
    widths_a = [b["bin_range"][1] - b["bin_range"][0] for b in bins_a if b["count"] > 0]
    accs_a = [b["accuracy"] for b in bins_a if b["count"] > 0]
    confs_a = [b["confidence"] for b in bins_a if b["count"] > 0]
    
    ax2.plot([0, 1], [0, 1], "--", color="gray", label="Optimal")
    ax2.bar(centers_a, accs_a, width=widths_a, alpha=0.7, color="forestgreen", edgecolor="black", label="Accuracy")
    for c, a, w, conf in zip(centers_a, accs_a, widths_a, confs_a):
        if conf > a:
            ax2.bar(c, conf - a, bottom=a, width=w, alpha=0.3, color="crimson", edgecolor="red", hatch="//")
    opt_t = res_b_rand["temperature_scaling"]["optimal_temperature"]
    ax2.set_title(f"Temperature Scaled T={opt_t:.2f} (ECE = {ece_a['ece']*100:.2f}%)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Confidence", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.suptitle("Reliability Diagrams: Stratified Calibration (Before vs. After)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved updated reliability diagrams to: {fig_path}")

if __name__ == "__main__":
    run_calibration_study()
