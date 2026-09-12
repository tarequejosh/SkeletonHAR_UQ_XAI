import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import yaml
import numpy as np
import torch
import torch.nn.functional as F

from src.data.utd_dataset import get_utd_loaders, UTDMHADDataset
from torch.utils.data import DataLoader
from src.models.stgcn import STGCN
from src.uncertainty.mc_dropout import MCDropoutInference
from src.uncertainty.temperature_scaling import ModelWithTemperature
from src.eval.risk_coverage import compute_risk_coverage_curve, plot_risk_coverage_comparison

def run_selective_prediction_study(
    checkpoint_path="outputs/checkpoints/stgcn_utd_baseline.pt",
    config_path="configs/utd_baseline.yaml"
):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {checkpoint_path} for selective prediction study...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # 1. Base deterministic model
    model = STGCN(
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["num_classes"],
        num_nodes=config["model"]["num_nodes"],
        dropout=0.3, # supports both deterministic and MC-dropout
        channel_plan=config["model"]["channel_plan"]
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    
    # Datasets
    train_dataset = UTDMHADDataset(config["data"]["data_dir"], os.path.join(config["data"]["data_dir"], "metadata.json"), split="train")
    test_dataset = UTDMHADDataset(config["data"]["data_dir"], os.path.join(config["data"]["data_dir"], "metadata.json"), split="test")
    
    # Calibration split (Subject 7)
    cal_indices = [i for i, m in enumerate(train_dataset.meta) if m.get("subject") == 7]
    cal_loader = DataLoader(torch.utils.data.Subset(train_dataset, cal_indices), batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # --- 1. Fit Temperature Scaling on calibration split ---
    model.eval()
    cal_logits_list, cal_labels_list = [], []
    with torch.no_grad():
        for x, y, _ in cal_loader:
            x = x.to(device)
            cal_logits_list.append(model(x).cpu())
            cal_labels_list.append(y)
    cal_logits = torch.cat(cal_logits_list, dim=0)
    cal_labels = torch.cat(cal_labels_list, dim=0)
    
    ts = ModelWithTemperature()
    opt_temp, _, _ = ts.fit(cal_logits, cal_labels)
    
    # --- 2. Deterministic & Temperature-Scaled Evaluation on Test set ---
    test_logits_list, test_labels_list = [], []
    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.to(device)
            test_logits_list.append(model(x).cpu())
            test_labels_list.append(y)
    test_logits = torch.cat(test_logits_list, dim=0)
    test_labels = torch.cat(test_labels_list, dim=0).numpy()
    
    # Baseline uncalibrated softmax
    det_probs = F.softmax(test_logits, dim=1).numpy()
    det_preds = np.argmax(det_probs, axis=1)
    det_confs = np.max(det_probs, axis=1)
    
    # Temperature-scaled probabilities
    ts_probs = ts.predict_probs(test_logits)
    ts_preds = np.argmax(ts_probs, axis=1)
    ts_confs = np.max(ts_probs, axis=1)
    
    # --- 3. MC-Dropout Evaluation on Test set ---
    mc_engine = MCDropoutInference(model, num_passes=25, device=device)
    mc_res = mc_engine.predict_loader(test_loader, desc="MC-Dropout for Risk-Coverage")
    mc_probs = mc_res["mean_probs"]
    mc_preds = np.argmax(mc_probs, axis=1)
    mc_confs = np.max(mc_probs, axis=1)
    mc_neg_entropy = -mc_res["pred_entropy"]
    
    # --- 4. Compute Risk-Coverage Curves ---
    print("\n--- Computing Risk-Coverage Curves ---")
    curve_det = compute_risk_coverage_curve(det_confs, det_preds, test_labels)
    curve_ts = compute_risk_coverage_curve(ts_confs, ts_preds, test_labels)
    curve_mc_conf = compute_risk_coverage_curve(mc_confs, mc_preds, test_labels)
    curve_mc_ent = compute_risk_coverage_curve(mc_neg_entropy, mc_preds, test_labels)
    
    print(f"AURC Baseline (Uncalibrated Softmax): {curve_det['aurc']*100:.2f}%")
    print(f"AURC Temperature-Scaled Confidence:  {curve_ts['aurc']*100:.2f}%")
    print(f"AURC MC-Dropout Predictive Conf:      {curve_mc_conf['aurc']*100:.2f}%")
    print(f"AURC MC-Dropout Neg Entropy:          {curve_mc_ent['aurc']*100:.2f}%")
    
    curves = {
        "Uncalibrated Softmax": curve_det,
        "Temperature-Scaled": curve_ts,
        "MC-Dropout Confidence": curve_mc_conf,
        "MC-Dropout (-Entropy)": curve_mc_ent
    }
    
    figures_dir = "outputs/figures"
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = os.path.join(figures_dir, "risk_coverage_curve.png")
    plot_risk_coverage_comparison(curves, save_path=fig_path)
    
    # Save results json
    results_dir = "outputs/results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "phase6_risk_coverage.json")
    
    summary = {
        "aurc_comparison": {
            "uncalibrated_softmax": curve_det["aurc"],
            "temperature_scaled": curve_ts["aurc"],
            "mc_dropout_confidence": curve_mc_conf["aurc"],
            "mc_dropout_neg_entropy": curve_mc_ent["aurc"]
        },
        "aurc_reduction_pct": {
            "mc_conf_vs_uncal": float((curve_det["aurc"] - curve_mc_conf["aurc"]) / curve_det["aurc"] * 100),
            "mc_ent_vs_uncal": float((curve_det["aurc"] - curve_mc_ent["aurc"]) / curve_det["aurc"] * 100)
        },
        "curves": curves
    }
    
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved Phase 6 risk-coverage results to {results_file}")
    return summary

if __name__ == "__main__":
    run_selective_prediction_study()
