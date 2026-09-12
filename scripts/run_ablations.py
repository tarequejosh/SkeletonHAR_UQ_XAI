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
from src.eval.metrics import compute_classification_metrics, compute_ece
from src.eval.risk_coverage import compute_risk_coverage_curve

def run_ablations(
    checkpoint_path="outputs/checkpoints/stgcn_utd_baseline.pt",
    config_path="configs/utd_baseline.yaml"
):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {checkpoint_path} for ablation studies...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Load dataset
    _, test_loader = get_utd_loaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=2,
        augment=False
    )
    
    cal_dataset = UTDMHADDataset(config["data"]["data_dir"], os.path.join(config["data"]["data_dir"], "metadata.json"), split="cal_stratified")
    cal_loader = DataLoader(cal_dataset, batch_size=32, shuffle=False)
    
    # Base deterministic model
    base_model = STGCN(
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["num_classes"],
        num_nodes=config["model"]["num_nodes"],
        dropout=0.0,
        channel_plan=config["model"]["channel_plan"]
    ).to(device)
    base_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    base_model.eval()
    
    # Fit temperature scaler
    cal_logits_list, cal_labels_list = [], []
    with torch.no_grad():
        for x, y, _ in cal_loader:
            cal_logits_list.append(base_model(x.to(device)).cpu())
            cal_labels_list.append(y)
    cal_logits = torch.cat(cal_logits_list, dim=0)
    cal_labels = torch.cat(cal_labels_list, dim=0)
    ts = ModelWithTemperature()
    opt_temp, _, _ = ts.fit(cal_logits, cal_labels)
    
    # Get test logits
    test_logits_list, test_labels_list = [], []
    with torch.no_grad():
        for x, y, _ in test_loader:
            test_logits_list.append(base_model(x.to(device)).cpu())
            test_labels_list.append(y)
    test_logits = torch.cat(test_logits_list, dim=0)
    test_labels = torch.cat(test_labels_list, dim=0).numpy()
    
    # 1. Deterministic Uncalibrated
    det_probs = F.softmax(test_logits, dim=1).numpy()
    det_preds = np.argmax(det_probs, axis=1)
    det_metrics = compute_classification_metrics(test_labels, det_preds, det_probs)
    det_ece = compute_ece(det_probs, test_labels)["ece"]
    det_aurc = compute_risk_coverage_curve(np.max(det_probs, axis=1), det_preds, test_labels)["aurc"]
    
    # 2. Deterministic + Temperature Scaled
    ts_probs = ts.predict_probs(test_logits)
    ts_preds = np.argmax(ts_probs, axis=1)
    ts_metrics = compute_classification_metrics(test_labels, ts_preds, ts_probs)
    ts_ece = compute_ece(ts_probs, test_labels)["ece"]
    ts_aurc = compute_risk_coverage_curve(np.max(ts_probs, axis=1), ts_preds, test_labels)["aurc"]
    
    ablation_results = [
        {
            "configuration": "Deterministic (Uncalibrated)",
            "dropout_p": 0.0,
            "mc_passes": 1,
            "temp_scaled": False,
            "accuracy": det_metrics["accuracy"],
            "macro_f1": det_metrics["macro_f1"],
            "ece": det_ece,
            "brier": det_metrics["brier_score"],
            "aurc": det_aurc
        },
        {
            "configuration": f"Deterministic + Temp Scaling (T={opt_temp:.2f})",
            "dropout_p": 0.0,
            "mc_passes": 1,
            "temp_scaled": True,
            "accuracy": ts_metrics["accuracy"],
            "macro_f1": ts_metrics["macro_f1"],
            "ece": ts_ece,
            "brier": ts_metrics["brier_score"],
            "aurc": ts_aurc
        }
    ]
    
    # 3. Sensitivity to Dropout rate: p \in [0.1, 0.3, 0.5] with fixed N=25
    for p in [0.1, 0.3, 0.5]:
        mc_model = STGCN(
            in_channels=config["model"]["in_channels"],
            num_classes=config["model"]["num_classes"],
            num_nodes=config["model"]["num_nodes"],
            dropout=p,
            channel_plan=config["model"]["channel_plan"]
        ).to(device)
        mc_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        mc_engine = MCDropoutInference(mc_model, num_passes=25, device=device)
        mc_res = mc_engine.predict_loader(test_loader, desc=f"Ablation Dropout p={p}")
        
        probs = mc_res["mean_probs"]
        preds = np.argmax(probs, axis=1)
        m = compute_classification_metrics(test_labels, preds, probs)
        ece = compute_ece(probs, test_labels)["ece"]
        aurc = compute_risk_coverage_curve(np.max(probs, axis=1), preds, test_labels)["aurc"]
        
        ablation_results.append({
            "configuration": f"MC-Dropout (p={p}, N=25)",
            "dropout_p": p,
            "mc_passes": 25,
            "temp_scaled": False,
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "ece": ece,
            "brier": m["brier_score"],
            "aurc": aurc
        })
        
    # 4. Sensitivity to Number of MC Passes: N \in [5, 15, 30] with fixed p=0.3
    mc_model_std = STGCN(
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["num_classes"],
        num_nodes=config["model"]["num_nodes"],
        dropout=0.3,
        channel_plan=config["model"]["channel_plan"]
    ).to(device)
    mc_model_std.load_state_dict(ckpt["model_state_dict"], strict=False)
    
    for n_passes in [5, 15, 30]:
        mc_engine = MCDropoutInference(mc_model_std, num_passes=n_passes, device=device)
        mc_res = mc_engine.predict_loader(test_loader, desc=f"Ablation Passes N={n_passes}")
        
        probs = mc_res["mean_probs"]
        preds = np.argmax(probs, axis=1)
        m = compute_classification_metrics(test_labels, preds, probs)
        ece = compute_ece(probs, test_labels)["ece"]
        aurc = compute_risk_coverage_curve(np.max(probs, axis=1), preds, test_labels)["aurc"]
        
        ablation_results.append({
            "configuration": f"MC-Dropout (p=0.3, N={n_passes})",
            "dropout_p": 0.3,
            "mc_passes": n_passes,
            "temp_scaled": False,
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "ece": ece,
            "brier": m["brier_score"],
            "aurc": aurc
        })
        
    # Save results
    results_dir = "outputs/results"
    os.makedirs(results_dir, exist_ok=True)
    res_path = os.path.join(results_dir, "phase8_ablations.json")
    with open(res_path, "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"Saved ablation results to {res_path}")
    
    # Print formatted markdown table
    print("\n| Configuration | Acc (%) | Macro F1 (%) | ECE (%) | Brier | AURC (%) |")
    print("|---|---|---|---|---|---|")
    for r in ablation_results:
        print(f"| {r['configuration']} | {r['accuracy']*100:.2f} | {r['macro_f1']*100:.2f} | {r['ece']*100:.2f} | {r['brier']:.4f} | {r['aurc']*100:.2f} |")
        
    return ablation_results

if __name__ == "__main__":
    run_ablations()
