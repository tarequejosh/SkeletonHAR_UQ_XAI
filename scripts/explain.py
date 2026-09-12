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
from src.models.stgcn import STGCN
from src.explainability.perturbation import PerturbationExplainer
from src.explainability.integrated_gradients import IntegratedGradientsExplainer
from src.explainability.faithfulness import FaithfulnessEvaluator
from src.utils.visualizer import plot_skeleton_sequence
from src.data.skeleton_graph import UTD_JOINT_NAMES
from scripts.preprocess_utd import ACTION_NAMES

def run_explainability_study(
    checkpoint_path="outputs/checkpoints/stgcn_utd_baseline.pt",
    config_path="configs/utd_baseline.yaml",
    num_samples_eval=100
):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {checkpoint_path} for explainability analysis...")
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
    
    test_dataset = UTDMHADDataset(
        config["data"]["data_dir"],
        os.path.join(config["data"]["data_dir"], "metadata.json"),
        split="test"
    )
    
    pert_explainer = PerturbationExplainer(model, device=device)
    ig_explainer = IntegratedGradientsExplainer(model, device=device)
    faith_eval = FaithfulnessEvaluator(model, device=device)
    
    print("\n--- 1. Evaluating Faithfulness (Deletion Curves) ---")
    del_method_curves = []
    del_rand_curves = []
    audc_methods = []
    audc_rands = []
    
    # Run faithfulness evaluation on subset of test samples
    faith_samples = min(50, len(test_dataset))
    for i in range(faith_samples):
        x, y, meta = test_dataset[i]
        attr = pert_explainer.attribute_joints(x, target_class=y)
        faith_res = faith_eval.compute_deletion_curve(x, attr["joint_importance"], target_class=y)
        
        del_method_curves.append(faith_res["method_curve"])
        del_rand_curves.append(faith_res["random_curve"])
        audc_methods.append(faith_res["audc_method"])
        audc_rands.append(faith_res["audc_random"])
        
    mean_method_curve = np.mean(del_method_curves, axis=0)
    mean_rand_curve = np.mean(del_rand_curves, axis=0)
    mean_audc_method = float(np.mean(audc_methods))
    mean_audc_rand = float(np.mean(audc_rands))
    
    print(f"Mean AUDC Method: {mean_audc_method:.4f} vs. Mean AUDC Random: {mean_audc_rand:.4f}")
    is_faithful = mean_audc_method < mean_audc_rand
    print(f"Faithfulness Sanity Check: {'PASSED (Method collapses faster than random)' if is_faithful else 'FAILED'}")
    
    figures_dir = "outputs/figures"
    os.makedirs(figures_dir, exist_ok=True)
    del_fig_path = os.path.join(figures_dir, "explainability_deletion_curve.png")
    
    fig, ax = plt.subplots(figsize=(7, 5))
    steps = np.arange(len(mean_method_curve))
    ax.plot(steps, mean_method_curve, "o-", color="crimson", lw=2, label=f"Joint Perturbation (AUDC: {mean_audc_method:.3f})")
    ax.plot(steps, mean_rand_curve, "s--", color="gray", lw=2, label=f"Random Joint Deletion (AUDC: {mean_audc_rand:.3f})")
    ax.set_xlabel("Number of Joints Removed", fontsize=11, fontweight='bold')
    ax.set_ylabel("Predicted Class Confidence", fontsize=11, fontweight='bold')
    ax.set_title("Attribution Faithfulness: Joint Deletion Curves", fontsize=13, pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(del_fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved deletion curve plot to: {del_fig_path}")

    # --- 2. Comparative Analysis across Uncertainty Tiers ---
    print("\n--- 2. Comparative Attribution Analysis Across Uncertainty Tiers ---")
    all_confs = []
    all_entropies = []
    all_correct = []
    all_joint_entropies = []
    all_joint_attributions = []
    all_preds = []
    all_targets = []
    
    N = min(num_samples_eval, len(test_dataset))
    for i in range(N):
        x, y, meta = test_dataset[i]
        with torch.no_grad():
            x_cuda = x.unsqueeze(0).to(device)
            logits = model(x_cuda)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            
        pred = np.argmax(probs)
        conf = probs[pred]
        entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)))
        
        # Attribute joints
        attr = pert_explainer.attribute_joints(x, target_class=pred)
        w = attr["joint_importance"]
        # Joint attribution dispersion / entropy: H(w) = - \sum w_j log(w_j)
        w_entropy = -np.sum(w * np.log(np.clip(w, 1e-12, 1.0)))
        
        all_confs.append(conf)
        all_entropies.append(entropy)
        all_correct.append(pred == y)
        all_joint_entropies.append(w_entropy)
        all_joint_attributions.append(w)
        all_preds.append(pred)
        all_targets.append(y)
        
    all_confs = np.array(all_confs)
    all_entropies = np.array(all_entropies)
    all_correct = np.array(all_correct)
    all_joint_entropies = np.array(all_joint_entropies)
    all_joint_attributions = np.array(all_joint_attributions)
    
    # Define three groups:
    # 1. Confident-Correct: correct & confidence >= median confidence of correct
    # 2. Confident-Incorrect: incorrect & confidence >= 0.5
    # 3. Uncertain: confidence < median or high entropy
    conf_correct_mask = all_correct & (all_confs >= np.median(all_confs[all_correct]))
    uncertain_mask = all_confs < 0.65
    conf_incorrect_mask = (~all_correct) & (all_confs >= 0.5)
    
    ent_conf_correct = all_joint_entropies[conf_correct_mask]
    ent_uncertain = all_joint_entropies[uncertain_mask]
    ent_conf_incorrect = all_joint_entropies[conf_incorrect_mask] if np.sum(conf_incorrect_mask) > 0 else np.array([0.0])
    
    print(f"Confident-Correct (n={len(ent_conf_correct)}): "
          f"Mean Joint Attribution Entropy = {np.mean(ent_conf_correct):.4f} +/- {np.std(ent_conf_correct):.4f}")
    print(f"Uncertain (n={len(ent_uncertain)}): "
          f"Mean Joint Attribution Entropy = {np.mean(ent_uncertain):.4f} +/- {np.std(ent_uncertain):.4f}")
    if len(ent_conf_incorrect) > 0 and np.sum(conf_incorrect_mask) > 0:
        print(f"Confident-Incorrect (n={len(ent_conf_incorrect)}): "
              f"Mean Joint Attribution Entropy = {np.mean(ent_conf_incorrect):.4f} +/- {np.std(ent_conf_incorrect):.4f}")
              
    # --- 3. Save Qualitative Side-by-Side Skeleton Overlays ---
    # Find one prototypical Confident-Correct example and one Uncertain example
    conf_idx = np.where(conf_correct_mask)[0][0]
    unc_idx = np.where(uncertain_mask)[0][0] if np.sum(uncertain_mask) > 0 else 0
    
    # Save visualizer comparisons
    x_conf, y_conf, m_conf = test_dataset[conf_idx]
    w_conf = all_joint_attributions[conf_idx]
    plot_skeleton_sequence(
        x_conf.numpy(),
        action_name=f"Confident-Correct: {m_conf['action_name']} (Conf={all_confs[conf_idx]:.2f})",
        save_path=os.path.join(figures_dir, "explainability_confident_correct.png"),
        joint_weights=w_conf
    )
    
    x_unc, y_unc, m_unc = test_dataset[unc_idx]
    w_unc = all_joint_attributions[unc_idx]
    plot_skeleton_sequence(
        x_unc.numpy(),
        action_name=f"Uncertain: True={ACTION_NAMES[y_unc]} Pred={ACTION_NAMES[all_preds[unc_idx]]} (Conf={all_confs[unc_idx]:.2f})",
        save_path=os.path.join(figures_dir, "explainability_uncertain.png"),
        joint_weights=w_unc
    )
    
    # Save summary results
    results_dir = "outputs/results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "phase5_explainability.json")
    
    results_data = {
        "faithfulness": {
            "mean_audc_method": mean_audc_method,
            "mean_audc_random": mean_audc_rand,
            "is_faithful": bool(is_faithful)
        },
        "uncertainty_tiers_attribution": {
            "confident_correct": {
                "count": int(len(ent_conf_correct)),
                "mean_joint_entropy": float(np.mean(ent_conf_correct)),
                "std_joint_entropy": float(np.std(ent_conf_correct))
            },
            "uncertain": {
                "count": int(len(ent_uncertain)),
                "mean_joint_entropy": float(np.mean(ent_uncertain)),
                "std_joint_entropy": float(np.std(ent_uncertain))
            },
            "diffuse_attribution_in_uncertain": bool(np.mean(ent_uncertain) > np.mean(ent_conf_correct))
        }
    }
    
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Saved Phase 5 explainability results to {results_file}")
    return results_data

if __name__ == "__main__":
    run_explainability_study()
