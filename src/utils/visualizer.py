import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import itertools
from src.data.skeleton_graph import UTD_EDGES, UTD_JOINT_NAMES

def plot_skeleton_sequence(
    sequence, 
    action_name="Action", 
    save_path="outputs/figures/sanity_check_utd_mhad.png", 
    num_frames=6,
    joint_weights=None
):
    """
    Plots multiple 3D snapshots of a skeleton sequence over time.
    sequence: (C, T, V) or (T, V, C) where C=3, V=20.
    joint_weights: (V,) optional weights to color joints (for explainability heatmaps).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if sequence.shape[0] == 3:
        sequence = np.transpose(sequence, (1, 2, 0))
    
    T, V, C = sequence.shape
    frame_indices = np.linspace(0, T - 1, num_frames, dtype=int)
    
    fig = plt.figure(figsize=(3.5 * num_frames, 4.5))
    
    for idx, f_idx in enumerate(frame_indices):
        ax = fig.add_subplot(1, num_frames, idx + 1, projection='3d')
        frame_data = sequence[f_idx]
        
        xs = frame_data[:, 0]
        ys = frame_data[:, 2] # Z mapped to Y for intuitive 3D viewpoint
        zs = frame_data[:, 1] # Y mapped to Z (height)
        
        for u, v in UTD_EDGES:
            ax.plot([xs[u], xs[v]], [ys[u], ys[v]], [zs[u], zs[v]], color='royalblue', lw=2.0, alpha=0.8)
        
        if joint_weights is not None:
            norm_w = (joint_weights - joint_weights.min()) / (joint_weights.max() - joint_weights.min() + 1e-6)
            sc = ax.scatter(xs, ys, zs, c=norm_w, cmap='plasma', s=70, edgecolors='black', vmin=0, vmax=1)
        else:
            ax.scatter(xs, ys, zs, color='crimson', s=45, edgecolors='black')
        
        ax.set_title(f"t = {f_idx}/{T-1}", fontsize=11)
        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Z (m)", fontsize=8)
        ax.set_zlabel("Y (m)", fontsize=8)
        ax.view_init(elev=15, azim=-75)
        
        max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0
        if max_range > 0:
            mid_x = (xs.max()+xs.min()) * 0.5
            mid_y = (ys.max()+ys.min()) * 0.5
            mid_z = (zs.max()+zs.min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

    fig.suptitle(f"UTD-MHAD Skeleton Sequence: {action_name}", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved skeleton sequence visualization to: {save_path}")

def plot_confusion_matrix(cm, class_names, save_path="outputs/figures/confusion_matrix_baseline.png", normalize=True):
    """
    Plots and saves a high-resolution confusion matrix heatmap.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm_plot = np.array(cm)
    if normalize:
        cm_norm = cm_plot.astype('float') / (cm_plot.sum(axis=1, keepdims=True) + 1e-8)
    else:
        cm_norm = cm_plot

    fig, ax = plt.subplots(figsize=(14, 12))
    cax = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=8)

    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_title('Normalized Confusion Matrix: UTD-MHAD ST-GCN', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved confusion matrix plot to: {save_path}")

def plot_reliability_diagram(ece_dict, title="Reliability Diagram", save_path="outputs/figures/reliability_diagram.png"):
    """
    Plots reliability diagram with accuracy vs confidence across probability bins.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    bin_stats = ece_dict["bin_stats"]
    ece = ece_dict["ece"]
    
    confs = [b["confidence"] for b in bin_stats if b["count"] > 0]
    accs = [b["accuracy"] for b in bin_stats if b["count"] > 0]
    props = [b["prop"] for b in bin_stats if b["count"] > 0]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Diagonal perfect calibration
    ax1.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
    # Accuracy bars
    centers = [(b["bin_range"][0] + b["bin_range"][1]) / 2.0 for b in bin_stats if b["count"] > 0]
    widths = [b["bin_range"][1] - b["bin_range"][0] for b in bin_stats if b["count"] > 0]
    
    ax1.bar(centers, accs, width=widths, alpha=0.7, color="royalblue", edgecolor="black", label="Outputs")
    # Calibration gap
    for c, a, w in zip(centers, accs, widths):
        ax1.bar(c, c - a, bottom=a, width=w, alpha=0.3, color="crimson", edgecolor="red", hatch="//", label="Calibration Gap" if c == centers[0] else "")
        
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title(f"{title} (ECE = {ece*100:.2f}%)", fontsize=13, fontweight='bold')
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Proportion histogram
    ax2.bar(centers, props, width=widths, color="slategray", edgecolor="black", alpha=0.8)
    ax2.set_xlabel("Confidence", fontsize=11)
    ax2.set_ylabel("Proportion", fontsize=11)
    ax2.set_xlim(0, 1)
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved reliability diagram to: {save_path}")
