import torch
import torch.nn.functional as F
import numpy as np

class PerturbationExplainer:
    """
    Perturbation-based joint and temporal attribution for Skeleton HAR.
    Measures confidence drop when masking individual joint trajectories.
    """
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def attribute_joints(self, x, target_class=None, mode="zero"):
        """
        x: (1, C, T, V) or (C, T, V) torch.Tensor or numpy array.
        target_class: int, target action class to explain (if None, uses predicted class).
        mode: "zero" or "mean" reference pose.
        returns:
          joint_importance: (V,) normalized importance array
          raw_drops: (V,) confidence drops
          pred_class: int
          orig_conf: float
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        x = x.to(self.device)
        C, T, V = x.size(1), x.size(2), x.size(3)
        
        # Base prediction
        orig_logits = self.model(x)
        orig_probs = F.softmax(orig_logits, dim=1).squeeze(0)
        
        if target_class is None:
            target_class = torch.argmax(orig_probs).item()
        orig_conf = orig_probs[target_class].item()
        
        raw_drops = np.zeros(V, dtype=np.float32)
        
        for j in range(V):
            x_perturbed = x.clone()
            if mode == "zero":
                # Mask joint j trajectory to 0
                x_perturbed[:, :, :, j] = 0.0
            elif mode == "mean":
                # Replace with mean across all joints in that frame
                mean_pos = torch.mean(x[:, :, :, :], dim=3, keepdim=True)
                x_perturbed[:, :, :, j] = mean_pos[:, :, :, 0]
                
            pert_logits = self.model(x_perturbed)
            pert_conf = F.softmax(pert_logits, dim=1).squeeze(0)[target_class].item()
            raw_drops[j] = max(0.0, orig_conf - pert_conf)
            
        sum_drops = np.sum(raw_drops)
        if sum_drops > 1e-6:
            joint_importance = raw_drops / sum_drops
        else:
            joint_importance = np.ones(V, dtype=np.float32) / V
            
        return {
            "joint_importance": joint_importance,
            "raw_drops": raw_drops,
            "target_class": target_class,
            "orig_confidence": orig_conf
        }

    @torch.no_grad()
    def attribute_temporal_segments(self, x, num_segments=4, target_class=None):
        """
        Measures confidence drop when masking temporal chunks (e.g. quarters of motion).
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        T = x.size(2)
        
        orig_logits = self.model(x)
        orig_probs = F.softmax(orig_logits, dim=1).squeeze(0)
        if target_class is None:
            target_class = torch.argmax(orig_probs).item()
        orig_conf = orig_probs[target_class].item()
        
        seg_size = T // num_segments
        temporal_drops = np.zeros(num_segments, dtype=np.float32)
        
        for s in range(num_segments):
            start_t = s * seg_size
            end_t = (s + 1) * seg_size if s < num_segments - 1 else T
            x_pert = x.clone()
            x_pert[:, :, start_t:end_t, :] = 0.0
            
            pert_logits = self.model(x_pert)
            pert_conf = F.softmax(pert_logits, dim=1).squeeze(0)[target_class].item()
            temporal_drops[s] = max(0.0, orig_conf - pert_conf)
            
        sum_drops = np.sum(temporal_drops)
        norm_importance = temporal_drops / sum_drops if sum_drops > 1e-6 else np.ones(num_segments) / num_segments
        return {
            "temporal_importance": norm_importance,
            "raw_drops": temporal_drops,
            "target_class": target_class,
            "orig_confidence": orig_conf
        }
