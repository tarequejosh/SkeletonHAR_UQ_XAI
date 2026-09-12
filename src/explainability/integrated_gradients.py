import torch
import torch.nn.functional as F
import numpy as np

class IntegratedGradientsExplainer:
    """
    Computes Integrated Gradients attribution for Skeleton HAR.
    A path-integral attribution method comparing the input against a baseline pose.
    """
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.model.to(device)

    def attribute(self, x, target_class=None, baseline=None, steps=20):
        """
        x: (1, C, T, V) or (C, T, V)
        target_class: int
        baseline: reference input (defaults to zeros)
        steps: number of Riemann approximation steps
        returns:
          joint_importance: (V,) normalized attribution vector
          raw_attributions: (C, T, V)
        """
        self.model.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        x = x.to(self.device)
        
        if baseline is None:
            baseline = torch.zeros_like(x)
        else:
            if isinstance(baseline, np.ndarray):
                baseline = torch.from_numpy(baseline).float()
            if baseline.dim() == 3:
                baseline = baseline.unsqueeze(0)
            baseline = baseline.to(self.device)
            
        # Determine target class
        with torch.no_grad():
            orig_logits = self.model(x)
            if target_class is None:
                target_class = torch.argmax(orig_logits, dim=1).item()
                
        # Generate interpolated inputs: x_0 + alpha * (x - x_0)
        alphas = torch.linspace(0.0, 1.0, steps + 1, device=self.device)
        total_grads = torch.zeros_like(x)
        
        for alpha in alphas[1:]:
            interpolated = baseline + alpha * (x - baseline)
            interpolated.requires_grad_(True)
            
            logits = self.model(interpolated)
            target_logit = logits[0, target_class]
            
            grad = torch.autograd.grad(target_logit, interpolated)[0]
            total_grads += grad.detach()
            
        avg_grads = total_grads / steps
        # IG = (x - baseline) * avg_grads
        integrated_grads = (x - baseline) * avg_grads # (1, C, T, V)
        ig_np = integrated_grads.squeeze(0).cpu().numpy() # (C, T, V)
        
        # Joint importance: aggregate magnitude across channels (C) and time (T)
        # Shape: (V,)
        joint_energy = np.sum(np.abs(ig_np), axis=(0, 1))
        sum_energy = np.sum(joint_energy)
        if sum_energy > 1e-6:
            joint_importance = joint_energy / sum_energy
        else:
            joint_importance = np.ones(x.size(3), dtype=np.float32) / x.size(3)
            
        return {
            "joint_importance": joint_importance,
            "raw_attribution": ig_np,
            "target_class": target_class
        }
