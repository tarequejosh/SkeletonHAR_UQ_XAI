import torch
import torch.nn.functional as F
import numpy as np

class FaithfulnessEvaluator:
    """
    Evaluates faithfulness of joint attribution through deletion and insertion curves.
    """
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def compute_deletion_curve(self, x, joint_importance, target_class=None, num_random_trials=10, seed=42):
        """
        Progressively removes joints in order of highest importance vs random order.
        x: (1, C, T, V) or (C, T, V)
        joint_importance: (V,) array of importance scores
        returns:
          del_curve_method: (V+1,) confidence at each deletion step
          del_curve_random: (V+1,) mean confidence at each deletion step
        """
        rng = np.random.default_rng(seed)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        V = x.size(3)
        
        orig_logits = self.model(x)
        orig_probs = F.softmax(orig_logits, dim=1).squeeze(0)
        if target_class is None:
            target_class = torch.argmax(orig_probs).item()
        base_conf = orig_probs[target_class].item()
        
        # 1. Method order: descending importance
        sorted_joints = np.argsort(-joint_importance) # highest first
        
        method_confs = [base_conf]
        x_curr = x.clone()
        for j in sorted_joints:
            x_curr[:, :, :, j] = 0.0
            logits = self.model(x_curr)
            conf = F.softmax(logits, dim=1).squeeze(0)[target_class].item()
            method_confs.append(conf)
            
        # 2. Random order baseline
        random_trials_confs = []
        for _ in range(num_random_trials):
            rand_joints = rng.permutation(V)
            r_confs = [base_conf]
            x_rand = x.clone()
            for j in rand_joints:
                x_rand[:, :, :, j] = 0.0
                logits = self.model(x_rand)
                conf = F.softmax(logits, dim=1).squeeze(0)[target_class].item()
                r_confs.append(conf)
            random_trials_confs.append(r_confs)
            
        mean_random_confs = np.mean(random_trials_confs, axis=0)
        
        # Area Under Deletion Curve (AUDC)
        steps = np.linspace(0, 1, V + 1)
        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        audc_method = trapz_fn(method_confs, steps)
        audc_random = trapz_fn(mean_random_confs, steps)
        
        return {
            "method_curve": method_confs,
            "random_curve": mean_random_confs.tolist(),
            "audc_method": float(audc_method),
            "audc_random": float(audc_random),
            "faithful": bool(audc_method < audc_random) # Lower curve area = faster drop = more faithful
        }
