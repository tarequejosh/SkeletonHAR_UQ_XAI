import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class MCDropoutInference:
    """
    Performs Monte Carlo Dropout inference over a PyTorch model.
    Runs N stochastic forward passes sequentially in a memory-efficient loop.
    """
    def __init__(self, model, num_passes=25, device="cuda"):
        self.model = model
        self.num_passes = num_passes
        self.device = device
        self.model.to(self.device)

    @torch.no_grad()
    def predict_batch(self, x):
        """
        x: (B, C, T, V)
        returns dict with:
          - mean_probs: (B, num_classes)
          - predictive_entropy: (B,)
          - mutual_information: (B,) [epistemic uncertainty]
          - expected_entropy: (B,) [aleatoric uncertainty]
          - variance: (B, num_classes)
          - total_variance: (B,)
          - all_probs: (num_passes, B, num_classes)
        """
        self.model.eval()
        # Enable dropout layers while keeping BatchNorm in eval mode
        self.model.enable_mc_dropout(True)
        
        B = x.size(0)
        x = x.to(self.device)
        
        all_probs_list = []
        all_entropies_list = []
        
        # Loop sequentially over passes to conserve VRAM on 8GB GPU
        for pass_idx in range(self.num_passes):
            logits = self.model(x) # (B, num_classes)
            probs = F.softmax(logits, dim=1) # (B, num_classes)
            all_probs_list.append(probs.cpu())
            
            # Entropy of individual pass: - \sum p * log(p)
            eps = 1e-12
            pass_ent = -torch.sum(probs * torch.log(torch.clamp(probs, min=eps)), dim=1)
            all_entropies_list.append(pass_ent.cpu())
            
        # Stack passes: (num_passes, B, num_classes)
        all_probs = torch.stack(all_probs_list, dim=0).numpy() # (N, B, C)
        all_entropies = torch.stack(all_entropies_list, dim=0).numpy() # (N, B)
        
        # 1. Predictive Mean
        mean_probs = np.mean(all_probs, axis=0) # (B, C)
        
        # 2. Predictive Total Entropy H(\bar{p})
        eps = 1e-12
        mean_probs_clamped = np.clip(mean_probs, eps, 1.0)
        pred_entropy = -np.sum(mean_probs_clamped * np.log(mean_probs_clamped), axis=1) # (B,)
        
        # 3. Expected Entropy (Aleatoric uncertainty) \mathbb{E}[H(p)]
        expected_entropy = np.mean(all_entropies, axis=0) # (B,)
        
        # 4. Mutual Information (Epistemic uncertainty) I = H - E[H]
        mutual_info = np.maximum(pred_entropy - expected_entropy, 0.0) # (B,)
        
        # 5. Variance across stochastic passes
        var_per_class = np.var(all_probs, axis=0) # (B, C)
        total_var = np.sum(var_per_class, axis=1) # (B,)
        
        return {
            "mean_probs": mean_probs,
            "pred_entropy": pred_entropy,
            "mutual_info": mutual_info,
            "expected_entropy": expected_entropy,
            "variance": var_per_class,
            "total_variance": total_var,
            "all_probs": all_probs
        }

    def predict_loader(self, dataloader, desc="MC-Dropout Inference"):
        """
        Runs MC-Dropout over an entire DataLoader.
        """
        all_mean_probs = []
        all_pred_entropy = []
        all_mutual_info = []
        all_expected_entropy = []
        all_total_variance = []
        all_targets = []
        
        for x, y, _ in tqdm(dataloader, desc=desc):
            results = self.predict_batch(x)
            all_mean_probs.append(results["mean_probs"])
            all_pred_entropy.append(results["pred_entropy"])
            all_mutual_info.append(results["mutual_info"])
            all_expected_entropy.append(results["expected_entropy"])
            all_total_variance.append(results["total_variance"])
            all_targets.append(y.numpy())
            
        return {
            "mean_probs": np.concatenate(all_mean_probs, axis=0),
            "pred_entropy": np.concatenate(all_pred_entropy, axis=0),
            "mutual_info": np.concatenate(all_mutual_info, axis=0),
            "expected_entropy": np.concatenate(all_expected_entropy, axis=0),
            "total_variance": np.concatenate(all_total_variance, axis=0),
            "targets": np.concatenate(all_targets, axis=0)
        }
