import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class ModelWithTemperature(nn.Module):
    """
    A thin decorator around a model or logit array that learns a temperature scaling parameter.
    Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        """
        logits: (N, C)
        returns scaled logits: (N, C)
        """
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        # Expand temperature to match logits size
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def fit(self, val_logits, val_labels, max_iter=50, lr=0.01):
        """
        Tunes temperature on validation set logits using L-BFGS optimizer.
        val_logits: torch.Tensor of shape (N, C)
        val_labels: torch.Tensor of shape (N,)
        """
        if not isinstance(val_logits, torch.Tensor):
            val_logits = torch.from_numpy(val_logits).float()
        if not isinstance(val_labels, torch.Tensor):
            val_labels = torch.from_numpy(val_labels).long()

        val_logits = val_logits.clone().detach()
        val_labels = val_labels.clone().detach()
        
        nll_criterion = nn.CrossEntropyLoss()
        
        # Calculate NLL before scaling
        before_nll = nll_criterion(val_logits, val_labels).item()
        
        # L-BFGS optimizer to fit single parameter
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        
        after_nll = nll_criterion(self.temperature_scale(val_logits), val_labels).item()
        optimal_temp = self.temperature.item()
        
        print(f"Temperature Scaling Fit: T={optimal_temp:.4f} | NLL Before: {before_nll:.4f} -> After: {after_nll:.4f}")
        return optimal_temp, before_nll, after_nll

    def predict_probs(self, logits):
        """
        Returns calibrated probabilities from logits.
        """
        with torch.no_grad():
            if not isinstance(logits, torch.Tensor):
                logits = torch.from_numpy(logits).float()
            scaled_logits = self.temperature_scale(logits)
            probs = F.softmax(scaled_logits, dim=1).cpu().numpy()
        return probs
