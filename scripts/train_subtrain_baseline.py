import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.utd_dataset import get_utd_loaders
from src.models.stgcn import STGCN
from src.eval.metrics import compute_classification_metrics, compute_ece

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for x, y, _ in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []
    
    for x, y, _ in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        probs = torch.softmax(logits, dim=1)
        
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    avg_loss = total_loss / len(all_targets)
    metrics = compute_classification_metrics(all_targets, all_preds, all_probs)
    ece_info = compute_ece(all_probs, all_targets)
    metrics["ece"] = ece_info["ece"]
    metrics["loss"] = avg_loss
    return metrics

def train_subtrain(config_path="configs/utd_baseline.yaml", epochs=60):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    set_seed(config["training"]["seed"])
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Training ST-GCN on sub-train split (323 samples) on {device}...")
    
    train_loader, test_loader = get_utd_loaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=2,
        augment=True,
        train_split="train_sub"
    )
    
    model = STGCN(
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["num_classes"],
        num_nodes=config["model"]["num_nodes"],
        dropout=config["model"]["dropout"],
        channel_plan=config["model"]["channel_plan"]
    ).to(device)
    
    epochs = config["training"].get("epochs", epochs)
    label_smooth = config["training"].get("label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    
    opt_type = config["training"].get("optimizer", "adam").lower()
    if opt_type == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"]
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"]
        )
        
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=config["training"]["min_lr"])
    
    checkpoint_dir = config["training"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "stgcn_utd_subtrain.pt")
    
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        
        val_acc = val_metrics["accuracy"]
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_acc": best_acc,
                "config": config,
                "trained_on": "train_sub (323 samples across subjects 1,3,5,7)"
            }, checkpoint_path)
            
        if epoch % 10 == 0 or epoch == epochs or val_acc == best_acc:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Acc: {tr_acc*100:.1f}% | Test Acc: {val_acc*100:.2f}% (Best: {best_acc*100:.2f}%)")
            
    print(f"\nSub-train model finished. Best checkpoint saved to {checkpoint_path} with {best_acc*100:.2f}% test accuracy.")
    return checkpoint_path

if __name__ == "__main__":
    train_subtrain()
