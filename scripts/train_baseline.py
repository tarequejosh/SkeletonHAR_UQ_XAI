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
from src.utils.visualizer import plot_confusion_matrix
from scripts.preprocess_utd import ACTION_NAMES

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
    all_preds = []
    all_targets = []
    all_probs = []
    
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
    
    return metrics, all_targets, all_preds, all_probs

def main(config_path="configs/utd_baseline.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    set_seed(config["training"]["seed"])
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Training ST-GCN on {device}...")
    
    train_loader, test_loader = get_utd_loaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        augment=config["data"]["augment"]
    )
    
    model = STGCN(
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["num_classes"],
        num_nodes=config["model"]["num_nodes"],
        dropout=config["model"]["dropout"],
        channel_plan=config["model"]["channel_plan"]
    ).to(device)
    
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
    
    epochs = config["training"]["epochs"]
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=config["training"]["min_lr"]
    )
    
    best_acc = 0.0
    checkpoint_dir = config["training"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, config["training"]["checkpoint_name"])
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, _, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        
        val_acc = val_metrics["accuracy"]
        val_loss = val_metrics["loss"]
        
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
                "config": config
            }, best_checkpoint_path)
            
        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
              f"Test Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% "
              f"{'*** BEST ***' if is_best else ''}")
              
    print(f"\nTraining complete. Loading best checkpoint from {best_checkpoint_path}...")
    ckpt = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    
    final_metrics, y_true, y_pred, y_probs = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Evaluation: Accuracy={final_metrics['accuracy']*100:.2f}%, "
          f"Macro F1={final_metrics['macro_f1']*100:.2f}%, "
          f"ECE={final_metrics['ece']*100:.2f}%, "
          f"Brier={final_metrics['brier_score']:.4f}")
          
    # Save results
    results_dir = config["training"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "phase2_baseline.json")
    with open(results_file, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Saved metrics to {results_file}")
    
    # Save confusion matrix figure
    figures_dir = config["training"]["figures_dir"]
    cm_path = os.path.join(figures_dir, "confusion_matrix_baseline.png")
    plot_confusion_matrix(final_metrics["confusion_matrix"], ACTION_NAMES, save_path=cm_path)

if __name__ == "__main__":
    main()
