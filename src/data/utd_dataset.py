import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.data.transforms import RandomRotation3D, RandomJitter, Compose

class UTDMHADDataset(Dataset):
    """
    UTD-MHAD Skeleton Dataset.
    Loads preprocessed (C, T, V) skeleton sequences.
    """
    def __init__(self, data_path, meta_path=None, split="train", transform=None):
        super().__init__()
        self.split = split
        self.transform = transform
        
        npz_file = os.path.join(data_path, f"{split}.npz")
        if not os.path.exists(npz_file):
            raise FileNotFoundError(f"Processed file not found: {npz_file}. Run scripts/preprocess_utd.py first.")
            
        loaded = np.load(npz_file)
        self.data = loaded["data"]       # (N, C, T, V)
        self.labels = loaded["labels"]   # (N,)
        
        self.meta = []
        if meta_path and os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                full_meta = json.load(f)
                self.meta = full_meta.get(f"{split}_metadata", [])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx] # (3, T, 20)
        y = self.labels[idx]
        
        if self.transform is not None:
            x = self.transform(x)
            
        x_tensor = torch.from_numpy(x).float()
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        meta = self.meta[idx] if idx < len(self.meta) else {}
        return x_tensor, y_tensor, meta

def get_utd_loaders(
    data_dir="data/utd_mhad/processed", 
    batch_size=32, 
    num_workers=2, 
    augment=True,
    train_split="train"
):
    """
    Returns (train_loader, test_loader) for UTD-MHAD cross-subject split.
    num_workers defaults to 2 to avoid Windows WinError 1455.
    """
    meta_path = os.path.join(data_dir, "metadata.json")
    
    train_transform = None
    if augment:
        train_transform = Compose([
            RandomRotation3D(max_angle_deg=15.0, prob=0.5),
            RandomJitter(sigma=0.005, prob=0.5)
        ])
        
    train_dataset = UTDMHADDataset(data_dir, meta_path, split=train_split, transform=train_transform)
    test_dataset = UTDMHADDataset(data_dir, meta_path, split="test", transform=None)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, test_loader
