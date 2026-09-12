import os
import json
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def create_stratified_calibration_split(
    data_dir="data/utd_mhad/processed",
    test_size=0.25,
    seed=42
):
    train_npz_path = os.path.join(data_dir, "train.npz")
    meta_path = os.path.join(data_dir, "metadata.json")
    
    loaded = np.load(train_npz_path)
    train_data = loaded["data"]       # (431, 3, 64, 20)
    train_labels = loaded["labels"]   # (431,)
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
    train_meta = meta["train_metadata"]
    
    cal_indices = []
    subtrain_indices = []
    
    # Stratified 25% slice from EACH of the 4 training subjects {1, 3, 5, 7}
    for s in [1, 3, 5, 7]:
        s_idx = [i for i, m in enumerate(train_meta) if m["subject"] == s]
        s_labels = [train_meta[i]["label"] for i in s_idx]
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        for train_local, val_local in sss.split(s_idx, s_labels):
            subtrain_indices.extend([s_idx[i] for i in train_local])
            cal_indices.extend([s_idx[i] for i in val_local])
            
    subtrain_indices = np.array(subtrain_indices)
    cal_indices = np.array(cal_indices)
    
    # Save arrays
    subtrain_data = train_data[subtrain_indices]
    subtrain_labels = train_labels[subtrain_indices]
    subtrain_meta = [train_meta[i] for i in subtrain_indices]
    
    cal_data = train_data[cal_indices]
    cal_labels = train_labels[cal_indices]
    cal_meta = [train_meta[i] for i in cal_indices]
    
    np.savez_compressed(os.path.join(data_dir, "train_sub.npz"), data=subtrain_data, labels=subtrain_labels)
    np.savez_compressed(os.path.join(data_dir, "cal_stratified.npz"), data=cal_data, labels=cal_labels)
    
    with open(os.path.join(data_dir, "calibration_splits_meta.json"), "w") as f:
        json.dump({
            "subtrain_indices": subtrain_indices.tolist(),
            "cal_indices": cal_indices.tolist(),
            "subtrain_count": len(subtrain_indices),
            "cal_count": len(cal_indices),
            "subtrain_meta": subtrain_meta,
            "cal_meta": cal_meta
        }, f, indent=2)
        
    print(f"Stratified calibration split created:")
    print(f"Sub-train samples: {len(subtrain_indices)} (trained model never sees calibration data)")
    print(f"Calibration samples: {len(cal_indices)} (27 samples per subject across {sorted(list(set(m['subject'] for m in cal_meta)))})")
    return subtrain_indices, cal_indices

if __name__ == "__main__":
    create_stratified_calibration_split()
