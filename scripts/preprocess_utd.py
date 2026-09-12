import os
import glob
import re
import scipy.io as sio
import numpy as np
import json
from scipy.interpolate import interp1d

RAW_DIR = os.path.join("data", "utd_mhad", "raw", "Skeleton")
PROCESSED_DIR = os.path.join("data", "utd_mhad", "processed")
TARGET_FRAMES = 64

# Action names mapped from 1-indexed to 0-indexed action labels
ACTION_NAMES = [
    "swipe_left", "swipe_right", "wave", "clap", "throw", "arm_cross",
    "basketball_shoot", "draw_x", "draw_circle_cw", "draw_circle_ccw",
    "draw_triangle", "bowling", "boxing", "baseball_swing", "tennis_swing",
    "arm_curl", "tennis_serve", "push", "knock", "catch",
    "pickup_and_throw", "jog", "walk", "sit_to_stand", "stand_to_sit",
    "lunge", "squat"
]

def resample_sequence(skeleton, target_len=TARGET_FRAMES):
    """
    Interpolates skeleton sequence along time axis to uniform length.
    skeleton: (C, T, V)
    returns: (C, target_len, V)
    """
    C, T, V = skeleton.shape
    if T == target_len:
        return skeleton
    
    orig_t = np.linspace(0, 1, T)
    new_t = np.linspace(0, 1, target_len)
    
    f = interp1d(orig_t, skeleton, axis=1, kind='linear')
    resampled = f(new_t)
    return resampled.astype(np.float32)

def normalize_skeleton(skeleton):
    """
    Normalizes 3D coordinates:
    Center relative to root joint (Hip Center, joint 0) at each frame.
    skeleton: (C, T, V) where C=3, V=20
    """
    C, T, V = skeleton.shape
    # Joint 0 is Hip_Center
    root = skeleton[:, :, 0:1] # (3, T, 1)
    centered = skeleton - root
    return centered.astype(np.float32)

def preprocess_utd():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.mat")))
    print(f"Found {len(files)} raw skeleton files.")
    
    train_data, train_labels, train_meta = [], [], []
    test_data, test_labels, test_meta = [], [], []
    
    # Standard cross-subject evaluation split:
    # Train subjects: 1, 3, 5, 7
    # Test subjects: 2, 4, 6, 8
    train_subjects = {1, 3, 5, 7}
    test_subjects = {2, 4, 6, 8}
    
    dropped_files = []
    
    for f in files:
        base = os.path.basename(f)
        m = re.match(r"a(\d+)_s(\d+)_t(\d+)_skeleton\.mat", base)
        if not m:
            print(f"Warning: unrecognized file format {base}")
            dropped_files.append((base, "unrecognized_format"))
            continue
        
        action_idx = int(m.group(1)) # 1 to 27
        subject_idx = int(m.group(2)) # 1 to 8
        trial_idx = int(m.group(3)) # 1 to 4
        
        try:
            mat = sio.loadmat(f)
            # raw shape is (20, 3, T)
            d_skel = mat["d_skel"]
            if np.isnan(d_skel).any() or d_skel.size == 0:
                dropped_files.append((base, "nan_or_empty"))
                continue
            
            # Transpose to (C, T, V) -> (3, T, 20)
            skeleton = np.transpose(d_skel, (1, 2, 0)).astype(np.float32)
            
            # Resample and normalize
            skeleton = resample_sequence(skeleton, TARGET_FRAMES)
            skeleton = normalize_skeleton(skeleton)
            
            # Zero-indexed action label (0 to 26)
            label = action_idx - 1
            
            meta = {
                "filename": base,
                "action": action_idx,
                "action_name": ACTION_NAMES[label],
                "subject": subject_idx,
                "trial": trial_idx,
                "label": label
            }
            
            if subject_idx in train_subjects:
                train_data.append(skeleton)
                train_labels.append(label)
                train_meta.append(meta)
            elif subject_idx in test_subjects:
                test_data.append(skeleton)
                test_labels.append(label)
                test_meta.append(meta)
            else:
                dropped_files.append((base, f"unknown_subject_{subject_idx}"))
                
        except Exception as e:
            print(f"Error loading {base}: {e}")
            dropped_files.append((base, str(e)))

    train_data = np.stack(train_data, axis=0) # (N_train, 3, T, 20)
    train_labels = np.array(train_labels, dtype=np.int64)
    test_data = np.stack(test_data, axis=0)   # (N_test, 3, T, 20)
    test_labels = np.array(test_labels, dtype=np.int64)
    
    print(f"Preprocessing completed:")
    print(f"Train samples: {train_data.shape[0]}, Shape: {train_data.shape}")
    print(f"Test samples: {test_data.shape[0]}, Shape: {test_data.shape}")
    print(f"Dropped files: {len(dropped_files)}")
    if dropped_files:
        print(f"Details on dropped files: {dropped_files}")
        
    # Save npz files
    train_npz = os.path.join(PROCESSED_DIR, "train.npz")
    test_npz = os.path.join(PROCESSED_DIR, "test.npz")
    meta_json = os.path.join(PROCESSED_DIR, "metadata.json")
    
    np.savez_compressed(train_npz, data=train_data, labels=train_labels)
    np.savez_compressed(test_npz, data=test_data, labels=test_labels)
    
    with open(meta_json, "w") as f:
        json.dump({
            "target_frames": TARGET_FRAMES,
            "num_classes": 27,
            "action_names": ACTION_NAMES,
            "train_samples": len(train_meta),
            "test_samples": len(test_meta),
            "train_metadata": train_meta,
            "test_metadata": test_meta,
            "dropped_files": dropped_files
        }, f, indent=2)
        
    print(f"Saved preprocessed data to {PROCESSED_DIR}")

if __name__ == "__main__":
    preprocess_utd()
