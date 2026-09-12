import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.data.skeleton_graph import SkeletonGraph

# NTU RGB+D 25 Kinect v2 joints
NTU_JOINT_NAMES = [
    "Spine_Base",      # 0
    "Spine_Mid",       # 1
    "Neck",            # 2
    "Head",            # 3
    "Shoulder_Left",   # 4
    "Elbow_Left",      # 5
    "Wrist_Left",      # 6
    "Hand_Left",       # 7
    "Shoulder_Right",  # 8
    "Elbow_Right",     # 9
    "Wrist_Right",     # 10
    "Hand_Right",      # 11
    "Hip_Left",        # 12
    "Knee_Left",       # 13
    "Ankle_Left",      # 14
    "Foot_Left",       # 15
    "Hip_Right",       # 16
    "Knee_Right",      # 17
    "Ankle_Right",     # 18
    "Foot_Right",      # 19
    "Spine_Shoulder",  # 20
    "Hand_Tip_Left",   # 21
    "Thumb_Left",      # 22
    "Hand_Tip_Right",  # 23
    "Thumb_Right"      # 24
]

NTU_EDGES = [
    (0, 1), (1, 20), (20, 2), (2, 3),        # Spine chain
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22), # Left arm & hand
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24), # Right arm & hand
    (0, 12), (12, 13), (13, 14), (14, 15),   # Left leg
    (0, 16), (16, 17), (17, 18), (18, 19)    # Right leg
]

# Standard NTU Cross-Subject (CS) training subjects
NTU_CS_TRAIN_SUBJECTS = {
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16,
    17, 18, 19, 25, 27, 28, 31, 34, 35, 38
}

# Standard NTU Cross-View (CV) training cameras
NTU_CV_TRAIN_CAMERAS = {2, 3}

def parse_ntu_filename(filename):
    """
    Parses SsssCcccPpppRrrrAaaa from filename.
    """
    base = os.path.basename(filename)
    m = re.match(r"S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})", base)
    if not m:
        raise ValueError(f"Invalid NTU filename format: {base}")
    setup = int(m.group(1))
    camera = int(m.group(2))
    performer = int(m.group(3))
    replication = int(m.group(4))
    action = int(m.group(5))
    return {
        "setup": setup,
        "camera": camera,
        "performer": performer,
        "replication": replication,
        "action": action, # 1 to 60
        "label": action - 1 # 0 to 59
    }

class NTUDataset(Dataset):
    """
    PyTorch Dataset for NTU RGB+D 60 skeleton modality.
    Supports benchmark split protocols:
      - 'xsub' / 'cs': Cross-Subject
      - 'xview' / 'cv': Cross-View
    """
    def __init__(
        self,
        data_dir="data/ntu60",
        benchmark="xsub",
        split="train",
        target_frames=64,
        transform=None
    ):
        super().__init__()
        self.data_dir = data_dir
        self.benchmark = benchmark.lower()
        self.split = split.lower()
        self.target_frames = target_frames
        self.transform = transform
        
        self.samples = []
        self._scan_dataset()

    def _scan_dataset(self):
        if not os.path.exists(self.data_dir):
            return
            
        for root, _, files in os.walk(self.data_dir):
            for f in files:
                if f.endswith(".skeleton") or f.endswith(".npy"):
                    try:
                        meta = parse_ntu_filename(f)
                        meta["path"] = os.path.join(root, f)
                        
                        # Filter by protocol split
                        if self.benchmark in ["xsub", "cs"]:
                            is_train = meta["performer"] in NTU_CS_TRAIN_SUBJECTS
                        elif self.benchmark in ["xview", "cv"]:
                            is_train = meta["camera"] in NTU_CV_TRAIN_CAMERAS
                        else:
                            is_train = True
                            
                        if (self.split == "train" and is_train) or (self.split == "test" and not is_train):
                            self.samples.append(meta)
                    except ValueError:
                        continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta = self.samples[idx]
        file_path = meta["path"]
        
        # If preprocessed .npy
        if file_path.endswith(".npy"):
            data = np.load(file_path).astype(np.float32) # (3, T, 25)
        else:
            # Parse raw .skeleton text file
            data = self._read_raw_skeleton(file_path)
            
        if self.transform is not None:
            data = self.transform(data)
            
        return torch.from_numpy(data).float(), torch.tensor(meta["label"], dtype=torch.long), meta

    def _read_raw_skeleton(self, file_path):
        """Reads raw NTU RGB+D text skeleton file format."""
        with open(file_path, "r") as f:
            lines = f.readlines()
        num_frames = int(lines[0].strip())
        current_line = 1
        
        skeleton_frames = []
        for f in range(num_frames):
            if current_line >= len(lines):
                break
            num_bodies = int(lines[current_line].strip())
            current_line += 1
            if num_bodies == 0:
                skeleton_frames.append(np.zeros((3, 25), dtype=np.float32))
                continue
            # Body info
            current_line += 1
            num_joints = int(lines[current_line].strip())
            current_line += 1
            
            joint_coords = np.zeros((3, 25), dtype=np.float32)
            for j in range(min(num_joints, 25)):
                tokens = lines[current_line].strip().split()
                current_line += 1
                joint_coords[0, j] = float(tokens[0]) # x
                joint_coords[1, j] = float(tokens[1]) # y
                joint_coords[2, j] = float(tokens[2]) # z
                
            # Skip any extra bodies
            for b in range(1, num_bodies):
                current_line += 1 # body info
                n_j = int(lines[current_line].strip())
                current_line += 1 + n_j
                
            skeleton_frames.append(joint_coords)
            
        # (T, 3, 25) -> (3, T, 25)
        arr = np.stack(skeleton_frames, axis=1)
        # Resample to target_frames if necessary
        from src.data.transforms import Compose
        # Basic interpolation
        if arr.shape[1] != self.target_frames:
            from scipy.interpolate import interp1d
            orig_t = np.linspace(0, 1, arr.shape[1])
            new_t = np.linspace(0, 1, self.target_frames)
            f = interp1d(orig_t, arr, axis=1, kind='linear', fill_value='extrapolate')
            arr = f(new_t).astype(np.float32)
        return arr
