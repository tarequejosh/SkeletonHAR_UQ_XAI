import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch

from src.data.ntu_dataset import parse_ntu_filename, NTUDataset, NTU_EDGES
from src.data.skeleton_graph import SkeletonGraph
from src.models.stgcn import STGCN

def test_ntu_synthetic():
    print("=== Testing NTU RGB+D 60 Loader & ST-GCN Compatibility ===")
    
    # 1. Filename parser test
    fn = "S001C002P004R001A012.skeleton"
    meta = parse_ntu_filename(fn)
    assert meta["setup"] == 1
    assert meta["camera"] == 2
    assert meta["performer"] == 4
    assert meta["action"] == 12
    assert meta["label"] == 11
    print(f"Parser Check Passed: {fn} -> Performer={meta['performer']}, Camera={meta['camera']}, Action={meta['action']}")
    
    # 2. Skeleton Graph definition for 25 joints
    graph_25 = SkeletonGraph(num_nodes=25, edges=NTU_EDGES)
    A_25 = graph_25.A
    assert A_25.shape == (3, 25, 25)
    print(f"NTU-25 Spatial Graph Adjacency Shape: {A_25.shape} (3 partitions, 25 nodes)")
    
    # 3. Model forward pass with 25 nodes and 60 classes
    model_ntu = STGCN(
        in_channels=3,
        num_classes=60,
        num_nodes=25,
        edges=NTU_EDGES,
        dropout=0.3,
        channel_plan="standard"
    )
    
    dummy_x = torch.randn(4, 3, 64, 25) # (Batch=4, Channels=3, Frames=64, Joints=25)
    logits = model_ntu(dummy_x)
    assert logits.shape == (4, 60)
    print(f"NTU ST-GCN Forward Pass Succeeded: Output shape = {logits.shape}")
    
    print("=== All NTU RGB+D 60 Unit Tests Passed! ===")

if __name__ == "__main__":
    test_ntu_synthetic()
