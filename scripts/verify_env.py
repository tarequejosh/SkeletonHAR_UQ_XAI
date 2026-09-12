import sys
import torch
import numpy as np
import scipy
import sklearn
import matplotlib
import pandas
import yaml
import tqdm
import h5py
import captum

def main():
    print("=== Environment Verification ===")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU Count: {device_count}")
        print(f"Active Device: {current_device} - {device_name}")
        print(f"Total VRAM: {vram_total:.2f} GB")
        
        # Test GPU computation
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
        print(f"GPU MatMul Test Successful. Tensor Norm: {y.norm().item():.4f}")
    else:
        raise RuntimeError("CUDA is not available on this machine!")

    print(f"NumPy Version: {np.__version__}")
    print(f"SciPy Version: {scipy.__version__}")
    print(f"Scikit-Learn Version: {sklearn.__version__}")
    print(f"Matplotlib Version: {matplotlib.__version__}")
    print(f"Pandas Version: {pandas.__version__}")
    print(f"PyYAML Version: {yaml.__version__}")
    print(f"Tqdm Version: {tqdm.__version__}")
    print(f"H5Py Version: {h5py.__version__}")
    print(f"Captum Version: {captum.__version__}")
    print("=== All Verification Checks Passed! ===")

if __name__ == "__main__":
    main()
