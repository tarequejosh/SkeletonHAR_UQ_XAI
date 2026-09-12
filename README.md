# Explainable & Uncertainty-Calibrated Skeleton HAR (ST-GCN)

A research codebase for **Explainable and Uncertainty-Calibrated Skeleton-Based Human Action Recognition** combining:
- **Spatial-Temporal Graph Convolutional Networks (ST-GCN)** for 3D skeleton sequences.
- **Monte Carlo (MC) Dropout** for epistemic and predictive uncertainty quantification.
- **Post-Hoc Probability Calibration** via Temperature Scaling (Platt scaling on graph logits).
- **Conformal Prediction** via Adaptive Prediction Sets (APS) with finite-sample coverage guarantees.
- **Kinematic Explainability** via Perturbation-Based Joint Attribution and Integrated Gradients (cross-checked with Faithfulness Deletion Curves).
- **Selective Prediction / Risk-Coverage Analysis** demonstrating the reliability gains of uncertainty-aware deferral policies.

---

## 1. Environment & Hardware Specifications

- **OS:** Windows 11 (64-bit)
- **GPU:** NVIDIA GeForce RTX 5060 (8 GB VRAM)
- **CUDA Toolkit:** Compatible with CUDA 12.8 / 13.2
- **PyTorch:** 2.11.0+cu128
- **Python:** 3.11.15

### Windows Gotchas & Applied Safeguards
- **WinError 1455 Prevention:** PyTorch `DataLoader` instances use `num_workers=2` (maximum $\le 4$) to prevent Windows paging file exhaustion.
- **Numerical Stability:** Strict **FP32** training (no AMP / mixed-precision) to avoid NaN instability observed on this machine.
- **VRAM Discipline:** MC-Dropout inference runs sequentially over $N=25\text{--}30$ stochastic passes rather than batch-expanding across passes, keeping VRAM strictly below 2 GB.

---

## 2. Setup & Installation

```bash
# Clone the repository and activate environment
git clone <repo-url>
cd SkeletonHAR_UQ_XAI

# Using existing conda env or create from environment.yml
conda env create -f environment.yml
conda activate research

# Verify CUDA and dependencies
python scripts/verify_env.py
```

---

## 3. Dataset Instructions

### UTD-MHAD (Primary Benchmark)
UTD-MHAD is freely accessible from UT Dallas. Run the automated download and preprocessing pipeline:
```bash
# Download official Skeleton archive (~14.7 MB)
python scripts/download_utd.py

# Preprocess, normalize, resample to T=64, and create cross-subject splits
python scripts/preprocess_utd.py
```
- **Cross-Subject Split:**
  - Train: Subjects 1, 3, 5, 7 (431 sequences)
  - Test: Subjects 2, 4, 6, 8 (430 sequences)
  - Classes: 27 human actions (swipe, wave, clap, throw, squat, walk, etc.)
  - Modality: Kinect v1 3D skeleton (20 joints)

### NTU RGB+D 60 (Conditional Scale-Up)
NTU RGB+D 60 requires institutional access from the [ROSE Lab at NTU Singapore](https://rose1.ntu.edu.sg/dataset/actionRecognition/).
1. Request access from the official portal.
2. Once approved, place `.skeleton` files inside `data/ntu60/`.
3. The dataset loader and split parser (`src/data/ntu_dataset.py`) supports Cross-Subject (CS) and Cross-View (CV) protocols out-of-the-box.
4. Verify loader integrity:
   ```bash
   python scripts/test_ntu_loader.py
   ```

---

## 4. End-to-End Experiment Pipeline

Each phase is self-contained, reproducible, and runnable from saved checkpoints:

### Phase 2: Train ST-GCN Baseline
```bash
python scripts/train_baseline.py
# Saves: outputs/checkpoints/stgcn_utd_baseline.pt
# Outputs: outputs/results/phase2_baseline.json, outputs/figures/confusion_matrix_baseline.png
```

### Phase 3: MC-Dropout Uncertainty Quantification
```bash
python scripts/evaluate_uncertainty.py
# Outputs: outputs/results/phase3_mc_dropout.json, outputs/figures/uncertainty_entropy_boxplot.png
```

### Phase 4: Calibration (Temperature Scaling + Conformal Prediction)
```bash
python scripts/calibrate.py
# Outputs: outputs/results/phase4_calibration.json, outputs/figures/reliability_diagram_pre_post.png
```

### Phase 5: Explainability & Faithfulness Sanity Check
```bash
python scripts/explain.py
# Outputs: outputs/results/phase5_explainability.json, outputs/figures/explainability_deletion_curve.png
```

### Phase 6: Selective Prediction (Risk-Coverage Curves)
```bash
python scripts/run_risk_coverage.py
# Outputs: outputs/results/phase6_risk_coverage.json, outputs/figures/risk_coverage_curve.png
```

### Phase 8: Full Ablation Matrix
```bash
python scripts/run_ablations.py
# Outputs: outputs/results/phase8_ablations.json
```

---

## 5. Repository Structure

```
skeleton-har-uncertainty/
├── PROGRESS.md                 # Living status log
├── README.md                   # Setup & reproduction guide
├── environment.yml             # Conda environment spec
├── configs/                    # YAML configuration files
├── data/
│   ├── utd_mhad/               # Raw and preprocessed UTD-MHAD
│   └── ntu60/                  # NTU-60 dataset location
├── src/
│   ├── data/                   # Graph topology, datasets, augmentations
│   ├── models/                 # ST-GCN backbone, blocks, dropout
│   ├── uncertainty/            # MC-Dropout, Temperature Scaling, Conformal APS
│   ├── explainability/         # Perturbation, Integrated Gradients, Faithfulness
│   ├── eval/                   # Metrics, ECE, Brier, Risk-Coverage
│   └── utils/                  # 3D stick-figure visualizer, plotters
├── scripts/                    # Independent runnable experiment scripts
└── outputs/
    ├── checkpoints/            # Model weights
    ├── figures/                # Publication-quality plots
    └── results/                # JSON/CSV metric dumps
```
