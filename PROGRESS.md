# Project Progress Log: Explainable and Uncertainty-Calibrated Skeleton HAR

**Last Updated:** 2026-09-13  
**Status:** **ALL PHASES COMPLETED (0 THROUGH 8)**  
**Machine:** Windows 11, RTX 5060 (8GB VRAM), PyTorch 2.11.0+cu128, Python 3.11.15  
**Hardware Safeguards Applied:**
- `num_workers = 2` (prevented WinError 1455)
- Strict FP32 training (no AMP / mixed precision; 0 NaNs)
- MC-dropout sequential stochastic inference loop to preserve 8GB VRAM

---

## Phase Status Summary

| Phase | Description | Status | Key Results / Checkpoints |
|---|---|:---:|---|
| **Phase 0** | Environment Setup & Verification | **COMPLETED** | GPU verified: RTX 5060 (8GB), PyTorch 2.11.0+cu128, Captum installed, matmul verified |
| **Phase 1** | UTD-MHAD Acquisition & Preprocessing | **COMPLETED** | 861 `.mat` files; 27 classes; Train (431), Test (430); 3D stick-figure sanity check generated |
| **Phase 2** | ST-GCN Baseline Implementation & Training | **COMPLETED** | Cross-Subject Acc: **82.79%**, Macro F1: **81.84%**, ECE: 7.92%, Brier: 0.2782 (`stgcn_utd_baseline.pt`) |
| **Phase 3** | MC-Dropout Uncertainty Quantification | **COMPLETED** | Statistically significant error separation ($p = 5.32 \times 10^{-25}$); error entropy ($1.064$) vs correct ($0.391$) |
| **Phase 4** | Calibration (Temperature Scaling + Conformal Prediction) | **COMPLETED** | Temperature scaling $T=1.360$ reduced ECE from **7.92%** to **6.68%**; Conformal APS coverage = **99.07%** |
| **Phase 5** | Explainability (Perturbation + Faithfulness) | **COMPLETED** | Faithfulness check passed (AUDC $0.0210$ vs $0.0349$ random); Uncertain entropy higher ($2.70$ vs $2.26$) |
| **Phase 6** | Selective Prediction / Risk-Coverage Analysis | **COMPLETED** | Generated risk-coverage curves; AURC evaluated for uncalibrated, temperature scaled, and MC-dropout |
| **Phase 7** | NTU RGB+D 60 Scale-Up Preparation | **COMPLETED** | Dataset loader supporting CS/CV protocols and 25-joint ST-GCN graph architecture unit-tested on synthetic data |
| **Phase 8** | Ablations & Final Report Assembly | **COMPLETED** | Multi-condition ablation matrix generated; `RESULTS_SUMMARY.md` authored answering all 3 research questions |

---

## Detailed Chronological Log

### Phase 0: Environment Setup
- Verified Conda environment `research` with Python 3.11.15 and PyTorch 2.11.0+cu128.
- Tested CUDA execution on RTX 5060; GPU matrix multiplication test passed.
- Installed `captum` 0.9.0 via pip into `research` environment.
- Created repository directories (`data/`, `src/`, `scripts/`, `configs/`, `outputs/`).
- Authored `environment.yml` and `scripts/verify_env.py`.

### Phase 1: Dataset Acquisition & Preprocessing (UTD-MHAD)
- Downloaded `Skeleton.zip` (14.7 MB) from UT Dallas official site.
- Verified 861 raw `.mat` files spanning 27 actions, 8 subjects, 4 trials (0 NaNs).
- Preprocessed to uniform $T=64$ frames via linear interpolation and normalized relative to Hip Center (root joint 0).
- Standard cross-subject split built:
  - Train subjects: {1, 3, 5, 7} -> 431 samples
  - Test subjects: {2, 4, 6, 8} -> 430 samples
- Implemented `UTDMHADDataset` and `get_utd_loaders` with 3D rotation and jitter augmentations, verified with `num_workers=2`.
- Generated 3D stick-figure sequence sanity check to `outputs/figures/sanity_check_utd_mhad.png`.

### Phase 2: ST-GCN Baseline Implementation & Training
- Implemented Spatial-Temporal Graph Convolutional Network (ST-GCN) with Kinect v1 20-joint kinematic graph and 3-partition spatial configuration.
- Trained in FP32 on RTX 5060 for 60 epochs with Cosine Annealing learning rate schedule.
- Final test performance on cross-subject split:
  - Top-1 Accuracy: **82.79%**
  - Macro F1-score: **81.84%**
  - Weighted F1-score: **82.79%**
  - Expected Calibration Error (ECE): **7.92%**
  - Brier Score: **0.2782**
- Saved best checkpoint to `outputs/checkpoints/stgcn_utd_baseline.pt`.
- Generated normalized confusion matrix to `outputs/figures/confusion_matrix_baseline.png`.
- Saved metric report to `outputs/results/phase2_baseline.json`.

### Phase 3: MC-Dropout Uncertainty Quantification
- Evaluated stochastic forward passes ($N=25$) with dropout.
- Demonstrated strong predictive entropy separation between correct predictions ($0.3908$) and misclassifications ($1.0641$).
- Mann-Whitney U test confirmed statistical significance ($p = 5.32 \times 10^{-25}$).
- Generated boxplot to `outputs/figures/uncertainty_entropy_boxplot.png` and logged results to `outputs/results/phase3_mc_dropout.json`.

### Phase 4: Calibration (Temperature Scaling & Conformal Prediction)
- Partitioned Subject 7 (108 samples) from the training split as a held-out calibration set.
- Fit optimal scalar temperature $T = 1.3604$ on calibration logits via NLL minimization.
- Post-hoc temperature scaling reduced Test ECE from **7.92%** to **6.68%** and MCE from **35.47%** to **29.85%**.
- Implemented Adaptive Prediction Sets (APS) conformal prediction: target coverage 90% yielded empirical test coverage of **99.07%**.
- Saved reliability diagrams to `outputs/figures/reliability_diagram_pre_post.png` and results to `outputs/results/phase4_calibration.json`.

### Phase 5: Explainability & Faithfulness Sanity Check
- Implemented perturbation-based joint importance and Integrated Gradients.
- Evaluated progressive joint deletion curves: method AUDC (**0.0210**) was substantially lower than random deletion AUDC (**0.0349**), passing the faithfulness sanity check.
- Comparative analysis across uncertainty tiers showed that uncertain predictions have significantly higher joint attribution entropy (**2.7027** vs. **2.2632**), demonstrating diffuse and unstable explanations under uncertainty.
- Saved deletion curve to `outputs/figures/explainability_deletion_curve.png`, qualitative skeleton heatmaps to `outputs/figures/`, and metrics to `outputs/results/phase5_explainability.json`.

### Phase 6: Selective Prediction / Risk-Coverage Analysis
- Swept rejection threshold across deferral signals: uncalibrated softmax confidence, temperature-scaled confidence, and MC-dropout predictive entropy.
- Generated risk-coverage curves showing monotonic error reduction on retained predictions as rejection threshold increases.
- Saved curve plot to `outputs/figures/risk_coverage_curve.png` and data to `outputs/results/phase6_risk_coverage.json`.

### Phase 7: NTU RGB+D 60 Scale-Up Preparation (Gated Dataset)
- Verified `data/ntu60/` status (awaiting human academic license approval from ROSE Lab NTU Singapore).
- Implemented complete dataset loader in `src/data/ntu_dataset.py` supporting 25 Kinect v2 joints, Cross-Subject (CS), and Cross-View (CV) protocols.
- Authored and executed unit test `scripts/test_ntu_loader.py` validating filename parsing, 25-joint graph topology, and 60-class ST-GCN forward passes.

### Phase 8: Full Ablation Studies & Final Synthesis
- Executed ablation grid over dropout rates ($p \in \{0.1, 0.3, 0.5\}$), number of MC passes ($N \in \{5, 15, 30\}$), and calibration methods.
- Authored `outputs/RESULTS_SUMMARY.md` answering all three research questions with quantitative tables and figure references.
- Authored `README.md` with complete reproduction commands.
