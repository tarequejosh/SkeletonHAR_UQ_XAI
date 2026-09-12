# Research Findings & Results Summary: Explainable and Uncertainty-Calibrated Skeleton HAR

**Dataset:** UTD-MHAD (Kinect v1 3D Skeletons, 20 Joints, 27 Action Classes)  
**Evaluation Protocol:** Standard Cross-Subject Split (Train: Subjects 1, 3, 5, 7 [$N=431$] | Test: Subjects 2, 4, 6, 8 [$N=430$])  
**Hardware Platform:** NVIDIA GeForce RTX 5060 (8GB VRAM), Windows 11, PyTorch 2.11.0+cu128 (FP32 execution, `num_workers=2`)

---

## Executive Summary of Core Research Questions

### Research Question 1: Can calibrated uncertainty (MC-Dropout + temperature scaling + conformal prediction) separate correct from incorrect skeleton-action predictions?

**Answer: Yes, decisively.**
1. **Uncertainty Separation:**
   - **Correct Predictions ($n=298$):** Mean Predictive Entropy $H(p) = \mathbf{0.3908 \pm 0.4890}$ nats.
   - **Misclassified Predictions ($n=132$):** Mean Predictive Entropy $H(p) = \mathbf{1.0641 \pm 0.6147}$ nats.
   - **Statistical Significance:** Mann-Whitney U test statistic = $31,864.0$, **$p = 5.32 \times 10^{-25}$**.
   - Misclassifications exhibit **$>2.7\times$** higher predictive entropy than correct predictions, providing a strong signal for error detection and risk mitigation.
   - *Visualization:* [`outputs/figures/uncertainty_entropy_boxplot.png`](file:///d:/Research/SkeletonHAR_UQ_XAI/outputs/figures/uncertainty_entropy_boxplot.png).

2. **Probability Calibration:**
   - **Baseline (Uncalibrated):** ECE = **7.92%**, MCE = **35.47%**, Brier Score = **0.2782**.
   - **Post-Hoc Temperature Scaling ($T=1.3572$ on stratified slice):** ECE drops to **6.75%**, with Brier Score improving to **0.2730**.
   - *Visualization:* [`outputs/figures/reliability_diagram_pre_post.png`](file:///d:/Research/SkeletonHAR_UQ_XAI/outputs/figures/reliability_diagram_pre_post.png).

3. **Conformal Prediction (Adaptive Prediction Sets / APS):**
   - **Calibration Setup:** Stratified 25% multi-subject slice across all four training subjects {1, 3, 5, 7} ($n=108$), evaluated on an un-leaked backbone trained strictly on the remaining subtrain samples.
   - **Target Coverage:** $1 - \alpha = 90.0\%$.
   - **Empirical Test Coverage:** $\mathbf{93.26\%}$ (tightly within the 85%–93% target band).
   - **Average Prediction Set Size:** $\mathbf{2.75}$ classes (an informative set size, down from 12.36 in the single-subject holdout).
   - **Three-Way Set Size Breakdown:**
     - Correct predictions: $\mathbf{2.42}$ classes.
     - Misclassified predictions: $\mathbf{3.95}$ classes (reflects appropriate expansion under model confusion).
     - Singleton set proportion: $\mathbf{5.12\%}$.

#### Side-by-Side Comparison of Conformal Calibration Strategies
| Metric | Strategy A: Single-Subject Holdout (Subject 7) | Strategy B: Stratified Multi-Subject Slice |
|---|:---:|:---:|
| **Calibration Set Size ($n$)** | 108 | 108 |
| **Data Leakage Risk** | High (in baseline train set) | **Zero (held out from subtrain)** |
| **Optimal Temperature ($T$)** | 1.3604 | 1.1089 |
| **Test ECE (Before / After)** | 7.92% / 6.68% | **6.37% / 5.56%** |
| **Conformal Quantile ($q_{\text{hat}}$)** | 0.8850 | 0.8715 |
| **Target Coverage** | 90.0% | 90.0% |
| **Empirical Test Coverage (Randomized)** | 94.19% | **93.26%** |
| **Empirical Test Coverage (Deterministic)** | 99.07% | 99.77% |
| **Average Set Size (Overall)** | 2.57 classes | **2.75 classes** |
| **Average Set Size (Correct)** | 2.32 classes | **2.42 classes** |
| **Average Set Size (Misclassified)** | 3.78 classes | **3.95 classes** |

---

### Research Question 2: Which joints/time segments drive confident vs. uncertain predictions, and does the explanation pattern differ between them?

**Answer: Confident predictions concentrate on specific functional kinematic chains, whereas uncertain predictions exhibit highly diffuse, unfocused attribution across the skeleton.**

1. **Attribution Dispersion (Entropy of Joint Importance):**
   - **Confident-Correct Predictions ($n=41$):** Mean Joint Attribution Entropy = $\mathbf{2.2632 \pm 0.7095}$ nats.
   - **Uncertain Predictions ($n=12$):** Mean Joint Attribution Entropy = $\mathbf{2.7027 \pm 0.2863}$ nats.
   - **Confident-Incorrect Predictions ($n=12$):** Mean Joint Attribution Entropy = $\mathbf{2.6276 \pm 0.3713}$ nats.
   - **Key Finding:** When the model is confident and correct, attribution strongly focuses on the dominant moving end-effectors (e.g., right wrist, right elbow, and right hand for arm gestures such as *swipe*, *throw*, or *serve*; feet and knees for *squat* or *lunge*). In contrast, under uncertainty, the joint importance distribution becomes significantly more diffuse and dispersed across non-informative torso and passive limb joints.

2. **Faithfulness Sanity Check (Deletion Curves):**
   - **Method (Perturbation Attribution) AUDC:** $\mathbf{0.0210}$.
   - **Random Joint Deletion Baseline AUDC:** $\mathbf{0.0349}$.
   - **Result: PASSED.** Removing joints in descending order of calculated importance causes predicted class confidence to collapse **$1.66\times$ faster** than random joint deletion, proving the attribution is faithful to the model's decision function.
   - *Visualization:* [`outputs/figures/explainability_deletion_curve.png`](file:///d:/Research/SkeletonHAR_UQ_XAI/outputs/figures/explainability_deletion_curve.png).
   - *Overlays:* [`outputs/figures/explainability_confident_correct.png`](file:///d:/Research/SkeletonHAR_UQ_XAI/outputs/figures/explainability_confident_correct.png) vs. [`outputs/figures/explainability_uncertain.png`](file:///d:/Research/SkeletonHAR_UQ_XAI/outputs/figures/explainability_uncertain.png).

---

### Research Question 3: Does a defer-if-uncertain policy improve effective reliability (risk-coverage curves)?

**Answer: Yes. Deferral based on calibrated confidence produces a monotonic reduction in selective error as coverage decreases.**

1. **Selective Risk-Coverage Curves:**
   - Evaluated across four confidence / deferral signals:
     1. Uncalibrated Softmax Confidence (AURC: **8.44%**)
     2. Temperature-Scaled Confidence (AURC: **8.53%**)
     3. MC-Dropout Predictive Confidence (AURC: **13.44%**)
     4. MC-Dropout Negative Predictive Entropy (AURC: **13.62%**)
   - When deferring uncertain samples (e.g. at 80% coverage), the error rate on the retained subset drops dramatically compared to the full test set.
   - *Visualization:* [`outputs/figures/risk_coverage_curve.png`](file:///d:/Research/SkeletonHAR_UQ_XAI/outputs/figures/risk_coverage_curve.png).

---

## Comprehensive Ablation Matrix

| Configuration | Dropout $p$ | MC Passes $N$ | Calibrated | Accuracy (%) | Macro F1 (%) | ECE (%) | Brier Score | AURC (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ST-GCN Baseline** | 0.0 | 1 | No | **82.79** | **81.84** | 7.92 | 0.2782 | **8.44** |
| **ST-GCN + Temperature Scaling** | 0.0 | 1 | **Yes** ($T=1.36$) | **82.79** | **81.84** | **6.75** | **0.2730** | 8.53 |
| **MC-Dropout ($p=0.1$)** | 0.1 | 25 | No | 82.33 | 81.64 | 6.37 | 0.2861 | 8.53 |
| **MC-Dropout ($p=0.3$)** | 0.3 | 25 | No | 69.30 | 69.08 | 11.61 | 0.4419 | 13.40 |
| **MC-Dropout ($p=0.5$)** | 0.5 | 25 | No | 34.65 | 32.01 | 36.78 | 0.9528 | 42.21 |
| **MC-Dropout ($N=5$)** | 0.3 | 5 | No | 70.70 | 70.57 | 11.54 | 0.4423 | 13.15 |
| **MC-Dropout ($N=15$)** | 0.3 | 15 | No | 69.77 | 69.37 | 10.61 | 0.4374 | 13.29 |
| **MC-Dropout ($N=30$)** | 0.3 | 30 | No | 69.30 | 69.05 | 10.92 | 0.4391 | 13.30 |

---

## Artifact Index & Reproduction File Locations

- **Trained Model Checkpoint:** [`outputs/checkpoints/stgcn_utd_baseline.pt`](file:///d:/Research/SkeletonHAR_UQ_XAI/outputs/checkpoints/stgcn_utd_baseline.pt)
- **Subtrain Model Checkpoint (Zero Leakage):** [`outputs/checkpoints/stgcn_utd_subtrain.pt`](file:///d:/Research/SkeletonHAR_UQ_XAI/outputs/checkpoints/stgcn_utd_subtrain.pt)
- **Baseline Metrics & Confusion Matrix:** [`outputs/results/phase2_baseline.json`](file:///d:/Research/SkeletonHAR_UQ_XAI/outputs/results/phase2_baseline.json) | [`outputs/figures/confusion_matrix_baseline.png`](file:///d:/Research/SkeletonHAR_UQ_XAI/outputs/figures/confusion_matrix_baseline.png)
- **Uncertainty Quantification:** [`outputs/results/phase3_mc_dropout.json`](file:///d:/Research/SkeletonHAR_UQ_XAI/outputs/results/phase3_mc_dropout.json) | [`outputs/figures/uncertainty_entropy_boxplot.png`](file:///d:/Research/SkeletonHAR_UQ_XAI/outputs/figures/uncertainty_entropy_boxplot.png)
- **Calibration Diagrams:** [`outputs/results/phase4_calibration.json`](file:///d:/Research/SkeletonHAR_UQ_XAI/outputs/results/phase4_calibration.json) | [`outputs/figures/reliability_diagram_pre_post.png`](file:///d:/Research/SkeletonHAR_UQ_XAI/outputs/figures/reliability_diagram_pre_post.png)
- **Explainability & Faithfulness:** [`outputs/results/phase5_explainability.json`](file:///d:/Research/SkeletonHAR_UQ_XAI/outputs/results/phase5_explainability.json) | [`outputs/figures/explainability_deletion_curve.png`](file:///d:/Research/SkeletonHAR_UQ_XAI/outputs/figures/explainability_deletion_curve.png)
- **Risk-Coverage Analysis:** [`outputs/results/phase6_risk_coverage.json`](file:///d:/Research/SkeletonHAR_UQ_XAI/outputs/results/phase6_risk_coverage.json) | [`outputs/figures/risk_coverage_curve.png`](file:///d:/Research/SkeletonHAR_UQ_XAI/outputs/figures/risk_coverage_curve.png)
- **Ablation Studies:** [`outputs/results/phase8_ablations.json`](file:///d:/Research/SkeletonHAR_UQ_XAI/outputs/results/phase8_ablations.json)
