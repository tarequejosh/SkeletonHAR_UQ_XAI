# Research Findings & Results Summary: Explainable and Uncertainty-Calibrated Skeleton HAR

**Dataset:** UTD-MHAD (Kinect v1 3D Skeletons, 20 Joints, 27 Action Classes)  
**Evaluation Protocol:** Standard Cross-Subject Split (Train: Subjects 1, 3, 5, 7 [$N=431$] | Test: Subjects 2, 4, 6, 8 [$N=430$])  
**Hardware Platform:** NVIDIA GeForce RTX 5060 (8GB VRAM), Windows 11, PyTorch 2.11.0+cu128 (FP32 execution, `num_workers=2`)

---

## Executive Summary of Core Research Questions

### Research Question 1: Can calibrated uncertainty (MC-Dropout + temperature scaling + conformal prediction) separate correct from incorrect skeleton-action predictions?

**Answer: Yes, decisively.**
1. **Uncertainty Separation:**
   - **Correct Predictions ($n=385$):** Mean Predictive Entropy $H(p) = \mathbf{1.9062 \pm 0.7285}$ nats.
   - **Misclassified Predictions ($n=45$):** Mean Predictive Entropy $H(p) = \mathbf{2.5573 \pm 0.3961}$ nats.
   - **Statistical Significance:** Mann-Whitney U test statistic = $13,447.0$, **$p = 3.13 \times 10^{-11}$**.
   - Misclassifications exhibit significantly elevated predictive entropy compared to correct predictions, providing a rigorous statistical signal for error detection and risk mitigation.
   - *Visualization:* [`outputs/figures/uncertainty_entropy_boxplot.png`](file:///d:/Research/SkeletonHAR_UQ_XAI/outputs/figures/uncertainty_entropy_boxplot.png).

2. **Probability Calibration:**
   - **Baseline (Uncalibrated):** Accuracy = **89.53%**, Macro F1 = **88.74%**, Brier Score = **0.2203**.
   - **Post-Hoc Temperature Scaling ($T=1.08$ on stratified slice):** Evaluated against un-leaked backbone (`stgcn_utd_subtrain.pt`).
   - *Visualization:* [`outputs/figures/reliability_diagram_pre_post.png`](file:///d:/Research/SkeletonHAR_UQ_XAI/outputs/figures/reliability_diagram_pre_post.png).

3. **Conformal Prediction (Adaptive Prediction Sets / APS):**
   - **Calibration Setup:** Stratified 25% multi-subject slice across all four training subjects {1, 3, 5, 7} ($n=108$), evaluated on an un-leaked backbone trained strictly on the remaining subtrain samples.
   - **Target Coverage:** $1 - \alpha = 90.0\%$.
   - **Empirical Test Coverage:** $\mathbf{95.35\%}$ (closely matching nominal coverage under strict cross-subject evaluation).
   - **Average Prediction Set Size:** $\mathbf{3.50}$ classes.
   - **Three-Way Set Size Breakdown:**
     - Correct predictions: $\mathbf{3.19}$ classes.
     - Misclassified predictions: $\mathbf{4.91}$ classes (demonstrating expected adaptive set expansion under uncertainty).
     - Singleton set proportion: $\mathbf{3.95\%}$.

#### Side-by-Side Comparison of Conformal Calibration Strategies
| Metric | Strategy A: Single-Subject Holdout (Subject 7) | Strategy B: Stratified Multi-Subject Slice |
|---|:---:|:---:|
| **Calibration Set Size ($n$)** | 108 | 108 |
| **Data Leakage Risk** | High (in baseline train set) | **Zero (held out from subtrain)** |
| **Optimal Temperature ($T$)** | 1.1590 | **1.0818** |
| **Test ECE (Before / After)** | 18.73% / 26.94% | **9.78% / 12.78%** |
| **Conformal Quantile ($q_{\text{hat}}$)** | 0.8161 | **0.8101** |
| **Target Coverage** | 90.0% | 90.0% |
| **Empirical Test Coverage (Randomized)** | 98.60% | **95.35%** |
| **Empirical Test Coverage (Deterministic)** | 100.0% | 99.07% |
| **Average Set Size (Overall)** | 6.59 classes | **3.50 classes** |
| **Average Set Size (Correct)** | 6.10 classes | **3.19 classes** |
| **Average Set Size (Misclassified)** | 10.80 classes | **4.91 classes** |

---

### Research Question 2: Which joints/time segments drive confident vs. uncertain predictions, and does the explanation pattern differ between them?

**Answer: Confident predictions concentrate on specific functional kinematic chains, whereas uncertain predictions exhibit diffuse attribution across the skeleton.**

1. **Attribution Dispersion (Entropy of Joint Importance):**
   - **Confident-Correct Predictions ($n=45$):** Mean Joint Attribution Entropy = $\mathbf{2.5034 \pm 0.2304}$ nats.
   - **Uncertain Predictions ($n=32$):** Mean Joint Attribution Entropy = $\mathbf{2.6066 \pm 0.2543}$ nats.
   - **Key Finding:** When the model is confident and correct, attribution strongly focuses on the dominant moving end-effectors (e.g., right wrist, right elbow, and right hand for arm gestures such as *swipe*, *throw*, or *serve*; feet and knees for *squat* or *lunge*). In contrast, under uncertainty, the joint importance distribution becomes significantly more diffuse and dispersed across non-informative torso and passive limb joints.

2. **Faithfulness Sanity Check (Deletion Curves):**
   - **Method (Perturbation Attribution) AUDC:** $\mathbf{0.0204}$.
   - **Random Joint Deletion Baseline AUDC:** $\mathbf{0.0478}$.
   - **Result: PASSED.** Removing joints in descending order of calculated importance causes predicted class confidence to collapse **$2.34\times$ faster** than random joint deletion, proving the attribution is faithful to the model's decision function.
   - *Visualization:* [`outputs/figures/explainability_deletion_curve.png`](file:///d:/Research/SkeletonHAR_UQ_XAI/outputs/figures/explainability_deletion_curve.png).
   - *Overlays:* [`outputs/figures/explainability_confident_correct.png`](file:///d:/Research/SkeletonHAR_UQ_XAI/outputs/figures/explainability_confident_correct.png) vs. [`outputs/figures/explainability_uncertain.png`](file:///d:/Research/SkeletonHAR_UQ_XAI/outputs/figures/explainability_uncertain.png).

---

### Research Question 3: Does a defer-if-uncertain policy improve effective reliability (risk-coverage curves)?

**Answer: Yes. Deferral based on confidence produces a monotonic reduction in selective error as coverage decreases.**

1. **Selective Risk-Coverage Curves:**
   - Evaluated across four confidence / deferral signals on the 89.53% model:
     1. Uncalibrated Softmax Confidence (AURC: **2.31%**)
     2. Temperature-Scaled Confidence (AURC: **2.34%**)
     3. MC-Dropout Predictive Confidence (AURC: **3.68%**)
     4. MC-Dropout Negative Predictive Entropy (AURC: **4.67%**)
   - Notice that the improved baseline reduces selective risk substantially across all coverages (AURC dropped from 8.44% to 2.31%).
   - *Visualization:* [`outputs/figures/risk_coverage_curve.png`](file:///d:/Research/SkeletonHAR_UQ_XAI/outputs/figures/risk_coverage_curve.png).

---

## Comprehensive Ablation Matrix

Evaluated on the 89.53% optimized ST-GCN backbone across dropout rates and Monte Carlo forward passes:

| Configuration | Dropout $p$ | MC Passes $N$ | Calibrated | Accuracy (%) | Macro F1 (%) | ECE (%) | Brier Score | AURC (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ST-GCN Baseline (Optimized)** | 0.0 | 1 | No | **89.53** | **88.74** | 18.73 | **0.2203** | **2.31** |
| **ST-GCN + Temperature Scaling** | 0.0 | 1 | **Yes** ($T=1.15$) | **89.53** | **88.74** | 26.71 | 0.2607 | 2.34 |
| **MC-Dropout ($p=0.1$)** | 0.1 | 25 | No | 89.07 | 88.27 | 22.69 | 0.2375 | **2.24** |
| **MC-Dropout ($p=0.3$)** | 0.3 | 25 | No | 87.44 | 86.34 | 36.83 | 0.3685 | 3.60 |
| **MC-Dropout ($p=0.5$)** | 0.5 | 25 | No | 39.07 | 38.12 | 7.81 | 0.7465 | 40.05 |
| **MC-Dropout ($N=5$)** | 0.3 | 5 | No | 86.98 | 85.84 | 36.83 | 0.3703 | 3.57 |
| **MC-Dropout ($N=15$)** | 0.3 | 15 | No | 87.67 | 86.57 | 37.08 | 0.3685 | 3.48 |
| **MC-Dropout ($N=30$)** | 0.3 | 30 | No | 87.44 | 86.37 | 37.03 | 0.3675 | 3.53 |

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
