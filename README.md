<<<<<<< HEAD
# Diabetic Retinopathy Detection From Retinal Images (LHTViT)

This repository contains the official implementation of the **LHTViT** (Lightweight Hybrid Transformer-Vision Network) framework developed for automated multi-class grading of Diabetic Retinopathy (DR) from retinal fundus images.

The architecture elegantly fuses an **EfficientNet** branch (for fine-grained local lesions like microaneurysms) with a **Vision Transformer (ViT)** branch (for global retinal context and spatial geometries) using **Multimodal Factorized Bilinear Pooling (MFB)**.

---

## 📂 Repository Structure

```text
├── Exp 6.2/                # Core folder for the proposed model configuration
│   ├── training_history_exp6_2.png   # Convergence, loss, and QWK history curves
│   └── confusion_matrix_multiclass.png # Five-category evaluation matrix 
├── src/                    # Source code directory containing modular python scripts
│   ├── model.py            # Dual-branch LHTViT network architecture with MFB fusion
│   ├── train.py            # Training loops, optimization routines, and early stopping
│   └── dataset.py          # Custom PyTorch DataLoader, image augmentation, and balancing
└── comparison/             # Cross-dataset benchmarking and validation scripts
    └── evaluate_sota.py    # Zero-shot inference scripts across external test sets
=======
\# Dual-Branch LHTViT Framework for Automated Multi-Class Diabetic Retinopathy Grading



\[!\[Framework](https://img.shields.io/badge/Framework-PyTorch%202.x-EE4C2C.svg)](https://pytorch.org/)

\[!\[GPU Acceleration](https://img.shields.io/badge/Compute-NVIDIA%20L4%2024GB-76B900.svg)](https://www.nvidia.com/)

\[!\[Academic Project](https://img.shields.io/badge/Research-Wrocław%20University%20of%20Science%20and%20Technology-005EA2.svg)](https://pwr.edu.pl/)



Official production-grade implementation of the \*\*LHTViT\*\* (Lightweight Hybrid Transformer-Vision Network) framework designed for multi-class Diabetic Retinopathy (DR) severity classification across five clinical categories (Grades 0–4).



\---



\## 📖 Architectural Paradigm

The LHTViT architecture solves the traditional performance trade-off between \*\*Local Feature Tracking\*\* and \*\*Global Context Awareness\*\* by utilizing a dual-backbone parallel processing topology:

1\. \*\*CNN Extractor Branch (EfficientNet-B0):\*\* Optimizes feature visibility of fine-grained localized retinal lesions (microaneurysms, hemorrhages, and hard exudates).

2\. \*\*Vision Transformer Branch (ViT-Tiny):\*\* Implements multi-head self-attention mechanism layers to extract global spatial orientation and macro-vascular geometries across the fundus.

3\. \*\*Multimodal Factorized Bilinear Pooling (MFB):\*\* Drives high-order bilinear interactions to cleanly fuse feature maps without triggering dimension conflicts or parameter explosions.



\---



\## 📂 Repository Layout and Deliverables

```text

LHTViT\_Project\_Hub/

├── src/

│   ├── models.py                   # LHTViT parallel network graph and MFB module

│   ├── dataset.py                  # PyTorch DataLoader with CLAHE and Stratified Sampling

│   ├── train.py                    # Optimization schedules and early-stopping parameters

│   └── utils.py                    # Evaluation metrics and utility scripts

├── Experiment\_6.2/

│   ├── confusion\_matrix\_multiclass.png  # Primary 5-class evaluation performance matrix

│   ├── ROC\_curves\_exp6\_2.png            # Per-class True Positive/False Positive rates

│   ├── train\_log\_exp6\_2.json            # Frame-by-frame epoch loss telemetry

│   └── training\_history\_exp6\_2.png      # Optimization loss vs convergence loops

└── comparison/

&#x20;   ├── benchmark\_comparison\_chart.png   # Cross-dataset visualization histogram

&#x20;   ├── benchmark\_loss\_curves.png        # Domain validation convergence behaviors

&#x20;   └── comparison\_table.png             # Full benchmarking matrix against SOTA baseline

>>>>>>> dbdce88 (feat: complete research workspace integration for LHTViT framework)
