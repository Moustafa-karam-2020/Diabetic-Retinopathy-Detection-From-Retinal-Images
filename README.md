# Dual-Branch LHTViT Framework for Automated Multi-Class Diabetic Retinopathy Grading

[![Framework](https://img.shields.io/badge/Framework-PyTorch%202.x-EE4C2C.svg)](https://pytorch.org/)
[![GPU Acceleration](https://img.shields.io/badge/Compute-NVIDIA%20L4%2024GB-76B900.svg)](https://www.nvidia.com/)
[![Academic Project](https://img.shields.io/badge/Research-Wrocław%20University%20of%20Science%20and%20Technology-005EA2.svg)](https://pwr.edu.pl/)

Official production-grade implementation of the **LHTViT** (Lightweight Hybrid Transformer-Vision Network) framework designed for multi-class Diabetic Retinopathy (DR) severity classification across five clinical categories (Grades 0–4).

---

## 📖 Architectural Paradigm
The LHTViT architecture solves the traditional performance trade-off between **Local Feature Tracking** and **Global Context Awareness** by utilizing a dual-backbone parallel processing topology:
1. **CNN Extractor Branch (EfficientNet-B0):** Optimizes feature visibility of fine-grained localized retinal lesions (microaneurysms, hemorrhages, and hard exudates).
2. **Vision Transformer Branch (ViT-Tiny):** Implements multi-head self-attention mechanism layers to extract global spatial orientation and macro-vascular geometries across the fundus.
3. **Multimodal Factorized Bilinear Pooling (MFB):** Drives high-order bilinear interactions to cleanly fuse feature maps without triggering dimension conflicts or parameter explosions.

---

## 📂 Repository Layout and Deliverables
```text
LHTViT_Project_Hub/
├── src/
│   ├── models.py                   # LHTViT parallel network graph and MFB module
│   ├── dataset.py                  # PyTorch DataLoader with CLAHE and Stratified Sampling
│   ├── train.py                    # Optimization schedules and early-stopping parameters
│   └── utils.py                    # Evaluation metrics and utility scripts
├── Experiment_6.2/
│   ├── confusion_matrix_multiclass.png  # Primary 5-class evaluation performance matrix
│   ├── ROC_curves_exp6_2.png            # Per-class True Positive/False Positive rates
│   ├── train_log_exp6_2.json            # Frame-by-frame epoch loss telemetry
│   └── training_history_exp6_2.png      # Optimization loss vs convergence loops
└── comparison/
    ├── benchmark_comparison_chart.png   # Cross-dataset visualization histogram
    ├── benchmark_loss_curves.png        # Domain validation convergence behaviors
    └── comparison_table.png             # Full benchmarking matrix against SOTA baseline