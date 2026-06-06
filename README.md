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
