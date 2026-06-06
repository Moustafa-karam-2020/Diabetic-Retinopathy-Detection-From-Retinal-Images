
"""
============================================================================
EXPERIMENT 7 — LHT-ViT  Final Master Run  (Parallel-Curve Generalization)
   EfficientNet-B0 + ViT-Small/16  ·  Bidirectional Co-Attention
   MFB Pooling (K=5, inner Dropout=0.5)
   Standard CrossEntropyLoss (no label smoothing)
   AdamW(lr=3e-5, weight_decay=5e-4)  ·  Dropout(0.40) before head
   LambdaLR : 3-epoch Linear Warmup → Cosine Annealing
   40-Epoch Max  ·  Early Stopping on VAL LOSS (patience=6)
   Checkpoint on best VAL QWK → best_model_exp7.pth
                   Google Colab Pro  (NVIDIA L4)
============================================================================

CRITICAL: the core dual-stream LHT-ViT architecture is UNCHANGED from the
previous build — the pretrained EfficientNet-B0 + ViT-Small extractors, the
bidirectional Co-Attention alignment, and the MFB pooling layer (rank K=5)
are all preserved verbatim.  Only the training loop, augmentations, and
regularisation knobs around the model have been modified.

ARCHITECTURE OVERVIEW
---------------------
Fundus image  (B, 3, 224, 224)
    │
    ├── CNN BRANCH ── EfficientNet-B0  (~4 M params, ImageNet pretrained)
    │       └── forward_features → (B, 1280, 7, 7)
    │             → flatten → 49 spatial tokens × 1280-dim
    │
    ├── TRANSFORMER BRANCH ── ViT-Small/16-224  (~22 M params, ImageNet)
    │       └── forward_features → (B, 197, 384) → drop CLS → 196 patches
    │             drop_path_rate = 0.40  (stochastic depth per block)
    │             attn_drop_rate = 0.15  (attention-weight dropout)
    │
    ├── CO-ATTENTION BLOCK ── bidirectional spatial cross-modal attention
    │       cnn_p [B,49,512]  ←→  vit_p [B,196,512]
    │       residual + LayerNorm → mean-pool → cnn_vec, vit_vec ∈ ℝ^512
    │
    ├── MFB POOLING ── Multi-modal Factorized Bilinear (K=5, out=1024)
    │       z = (Wq·q ⊙ Wv·v) summed over K ranks
    │       → Dropout(0.5)   ← projection-layer regulariser
    │       → sign(z)·√|z|  then L2-norm  →  fused ∈ ℝ^1024
    │
    └── HEAD : LayerNorm → Linear(1024→512) → GELU → Dropout(0.40)
               → Linear(512→5)   raw logits (standard CrossEntropyLoss)

TRAINING-EXECUTION CHANGES  (anti-overfitting stabilizers)
----------------------------------------------------------
1. Augmentation   : NONE — Mixup and CutMix removed.  Generalization is
                    handled structurally via Dropout(0.40) + weight_decay.

2. Regularisation : standard CrossEntropyLoss (no label smoothing);
                    nn.Dropout(p=0.40) before the final classifier;
                    MFB internal Dropout(0.50); ViT drop_path=0.40,
                    attn_drop=0.15.

3. Optimizer      : AdamW(lr=3e-5, weight_decay=5e-4).

4. Schedule       : 40 epochs max; LambdaLR linear-warmup(3) → cosine decay.

5. Early Stopping : monitors VAL LOSS, patience=6.  Halts before the loss
                    gap can widen irreversibly.
   Checkpoint     : 'best_model_exp7.pth' saved on best VALIDATION QWK,
                    ensuring the stored weights are the peak-thesis weights.

6. Metrics        : Accuracy and QWK computed purely from model logits via
                    scikit-learn — no multipliers or artificial adjustments.

7. AMP            : autocast + GradScaler in both train and eval loops.
   DataLoaders    : num_workers=4, pin_memory=True, persistent_workers=True.

Outputs → /content/drive/MyDrive/DR_Experiment_7_Outputs/
============================================================================
"""

# =============================================================================
# STEP 0 — ENVIRONMENT
# =============================================================================

import gc
import json
import math
import os
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

import timm

from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import label_binarize
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ── Google Drive mount (Colab only) ───────────────────────────────────────────
#try:
#    from google.colab import drive
#    drive.mount("/content/drive", force_remount=False)
#    print("Google Drive mounted.")
#except ImportError:
#    print("Not running on Colab — Drive mount skipped.")

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# =============================================================================
# STEP 1 — CONFIGURATION
# =============================================================================

CFG = dict(
    # ── Paths ─────────────────────────────────────────────────────────────
    data_dir   = "/content/dataset/augmented_resized_V2/train",
    output_dir = "/content/drive/MyDrive/DR_Experiment_7_Outputs",
    cache_dir  = "/content/cache/preproc",   # reuse prior cache — no rebuild

    # ── Data ──────────────────────────────────────────────────────────────
    num_samples = 80_000,
    image_size  = 224,
    batch_size  = 64,
    num_workers = 4,
    pin_memory  = True,
    seed        = SEED,

    # ── Training ──────────────────────────────────────────────────────────
    epochs        = 40,              # max boundary; early stopping cuts earlier
    lr            = 3e-5,            # stable global LR (after warmup)
    weight_decay  = 5e-4,            # AdamW L2 penalty to close the loss gap
    eta_min       = 1e-7,
    patience      = 6,               # early-stop patience on VAL LOSS

    # LambdaLR schedule
    warmup_epochs = 3,               # linear ramp 0 → lr over first 3 epochs

    # ── Architecture (UNCHANGED — base network intact) ─────────────────────
    attn_dim       = 512,
    mfb_k          = 5,
    mfb_out        = 1024,
    num_classes    = 5,
    mfb_dropout    = 0.50,           # inside MFB after K-rank sum-pool
    head_dropout   = 0.40,           # nn.Dropout(p=0.4) before final classifier
    drop_path_rate = 0.40,
    attn_drop_rate = 0.15,
)

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

out_dir = Path(CFG["output_dir"])
out_dir.mkdir(parents=True, exist_ok=True)
Path(CFG["cache_dir"]).mkdir(parents=True, exist_ok=True)

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available()

print("=" * 72)
print("  EXPERIMENT 7 — LHT-ViT  Final Master Run")
print("  Standard CrossEntropyLoss  ·  Dropout(0.40)  ·  LambdaLR Warmup+Cosine")
print(f"  40-Epoch Max  ·  L4 GPU  ·  AMP: {use_amp}")
print("=" * 72)
print(f"  Device       : {device}")
print(f"  Output dir   : {CFG['output_dir']}")
print(f"  Data         : {CFG['num_samples']:,} images  |  "
      f"batch={CFG['batch_size']}  workers={CFG['num_workers']}")
print(f"  Epochs (max) : {CFG['epochs']}   "
      f"Patience: {CFG['patience']} (early-stop on val loss)")
print(f"  Optimizer    : AdamW(lr={CFG['lr']}, wd={CFG['weight_decay']})")
print(f"  Scheduler    : LambdaLR — linear warmup {CFG['warmup_epochs']} ep → cosine")
print(f"  Loss         : CrossEntropyLoss()  (no label smoothing)")
print(f"  Augmentation : none  (pure model generalization via Dropout + weight_decay)")
print(f"  ViT drops    : drop_path={CFG['drop_path_rate']}  "
      f"attn_drop={CFG['attn_drop_rate']}")
print(f"  MFB dropout  : {CFG['mfb_dropout']}   "
      f"Head dropout: {CFG['head_dropout']}")
print(f"  Checkpoint   : best_model_exp7.pth — saved on best VAL QWK")
print()


# =============================================================================
# STEP 2 — MODEL
# =============================================================================

class CoAttention(nn.Module):
    """
    Bidirectional spatial cross-modal attention (CNN ↔ ViT token streams).

    Projects both modalities into a shared attn_dim space, computes scaled
    dot-product attention in each direction, and returns a mean-pooled
    vector per modality.
    """

    def __init__(self, cnn_dim: int, vit_dim: int, attn_dim: int = 512):
        super().__init__()
        self.proj_cnn = nn.Linear(cnn_dim, attn_dim)
        self.proj_vit = nn.Linear(vit_dim, attn_dim)
        self.norm_cnn = nn.LayerNorm(attn_dim)
        self.norm_vit = nn.LayerNorm(attn_dim)

    def forward(
        self,
        cnn_tokens: torch.Tensor,   # [B, 49,  cnn_dim]
        vit_tokens: torch.Tensor,   # [B, 196, vit_dim]
    ):
        cnn_p = self.proj_cnn(cnn_tokens)   # [B, 49,  attn_dim]
        vit_p = self.proj_vit(vit_tokens)   # [B, 196, attn_dim]
        scale = cnn_p.size(-1) ** 0.5

        # CNN attends to ViT patches
        attn_c2v = torch.softmax(
            torch.bmm(cnn_p, vit_p.transpose(1, 2)) / scale, dim=-1)  # [B,49,196]
        cnn_att  = self.norm_cnn(cnn_p + torch.bmm(attn_c2v, vit_p))  # [B,49,D]

        # ViT patches attend to CNN spatial tokens
        attn_v2c = torch.softmax(
            torch.bmm(vit_p, cnn_p.transpose(1, 2)) / scale, dim=-1)  # [B,196,49]
        vit_att  = self.norm_vit(vit_p + torch.bmm(attn_v2c, cnn_p))  # [B,196,D]

        # Mean-pool over token dimension → one vector per modality
        return cnn_att.mean(dim=1), vit_att.mean(dim=1)                # [B, D] each


class MFBPooling(nn.Module):
    """
    Multi-modal Factorized Bilinear pooling  (rank K=5, output=1024).

    Experiment 7 addition
    ---------------------
    nn.Dropout(p=mfb_dropout) is applied immediately after the K-rank
    sum-pooling step and before the sign-sqrt normalisation.  This forces
    different subsets of bilinear rank factors to be active on each forward
    pass, preventing the factors from co-adapting to static token patterns
    that appear only in the training set.

    Forward
    -------
    z_q = Wq · q    [B, mfb_out × K]  → reshape → [B, mfb_out, K]
    z_v = Wv · v    [B, mfb_out × K]  → reshape → [B, mfb_out, K]
    z   = sum_K(z_q ⊙ z_v)            [B, mfb_out]
    z   = Dropout(z)                   ← Exp 7 regulariser
    z   = sign(z) · √|z|               (signed square-root trick)
    z   = L2-normalise(z)              [B, mfb_out]
    """

    def __init__(
        self,
        dim_q      : int   = 512,
        dim_v      : int   = 512,
        mfb_k      : int   = 5,
        mfb_out    : int   = 1024,
        mfb_dropout: float = 0.40,
    ):
        super().__init__()
        self.K       = mfb_k
        self.mfb_out = mfb_out
        self.proj_q  = nn.Linear(dim_q, mfb_out * mfb_k)
        self.proj_v  = nn.Linear(dim_v, mfb_out * mfb_k)
        self.dropout = nn.Dropout(p=mfb_dropout)

    def forward(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B    = q.size(0)
        z_q  = self.proj_q(q).view(B, self.mfb_out, self.K)   # [B, D, K]
        z_v  = self.proj_v(v).view(B, self.mfb_out, self.K)   # [B, D, K]
        z    = (z_q * z_v).sum(dim=-1)                         # [B, D]  sum-pool
        z    = self.dropout(z)                                  # ← Exp 7
        z    = torch.sign(z) * torch.sqrt(torch.abs(z) + 1e-8) # signed sqrt
        z    = F.normalize(z, p=2, dim=-1)                     # L2 norm
        return z                                                # [B, mfb_out]


class HybridLHT_ViT(nn.Module):
    """
    Hybrid Local-Hierarchical Transformer + Vision Transformer.
    5-class classifier for Diabetic Retinopathy grading  (Experiment 7).

    Returns raw logits [B, 5].  Loss is applied externally.
    """

    def __init__(
        self,
        attn_dim       : int   = 512,
        mfb_k          : int   = 5,
        mfb_out        : int   = 1024,
        num_classes    : int   = 5,
        mfb_dropout    : float = 0.40,
        head_dropout   : float = 0.60,
        drop_path_rate : float = 0.40,
        attn_drop_rate : float = 0.15,
        pretrained     : bool  = True,
    ):
        super().__init__()

        # ── Backbones ──────────────────────────────────────────────────────
        self.cnn = timm.create_model(
            "efficientnet_b0",
            pretrained  = pretrained,
            num_classes = 0,        # remove classifier head
        )
        self.vit = timm.create_model(
            "vit_small_patch16_224",
            pretrained      = pretrained,
            num_classes     = 0,
            drop_path_rate  = drop_path_rate,
            attn_drop_rate  = attn_drop_rate,
        )
        cnn_dim = self.cnn.num_features   # 1280
        vit_dim = self.vit.embed_dim      # 384

        # ── Fusion ─────────────────────────────────────────────────────────
        self.coattn = CoAttention(cnn_dim, vit_dim, attn_dim=attn_dim)
        self.mfb    = MFBPooling(
            dim_q       = attn_dim,
            dim_v       = attn_dim,
            mfb_k       = mfb_k,
            mfb_out     = mfb_out,
            mfb_dropout = mfb_dropout,
        )

        # ── Classification head ────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(mfb_out),
            nn.Linear(mfb_out, 512),
            nn.GELU(),
            nn.Dropout(p=head_dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN stream
        cnn_map    = self.cnn.forward_features(x)            # [B, 1280, 7, 7]
        cnn_tokens = cnn_map.flatten(2).transpose(1, 2)      # [B, 49,   1280]

        # ViT stream  (drop the [CLS] token at index 0)
        vit_seq    = self.vit.forward_features(x)            # [B, 197, 384]
        vit_tokens = vit_seq[:, 1:, :]                       # [B, 196, 384]

        # Cross-modal fusion
        cnn_vec, vit_vec = self.coattn(cnn_tokens, vit_tokens)
        fused = self.mfb(cnn_vec, vit_vec)                   # [B, 1024]

        return self.head(fused)                               # [B, 5] logits


# =============================================================================
# STEP 3 — PREPROCESSING & SSD CACHE
# =============================================================================

def ben_graham_preprocess(image: np.ndarray, sigma_x: int = 10) -> np.ndarray:
    """Circular crop + Gaussian-blend contrast normalisation (Graham 2015)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        image = image[y:y + h, x:x + w]
    return cv2.addWeighted(image, 4,
                           cv2.GaussianBlur(image, (0, 0), sigma_x), -4, 128)


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Per-channel CLAHE contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out   = np.empty_like(image)
    for c in range(3):
        out[:, :, c] = clahe.apply(image[:, :, c])
    return out


def _preprocess_single(img_path: str, target_size: int = 224) -> np.ndarray:
    """Full preprocessing pipeline for a single image. Returns uint8 RGB."""
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    img = ben_graham_preprocess(img)
    img = apply_clahe(img)
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return img.astype(np.uint8)


# =============================================================================
# STEP 4 — DATASET & DATALOADERS
# =============================================================================

def get_train_transforms() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ColorJitter(brightness=0.10, contrast=0.10, p=0.30),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class LazyDRDataset(Dataset):
    """
    On-demand dataset for Diabetic Retinopathy images.

    Only the list of (path, label) pairs is held in RAM.
    Each __getitem__ reads the preprocessed .npy file from the SSD cache
    (or runs the preprocessing pipeline on first access and saves to cache).
    This keeps the RAM footprint well below the DataLoader fork budget.
    """

    def __init__(
        self,
        records    : list,
        cache_dir  : str,
        transforms : A.Compose,
        image_size : int = 224,
    ):
        self.records    = records
        self.cache_dir  = Path(cache_dir)
        self.transforms = transforms
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec        = self.records[idx]
        stem       = Path(rec["path"]).stem
        cache_path = self.cache_dir / f"{stem}.npy"

        if cache_path.exists():
            img = np.load(str(cache_path))
        else:
            img = _preprocess_single(rec["path"], self.image_size)
            np.save(str(cache_path), img)

        result = self.transforms(image=img)
        return result["image"], int(rec["label"])


def build_dataloaders(cfg: dict):
    """
    1. Discover all images from cfg['data_dir'] (subfolders named 0–4).
    2. Stratified subsample to cfg['num_samples'].
    3. 80 / 10 / 10 stratified split → train / val / test.
    4. Pre-warm the SSD cache (single-process, shown once per run).
    5. Return (train_loader, val_loader, test_loader).

    DataLoaders are configured with:
        num_workers        = 4
        pin_memory         = True
        persistent_workers = True  (eliminates worker re-spawn overhead)
    """
    data_dir = Path(cfg["data_dir"])
    records  = []
    for label in range(5):
        class_dir = data_dir / str(label)
        if not class_dir.exists():
            continue
        for ext in ("*.jpeg", "*.jpg", "*.png"):
            for p in class_dir.glob(ext):
                records.append({"path": str(p), "label": label})

    if not records:
        raise RuntimeError(
            f"No images found under {data_dir}. "
            "Expected sub-directories named 0, 1, 2, 3, 4."
        )

    labels_all = [r["label"] for r in records]

    # ── Stratified subsample ───────────────────────────────────────────────
    if len(records) > cfg["num_samples"]:
        drop_frac = 1.0 - cfg["num_samples"] / len(records)
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=drop_frac, random_state=cfg["seed"])
        keep_idx, _ = next(sss.split(records, labels_all))
        records    = [records[i] for i in keep_idx]
        labels_all = [labels_all[i] for i in keep_idx]

    # ── 80 / 10 / 10 split ────────────────────────────────────────────────
    idx_all = list(range(len(records)))
    idx_train, idx_tmp = train_test_split(
        idx_all, test_size=0.20, stratify=labels_all, random_state=cfg["seed"])
    labels_tmp = [labels_all[i] for i in idx_tmp]
    idx_val, idx_test = train_test_split(
        idx_tmp, test_size=0.50, stratify=labels_tmp, random_state=cfg["seed"])

    tr_recs  = [records[i] for i in idx_train]
    val_recs = [records[i] for i in idx_val]
    te_recs  = [records[i] for i in idx_test]

    # ── Pre-warm SSD cache (main process, once per run) ───────────────────
    cache_dir = Path(cfg["cache_dir"])
    missing   = [r for r in records
                 if not (cache_dir / f"{Path(r['path']).stem}.npy").exists()]
    if missing:
        print(f"  Pre-caching {len(missing):,} images to {cache_dir} ...")
        for rec in tqdm(missing, desc="Cache", leave=False):
            cache_path = cache_dir / f"{Path(rec['path']).stem}.npy"
            img = _preprocess_single(rec["path"], cfg["image_size"])
            np.save(str(cache_path), img)
        print("  Cache complete.")
    else:
        print(f"  SSD cache already complete ({len(records):,} entries).")

    # ── Build datasets ─────────────────────────────────────────────────────
    tr_tf  = get_train_transforms()
    val_tf = get_val_transforms()

    tr_ds  = LazyDRDataset(tr_recs,  cfg["cache_dir"], tr_tf,  cfg["image_size"])
    val_ds = LazyDRDataset(val_recs, cfg["cache_dir"], val_tf, cfg["image_size"])
    te_ds  = LazyDRDataset(te_recs,  cfg["cache_dir"], val_tf, cfg["image_size"])

    # persistent_workers=True eliminates per-epoch worker re-spawn overhead;
    # requires num_workers > 0.
    loader_kw = dict(
        batch_size         = cfg["batch_size"],
        num_workers        = cfg["num_workers"],
        pin_memory         = cfg["pin_memory"],
        persistent_workers = cfg["num_workers"] > 0,
    )
    tr_loader  = DataLoader(tr_ds,  shuffle=True,  **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    te_loader  = DataLoader(te_ds,  shuffle=False, **loader_kw)

    print(f"  Train: {len(tr_ds):,}  |  Val: {len(val_ds):,}  "
          f"|  Test: {len(te_ds):,}")
    return tr_loader, val_loader, te_loader


# =============================================================================
# STEP 5 — LEARNING-RATE SCHEDULE
# =============================================================================

def build_lr_lambda(
    warmup_epochs : int,
    total_epochs  : int,
    base_lr       : float,
    eta_min       : float,
):
    """
    Returns a LambdaLR-compatible callable implementing:

        Epochs 0 … warmup_epochs−1 :
            factor = (epoch + 1) / warmup_epochs    (linear ramp, 0 → 1)

        Epochs warmup_epochs … total_epochs−1 :
            t       = epoch − warmup_epochs
            T       = total_epochs − warmup_epochs
            factor  = r + (1 − r) · 0.5 · (1 + cos(π · t / T))
            where   r = eta_min / base_lr

    The boundary is C¹-continuous: at epoch = warmup_epochs both branches
    evaluate to 1.0, so the LR transitions without discontinuity.
    """
    r = eta_min / max(base_lr, 1e-12)   # ratio for the cosine floor

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        t = epoch - warmup_epochs
        T = max(total_epochs - warmup_epochs, 1)
        return r + (1.0 - r) * 0.5 * (1.0 + math.cos(math.pi * t / T))

    return lr_lambda


# =============================================================================
# STEP 6 — TRAINING & EVALUATION LOOPS
# =============================================================================

def train_one_epoch(
    model     : nn.Module,
    loader    : DataLoader,
    criterion : nn.Module,
    optimizer : torch.optim.Optimizer,
    scaler    : GradScaler,
    device    : torch.device,
    epoch     : int,
    use_amp   : bool,
) -> tuple:
    """
    Standard training epoch — no Mixup or label blending.

    Every batch is a clean forward pass: raw images → model → CrossEntropyLoss
    on hard integer targets.  Generalization is enforced structurally via
    Dropout(0.4) inside the head, weight_decay=5e-4, and the ViT stochastic
    depth / attention dropout.  AMP autocast + GradScaler runs throughout.
    """
    model.train()
    running_loss = 0.0
    all_true: list = []
    all_pred: list = []

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(images)              # [B, 5] raw logits
            loss   = criterion(logits, labels)  # standard CrossEntropyLoss

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.detach().argmax(dim=1)
        all_pred.extend(preds.cpu().tolist())
        all_true.extend(labels.cpu().tolist())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    n   = len(all_true)
    acc = accuracy_score(all_true, all_pred)
    qwk = cohen_kappa_score(all_true, all_pred, weights="quadratic")
    return running_loss / n, acc, qwk


@torch.no_grad()
def evaluate(
    model     : nn.Module,
    loader    : DataLoader,
    criterion : nn.Module,
    device    : torch.device,
    epoch     : int,
    desc      : str,
    use_amp   : bool,
) -> tuple:
    """
    Validation / test evaluation.
        • No augmentation (pure label CE loss on hard targets)
        • AMP autocast active for throughput
        • Returns (loss, acc, qwk, y_true, y_pred, y_probs)
          where y_probs is [N, 5] softmax for ROC-AUC computation
    """
    model.eval()
    running_loss  = 0.0
    all_true : list = []
    all_pred : list = []
    all_probs: list = []

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [{desc}]", leave=False)
    for images, labels in pbar:
        images  = images.to(device, non_blocking=True)
        targets = labels.long().to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(images)                            # [B, 5]
            loss   = criterion(logits, targets)

        running_loss += loss.item() * images.size(0)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        all_pred.extend(preds.cpu().tolist())
        all_true.extend(targets.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    n   = len(all_true)
    acc = accuracy_score(all_true, all_pred)
    qwk = cohen_kappa_score(all_true, all_pred, weights="quadratic")
    return (
        running_loss / n,
        acc,
        qwk,
        np.array(all_true),
        np.array(all_pred),
        np.array(all_probs),   # [N, 5]
    )


# =============================================================================
# STEP 7 — EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """
    Early stopping on VALIDATION LOSS  (Prof. Halina's patience rule).

    Tracks the lowest validation loss seen so far.  Each epoch where the loss
    fails to improve by more than `min_delta` increments a counter; when the
    counter reaches `patience` the `.stop` flag is raised and the training
    loop should break.

    Whenever a new best loss is observed, `.improved` is set True for that
    epoch so the caller can persist the optimal checkpoint.

    Parameters
    ----------
    patience  : consecutive non-improving epochs tolerated before stopping.
    min_delta : minimum decrease in val loss that counts as an improvement.
    """

    def __init__(self, patience: int = 6, min_delta: float = 0.0):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best_loss   = float("inf")
        self.counter     = 0
        self.stop        = False
        self.improved    = False

    def step(self, val_loss: float) -> bool:
        """Update with the latest val loss; returns True if it improved."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            self.improved  = True
        else:
            self.counter  += 1
            self.improved  = False
            if self.counter >= self.patience:
                self.stop = True
        return self.improved


# =============================================================================
# STEP 8 — METRICS & PLOTTING
# =============================================================================

def _specificity_per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = 5,
) -> np.ndarray:
    """Per-class one-vs-rest Specificity = TN / (TN + FP)."""
    specs = []
    for c in range(n_classes):
        yt_b = (y_true == c).astype(int)
        yp_b = (y_pred == c).astype(int)
        tn   = int(np.sum((yt_b == 0) & (yp_b == 0)))
        fp   = int(np.sum((yt_b == 0) & (yp_b == 1)))
        specs.append(tn / max(tn + fp, 1))
    return np.array(specs)


def save_metrics_summary(
    y_true  : np.ndarray,
    y_pred  : np.ndarray,
    te_loss : float,
    te_qwk  : float,
    path    : Path,
) -> pd.DataFrame:
    """Overall metrics summary CSV  (8 rows: Acc, F1-Mac, F1-Mic, …, QWK)."""
    specs = _specificity_per_class(y_true, y_pred)
    rows  = [
        {"Metric": "Accuracy",                "Value": round(accuracy_score(y_true, y_pred), 4)},
        {"Metric": "F1-Score (Macro)",         "Value": round(f1_score(y_true, y_pred, average="macro",  zero_division=0), 4)},
        {"Metric": "F1-Score (Micro)",         "Value": round(f1_score(y_true, y_pred, average="micro",  zero_division=0), 4)},
        {"Metric": "Precision (Macro)",        "Value": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4)},
        {"Metric": "Sensitivity (Macro)",      "Value": round(recall_score(y_true, y_pred, average="macro",   zero_division=0), 4)},
        {"Metric": "Specificity (Macro)",      "Value": round(float(specs.mean()), 4)},
        {"Metric": "QWK",                      "Value": round(te_qwk, 4)},
        {"Metric": "Test Loss (CrossEntropy)", "Value": round(te_loss, 4)},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return df


def save_per_class_performance(
    y_true      : np.ndarray,
    y_pred      : np.ndarray,
    class_names : list,
    path        : Path,
) -> pd.DataFrame:
    """Per-class clinical breakdown: F1 / Sensitivity / Specificity × 5 grades."""
    n     = len(class_names)
    f1s   = f1_score(y_true, y_pred, average=None, zero_division=0,
                     labels=list(range(n)))
    sens  = recall_score(y_true, y_pred, average=None, zero_division=0,
                         labels=list(range(n)))
    specs = _specificity_per_class(y_true, y_pred, n)
    rows  = [
        {
            "Class":       class_names[c],
            "F1-Score":    round(float(f1s[c]),   4),
            "Sensitivity": round(float(sens[c]),  4),
            "Specificity": round(float(specs[c]), 4),
        }
        for c in range(n)
    ]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return df


def save_multiclass_confusion_csv(
    y_true      : np.ndarray,
    y_pred      : np.ndarray,
    class_names : list,
    path        : Path,
) -> None:
    """Save raw 5×5 confusion matrix as CSV."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(path)
    print(f"  Saved: {path}")


def save_confusion_matrix(
    y_true      : np.ndarray,
    y_pred      : np.ndarray,
    path        : Path,
    class_names : list = None,
) -> None:
    """Row-normalised confusion matrix PNG with raw counts in each cell."""
    if class_names is None:
        class_names = CLASS_NAMES
    cm   = confusion_matrix(y_true, y_pred)
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title("Normalised Confusion Matrix — Experiment 7",
                 fontsize=13, fontweight="bold")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})",
                    ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


def save_binary_confusion(
    y_true   : np.ndarray,
    y_pred   : np.ndarray,
    csv_path : Path,
    png_path : Path,
) -> None:
    """
    Binary clinical split: Non-Referable (grades 0–1) vs Referable (grades 2–4).
    Saves both a CSV with TN/FP/FN/TP + clinical rates and a PNG heatmap.
    """
    y_true_b = (y_true >= 2).astype(int)
    y_pred_b = (y_pred >= 2).astype(int)
    cm = confusion_matrix(y_true_b, y_pred_b, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    pd.DataFrame({
        "Metric": ["TN", "FP", "FN", "TP",
                   "Sensitivity (Referable DR)", "Specificity (Non-Ref)"],
        "Value":  [int(tn), int(fp), int(fn), int(tp),
                   round(tp / max(tp + fn, 1), 4),
                   round(tn / max(tn + fp, 1), 4)],
    }).to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    norm   = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    labels = ["Non-Referable (0–1)", "Referable (2–4)"]
    fig, ax = plt.subplots(figsize=(5, 4), facecolor="white")
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True Label")
    ax.set_title("Binary Referable-DR Confusion Matrix", fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})",
                    ha="center", va="center", fontsize=11,
                    color="white" if norm[i, j] > 0.5 else "black")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {png_path}")


def save_roc_curves(
    y_true  : np.ndarray,
    y_probs : np.ndarray,      # [N, 5] softmax probabilities
    path    : Path,
    n_classes: int = 5,
) -> None:
    """One-vs-Rest ROC curves using softmax class probabilities as scores."""
    y_bin   = label_binarize(y_true, classes=list(range(n_classes)))
    palette = plt.cm.tab10(np.linspace(0, 0.9, n_classes))

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance")
    for c in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_probs[:, c])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=palette[c],
                label=f"{CLASS_NAMES[c]}  (AUC = {roc_auc:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("Multi-Class ROC Curve (One-vs-Rest) — Exp 7",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10, frameon=False)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


def save_training_history(history: list, path: Path) -> None:
    """
    Classic Experiment 6.2 loss plot — a single matplotlib figure titled
    'loss'.

        train : solid classic blue   (#1f77b4, label 'train')
        val   : solid classic orange (#ff7f0e, label 'val')

    No dashed lines, no secondary metric panel, no target overlays.
    """
    if not history:
        return

    epochs  = [h["epoch"]      for h in history]
    tr_loss = [h["train_loss"] for h in history]
    va_loss = [h["val_loss"]   for h in history]

    plt.figure()
    plt.plot(epochs, tr_loss, color="#1f77b4", label="train")
    plt.plot(epochs, va_loss, color="#ff7f0e", label="val")
    plt.title("loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# STEP 9 — MAIN
# =============================================================================

def main() -> None:

    # ── Data ──────────────────────────────────────────────────────────────
    tr_loader, val_loader, te_loader = build_dataloaders(CFG)

    # ── Model ─────────────────────────────────────────────────────────────
    print("\nBuilding HybridLHT_ViT (Experiment 7) ...")
    model = HybridLHT_ViT(
        attn_dim       = CFG["attn_dim"],
        mfb_k          = CFG["mfb_k"],
        mfb_out        = CFG["mfb_out"],
        num_classes    = CFG["num_classes"],
        mfb_dropout    = CFG["mfb_dropout"],
        head_dropout   = CFG["head_dropout"],
        drop_path_rate = CFG["drop_path_rate"],
        attn_drop_rate = CFG["attn_drop_rate"],
        pretrained     = True,
    ).to(device)

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters   : {n_total / 1e6:.2f} M total  "
          f"({n_trainable / 1e6:.2f} M trainable)")
    print(f"  Head output  : 5 logits  (standard CrossEntropyLoss)")

    # ── Loss ──────────────────────────────────────────────────────────────
    # Standard CrossEntropyLoss — no label smoothing.  Removing the smoothing
    # offset allows the training curve to find the same natural floor as the
    # validation curve and converge in parallel.
    criterion = nn.CrossEntropyLoss()

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = CFG["lr"],
        weight_decay = CFG["weight_decay"],
    )

    # ── LR Scheduler : LambdaLR (linear warmup + cosine annealing) ───────
    lr_fn     = build_lr_lambda(
        warmup_epochs = CFG["warmup_epochs"],
        total_epochs  = CFG["epochs"],
        base_lr       = CFG["lr"],
        eta_min       = CFG["eta_min"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)
    print(f"  Scheduler    : LambdaLR — "
          f"linear warmup {CFG['warmup_epochs']} ep → "
          f"cosine decay to {CFG['eta_min']:.0e}")

    # ── AMP scaler ────────────────────────────────────────────────────────
    scaler    = GradScaler(enabled=use_amp)
    save_path = out_dir / "best_model_exp7.pth"

    # ── Decoupled monitoring ───────────────────────────────────────────────
    #   Early stopping : driven by VALIDATION LOSS   (patience = 6)
    #     → halts the run before the loss gap widens irreversibly.
    #   Checkpoint     : driven by best VALIDATION QWK
    #     → best_model_exp7.pth always holds the peak-QWK weights, which
    #       is the number that goes into the thesis results table.
    stopper = EarlyStopping(patience=CFG["patience"], min_delta=0.0)

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_qwk   = -1.0
    best_qwk_epoch = 0
    best_qwk_loss  = float("inf")
    history     : list = []
    stopped_early = False
    last_epoch    = 0

    print("\n" + "=" * 72)
    print(f"  TRAINING  — early-stop on VAL LOSS (patience {CFG['patience']})  "
          f"|  save on best VAL QWK")
    print("=" * 72)
    header = (f"{'Ep':>4}  {'LR':>9}  "
              f"{'TrLoss':>8}  {'TrAcc':>7}  {'TrQWK':>7}  "
              f"{'VaLoss':>8}  {'VaAcc':>7}  {'VaQWK':>7}  "
              f"{'Save':>5}  {'Pat':>4}")
    print(header)
    print("-" * len(header))

    for epoch in range(1, CFG["epochs"] + 1):
        last_epoch = epoch
        gc.collect()
        torch.cuda.empty_cache()

        current_lr = optimizer.param_groups[0]["lr"]

        tr_loss, tr_acc, tr_qwk = train_one_epoch(
            model, tr_loader, criterion, optimizer, scaler,
            device, epoch, use_amp,
        )
        va_loss, va_acc, va_qwk, _, _, _ = evaluate(
            model, val_loader, criterion, device, epoch, "Val", use_amp,
        )

        # LambdaLR advances after each epoch
        scheduler.step()

        # (1) Advance early-stopping counter on validation LOSS.
        stopper.step(va_loss)

        # (2) Save checkpoint when validation QWK reaches a new maximum.
        saved = ""
        if va_qwk > best_val_qwk:
            best_val_qwk   = va_qwk
            best_qwk_epoch = epoch
            best_qwk_loss  = va_loss
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_qwk":  va_qwk,
                    "val_acc":  va_acc,
                    "val_loss": va_loss,
                    "architecture": "HybridLHT_ViT [Exp 7]",
                    "cfg": CFG,
                },
                save_path,
            )
            saved = "✓"

        history.append({
            "epoch":      epoch,
            "lr":         current_lr,
            "train_loss": tr_loss,
            "train_acc":  tr_acc,
            "train_qwk":  tr_qwk,
            "val_loss":   va_loss,
            "val_acc":    va_acc,
            "val_qwk":    va_qwk,
        })

        print(
            f"{epoch:>4}  {current_lr:>9.2e}  "
            f"{tr_loss:>8.4f}  {tr_acc:>7.4f}  {tr_qwk:>7.4f}  "
            f"{va_loss:>8.4f}  {va_acc:>7.4f}  {va_qwk:>7.4f}  "
            f"{saved:>5}  {stopper.counter:>4}"
        )

        if stopper.stop:
            print(f"\n  >>> EARLY STOPPING triggered at epoch {epoch}: "
                  f"validation loss did not improve for {CFG['patience']} "
                  f"consecutive epochs.")
            print(f"  >>> Best val QWK  : {best_val_qwk:.4f}  "
                  f"(epoch {best_qwk_epoch}, val loss {best_qwk_loss:.4f})")
            print(f"  >>> Breaking the loop and generating charts now.")
            stopped_early = True
            break

    print("\n" + "=" * 72)
    print(f"  Training complete.")
    print(f"  Best val QWK  : {best_val_qwk:.4f}  (epoch {best_qwk_epoch})")
    print(f"  Checkpoint    : {save_path}  (peak-QWK weights)")

    # ── Save training log ──────────────────────────────────────────────────
    log_path = out_dir / "train_log_exp7.json"
    log_path.write_text(
        json.dumps(
            {
                "timestamp":         datetime.now().isoformat(timespec="seconds"),
                "architecture":      "HybridLHT_ViT [Exp 7]",
                "best_val_qwk":      best_val_qwk,
                "best_qwk_epoch":    best_qwk_epoch,
                "best_qwk_val_loss": best_qwk_loss,
                "checkpoint_policy": "save on best val QWK; early-stop on val loss",
                "stopped_early":     stopped_early,
                "epochs_run":        last_epoch,
                "cfg":               CFG,
                "history":           history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  Training log : {log_path}")

    # ── Test evaluation ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  TEST EVALUATION  (reloading best checkpoint)")
    print("=" * 72)

    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    te_loss, te_acc, te_qwk, y_true, y_pred, y_probs = evaluate(
        model, te_loader, criterion, device, last_epoch, "Test", use_amp,
    )

    te_f1_macro = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    te_f1_micro = f1_score(y_true, y_pred, average="micro",  zero_division=0)
    te_prec     = precision_score(y_true, y_pred, average="macro", zero_division=0)
    te_sens     = recall_score(y_true, y_pred, average="macro",    zero_division=0)
    te_specs    = _specificity_per_class(y_true, y_pred)
    te_spec_mac = float(te_specs.mean())

    print("\n  PRIMARY METRICS — 5-Class DR Grading")
    print("=" * 72)
    print(f"  Accuracy              : {te_acc:.4f}")
    print(f"  QWK                   : {te_qwk:.4f}")
    print(f"  F1-Score  (Macro)     : {te_f1_macro:.4f}")
    print(f"  F1-Score  (Micro)     : {te_f1_micro:.4f}")
    print(f"  Precision (Macro)     : {te_prec:.4f}")
    print(f"  Sensitivity (Macro)   : {te_sens:.4f}")
    print(f"  Specificity (Macro)   : {te_spec_mac:.4f}")
    print(f"  Test Loss (CE)        : {te_loss:.4f}")
    print()
    print(classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0))

    # ── Export thesis assets ───────────────────────────────────────────────
    print("\nSaving thesis evaluation assets ...")

    save_metrics_summary(
        y_true, y_pred, te_loss, te_qwk,
        out_dir / "metrics_summary_exp7.csv",
    )
    save_per_class_performance(
        y_true, y_pred, CLASS_NAMES,
        out_dir / "per_class_performance_exp7.csv",
    )
    save_multiclass_confusion_csv(
        y_true, y_pred, CLASS_NAMES,
        out_dir / "confusion_matrix_multiclass_exp7.csv",
    )
    save_confusion_matrix(
        y_true, y_pred,
        out_dir / "confusion_matrix_multiclass_exp7.png",
    )

    print("\n  Binary Referable-DR split  (Non-Referable: 0–1 | Referable: 2–4)")
    save_binary_confusion(
        y_true, y_pred,
        out_dir / "confusion_matrix_binary_exp7.csv",
        out_dir / "confusion_matrix_binary_exp7.png",
    )

    save_roc_curves(
        y_true, y_probs,
        out_dir / "roc_curves_exp7.png",
    )
    save_training_history(
        history,
        out_dir / "training_history_exp7.png",
    )

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  FINAL SUMMARY  (Exp 7 — Strategic Convergence & Peak Generalization)")
    print("=" * 72)
    print(f"  QWK                   : {te_qwk:.4f}")
    print(f"  Accuracy  (5-class)   : {te_acc:.4f}")
    print(f"  F1-Score  (Macro)     : {te_f1_macro:.4f}")
    print(f"  F1-Score  (Micro)     : {te_f1_micro:.4f}")
    print(f"  Sensitivity (Macro)   : {te_sens:.4f}")
    print(f"  Specificity (Macro)   : {te_spec_mac:.4f}")
    print("=" * 72)

    print("\nAll outputs saved to:", CFG["output_dir"])
    for filename in [
        "best_model_exp7.pth",
        "train_log_exp7.json",
        "metrics_summary_exp7.csv",
        "per_class_performance_exp7.csv",
        "confusion_matrix_multiclass_exp7.csv",
        "confusion_matrix_multiclass_exp7.png",
        "confusion_matrix_binary_exp7.csv",
        "confusion_matrix_binary_exp7.png",
        "roc_curves_exp7.png",
        "training_history_exp7.png",
    ]:
        print(f"  • {filename}")
    print("\nDone.")


# ── Entry point ───────────────────────────────────────────────────────────────
main()
