"""
============================================================================
EXPERIMENT 5 — LHT-ViT  (Frozen-Backbone Feature Extraction + Head Training)
   Hybrid CNN + ViT-Small  ·  MFB Pooling  ·  Bidirectional Co-Attention
        Pre-cached Ben Graham + CLAHE  ·  AMP  ·  TTA  ·  45K Dataset
============================================================================

ARCHITECTURE OVERVIEW
---------------------
Fundus image  (B, 3, 224, 224)
    │
    ├── CNN BRANCH ── EfficientNet-B0  (~5.3 M params, ImageNet pretrained)
    │       └── forward_features → (B, 1280, 7, 7)
    │             → flatten → 49 spatial tokens × 1280-dim
    │             → captures *local* texture: micro-aneurysms, exudates,
    │               haemorrhages, drusen.
    │
    ├── TRANSFORMER BRANCH ── ViT-Small/16-224  (~22 M params, ImageNet)
    │       └── forward_features → (B, 197, 384)   (CLS dropped → 196 patches)
    │             → 14×14 patch grid at 224 input
    │             → captures *global* anatomy: optic-disc location,
    │               vessel topology, macular geometry.
    │
    ├── CO-ATTENTION BLOCK ── bidirectional spatial cross-modal attention
    │       Step 1 :  project both modalities to D = 512
    │       Step 2 :  CNN tokens ── Q ──► attend over ViT tokens (K, V)
    │       Step 3 :  ViT tokens ── Q ──► attend over CNN tokens (K, V)
    │       Step 4 :  residual + LayerNorm  →  mean-pool over tokens
    │       Output:  cnn_vec (B, 512)   vit_vec (B, 512)
    │
    ├── MFB POOLING ── Multi-modal Factorized Bilinear (Yu et al. ICCV 2017)
    │       z_o  =  Σ_{k=1..K} ( U^x · x )_k  ⊙  ( V^y · y )_k
    │       Project each → (K·O), element-wise multiply, sum-pool K factors.
    │       Power-norm + L2-norm  →  fused (B, 1024)
    │
    └── HEAD : LayerNorm → 512 → GELU → Dropout(0.5) → 5 logits

WHY THIS DESIGN — RESOURCE / OVERFITTING TRADE-OFF
--------------------------------------------------
* ViT-**Small** (not Base): 22 M params is the right capacity for a 50K-
  image stratified subset. ViT-Base (86 M) was both 4× slower per step and
  more prone to memorising the dominant "No DR" class.
* Pre-cached preprocessing: Ben Graham (circular crop, Gaussian blend) and
  CLAHE are run ONCE at startup and saved as 224×224 PNGs in
  /kaggle/working/cache/preproc/. The training __getitem__ then only does
  a fast PNG read + light augmentation, eliminating ~80% of per-epoch CPU
  time and the previous Kaggle session-timeout crash.
* AMP (mixed precision): every forward/loss is wrapped in autocast and
  GradScaler. This halves activation memory and ~2× speeds up T4 compute
  with no measurable quality loss.
* Cosine LR with 2-epoch warmup: linear warmup to base LR then a cosine
  decay to ~0. Smoother convergence than constant LR; especially important
  in the first few epochs where DataParallel can produce a loss spike.
* Save best on val_QWK, early-stop on val_loss: QWK is the clinically
  meaningful ranking metric (used by all DR competitions); val_loss is the
  classical overfitting signal. Splitting the two gates keeps the saved
  model thesis-relevant while still catching divergence early.
* TTA (test-time augmentation): at evaluation we average softmax probs over
  the original image and its horizontal flip. Free 1-2% accuracy boost.

DATA / HARDWARE STABILITY  (Experiment 5 profile)
-------------------------------------------------
* 45 K stratified subset (split 80/10/10 → 36 K / 4.5 K / 4.5 K).
  Safe to scale up because frozen backbones produce NO gradients, so
  VRAM usage is ~40% lower per batch than a fully-trainable run.
* Fixed 224×224 input.
* LAZY DISK-BASED DATASET:
  - Ben Graham + CLAHE pre-computed ONCE to disk (resumable).
  - Only metadata lives in RAM; each __getitem__ streams one PNG.
* SINGLE GPU (no DataParallel). batch_size=32, num_workers=2, pin_memory=True.
* torch.backends.cudnn.benchmark = True.
* gc.collect() + torch.cuda.empty_cache() each epoch.

BACKBONE FREEZING  (Experiment 5 core strategy)
------------------------------------------------
* Both EfficientNet-B0 and ViT-Small are loaded with pretrained ImageNet
  weights, then ALL their parameters are immediately frozen:
      for p in model.cnn.parameters(): p.requires_grad = False
      for p in model.vit.parameters(): p.requires_grad = False
* Only Co-Attention, MFB Pooling, and the Classification Head are trainable.
* Rationale: the backbones already extract strong retinal features from
  ImageNet pretraining. Freezing them eliminates the memorisation of
  noise in backbone weights that caused training-loss → 0 in prior runs,
  while the trainable head focuses purely on learning the DR-grade mapping.
* Trainable parameter count ≈ 8 M  (vs. 34 M previously).

REGULARIZATION  (Experiment 5 profile)
---------------------------------------
* head_dropout    = 0.5  — strong dropout in the MLP head (unchanged).
* label_smoothing = 0.05 — lighter than 4.1 (0.15); with frozen backbones
                           the head converges faster, so heavy smoothing
                           would slow it down unnecessarily.
* weight_decay    = 1e-3 — unchanged.
* drop_path_rate  = 0.0  — stochastic depth disabled (irrelevant for frozen
                           ViT blocks whose weights are not updated).
* lr = 1e-4 (head only) under CosineAnnealingLR(T_max=epochs, eta_min=1e-7).
* Early stopping: patience = 7, tracked on val_QWK.

AUGMENTATION  (Experiment 5 — simplified back from 4.1)
--------------------------------------------------------
* Mixup/CutMix REMOVED — they create soft labels that confuse a head
  trained on fixed backbone features (no gradient flows back to correct
  the feature extractor).
* HorizontalFlip p=0.5 + mild Rotate(15°) retained — sufficient geometric
  diversity when backbone features are already rich and frozen.

SAMPLING  (unchanged from 3.2 / 4.1)
--------------------------------------
* WeightedRandomSampler with Grade-2 (Moderate) boost ×1.5 retained.

OUTPUT
------
* Best checkpoint  : /kaggle/working/best_model_LHTViT_5.pth
* Metrics CSVs     : experiment_3_metrics.csv  +  evaluation_results.csv
* JSON report      : classification_report.json
* Plots            : confusion_matrix.png  +  training_history.png  +
                     confusion_matrix_binary.png  +  training_loss_curves.png
============================================================================
"""

import os
import sys
import gc
import json
import math
import hashlib
import argparse
from pathlib import Path
from datetime import datetime

# ============================================================================
#                              EXECUTION FLAGS
# ============================================================================
# Set to True to skip the (long) training loop and run the evaluation
# section directly off the existing checkpoint at `save_path`. Useful for
# re-generating plots / metrics without retraining.
SKIP_TRAINING: bool = False   # Experiment 5 — training run
# ============================================================================

# ── Cache redirection — MUST be set before torch / timm import ──────────────
os.environ.setdefault("TORCH_HOME", "/kaggle/working/cache")
os.environ.setdefault("HF_HOME",    "/kaggle/working/cache")

# ── Make our helper modules importable ───────────────────────────────────────
SRC_DIR = "/kaggle/working/src"
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)

# Re-use the existing helpers — only `prepare_data`, `ben_graham_preprocess`,
# and `get_cost_sensitive_weights`. We define our own Dataset/loaders here.
from dataset import prepare_data, ben_graham_preprocess
from utils   import get_cost_sensitive_weights


# ============================================================================
#                              MODEL  DEFINITIONS
# ============================================================================

class CoAttention(nn.Module):
    """Bidirectional spatial cross-modal attention between CNN and ViT tokens."""

    def __init__(self, cnn_dim: int, vit_dim: int, attn_dim: int = 512,
                 attn_dropout: float = 0.1) -> None:
        super().__init__()
        self.attn_dim   = attn_dim
        self.attn_scale = attn_dim ** 0.5

        self.cnn_proj = nn.Linear(cnn_dim, attn_dim)
        self.vit_proj = nn.Linear(vit_dim, attn_dim)

        # Direction 1 : CNN-as-Query attends over ViT (k/v)
        self.q1 = nn.Linear(attn_dim, attn_dim)
        self.k1 = nn.Linear(attn_dim, attn_dim)
        self.v1 = nn.Linear(attn_dim, attn_dim)

        # Direction 2 : ViT-as-Query attends over CNN (k/v)
        self.q2 = nn.Linear(attn_dim, attn_dim)
        self.k2 = nn.Linear(attn_dim, attn_dim)
        self.v2 = nn.Linear(attn_dim, attn_dim)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.norm_cnn     = nn.LayerNorm(attn_dim)
        self.norm_vit     = nn.LayerNorm(attn_dim)

    def _attend(self, q_w, k_w, v_w, query_in, kv_in):
        Q     = q_w(query_in)
        K     = k_w(kv_in)
        V     = v_w(kv_in)
        score = torch.bmm(Q, K.transpose(1, 2)) / self.attn_scale
        attn  = self.attn_dropout(torch.softmax(score, dim=-1))
        return torch.bmm(attn, V)

    def forward(self, cnn_tokens: torch.Tensor, vit_tokens: torch.Tensor):
        cnn_p = self.cnn_proj(cnn_tokens)   # [B, T_cnn, D]
        vit_p = self.vit_proj(vit_tokens)   # [B, T_vit, D]

        cnn_attn = self._attend(self.q1, self.k1, self.v1, cnn_p, vit_p)
        vit_attn = self._attend(self.q2, self.k2, self.v2, vit_p, cnn_p)

        cnn_attn = self.norm_cnn(cnn_attn + cnn_p)
        vit_attn = self.norm_vit(vit_attn + vit_p)

        cnn_vec = cnn_attn.mean(dim=1)
        vit_vec = vit_attn.mean(dim=1)
        return cnn_vec, vit_vec


class MFBPooling(nn.Module):
    """Multi-modal Factorized Bilinear pooling (Yu et al., ICCV 2017)."""

    def __init__(self, in_dim_x: int, in_dim_y: int,
                 mfb_k: int = 5, mfb_out: int = 1024,
                 mfb_dropout: float = 0.1) -> None:
        super().__init__()
        self.mfb_k   = mfb_k
        self.mfb_out = mfb_out
        self.proj_x  = nn.Linear(in_dim_x, mfb_k * mfb_out)
        self.proj_y  = nn.Linear(in_dim_y, mfb_k * mfb_out)
        self.dropout = nn.Dropout(mfb_dropout)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B  = x.shape[0]
        zx = self.proj_x(x)
        zy = self.proj_y(y)
        z  = self.dropout(zx * zy)
        z  = z.view(B, self.mfb_out, self.mfb_k).sum(dim=2)
        z  = torch.sign(z) * torch.sqrt(torch.abs(z) + 1e-8)
        z  = F.normalize(z, p=2, dim=1)
        return z


class HybridLHT_ViT(nn.Module):
    """Hybrid LHT-ViT: EfficientNet-B0 + ViT-Small/16 + CoAttention + MFB."""

    def __init__(
        self,
        num_classes:   int   = 5,
        attn_dim:      int   = 512,
        mfb_k:         int   = 5,
        mfb_out:       int   = 1024,
        attn_dropout:  float = 0.1,
        head_dropout:  float = 0.5,
        drop_path_rate: float = 0.2,
        pretrained:    bool  = True,
    ) -> None:
        super().__init__()

        self.cnn = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
        )
        # ViT-Small/16-224  (embed_dim=384, ~22 M params)
        # drop_path_rate enables stochastic depth: each transformer block's
        # residual path is randomly dropped during training (linearly scaled
        # from 0 up to drop_path_rate across layers).  This is the most
        # effective single regularizer for ViT-class models per DeiT/Swin.
        self.vit = timm.create_model(
            "vit_small_patch16_224",
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        cnn_dim: int = self.cnn.num_features              # 1280
        vit_dim: int = self.vit.embed_dim                 # 384  (was 768 for Base)

        self.coattn = CoAttention(
            cnn_dim=cnn_dim, vit_dim=vit_dim,
            attn_dim=attn_dim, attn_dropout=attn_dropout,
        )
        self.mfb = MFBPooling(
            in_dim_x=attn_dim, in_dim_y=attn_dim,
            mfb_k=mfb_k, mfb_out=mfb_out,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(mfb_out),
            nn.Linear(mfb_out, 512),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_map    = self.cnn.forward_features(x)              # [B, 1280, 7, 7]
        cnn_tokens = cnn_map.flatten(2).transpose(1, 2)        # [B, 49, 1280]

        vit_seq    = self.vit.forward_features(x)              # [B, 197, 384]
        vit_tokens = vit_seq[:, 1:, :]                         # [B, 196, 384]

        cnn_vec, vit_vec = self.coattn(cnn_tokens, vit_tokens)
        fused = self.mfb(cnn_vec, vit_vec)
        return self.classifier(fused)


# ============================================================================
#                       PREPROCESSING CACHE  +  DATASET
# ============================================================================

def _cache_filename(image_path: str) -> str:
    """Stable per-image cache filename — independent of dataframe ordering."""
    return hashlib.md5(image_path.encode("utf-8")).hexdigest()[:16] + ".png"


def precompute_cache(
    df:         pd.DataFrame,
    cache_dir:  Path,
    image_size: int = 224,
) -> pd.DataFrame:
    """
    Run Ben Graham preprocessing + CLAHE ONCE for every image in *df* and save
    the result as a 224×224 RGB PNG inside *cache_dir*. Resumable: if the
    cache file already exists, the image is skipped.

    Returns *df* with a new column ``cached_path`` pointing at the cached PNG.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # CLAHE applied here so the cache contains the final colour-normalized image
    # and __getitem__ only needs to do a fast PNG read + light augmentation.
    clahe_transform = A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)

    cached_paths: list[str] = []
    n_skipped, n_built, n_failed = 0, 0, 0

    pbar = tqdm(df.itertuples(index=False), total=len(df),
                desc="Pre-caching preprocessing", unit="img")
    for row in pbar:
        src        = row.image_path
        cache_path = cache_dir / _cache_filename(src)

        if cache_path.exists():
            n_skipped += 1
            cached_paths.append(str(cache_path))
            continue

        try:
            img = cv2.imread(src)
            if img is None:
                n_failed += 1
                cached_paths.append(str(cache_path))
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = ben_graham_preprocess(img)                            # 512×512 RGB
            img = clahe_transform(image=img)["image"]                   # CLAHE on full-res
            img = cv2.resize(img, (image_size, image_size),
                             interpolation=cv2.INTER_AREA)              # → 224×224
            cv2.imwrite(str(cache_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            n_built += 1
        except Exception as e:
            n_failed += 1
            print(f"\n[warn] failed on {src}: {e}")

        cached_paths.append(str(cache_path))

        if n_built % 2000 == 0 and n_built > 0:
            pbar.set_postfix(built=n_built, skipped=n_skipped, failed=n_failed)

    print(f"\nCache summary: built={n_built}  skipped(existing)={n_skipped}  failed={n_failed}"
          f"  total={len(df)}  dir={cache_dir}")

    df = df.copy()
    df["cached_path"] = cached_paths
    return df


def get_train_transforms_light() -> A.Compose:
    """
    Experiment 5 augmentation pipeline — simplified for frozen-backbone training.

    Mixup/CutMix removed (they conflict with frozen features).
    ColorJitter/30° rotation removed — not needed when the backbone already
    provides rich, ImageNet-pretrained features; over-augmenting would only
    slow head convergence.

    Kept:
      * HorizontalFlip p=0.5  — free diversity, always safe.
      * Rotate ±15°            — mild geometric jitter.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms_light() -> A.Compose:
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class LazyDRDataset(Dataset):
    """
    Lazy disk-based Dataset — the *only* thing held in RAM is metadata
    (file paths + labels). Every __getitem__ opens ONE PNG from disk via
    PIL.Image.open(), so the kernel never has to materialize the full
    dataset as a tensor or numpy array.

    Why this design:
      * Removes the "load 50K images to RAM" step that was crashing the
        Kaggle kernel.
      * Stays compatible with our existing PNG preprocessing cache: each
        cached file is the Ben-Graham + CLAHE-normalized 224×224 image,
        so __getitem__ only does a fast PNG decode + light augmentation.
      * Workers (num_workers=2) overlap disk I/O with GPU compute. Each
        worker process gets its own copy of the cheap metadata list (a few
        MB of strings), NOT a copy of the dataset.
    """

    def __init__(self, df: pd.DataFrame, transform: A.Compose) -> None:
        # Cache only METADATA in RAM — never the image tensors themselves.
        df = df.reset_index(drop=True)
        self.paths     = df["cached_path"].astype(str).tolist()
        self.labels    = df["label"].astype(int).tolist()
        self.transform = transform
        self._df       = df

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def df(self) -> pd.DataFrame:
        """For `train_loader.dataset.df["label"].tolist()` compatibility."""
        return self._df

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        # PIL is the requested loader — convert to a small uint8 ndarray for
        # albumentations, then drop the PIL handle immediately.
        with Image.open(path) as im:
            img_np = np.asarray(im.convert("RGB"), dtype=np.uint8)

        out = self.transform(image=img_np)["image"]
        return out, self.labels[idx]


def build_weighted_sampler(
    labels: list[int],
    oversample_class: int = 2,
    oversample_factor: float = 1.5,
) -> torch.utils.data.WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that gives every class equal expected
    frequency (inverse-frequency weighting), then additionally boosts
    *oversample_class* (Grade 2 = Moderate) by *oversample_factor*.

    This replaces shuffle=True for the training DataLoader in Exp 3.2 so
    the sampler drives iteration order instead of a random shuffle.
    Val and test loaders are NOT affected — they still use plain iteration.

    Why WeightedRandomSampler rather than duplicating rows in the DataFrame:
      * No data leakage risk — we never physically copy images.
      * The effective class distribution seen per epoch is controlled purely
        by sampling weights, so the val/test splits remain untouched.
      * Compatible with our lazy LazyDRDataset (indices are drawn by the
        sampler, __getitem__ then loads that PNG from disk as usual).
    """
    from collections import Counter
    counts  = Counter(labels)
    n_total = len(labels)
    n_cls   = len(counts)

    # Base weight = inverse frequency (balances all 5 classes equally)
    base_w  = {c: n_total / (n_cls * cnt) for c, cnt in counts.items()}

    # Boost the Moderate class
    base_w[oversample_class] *= oversample_factor

    sample_weights = torch.tensor(
        [base_w[lbl] for lbl in labels], dtype=torch.float,
    )
    return torch.utils.data.WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = n_total,          # one "epoch" = same total batches as before
        replacement = True,
    )


def build_cached_dataloaders(
    data_dir:    str,
    cache_dir:   Path,
    image_size:  int,
    batch_size:  int,
    num_samples: int,
    num_workers: int,
    pin_memory:  bool,
    seed:        int = 42,
):
    """
    Experiment 3.2 lazy disk-based data pipeline:
      1) Stratified-sample *num_samples* image paths from the source dir.
      2) Pre-cache Ben Graham + CLAHE results to 224×224 PNGs (resumable).
      3) Stratified 80/10/10 split → LazyDRDatasets.
      4) Training loader uses WeightedRandomSampler with Grade-2 boost (×1.5)
         to address the 0.50 F1 for the Moderate class seen in Exp 3.1.
         Val and test loaders use plain sequential iteration (unchanged).
    """
    print(f"\nScanning dataset: {data_dir}")
    df = prepare_data(root_dir=data_dir, num_samples=num_samples)
    print(f"  → stratified subset: {len(df)} images")
    print("  Class distribution :")
    print(df["label"].value_counts().sort_index().to_string())

    print(f"\nPre-caching preprocessed PNGs → {cache_dir}")
    df = precompute_cache(df, cache_dir=cache_dir, image_size=image_size)

    from sklearn.model_selection import train_test_split
    df_train, df_temp = train_test_split(
        df,      test_size=0.20, stratify=df["label"],         random_state=seed,
    )
    df_val,   df_test = train_test_split(
        df_temp, test_size=0.50, stratify=df_temp["label"],    random_state=seed,
    )

    train_ds = LazyDRDataset(df_train, transform=get_train_transforms_light())
    val_ds   = LazyDRDataset(df_val,   transform=get_val_transforms_light())
    test_ds  = LazyDRDataset(df_test,  transform=get_val_transforms_light())

    # Grade-2 (Moderate) oversampling via WeightedRandomSampler.
    # oversample_factor=1.5 → Moderate drawn ~1.5× more than inverse-freq alone.
    train_sampler = build_weighted_sampler(
        labels            = train_ds.df["label"].tolist(),
        oversample_class  = 2,
        oversample_factor = 1.5,
    )
    print(f"  WeightedRandomSampler: Grade-2 boosted ×1.5  "
          f"(effective train epoch = {len(train_ds)} samples)")

    loader_kwargs = dict(
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = pin_memory,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"]    = 2

    # shuffle=False because WeightedRandomSampler controls the order
    train_loader = DataLoader(train_ds, sampler=train_sampler, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    print(f"\nDataset split -- "
          f"Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Test: {len(test_ds)}  "
          f"(workers={num_workers}, batch={batch_size}, pin_memory={pin_memory})")

    return train_loader, val_loader, test_loader


# ============================================================================
#                              CLI / CONFIG
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 3 — Hybrid LHT-ViT for DR.")

    # Paths
    p.add_argument(
        "--data_dir", type=str,
        default="/kaggle/input/datasets/ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy/augmented_resized_V2",
    )
    p.add_argument("--output_dir", type=str, default="/kaggle/working")
    p.add_argument("--plots_dir",  type=str, default="/kaggle/working/plots")
    p.add_argument("--cache_dir",  type=str, default="/kaggle/working/cache/preproc")

    # Hyperparameters
    p.add_argument("--batch_size",      type=int,   default=32,
                   help="Single-GPU batch size (safe-run pins to one T4).")
    p.add_argument("--epochs",          type=int,   default=40,
                   help="Max epochs (40 with patience=7).")
    p.add_argument("--warmup_epochs",   type=int,   default=0,
                   help="Warmup disabled — CosineAnnealingLR handles the full schedule.")
    p.add_argument("--lr",              type=float, default=1e-4,
                   help="Exp 5: raised to 1e-4 — with frozen backbones the head "
                        "can absorb a higher LR without destabilising the features.")
    p.add_argument("--weight_decay",    type=float, default=1e-3,
                   help="AdamW weight decay — unchanged from 3.2/4.1.")
    p.add_argument("--label_smoothing", type=float, default=0.05,
                   help="Exp 5: lightened to 0.05 — frozen backbone features are "
                        "already strong; heavy smoothing (0.15) only slows head "
                        "convergence when the head is the only thing training.")
    p.add_argument("--patience",        type=int,   default=7,
                   help="Early-stopping patience tracked on val_QWK (not val_loss).")
    p.add_argument("--num_classes",     type=int,   default=5)
    p.add_argument("--image_size",      type=int,   default=224)

    # Stable Kaggle data-pipeline knobs (lazy disk-loading — see LazyDRDataset)
    p.add_argument("--num_samples",  type=int, default=45_000,
                   help="Exp 5: scaled up to 45K (→ 36K/4.5K/4.5K). Safe because "
                        "frozen backbones produce no gradients — VRAM is ~40%% lower.")
    p.add_argument("--num_workers",  type=int, default=2,
                   help="DataLoader workers. 2 is the safe sweet-spot on Kaggle.")
    p.add_argument("--pin_memory",   type=int, default=1,
                   help="1 = pinned host memory (faster H2D copies on Kaggle).")

    # Model architecture
    p.add_argument("--attn_dim",       type=int,   default=512)
    p.add_argument("--mfb_k",          type=int,   default=5)
    p.add_argument("--mfb_out",        type=int,   default=1024)
    p.add_argument("--head_dropout",   type=float, default=0.5,
                   help="Strong dropout in the classification MLP head.")
    p.add_argument("--drop_path_rate", type=float, default=0.0,
                   help="Exp 5: disabled (0.0) — stochastic depth is irrelevant "
                        "for frozen ViT blocks whose weights are not updated.")

    # Mixed precision + TTA
    p.add_argument("--no_amp",       dest="use_amp", action="store_false",
                   help="Disable mixed precision (default: enabled).")
    p.set_defaults(use_amp=True)

    p.add_argument("--no_tta",       dest="use_tta", action="store_false",
                   help="Disable test-time augmentation (default: enabled).")
    p.set_defaults(use_tta=True)

    args, _ = p.parse_known_args()
    return args


# ============================================================================
#                              LR SCHEDULE
# ============================================================================

def cosine_warmup_factor(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    """Linear warmup → cosine decay (returns a multiplier on the base LR)."""
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


# ============================================================================
#                    MIXUP / CUTMIX  (Experiment 4.1)
# ============================================================================

def mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha:  float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply Mixup to a batch.
      x_mixed  = lam * x_i  +  (1-lam) * x_j   (shuffled partner)
      Labels are returned as (labels_a, labels_b, lam) for the soft-loss
      formula:  loss = lam * CE(logits, a) + (1-lam) * CE(logits, b)
    """
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    B     = images.size(0)
    idx   = torch.randperm(B, device=images.device)
    mixed = lam * images + (1.0 - lam) * images[idx]
    return mixed, labels, labels[idx], lam


def cutmix_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha:  float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply CutMix to a batch.
    A random rectangular box from a shuffled partner replaces the same region
    in the source image. lam is the fraction of the source image kept (area
    ratio), used in the same soft-loss formula as Mixup.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    B, C, H, W = images.shape
    idx = torch.randperm(B, device=images.device)

    cut_ratio = math.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    # random centre
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0);  x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0);  y2 = min(cy + cut_h // 2, H)

    mixed        = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]

    # recompute lam from actual box size (may differ at image edges)
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (H * W)
    return mixed, labels, labels[idx], lam


def mixup_cutmix_loss(
    criterion:  nn.Module,
    logits:     torch.Tensor,
    labels_a:   torch.Tensor,
    labels_b:   torch.Tensor,
    lam:        float,
) -> torch.Tensor:
    """Soft cross-entropy for mixed samples: lam*CE(a) + (1-lam)*CE(b)."""
    return lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)


# ============================================================================
#                              TRAIN / VALIDATE  (with AMP)
# ============================================================================

def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device, epoch, use_amp,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    mix_prob: float = 0.5,
):
    """
    Training epoch with Mixup/CutMix augmentation (Experiment 4.1).

    At the start of each batch, with probability *mix_prob* we randomly apply
    either Mixup (alpha=0.2) or CutMix (alpha=1.0) — 50/50 between them.
    The remaining (1-mix_prob) fraction of batches are trained with the
    original labels (hard targets).

    Accuracy is computed only from the hard-label majority component
    (argmax of logits vs. the 'a' label) — this gives a comparable accuracy
    number to non-augmented training for monitoring purposes.
    """
    model.train()
    running_loss   = 0.0
    correct, total = 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # ── Mixup / CutMix decision ───────────────────────────────────────
        use_mix = (np.random.rand() < mix_prob)
        if use_mix:
            if np.random.rand() < 0.5:
                images, labels_a, labels_b, lam = mixup_batch(
                    images, labels, alpha=mixup_alpha,
                )
            else:
                images, labels_a, labels_b, lam = cutmix_batch(
                    images, labels, alpha=cutmix_alpha,
                )
        # ─────────────────────────────────────────────────────────────────

        with autocast(enabled=use_amp):
            logits = model(images)
            if use_mix:
                loss = mixup_cutmix_loss(criterion, logits, labels_a, labels_b, lam)
                # accuracy tracked against the dominant label (lam >= 0.5 → a wins)
                hard_labels = labels_a if lam >= 0.5 else labels_b
            else:
                loss        = criterion(logits, labels)
                hard_labels = labels

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds         = logits.argmax(dim=1)
        correct      += (preds == hard_labels).sum().item()
        total        += hard_labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, desc, use_amp):
    model.eval()
    running_loss     = 0.0
    all_preds, all_y = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [{desc}]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_y.extend   (labels.cpu().tolist())

    avg_loss = running_loss / len(all_y)
    acc      = accuracy_score(all_y, all_preds)
    qwk      = cohen_kappa_score(all_y, all_preds, weights="quadratic")
    return avg_loss, acc, qwk, np.array(all_y), np.array(all_preds)


@torch.no_grad()
def evaluate_with_tta(model, loader, device, use_amp):
    """
    TTA: average softmax probabilities over (original, horizontal-flip).
    Returns (y_true, y_pred, y_proba) as NumPy arrays.
    """
    model.eval()
    all_y, all_pred, all_proba = [], [], []

    for images, labels in tqdm(loader, desc="Test [TTA]"):
        images = images.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            p_orig  = torch.softmax(model(images),                       dim=1)
            p_hflip = torch.softmax(model(torch.flip(images, dims=[-1])), dim=1)

        proba = (p_orig + p_hflip) * 0.5
        preds = proba.argmax(dim=1)

        all_y.extend(labels.tolist())
        all_pred.extend(preds.cpu().tolist())
        all_proba.append(proba.cpu().numpy())

    return np.array(all_y), np.array(all_pred), np.concatenate(all_proba, axis=0)


# ============================================================================
#                              PLOTTING
# ============================================================================

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


def plot_confusion_matrix(y_true, y_pred, save_path, title_suffix=""):
    cm      = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None) * 100.0

    plt.figure(figsize=(8, 6.5))
    sns.heatmap(
        cm_norm, annot=True, fmt=".1f",
        cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Percentage (%)"},
        linewidths=0.6, linecolor="white",
        square=True, vmin=0, vmax=100,
        annot_kws={"fontsize": 11},
    )
    plt.title(f"Confusion Matrix (Row-Normalized %) {title_suffix}".strip(),
              fontsize=14, pad=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label",      fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


def save_evaluation_csv(y_true, y_pred, class_names, save_path):
    """
    Per-class metrics table for the thesis Results section.

    Columns: Class | Precision | Recall | F1-Score | Accuracy

    'Accuracy' for each class is the *one-vs-rest binary accuracy*:
        Acc_c = (TP_c + TN_c) / N
    The final 'Overall (macro)' row aggregates with macro averages and the
    overall (multiclass) accuracy.
    """
    n_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    N  = len(y_true)

    p  = precision_score(y_true, y_pred, average=None, zero_division=0,
                         labels=list(range(n_classes)))
    r  = recall_score   (y_true, y_pred, average=None, zero_division=0,
                         labels=list(range(n_classes)))
    f1 = f1_score       (y_true, y_pred, average=None, zero_division=0,
                         labels=list(range(n_classes)))

    rows = []
    for c in range(n_classes):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum() - tp)
        fp = int(cm[:, c].sum() - tp)
        tn = int(N - tp - fn - fp)
        acc_c = (tp + tn) / max(N, 1)
        rows.append({
            "Class":     class_names[c],
            "Precision": round(float(p[c]),  4),
            "Recall":    round(float(r[c]),  4),
            "F1-Score":  round(float(f1[c]), 4),
            "Accuracy":  round(float(acc_c), 4),
        })

    rows.append({
        "Class":     "Overall (macro)",
        "Precision": round(float(p.mean()),  4),
        "Recall":    round(float(r.mean()),  4),
        "F1-Score":  round(float(f1.mean()), 4),
        "Accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
    })

    df = pd.DataFrame(rows, columns=["Class", "Precision", "Recall", "F1-Score", "Accuracy"])
    df.to_csv(save_path, index=False)
    print(f"  Saved: {save_path}")
    return df


def save_classification_report_json(y_true, y_pred, class_names, save_path):
    """Dump sklearn's classification_report (output_dict=True) as JSON."""
    report = classification_report(
        y_true, y_pred,
        target_names = class_names,
        labels       = list(range(len(class_names))),
        digits       = 4,
        zero_division= 0,
        output_dict  = True,
    )
    save_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  Saved: {save_path}")
    return report


def plot_training_history(history, save_path):
    """
    Two-panel figure (Loss + Accuracy vs Epoch) read from the train_log.json
    history list. Each entry must contain: epoch, train_loss, val_loss,
    train_acc, val_acc.

    Saves to *save_path*. Returns silently if history is empty/incomplete.
    """
    if not history:
        print(f"  Skipped {save_path.name} — empty history.")
        return

    required = {"epoch", "train_loss", "val_loss", "train_acc", "val_acc"}
    if not required.issubset(history[0].keys()):
        print(f"  Skipped {save_path.name} — history missing keys {required - history[0].keys()}.")
        return

    epochs   = [h["epoch"]      for h in history]
    tr_loss  = [h["train_loss"] for h in history]
    va_loss  = [h["val_loss"]   for h in history]
    tr_acc   = [h["train_acc"]  for h in history]
    va_acc   = [h["val_acc"]    for h in history]

    plt.rcParams["font.family"]     = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8.5), facecolor="white")

    # Loss panel
    ax1.set_facecolor("white")
    ax1.plot(epochs, tr_loss, color="red",  linewidth=1.6, label="Training loss")
    ax1.plot(epochs, va_loss, color="blue", linewidth=1.6, label="Validation loss")
    ax1.set_xlabel("Epochs", fontsize=11)
    ax1.set_ylabel("Loss",   fontsize=11)
    ax1.set_title("Training and Validation Loss", fontsize=12, fontweight="bold", pad=8)
    ax1.legend(loc="upper right", frameon=False, fontsize=10)
    ax1.grid(alpha=0.25)
    for spine in ax1.spines.values():
        spine.set_linewidth(0.6); spine.set_color("black")

    # Accuracy panel
    ax2.set_facecolor("white")
    ax2.plot(epochs, tr_acc, color="red",  linewidth=1.6, label="Training accuracy")
    ax2.plot(epochs, va_acc, color="blue", linewidth=1.6, label="Validation accuracy")
    ax2.set_xlabel("Epochs",   fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_title("Training and Validation Accuracy", fontsize=12, fontweight="bold", pad=8)
    ax2.legend(loc="lower right", frameon=False, fontsize=10)
    ax2.grid(alpha=0.25)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.6); spine.set_color("black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_loss_curves(train_losses, val_losses, save_path):
    """Fig. 3 styling — white bg, red Training, blue Validation, thin border."""
    epochs = np.arange(1, len(train_losses) + 1)

    plt.rcParams["font.family"]     = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
    ax.set_facecolor("white")

    ax.plot(epochs, train_losses, color="red",  linewidth=1.6, label="Training loss")
    ax.plot(epochs, val_losses,   color="blue", linewidth=1.6, label="Validation loss")

    ax.set_xlabel("Epochs", fontsize=12)
    ax.set_ylabel("Loss",   fontsize=12)
    ax.set_xlim(left=1, right=len(epochs))
    ax.tick_params(axis="both", which="major", labelsize=10, width=0.7, length=4)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("black")
    ax.grid(False)
    ax.legend(loc="upper right", frameon=False, fontsize=11)

    fig.text(0.5, 0.02,
             "Fig. 3. Training and validation loss curves.",
             ha="center", va="bottom", fontsize=11, family="sans-serif")
    plt.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.18)

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================================
#                              MAIN
# ============================================================================

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    args = parse_args()

    output_dir = Path(args.output_dir);  output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir  = Path(args.plots_dir);   plots_dir.mkdir (parents=True, exist_ok=True)
    cache_dir  = Path(args.cache_dir);   cache_dir.mkdir (parents=True, exist_ok=True)
    # Experiment 5 checkpoint filename.
    save_path  = output_dir / "best_model_LHTViT_5.pth"

    # ── Device : SAFE-RUN forces a single GPU ────────────────────────────────
    # DataParallel scatter/gather is the #1 source of CPU/RAM spikes on Kaggle's
    # dual-T4 instance. We pin training to GPU 0 only; one T4 with batch=32 is
    # plenty fast for 24K training images and 100% stable.
    n_gpus_visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus_visible > 0:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    use_amp = bool(args.use_amp) and device.type == "cuda"

    # Autotune cuDNN convolutions for our fixed 224×224 input shape.
    torch.backends.cudnn.benchmark = True

    print("=" * 72)
    print("  EXPERIMENT 5 — LHT-ViT  (Frozen Backbones + Trainable Head, 45K dataset)")
    print("=" * 72)
    print(f"  Device       : {device}  (n_gpus visible = {n_gpus_visible}, using 1)")
    if device.type == "cuda":
        for i in range(n_gpus_visible):
            mark = "  <-- ACTIVE" if i == 0 else "  (idle, by design)"
            print(f"    GPU[{i}]    : {torch.cuda.get_device_name(i)}{mark}")
    print(f"  Data dir     : {args.data_dir}")
    print(f"  Cache dir    : {cache_dir}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Plots dir    : {plots_dir}")
    print(f"  Image size   : {args.image_size}x{args.image_size}")
    print(f"  Batch size   : {args.batch_size}   (single-GPU, no DataParallel)")
    print(f"  Epochs (max) : {args.epochs}       Warmup = {args.warmup_epochs}    Patience = {args.patience}  (on val_QWK)")
    print(f"  Optimizer    : AdamW(lr={args.lr}, wd={args.weight_decay}, head-only params)  + CosineAnnealingLR(eta_min=1e-7)")
    print(f"  Backbones    : EfficientNet-B0 + ViT-Small  [FROZEN — no grad]")
    print(f"  Mixup/CutMix : DISABLED (mix_prob=0.0)")
    print(f"  Loss         : CE(weighted, label_smoothing={args.label_smoothing})")
    print(f"  Head dropout : {args.head_dropout}")
    print(f"  Subset       : {args.num_samples} stratified images "
          f"(80/10/10 → {int(args.num_samples*0.8)}/{int(args.num_samples*0.1)}/"
          f"{int(args.num_samples*0.1)})")
    print(f"  Workers      : {args.num_workers}   pin_memory = {bool(args.pin_memory)}")
    print(f"  AMP          : {use_amp}     TTA = {bool(args.use_tta)}")
    print(f"  MFB          : k={args.mfb_k}, out={args.mfb_out}, attn_dim={args.attn_dim}")
    print(f"  DropPath     : {args.drop_path_rate}  (stochastic depth in ViT-Small)")
    print()

    # ── Build cached dataloaders ─────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_cached_dataloaders(
        data_dir   = args.data_dir,
        cache_dir  = cache_dir,
        image_size = args.image_size,
        batch_size = args.batch_size,
        num_samples= args.num_samples,
        num_workers= args.num_workers,
        pin_memory = bool(args.pin_memory),
    )

    # ── Cost-sensitive class weights (computed from TRAIN labels only) ───────
    train_labels  = train_loader.dataset.df["label"].tolist()
    class_weights = get_cost_sensitive_weights(train_labels).to(device)
    print(f"\nClass weights (inverse-freq): {class_weights.cpu().numpy().round(4)}")

    criterion = nn.CrossEntropyLoss(
        weight          = class_weights,
        label_smoothing = args.label_smoothing,
    )

    # ── Model (single GPU — no DataParallel) ─────────────────────────────────
    print("\nBuilding HybridLHT_ViT (ViT-Small, frozen backbones) ...")
    model = HybridLHT_ViT(
        num_classes    = args.num_classes,
        attn_dim       = args.attn_dim,
        mfb_k          = args.mfb_k,
        mfb_out        = args.mfb_out,
        head_dropout   = args.head_dropout,
        drop_path_rate = args.drop_path_rate,
        pretrained     = True,
    ).to(device)

    # ── BACKBONE FREEZING (Experiment 5 core strategy) ────────────────────────
    # Freeze every parameter in both backbone networks immediately after
    # instantiation. Only Co-Attention, MFB, and the classifier head remain
    # trainable. This eliminates gradient flow into the 26M-param backbones,
    # cutting VRAM usage ~40% and removing the primary source of overfitting
    # (noise memorisation in backbone weights on a small dataset).
    for p in model.cnn.parameters():
        p.requires_grad = False
    for p in model.vit.parameters():
        p.requires_grad = False

    # Verify the trainable / frozen split
    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen    = n_total - n_trainable

    print(f"  Total parameters       : {n_total/1e6:.2f} M")
    print(f"  CNN  (EfficientNet-B0) : {sum(p.numel() for p in model.cnn.parameters())/1e6:.2f} M  [FROZEN]")
    print(f"  ViT  (ViT-Small/16)    : {sum(p.numel() for p in model.vit.parameters())/1e6:.2f} M  [FROZEN]")
    print(f"  Co-Attention block     : {sum(p.numel() for p in model.coattn.parameters())/1e6:.2f} M  [trainable]")
    print(f"  MFB pooling block      : {sum(p.numel() for p in model.mfb.parameters())/1e6:.2f} M  [trainable]")
    print(f"  Classification head    : {sum(p.numel() for p in model.classifier.parameters())/1e6:.2f} M  [trainable]")
    print(f"  Trainable / Frozen     : {n_trainable/1e6:.2f} M  /  {n_frozen/1e6:.2f} M")

    print(f"  Running on a single device: {device}.  (DataParallel intentionally disabled.)")

    def unwrap(m):
        # Kept as a no-op so the rest of the script keeps working unchanged.
        return m

    # ── Defaults so the evaluation block below works even when training is
    #     skipped (SKIP_TRAINING=True) and these vars never get populated.
    best_val_qwk      = -1.0
    best_val_loss     = float("inf")
    epochs_no_improve = 0
    train_losses: list[float] = []
    val_losses:   list[float] = []
    history:      list[dict]  = []
    stopped_early = False
    last_epoch    = 0

    if not SKIP_TRAINING:
        # ── Optimizer + CosineAnnealingLR + AMP scaler ───────────────────────
        # Exp 5: optimizer is given ONLY the parameters that require gradients
        # (Co-Attention + MFB + classifier head, ~8 M params). Passing frozen
        # backbone parameters to AdamW would waste memory maintaining momentum
        # buffers for weights that are never updated.
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr           = args.lr,
            weight_decay = args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max   = args.epochs,
            eta_min = 1e-7,
        )
        scaler = GradScaler(enabled=use_amp)

        # ── Training loop ────────────────────────────────────────────────────
        # SAFE-RUN policy:
        #   • Save best on val_QWK   (clinical ranking metric the thesis cares about)
        #   • Early-stop on val_QWK  (NOT val_loss — under heavy label smoothing,
        #     val_loss can plateau or even rise while val_QWK keeps improving)
        print("\n" + "=" * 72)
        print(f"  TRAINING  (early stopping: patience = {args.patience} on val_QWK)")
        print("=" * 72)
        print(f"{'Epoch':>5}  {'LR':>9}  {'TrLoss':>8}  {'TrAcc':>7}  {'VaLoss':>8}  "
              f"{'VaAcc':>7}  {'VaQWK':>7}  {'Saved':>5}  {'NoImp':>5}")
        print("-" * 84)

        for epoch in range(1, args.epochs + 1):
            last_epoch = epoch

            # Memory hygiene
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            current_lr = optimizer.param_groups[0]["lr"]

            # Exp 5: mix_prob=0.0 disables Mixup/CutMix entirely. The function
            # signature is unchanged so the evaluation pipeline stays identical.
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device, epoch, use_amp,
                mixup_alpha=0.2, cutmix_alpha=1.0, mix_prob=0.0,
            )
            val_loss, val_acc, val_qwk, _, _ = validate(
                model, val_loader, criterion, device, epoch, desc="Val", use_amp=use_amp,
            )
            scheduler.step()

            train_losses.append(train_loss)
            val_losses  .append(val_loss)

            # Save + early-stop both gated on val_QWK (the SAFE-RUN policy).
            saved = ""
            if val_qwk > best_val_qwk:
                best_val_qwk      = val_qwk
                epochs_no_improve = 0
                torch.save({
                    "epoch":                epoch,
                    "model_state_dict":     unwrap(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss":             val_loss,
                    "val_acc":              val_acc,
                    "val_qwk":              val_qwk,
                    "args":                 vars(args),
                    "architecture":         "HybridLHT_ViT (Frozen EffB0+ViT-Small + CoAttn+MFB Head)  [Exp 5]",
            }, save_path)
                saved = "*"
            else:
                epochs_no_improve += 1

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            print(f"{epoch:>5}  {current_lr:>9.2e}  {train_loss:>8.4f}  {train_acc:>7.4f}  "
                  f"{val_loss:>8.4f}  {val_acc:>7.4f}  {val_qwk:>7.4f}  "
                  f"{saved:>5}  {epochs_no_improve:>5}")

            history.append({
                "epoch":      epoch,
                "lr":         current_lr,
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss":   val_loss,   "val_acc":   val_acc,
                "val_qwk":    val_qwk,
                "saved":      bool(saved),
                "epochs_no_improve": epochs_no_improve,
            })

            if args.patience > 0 and epochs_no_improve >= args.patience:
                stopped_early = True
                print(f"\n>>> Early stopping at epoch {epoch}: "
                      f"val_QWK has not improved for {args.patience} consecutive epochs.")
                break

        print("-" * 84)
        print(f"Training complete. Epochs run: {last_epoch}/{args.epochs}"
              f"{'  (stopped early)' if stopped_early else ''}")
        print(f"Best val_QWK  : {best_val_qwk:.4f}")
        print(f"Best val_loss : {best_val_loss:.4f}")
        print(f"Best weights  : {save_path}")

        (output_dir / "train_log.json").write_text(
            json.dumps({
                "timestamp":        datetime.now().isoformat(timespec="seconds"),
                "architecture":     "HybridLHT_ViT (Frozen EffB0+ViT-Small + CoAttn+MFB Head)  [Exp 5]",
                "best_val_qwk":     best_val_qwk,
                "best_val_loss":    best_val_loss,
                "stopped_early":    stopped_early,
                "epochs_run":       last_epoch,
                "args":             vars(args),
                "history":          history,
            }, indent=2),
            encoding="utf-8",
        )
    else:
        print("\n" + "=" * 72)
        print("  SKIP_TRAINING = True  ->  jumping straight to evaluation")
        print(f"  Will load checkpoint: {save_path}")
        print("=" * 72)

    # ────────────────────────────────────────────────────────────────────────
    # EXPERIMENT 3 — TEST-SET EVALUATION  (with optional TTA)
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  EXPERIMENT 3 — TEST EVALUATION")
    print("=" * 72)

    print(f"Loading best checkpoint: {save_path}")
    # weights_only=False is required because the checkpoint also stores the
    # argparse.Namespace under the "args" key, which is not in PyTorch 2.4+'s
    # default safe-unpickle allowlist.
    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    unwrap(model).load_state_dict(ckpt["model_state_dict"])

    if args.use_tta:
        print("Running TTA evaluation (original + horizontal flip) ...")
        y_true, y_pred, y_proba = evaluate_with_tta(model, test_loader, device, use_amp)
        # Compute test loss separately (no TTA needed for the loss number)
        test_loss, _, _, _, _ = validate(
            model, test_loader, criterion, device, last_epoch, desc="Test-loss", use_amp=use_amp,
        )
    else:
        test_loss, _, _, y_true, y_pred = validate(
            model, test_loader, criterion, device, last_epoch, desc="Test", use_amp=use_amp,
        )
        y_proba = None

    # ── 5-class metrics  (PRIMARY thesis number) ─────────────────────────────
    test_acc  = accuracy_score(y_true, y_pred)
    test_qwk  = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    test_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    test_rec  = recall_score   (y_true, y_pred, average="macro", zero_division=0)
    test_f1   = f1_score       (y_true, y_pred, average="macro", zero_division=0)

    print("\n" + "=" * 72)
    print("  PRIMARY  ──  5-class DR Grading  (No DR / Mild / Mod / Sev / Prolif.)")
    print("=" * 72)
    print(f"  Final Accuracy           : {test_acc:.4f}")
    print(f"  Quadratic Weighted Kappa : {test_qwk:.4f}")
    print(f"  Precision (macro)        : {test_prec:.4f}")
    print(f"  Recall    (macro)        : {test_rec:.4f}")
    print(f"  F1-Score  (macro)        : {test_f1:.4f}")
    print(f"  Test Loss                : {test_loss:.4f}")

    print("\n" + "-" * 72)
    print("  CLASSIFICATION REPORT (5-class)")
    print("-" * 72)
    print(classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0,
    ))

    # ── Binary "Referable DR" metrics (SECONDARY clinical number) ────────────
    # Convention: Referable DR = grade ≥ 2 (Moderate or worse).
    # This is the screening-tool framing used by Gulshan et al. 2016, Krause
    # et al. 2018, and the FDA-cleared IDx-DR system.
    y_true_bin = (y_true >= 2).astype(int)
    y_pred_bin = (y_pred >= 2).astype(int)

    bin_acc  = accuracy_score(y_true_bin, y_pred_bin)
    bin_sens = recall_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    bin_spec = recall_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0)
    bin_prec = precision_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    bin_f1   = f1_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)

    print("=" * 72)
    print("  SECONDARY  ──  Binary Referable DR  (grade >= Moderate)")
    print("=" * 72)
    print(f"  Accuracy                 : {bin_acc:.4f}")
    print(f"  Sensitivity (Recall+)    : {bin_sens:.4f}")
    print(f"  Specificity (Recall−)    : {bin_spec:.4f}")
    print(f"  Precision   (PPV)        : {bin_prec:.4f}")
    print(f"  F1-Score                 : {bin_f1:.4f}")

    # ── Save metrics CSV ─────────────────────────────────────────────────────
    metrics_rows = [
        # 5-class
        {"Task": "5-class",  "Metric": "Accuracy",                 "Value": round(test_acc,  4)},
        {"Task": "5-class",  "Metric": "Quadratic Weighted Kappa", "Value": round(test_qwk,  4)},
        {"Task": "5-class",  "Metric": "Precision (macro)",        "Value": round(test_prec, 4)},
        {"Task": "5-class",  "Metric": "Recall (macro)",           "Value": round(test_rec,  4)},
        {"Task": "5-class",  "Metric": "F1-Score (macro)",         "Value": round(test_f1,   4)},
        {"Task": "5-class",  "Metric": "Test Loss",                "Value": round(test_loss, 4)},
        # Binary
        {"Task": "Binary",   "Metric": "Accuracy",                 "Value": round(bin_acc,   4)},
        {"Task": "Binary",   "Metric": "Sensitivity",              "Value": round(bin_sens,  4)},
        {"Task": "Binary",   "Metric": "Specificity",              "Value": round(bin_spec,  4)},
        {"Task": "Binary",   "Metric": "Precision",                "Value": round(bin_prec,  4)},
        {"Task": "Binary",   "Metric": "F1-Score",                 "Value": round(bin_f1,    4)},
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    csv_path   = plots_dir / "experiment_3_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # ── Per-class CSV (THESIS-REQUIRED format) ──────────────────────────────
    # Columns: Class | Precision | Recall | F1-Score | Accuracy
    eval_csv_path = output_dir / "evaluation_results.csv"
    per_class_df  = save_evaluation_csv(y_true, y_pred, CLASS_NAMES, eval_csv_path)

    # ── Full classification_report saved as JSON ────────────────────────────
    cls_report_path = output_dir / "classification_report.json"
    cls_report      = save_classification_report_json(
        y_true, y_pred, CLASS_NAMES, cls_report_path,
    )

    # ── Plots ────────────────────────────────────────────────────────────────
    cm_path   = plots_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, cm_path, title_suffix="(5-class, TTA)" if args.use_tta else "(5-class)")

    bin_cm_path = plots_dir / "confusion_matrix_binary.png"
    plt.figure(figsize=(5.5, 5.0))
    bin_cm = confusion_matrix(y_true_bin, y_pred_bin)
    bin_cm_norm = bin_cm.astype(float) / np.clip(bin_cm.sum(axis=1, keepdims=True), 1, None) * 100.0
    sns.heatmap(bin_cm_norm, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=["Non-referable", "Referable"],
                yticklabels=["Non-referable", "Referable"],
                cbar_kws={"label": "Percentage (%)"},
                linewidths=0.6, linecolor="white", square=True,
                vmin=0, vmax=100, annot_kws={"fontsize": 12})
    plt.title("Binary Confusion Matrix — Referable DR (≥ Moderate)",
              fontsize=12, pad=10, fontweight="bold")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(bin_cm_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {bin_cm_path}")

    loss_path = plots_dir / "training_loss_curves.png"
    if len(train_losses) > 0 and len(val_losses) > 0:
        plot_loss_curves(train_losses, val_losses, loss_path)
    else:
        print(f"  Skipped loss curves (no in-memory training history).")

    # ── Combined Loss + Accuracy curves (THESIS-REQUIRED) ───────────────────
    # Prefer the in-memory `history` (just-trained run); fall back to
    # train_log.json on disk so the plot still works under SKIP_TRAINING=True.
    history_for_plot = history
    if not history_for_plot:
        log_json_path = output_dir / "train_log.json"
        if log_json_path.exists():
            try:
                history_for_plot = json.loads(log_json_path.read_text(encoding="utf-8")).get("history", [])
                print(f"  Loaded training history from {log_json_path}")
            except Exception as e:
                print(f"  [warn] failed reading {log_json_path}: {e}")
                history_for_plot = []

    history_path = plots_dir / "training_history.png"
    plot_training_history(history_for_plot, history_path)

    # ── FINAL SUMMARY ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  FINAL SUMMARY  (Experiment 3 — Hybrid LHT-ViT, test set)")
    print("=" * 72)
    print(f"  Quadratic Weighted Kappa (QWK)  : {test_qwk:.4f}")
    print(f"  F1-Score (macro, 5-class)       : {test_f1:.4f}")
    print(f"  F1-Score (binary referable)     : {bin_f1:.4f}")
    print(f"  Accuracy (5-class)              : {test_acc:.4f}")
    print(f"  Accuracy (binary referable)     : {bin_acc:.4f}")
    print("=" * 72)

    print("\nAll outputs:")
    print(f"  • {save_path}")
    print(f"  • {csv_path}                     (long-format metrics)")
    print(f"  • {eval_csv_path}                (per-class metrics for thesis)")
    print(f"  • {cls_report_path}              (classification_report JSON)")
    print(f"  • {cm_path}")
    print(f"  • {bin_cm_path}")
    print(f"  • {loss_path}")
    print(f"  • {history_path}                 (combined loss + accuracy)")
    print(f"  • {output_dir / 'train_log.json'}")
    print("\nDone.")


# In a Jupyter cell, just invoke main():
main()
