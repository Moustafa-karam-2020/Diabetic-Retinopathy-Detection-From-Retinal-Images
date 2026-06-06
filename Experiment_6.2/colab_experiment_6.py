"""
============================================================================
EXPERIMENT 6.2 — LHT-ViT  Definitive Overfitting Elimination & Metrics Run
   EfficientNet-B0 + ViT-Small  ·  Bidirectional Co-Attention  ·  MFB
   Sigmoid-Bounded Ordinal Head  ·  OrdinalSmoothMSE  ·  LinearWarmup+Cosine
                   Google Colab Pro  (L4 / A100)
============================================================================

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
    │             drop_path_rate = 0.25  (stochastic depth per block)
    │             attn_drop_rate = 0.10  (attention-weight dropout)
    │
    ├── CO-ATTENTION BLOCK ── bidirectional spatial cross-modal attention
    │       cnn_p [B,49,512]  ←→  vit_p [B,196,512]
    │       residual + LayerNorm → mean-pool → cnn_vec, vit_vec ∈ ℝ^512
    │
    ├── MFB POOLING ── Multi-modal Factorized Bilinear (K=5, out=1024)
    │       z = sign(z)·√|z|  then L2-norm  →  fused ∈ ℝ^1024
    │
    └── HEAD : LayerNorm → 512 → GELU → Dropout(0.4) → Linear(512 → 1)
               Single continuous scalar (ordinal regression target: 0.0–4.0)

CHANGES FROM EXPERIMENT 6.1  (this is the definitive thesis benchmark build)
------------------------------------------------------------------------------
1. Bounded Ordinal Head  (NEW in 6.2):
   The raw Linear output is passed through  4 × sigmoid(x)  which maps
   (−∞, +∞) → (0, 4) strictly. Unbounded heads can memorise training targets
   by pushing logits to ±∞ (driving train loss → 0 while val loss stays
   high).  Sigmoid saturation physically prevents this: weights would have to
   grow to ±∞ to achieve extreme predictions, which AdamW + weight_decay=0.05
   actively resists.

2. weight_decay  : 1e-2  →  5e-2  (stronger L2 across all 33.88 M params)

3. LR Scheduler  : plain CosineAnnealingLR
                 → LinearWarmup (5 epochs, 0 → 2e-5) + CosineAnnealingLR
   Prevents early crystallisation of fragile attention maps.

4. Loss  : OrdinalSmoothMSELoss(smoothing=0.1)  — retained from 6.2 series.
   Smoothed targets structurally bound minimum training loss > 0.

5. All 6.1 regularisations retained  (drop_path=0.25, attn_drop=0.1,
   RandomErasing p=0.25).

6. Expanded evaluation suite  (NEW in 6.2):
   • metrics_summary.csv            (Acc, Macro-F1, Micro-F1, Precision,
                                      Sensitivity, Specificity, QWK)
   • per_class_performance.csv       (F1/Sensitivity/Specificity × 5 grades)
   • confusion_matrix_multiclass.csv + PNG
   • confusion_matrix_referable_binary.csv + PNG  (0–1 vs 2–4 clinical binary)

ENVIRONMENT
-----------
Google Colab Pro  (L4 or A100 GPU, High-RAM runtime)
All weights, CSVs, and PNGs are saved to Google Drive at:
  /content/drive/MyDrive/DR_Experiment_6_2_Outputs/
============================================================================
"""

# ============================================================================
#                         STEP 0 — COLAB SETUP
# ============================================================================

import subprocess, sys

def pip_install(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])

# Ensure required packages are present (pre-installed on Colab, but explicit
# versions guard against runtime image changes).
pip_install("timm>=0.9.0", "albumentations>=1.3.0", "scikit-learn>=1.3.0")

# Mount Google Drive — outputs go here so they survive session restarts.
#from google.colab import drive
#drive.mount("/content/drive")

# ============================================================================
#                         STEP 1 — IMPORTS
# ============================================================================

import os, gc, json, math, hashlib, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
#                         STEP 2 — CONFIGURATION
# ============================================================================

CFG = dict(
    # ── Paths ────────────────────────────────────────────────────────────────
    data_dir     = "/content/dataset/augmented_resized_V2/train",
    output_dir   = "/content/drive/MyDrive/DR_Experiment_6_2_Outputs",  # 6.2
    cache_dir    = "/content/cache/preproc",   # reuse prior cache — no rebuild

    # ── Data ─────────────────────────────────────────────────────────────────
    num_samples  = 80_000,           # stratified pool (unchanged)
    image_size   = 224,
    batch_size   = 64,
    num_workers  = 4,
    pin_memory   = True,
    seed         = 42,

    # ── Training ─────────────────────────────────────────────────────────────
    epochs        = 50,              # early stopping will cut this
    lr            = 2e-5,            # peak LR (reached after warmup)
    weight_decay  = 5e-2,            # 6.2: 1e-2 → 5e-2 (stronger L2 penalty)
    eta_min       = 1e-7,
    patience      = 10,              # unchanged from 6.1
    warmup_epochs = 5,               # 6.2: linear warmup for first 5 epochs

    # ── Architecture ─────────────────────────────────────────────────────────
    attn_dim       = 512,
    mfb_k          = 5,
    mfb_out        = 1024,
    head_dropout   = 0.4,
    drop_path_rate = 0.25,           # retained from 6.1
    attn_drop_rate = 0.10,           # retained from 6.1

    # ── Loss ─────────────────────────────────────────────────────────────────
    # 6.2: OrdinalSmoothMSELoss — MSE on label-smoothed float targets.
    # smoothing=0.1 pulls each integer grade target g slightly toward the
    # scale midpoint (2.0):  target = 0.9*g + 0.1*2.0
    # This prevents exact-integer overfitting and narrows train/val loss gap.
    label_smoothing = 0.10,
    huber_delta     = None,          # kept for JSON serialisation compatibility
)

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

# Create output directories
Path(CFG["output_dir"]).mkdir(parents=True, exist_ok=True)
Path(CFG["cache_dir"]).mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("  EXPERIMENT 6.2 — LHT-ViT  Definitive Benchmark Build")
print("  Sigmoid-Bounded Ordinal Head  ·  OrdinalSmoothMSE  ·  Warmup+Cosine")
print("=" * 72)
print(f"  Output dir   : {CFG['output_dir']}")
print(f"  Dataset      : {CFG['data_dir']}")
print(f"  Num samples  : {CFG['num_samples']:,}  →  80/10/10 split")
print(f"  Batch size   : {CFG['batch_size']}   Workers: {CFG['num_workers']}   "
      f"pin_memory: {CFG['pin_memory']}")
print(f"  Epochs (max) : {CFG['epochs']}   Patience: {CFG['patience']} (on val_QWK)")
print(f"  Optimizer    : AdamW(lr={CFG['lr']}, wd={CFG['weight_decay']})")
print(f"  Scheduler    : LinearWarmup({CFG['warmup_epochs']} ep) "
      f"→ CosineAnnealingLR(eta_min={CFG['eta_min']})")
print(f"  Loss         : OrdinalSmoothMSE(smoothing={CFG['label_smoothing']})")
print(f"  Head         : Linear → 4×sigmoid  (bounded output ∈ (0,4))")
print(f"  Regression   : bounded scalar → round → clip → grade 0–4")
print(f"  ViT drops    : drop_path={CFG['drop_path_rate']}  "
      f"attn_drop={CFG['attn_drop_rate']}")
print(f"  Augmentation : HFlip + Rotate15 + RandomErasing(p=0.25)")
print()

# ============================================================================
#                         STEP 3 — MODEL
# ============================================================================

class CoAttention(nn.Module):
    """Bidirectional spatial cross-modal attention (CNN ↔ ViT tokens)."""

    def __init__(self, cnn_dim: int, vit_dim: int,
                 attn_dim: int = 512, attn_dropout: float = 0.1):
        super().__init__()
        self.attn_dim   = attn_dim
        self.attn_scale = attn_dim ** 0.5

        self.cnn_proj = nn.Linear(cnn_dim, attn_dim)
        self.vit_proj = nn.Linear(vit_dim, attn_dim)

        self.q1 = nn.Linear(attn_dim, attn_dim)   # CNN-Q over ViT-K/V
        self.k1 = nn.Linear(attn_dim, attn_dim)
        self.v1 = nn.Linear(attn_dim, attn_dim)

        self.q2 = nn.Linear(attn_dim, attn_dim)   # ViT-Q over CNN-K/V
        self.k2 = nn.Linear(attn_dim, attn_dim)
        self.v2 = nn.Linear(attn_dim, attn_dim)

        self.drop  = nn.Dropout(attn_dropout)
        self.norm1 = nn.LayerNorm(attn_dim)
        self.norm2 = nn.LayerNorm(attn_dim)

    def _attend(self, q_w, k_w, v_w, query, kv):
        Q = q_w(query); K = k_w(kv); V = v_w(kv)
        score = torch.bmm(Q, K.transpose(1, 2)) / self.attn_scale
        return torch.bmm(self.drop(torch.softmax(score, dim=-1)), V)

    def forward(self, cnn_tokens, vit_tokens):
        cp = self.cnn_proj(cnn_tokens)     # [B, 49,  D]
        vp = self.vit_proj(vit_tokens)     # [B, 196, D]
        ca = self.norm1(self._attend(self.q1, self.k1, self.v1, cp, vp) + cp)
        va = self.norm2(self._attend(self.q2, self.k2, self.v2, vp, cp) + vp)
        return ca.mean(dim=1), va.mean(dim=1)   # [B, D], [B, D]


class MFBPooling(nn.Module):
    """Multi-modal Factorized Bilinear pooling (Yu et al., ICCV 2017)."""

    def __init__(self, in_dim_x, in_dim_y, mfb_k=5, mfb_out=1024,
                 dropout=0.1):
        super().__init__()
        self.mfb_k   = mfb_k
        self.mfb_out = mfb_out
        self.proj_x  = nn.Linear(in_dim_x, mfb_k * mfb_out)
        self.proj_y  = nn.Linear(in_dim_y, mfb_k * mfb_out)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x, y):
        B  = x.shape[0]
        z  = self.drop(self.proj_x(x) * self.proj_y(y))
        z  = z.view(B, self.mfb_out, self.mfb_k).sum(dim=2)
        z  = torch.sign(z) * torch.sqrt(torch.abs(z) + 1e-8)
        return F.normalize(z, p=2, dim=1)


class HybridLHT_ViT_Ordinal(nn.Module):
    """
    LHT-ViT with an ORDINAL REGRESSION head.

    The final Linear outputs a single continuous scalar ∈ [0, 4] (soft DR
    grade). SmoothL1Loss is computed against float targets.  At inference,
    round() + clip(0,4) gives the discrete grade.
    """

    def __init__(self, attn_dim=512, mfb_k=5, mfb_out=1024,
                 head_dropout=0.4, drop_path_rate=0.25,
                 attn_drop_rate=0.10, pretrained=True):
        super().__init__()

        # CNN backbone — local texture features
        self.cnn = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=0,
        )
        # ViT backbone — global anatomical context
        # drop_path_rate: stochastic depth — each transformer block's residual
        #   path is randomly disabled during training (linearly scaled per layer).
        #   Prevents deep-layer token memorisation.
        # attn_drop_rate: dropout on the softmax attention weights — forces each
        #   head to distribute attention across multiple tokens/regions.
        self.vit = timm.create_model(
            "vit_small_patch16_224",
            pretrained      = pretrained,
            num_classes     = 0,
            drop_path_rate  = drop_path_rate,
            attn_drop_rate  = attn_drop_rate,
        )
        cnn_dim = self.cnn.num_features   # 1280
        vit_dim = self.vit.embed_dim      # 384

        self.coattn = CoAttention(cnn_dim, vit_dim, attn_dim=attn_dim)
        self.mfb    = MFBPooling(attn_dim, attn_dim, mfb_k=mfb_k,
                                 mfb_out=mfb_out)

        # Bounded Ordinal Regression head — sigmoid-scaled to open interval (0,4).
        # Architecture: LayerNorm → Linear(1024→512) → GELU → Dropout → Linear(512→1)
        # The raw Linear output passes through 4 * sigmoid(x), which maps
        # (-∞, +∞)  →  (0, 4) strictly.
        #
        # Why sigmoid-bounded instead of unbounded MSE:
        #   • Unbounded heads can memorise training targets by pushing logits
        #     to ±∞, collapsing training loss while validation loss stays high.
        #   • Sigmoid saturation acts as an implicit regulariser: extreme
        #     predictions (near 0 or near 4) require infinitely large pre-sigmoid
        #     values, which AdamW + weight_decay actively prevents.
        #   • The bounded range matches the ordinal label range exactly, so the
        #     model's uncertainty is naturally expressed as values inside (0,4)
        #     rather than as unconstrained real numbers.
        self.head_linear = nn.Sequential(
            nn.LayerNorm(mfb_out),
            nn.Linear(mfb_out, 512),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(512, 1),
        )
        self._grade_scale = 4.0        # maps sigmoid(0,1) → (0, 4)

    def forward(self, x):
        cnn_map    = self.cnn.forward_features(x)          # [B,1280,7,7]
        cnn_tokens = cnn_map.flatten(2).transpose(1, 2)    # [B,49,1280]

        vit_seq    = self.vit.forward_features(x)          # [B,197,384]
        vit_tokens = vit_seq[:, 1:, :]                     # [B,196,384]

        cnn_vec, vit_vec = self.coattn(cnn_tokens, vit_tokens)
        fused  = self.mfb(cnn_vec, vit_vec)
        logit  = self.head_linear(fused).squeeze(1)        # [B]  unbounded
        return self._grade_scale * torch.sigmoid(logit)    # [B]  ∈ (0, 4)


# ============================================================================
#                         STEP 4 — PREPROCESSING
# ============================================================================

def ben_graham_preprocess(image: np.ndarray, sigmaX: int = 10) -> np.ndarray:
    """Circular crop + Gaussian-blend normalization (Ben Graham, 2015)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(largest)
        cx, cy, radius = int(cx), int(cy), int(radius)
        h, w = image.shape[:2]
        x1 = max(cx - radius, 0); y1 = max(cy - radius, 0)
        x2 = min(cx + radius, w); y2 = min(cy + radius, h)
        image = image[y1:y2, x1:x2]

    image   = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigmaX)
    return cv2.addWeighted(image, 4, blurred, -4, 128)


def _cache_name(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()[:16] + ".png"


def precompute_cache(df: pd.DataFrame, cache_dir: Path,
                     image_size: int = 224) -> pd.DataFrame:
    """Ben Graham + CLAHE once per image → 224×224 PNG on disk (resumable)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    clahe = A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)

    cached_paths, n_built, n_skip, n_fail = [], 0, 0, 0
    pbar = tqdm(df.itertuples(index=False), total=len(df),
                desc="Pre-caching", unit="img")
    for row in pbar:
        dst = cache_dir / _cache_name(row.image_path)
        if dst.exists():
            n_skip += 1; cached_paths.append(str(dst)); continue
        try:
            img = cv2.imread(row.image_path)
            if img is None: raise ValueError("cv2 returned None")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = ben_graham_preprocess(img)
            img = clahe(image=img)["image"]
            img = cv2.resize(img, (image_size, image_size),
                             interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dst), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            n_built += 1
        except Exception as e:
            n_fail += 1
            print(f"\n[warn] {row.image_path}: {e}")
        cached_paths.append(str(dst))
        if n_built % 5000 == 0 and n_built:
            pbar.set_postfix(built=n_built, skip=n_skip, fail=n_fail)

    print(f"\nCache: built={n_built}  skipped={n_skip}  failed={n_fail}")
    df = df.copy(); df["cached_path"] = cached_paths
    return df


def prepare_data(root_dir: str, num_samples: int,
                 seed: int = 42) -> pd.DataFrame:
    """Scan directory tree, assign labels, stratified-sample num_samples rows."""
    root = Path(root_dir)
    records = []
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir(): continue
        try: label = int(cls_dir.name)
        except ValueError: continue
        for img_path in cls_dir.glob("*"):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                records.append({"image_path": str(img_path), "label": label})

    df = pd.DataFrame(records)
    if len(df) == 0:
        raise RuntimeError(f"No images found under {root_dir}")

    # Stratified sample
    from sklearn.model_selection import train_test_split as tts
    if num_samples < len(df):
        _, df = tts(df, test_size=num_samples / len(df),
                    stratify=df["label"], random_state=seed)
    return df.reset_index(drop=True)


class _RandomErasingWrapper:
    """
    Wraps torchvision.transforms.v2.RandomErasing so it can sit inside an
    albumentations pipeline after ToTensorV2.

    RandomErasing must run on a float tensor (C, H, W) — i.e. AFTER
    ToTensorV2 and Normalize — which is why it cannot be an Albumentations
    transform directly.
    """
    def __init__(self, p=0.25, scale=(0.02, 0.08), value="random"):
        from torchvision.transforms import v2
        self._eraser = v2.RandomErasing(
            p=p, scale=scale, ratio=(0.3, 3.3), value=value, inplace=False,
        )

    def __call__(self, **data):
        # albumentations passes the tensor under the "image" key
        data["image"] = self._eraser(data["image"])
        return data


class _AlbuWrapper(A.DualTransform):
    """Bridges a plain callable (tensor → tensor) into an albumentations Compose."""
    def __init__(self, fn, p=1.0):
        super().__init__(p=p)
        self._fn = fn

    def apply(self, img, **_):
        return self._fn(img)

    def get_transform_init_args_names(self):
        return ()


def _make_random_erasing_transform():
    """Returns an albumentations-compatible RandomErasing transform."""
    from torchvision.transforms import v2
    eraser = v2.RandomErasing(
        p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3), value="random",
        inplace=False,
    )
    return _AlbuWrapper(eraser, p=1.0)


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """
    Experiment 6.1 training pipeline.

    Step-by-step:
      1. HorizontalFlip(p=0.5)
      2. Rotate ±15°
      3. Normalize to ImageNet stats
      4. ToTensorV2           → float tensor (C, H, W)
      5. RandomErasing(p=0.25, scale=(0.02,0.08), value='random')
         — removes 2–8 % of image area, forces attention to use global context.
         — runs on the tensor AFTER normalization (required by torchvision).
    """
    from torchvision.transforms import v2 as tv2

    # We apply RandomErasing as a post-processing step inside __getitem__
    # of LazyDRDataset, not inside the A.Compose, because torchvision's
    # RandomErasing expects a tensor, not a numpy array.
    # The _random_erasing_fn is stored as a module-level callable and called
    # explicitly after A.Compose returns in LazyDRDataset.__getitem__.
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# Module-level erasing callable — instantiated once, shared across workers.
_RANDOM_ERASING = None

def _get_random_erasing():
    global _RANDOM_ERASING
    if _RANDOM_ERASING is None:
        from torchvision.transforms import v2 as tv2
        _RANDOM_ERASING = tv2.RandomErasing(
            p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3),
            value="random", inplace=False,
        )
    return _RANDOM_ERASING


def get_val_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class LazyDRDataset(Dataset):
    """
    Lazy disk dataset. __init__ stores only metadata (paths + labels).
    __getitem__ opens one pre-cached PNG via PIL, no RAM accumulation.

    Experiment 6.1: when is_train=True, applies RandomErasing on the float
    tensor after albumentations normalization. Val/test splits set
    is_train=False and never touch RandomErasing.
    """
    def __init__(self, df: pd.DataFrame, transform: A.Compose,
                 is_train: bool = False):
        df = df.reset_index(drop=True)
        self.paths     = df["cached_path"].astype(str).tolist()
        self.labels    = df["label"].astype(float).tolist()  # float for MSE
        self.transform = transform
        self.is_train  = is_train
        self._df       = df

    def __len__(self): return len(self.paths)

    @property
    def df(self): return self._df

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as im:
            img = np.asarray(im.convert("RGB"), dtype=np.uint8)
        tensor = self.transform(image=img)["image"]   # float (C,H,W), normalised
        if self.is_train:
            tensor = _get_random_erasing()(tensor)
        return tensor, self.labels[idx]


def build_dataloaders(cfg: dict):
    """Full pipeline: scan → sample → cache → split → DataLoaders."""
    print(f"\nScanning dataset: {cfg['data_dir']}")
    df = prepare_data(cfg["data_dir"], cfg["num_samples"], cfg["seed"])
    print(f"  Stratified pool  : {len(df):,} images")
    print(df["label"].value_counts().sort_index()
          .rename(index={i: CLASS_NAMES[i] for i in range(5)}).to_string())

    cache_dir = Path(cfg["cache_dir"])
    print(f"\nPre-caching → {cache_dir}")
    df = precompute_cache(df, cache_dir, image_size=cfg["image_size"])

    df_tr, df_tmp = train_test_split(
        df, test_size=0.20, stratify=df["label"], random_state=cfg["seed"])
    df_val, df_te = train_test_split(
        df_tmp, test_size=0.50, stratify=df_tmp["label"],
        random_state=cfg["seed"])

    tr_ds  = LazyDRDataset(df_tr,  get_train_transforms(cfg["image_size"]),
                           is_train=True)   # RandomErasing active
    val_ds = LazyDRDataset(df_val, get_val_transforms(cfg["image_size"]))
    te_ds  = LazyDRDataset(df_te,  get_val_transforms(cfg["image_size"]))

    kw = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
              pin_memory=cfg["pin_memory"], persistent_workers=True,
              prefetch_factor=2)
    tr_loader  = DataLoader(tr_ds,  shuffle=True,  **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    te_loader  = DataLoader(te_ds,  shuffle=False, **kw)

    print(f"\n  Train: {len(tr_ds):,}  |  Val: {len(val_ds):,}  "
          f"|  Test: {len(te_ds):,}")
    return tr_loader, val_loader, te_loader


# ============================================================================
#                         STEP 5 — TRAIN / VALIDATE
# ============================================================================

class OrdinalSmoothMSELoss(nn.Module):
    """
    Label-smoothed MSE for ordinal regression (Experiment 6.2).

    Each integer grade target g ∈ {0,1,2,3,4} is softened toward the
    centre of the grade range (2.0, the midpoint of [0,4]):

        smoothed_target = (1 - smoothing) * g  +  smoothing * 2.0

    Effect on the loss surface
    --------------------------
    * With smoothing=0 the model can perfectly memorise training targets
      (loss → 0) by mapping every training image to exactly its integer
      grade — the root cause of the train/val gap.
    * With smoothing=0.1 no training target is an exact integer, so the
      model cannot achieve zero training loss even with perfect recall.
      This structurally bounds the minimum training loss above zero,
      pulling it closer to the irreducible validation loss and narrowing
      the divergence between the two curves.
    * The pull is toward the scale centre (2.0), not toward zero, so the
      gradient direction still encodes ordinal severity distance — extreme
      overconfidence on grade 0 or grade 4 is penalised extra.
    * MSE retains quadratic weighting, so a 2-grade error is still 4×
      costlier than a 1-grade error — consistent with QWK structure.

    Parameters
    ----------
    smoothing : float  — fraction of the target replaced by the midpoint
                         value. 0 = pure MSE. 0.1 = recommended.
    midpoint  : float  — the anchor value targets are softened toward.
                         Defaults to 2.0 (centre of [0,4]).
    """

    def __init__(self, smoothing: float = 0.1, midpoint: float = 2.0):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
        self.smoothing = smoothing
        self.midpoint  = midpoint

    def forward(self, predictions: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        smooth_targets = (
            (1.0 - self.smoothing) * targets
            + self.smoothing * self.midpoint
        )
        return F.mse_loss(predictions, smooth_targets)

    def extra_repr(self) -> str:
        return f"smoothing={self.smoothing}, midpoint={self.midpoint}"


def grade_from_scalar(raw: torch.Tensor) -> torch.Tensor:
    """Convert continuous scalar → integer grade in [0, 4]."""
    return raw.detach().round().clamp(0, 4).long()


def train_one_epoch(model, loader, criterion, optimizer, scaler,
                    device, epoch, use_amp):
    model.train()
    running_loss = 0.0
    all_true, all_pred = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        targets = labels.float().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            scalars = model(images)                 # [B] continuous
            loss    = criterion(scalars, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = grade_from_scalar(scalars)
        all_pred.extend(preds.cpu().tolist())
        all_true.extend(targets.long().cpu().tolist())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    n = len(all_true)
    acc = accuracy_score(all_true, all_pred)
    qwk = cohen_kappa_score(all_true, all_pred, weights="quadratic")
    return running_loss / n, acc, qwk


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch, desc, use_amp):
    model.eval()
    running_loss = 0.0
    all_true, all_pred, all_raw = [], [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [{desc}]", leave=False)
    for images, labels in pbar:
        images  = images.to(device, non_blocking=True)
        targets = labels.float().to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            scalars = model(images)
            loss    = criterion(scalars, targets)

        running_loss += loss.item() * images.size(0)
        preds = grade_from_scalar(scalars)
        all_pred.extend(preds.cpu().tolist())
        all_true.extend(targets.long().cpu().tolist())
        all_raw.extend(scalars.cpu().tolist())

    n    = len(all_true)
    acc  = accuracy_score(all_true, all_pred)
    qwk  = cohen_kappa_score(all_true, all_pred, weights="quadratic")
    return running_loss / n, acc, qwk, np.array(all_true), np.array(all_pred), np.array(all_raw)


# ============================================================================
#                         STEP 6 — PLOTTING
# ============================================================================

def _specificity_per_class(y_true: np.ndarray, y_pred: np.ndarray,
                            n_classes: int = 5) -> np.ndarray:
    """
    Per-class one-vs-rest Specificity  =  TN / (TN + FP).
    For each class c:  treat c as positive, all others as negative.
    """
    specs = []
    for c in range(n_classes):
        yt_bin = (y_true == c).astype(int)
        yp_bin = (y_pred == c).astype(int)
        tn = int(np.sum((yt_bin == 0) & (yp_bin == 0)))
        fp = int(np.sum((yt_bin == 0) & (yp_bin == 1)))
        specs.append(tn / max(tn + fp, 1))
    return np.array(specs)


def save_metrics_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    te_loss: float,
    te_qwk: float,
    path: Path,
) -> pd.DataFrame:
    """
    Overall metrics summary CSV  (metrics_summary.csv).

    Columns: Metric | Value
    Metrics : Accuracy, Macro-F1, Micro-F1, Macro-Precision,
              Macro-Sensitivity (Recall), Macro-Specificity, QWK, Test-Loss
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
    )
    n_classes = 5
    specs     = _specificity_per_class(y_true, y_pred, n_classes)

    rows = [
        {"Metric": "Accuracy",                "Value": round(accuracy_score(y_true, y_pred), 4)},
        {"Metric": "F1-Score (Macro)",         "Value": round(f1_score(y_true, y_pred, average="macro",    zero_division=0), 4)},
        {"Metric": "F1-Score (Micro)",         "Value": round(f1_score(y_true, y_pred, average="micro",    zero_division=0), 4)},
        {"Metric": "Precision (Macro)",        "Value": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4)},
        {"Metric": "Sensitivity (Macro)",      "Value": round(recall_score(y_true, y_pred, average="macro",  zero_division=0), 4)},
        {"Metric": "Specificity (Macro)",      "Value": round(float(specs.mean()), 4)},
        {"Metric": "QWK",                      "Value": round(te_qwk, 4)},
        {"Metric": "Test Loss (OrdSmoothMSE)", "Value": round(te_loss, 4)},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return df


def save_per_class_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    path: Path,
) -> pd.DataFrame:
    """
    Per-class clinical performance breakdown  (per_class_performance.csv).

    Columns: Class | F1-Score | Sensitivity | Specificity
    One row per DR grade (0 = No DR … 4 = Proliferative).
    """
    from sklearn.metrics import f1_score, recall_score
    n_classes = len(class_names)
    f1s  = f1_score(y_true, y_pred,     average=None, zero_division=0,
                    labels=list(range(n_classes)))
    sens = recall_score(y_true, y_pred, average=None, zero_division=0,
                        labels=list(range(n_classes)))
    specs = _specificity_per_class(y_true, y_pred, n_classes)

    rows = [
        {
            "Class":       class_names[c],
            "F1-Score":    round(float(f1s[c]),   4),
            "Sensitivity": round(float(sens[c]),  4),
            "Specificity": round(float(specs[c]), 4),
        }
        for c in range(n_classes)
    ]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return df


def save_binary_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    csv_path: Path,
    png_path: Path,
):
    """
    Referable-DR binary confusion matrix  (Classes 0–1 = Non-Referable,
    Classes 2–4 = Referable DR).

    Saves both a CSV of raw counts and a row-normalised heatmap PNG.
    Clinical rationale: detecting Referable DR (≥ Moderate) is the
    primary screening decision — sensitivity and specificity in this
    binary framing directly reflect the model's clinical utility.
    """
    y_true_bin = (y_true >= 2).astype(int)
    y_pred_bin = (y_pred >= 2).astype(int)
    bin_labels = ["Non-Referable (0–1)", "Referable DR (2–4)"]

    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])

    # ── CSV (raw counts) ──────────────────────────────────────────────────────
    df_cm = pd.DataFrame(cm, index=bin_labels, columns=bin_labels)
    df_cm.index.name = "True \\ Predicted"
    df_cm.to_csv(csv_path)
    print(f"  Saved: {csv_path}")

    # Derived binary clinical metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    ppv         = tp / max(tp + fp, 1)
    npv         = tn / max(tn + fn, 1)
    f1_ref      = 2 * tp / max(2 * tp + fp + fn, 1)
    print(f"    Referable DR  →  Sensitivity: {sensitivity:.4f}  "
          f"Specificity: {specificity:.4f}  PPV: {ppv:.4f}  "
          f"NPV: {npv:.4f}  F1: {f1_ref:.4f}")

    # ── PNG (row-normalised heatmap) ──────────────────────────────────────────
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1) * 100
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
    sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=bin_labels, yticklabels=bin_labels,
                cbar_kws={"label": "Percentage (%)"},
                linewidths=0.5, linecolor="white", square=True,
                vmin=0, vmax=100, annot_kws={"size": 12}, ax=ax)
    ax.set_title("Binary Confusion Matrix — Referable DR (≥ Moderate)",
                 fontsize=12, pad=10, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_xticklabels(bin_labels, rotation=15, ha="right")
    ax.set_yticklabels(bin_labels, rotation=0)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {png_path}")


def save_multiclass_confusion_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    path: Path,
):
    """Save the 5×5 confusion matrix as a CSV of raw counts."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    df = pd.DataFrame(cm, index=class_names, columns=class_names)
    df.index.name = "True \\ Predicted"
    df.to_csv(path)
    print(f"  Saved: {path}")


def save_confusion_matrix(y_true, y_pred, path):
    cm      = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1) * 100

    fig, ax = plt.subplots(figsize=(8, 6.5), facecolor="white")
    sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={"label": "Percentage (%)"},
                linewidths=0.5, linecolor="white", square=True,
                vmin=0, vmax=100, annot_kws={"size": 11}, ax=ax)
    ax.set_title("Confusion Matrix — Row Normalised (%)",
                 fontsize=14, pad=12, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES, rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


def save_roc_curves(y_true, y_raw_scalar, path, n_classes=5):
    """
    One-vs-Rest ROC curves using the continuous regression scalar as the
    ranking score. For class c the score is -(|scalar - c|) — the closer
    the prediction is to grade c, the higher the score.
    """
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    palette = plt.cm.tab10(np.linspace(0, 0.9, n_classes))
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance")

    for c in range(n_classes):
        scores = -np.abs(y_raw_scalar - c)   # higher → closer to grade c
        fpr, tpr, _ = roc_curve(y_bin[:, c], scores)
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=palette[c],
                label=f"{CLASS_NAMES[c]}  (AUC = {roc_auc:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("Multi-Class ROC Curve (One-vs-Rest)", fontsize=13,
                 fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=10, frameon=False)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


def save_training_history(history, path):
    if not history: return

    epochs   = [h["epoch"]      for h in history]
    tr_loss  = [h["train_loss"] for h in history]
    va_loss  = [h["val_loss"]   for h in history]
    tr_acc   = [h["train_acc"]  for h in history]
    va_acc   = [h["val_acc"]    for h in history]
    tr_qwk   = [h["train_qwk"] for h in history]
    va_qwk   = [h["val_qwk"]   for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor="white")

    for ax, (tr, va, ylabel, title) in zip(axes, [
        (tr_loss, va_loss, "OrdinalSmoothMSE Loss", "Loss"),
        (tr_acc,  va_acc,  "Accuracy",              "Accuracy"),
        (tr_qwk,  va_qwk,  "QWK",                   "Quadratic Weighted Kappa"),
    ]):
        ax.plot(epochs, tr, color="red",  lw=1.8, label="Train")
        ax.plot(epochs, va, color="blue", lw=1.8, label="Validation")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel,  fontsize=11)
        ax.set_title(title,    fontsize=12, fontweight="bold", pad=8)
        ax.legend(frameon=False, fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_facecolor("white")
        for sp in ax.spines.values():
            sp.set_linewidth(0.6); sp.set_color("black")

    fig.suptitle("Experiment 6 — Training History  (LHT-ViT Ordinal Regression)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
#                         STEP 7 — MAIN TRAINING LOOP
# ============================================================================

def main():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    torch.backends.cudnn.benchmark = True

    print(f"  Device       : {device}")
    if device.type == "cuda":
        for i in range(torch.cuda.device_count()):
            print(f"    GPU[{i}]  : {torch.cuda.get_device_name(i)}")
    print()

    out_dir = Path(CFG["output_dir"])

    # ── Data ─────────────────────────────────────────────────────────────────
    tr_loader, val_loader, te_loader = build_dataloaders(CFG)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding HybridLHT_ViT_Ordinal ...")
    model = HybridLHT_ViT_Ordinal(
        attn_dim       = CFG["attn_dim"],
        mfb_k          = CFG["mfb_k"],
        mfb_out        = CFG["mfb_out"],
        head_dropout   = CFG["head_dropout"],
        drop_path_rate = CFG["drop_path_rate"],
        attn_drop_rate = CFG["attn_drop_rate"],
        pretrained     = True,
    ).to(device)

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters   : {n_total/1e6:.2f} M total  "
          f"({n_trainable/1e6:.2f} M trainable)")
    print(f"  Head output  : 1 continuous scalar  (ordinal regression)")

    # ── Loss, optimizer, scheduler, AMP ──────────────────────────────────────
    # Exp 6.2 loss: OrdinalSmoothMSELoss — MSE on label-smoothed targets.
    criterion = OrdinalSmoothMSELoss(
        smoothing = CFG["label_smoothing"],   # 0.10
        midpoint  = 2.0,                      # centre of [0,4] grade range
    )
    print(f"  Loss         : {criterion}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = CFG["lr"],
        weight_decay = CFG["weight_decay"],
    )

    # Exp 6.2 scheduler: LinearWarmup(5 ep) → CosineAnnealingLR
    # SequentialLR chains two schedulers at a specified milestone.
    # LinearLR: start_factor=1e-6/lr ≈ near-zero, grows linearly to 1× over
    # warmup_epochs steps. After that, CosineAnnealingLR takes over and
    # decays the LR from the full base value down to eta_min.
    warmup_epochs      = CFG["warmup_epochs"]
    cosine_epochs      = CFG["epochs"] - warmup_epochs
    warmup_start_factor = 1e-6 / max(CFG["lr"], 1e-12)  # ≈ near-zero ratio

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor = warmup_start_factor,
        end_factor   = 1.0,
        total_iters  = warmup_epochs,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = cosine_epochs,
        eta_min = CFG["eta_min"],
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers = [warmup_sched, cosine_sched],
        milestones = [warmup_epochs],
    )

    scaler    = GradScaler(enabled=use_amp)
    save_path = out_dir / "best_model_LHTViT_Exp6_2.pth"

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_qwk      = -1.0
    best_val_loss     = float("inf")
    epochs_no_improve = 0
    history: list[dict] = []
    train_losses, val_losses = [], []

    print("\n" + "=" * 72)
    print("  TRAINING  (early stopping: patience = "
          f"{CFG['patience']} on val_QWK)")
    print("=" * 72)
    print(f"{'Ep':>4}  {'LR':>9}  "
          f"{'TrLoss':>8}  {'TrAcc':>7}  {'TrQWK':>7}  "
          f"{'VaLoss':>8}  {'VaAcc':>7}  {'VaQWK':>7}  "
          f"{'Best':>5}  {'NoImp':>5}")
    print("-" * 90)

    stopped_early = False
    last_epoch    = 0

    for epoch in range(1, CFG["epochs"] + 1):
        last_epoch = epoch
        gc.collect()
        torch.cuda.empty_cache()

        current_lr = optimizer.param_groups[0]["lr"]

        tr_loss, tr_acc, tr_qwk = train_one_epoch(
            model, tr_loader, criterion, optimizer, scaler,
            device, epoch, use_amp)
        va_loss, va_acc, va_qwk, _, _, _ = evaluate(
            model, val_loader, criterion, device, epoch, "Val", use_amp)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses  .append(va_loss)

        saved = ""
        if va_qwk > best_val_qwk:
            best_val_qwk      = va_qwk
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_qwk":  va_qwk, "val_loss": va_loss,
                "val_acc":  va_acc,
                "architecture": "HybridLHT_ViT_Ordinal [Exp 6.2]",
                "cfg": CFG,
            }, save_path)
            saved = "*"
        else:
            epochs_no_improve += 1

        if va_loss < best_val_loss:
            best_val_loss = va_loss

        print(f"{epoch:>4}  {current_lr:>9.2e}  "
              f"{tr_loss:>8.4f}  {tr_acc:>7.4f}  {tr_qwk:>7.4f}  "
              f"{va_loss:>8.4f}  {va_acc:>7.4f}  {va_qwk:>7.4f}  "
              f"{saved:>5}  {epochs_no_improve:>5}")

        history.append(dict(epoch=epoch, lr=current_lr,
                            train_loss=tr_loss, train_acc=tr_acc,
                            train_qwk=tr_qwk,
                            val_loss=va_loss,   val_acc=va_acc,
                            val_qwk=va_qwk,
                            saved=bool(saved)))

        if epochs_no_improve >= CFG["patience"]:
            stopped_early = True
            print(f"\n>>> Early stopping at epoch {epoch}: val_QWK did not "
                  f"improve for {CFG['patience']} consecutive epochs.")
            break

    print("-" * 90)
    print(f"Training done. Epochs run: {last_epoch}/{CFG['epochs']}"
          f"{'  (stopped early)' if stopped_early else ''}")
    print(f"  Best val_QWK  : {best_val_qwk:.4f}")
    print(f"  Best val_loss : {best_val_loss:.4f}")
    print(f"  Checkpoint    : {save_path}")

    # Save training log JSON
    log_path = out_dir / "train_log_exp6_2.json"
    log_path.write_text(json.dumps({
        "timestamp":     datetime.now().isoformat(timespec="seconds"),
        "architecture":  "HybridLHT_ViT_Ordinal",
        "best_val_qwk":  best_val_qwk,
        "best_val_loss": best_val_loss,
        "stopped_early": stopped_early,
        "epochs_run":    last_epoch,
        "cfg": CFG,
        "history": history,
    }, indent=2), encoding="utf-8")
    print(f"  Log           : {log_path}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  TEST EVALUATION  (loading best checkpoint)")
    print("=" * 72)

    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    te_loss, te_acc, te_qwk, y_true, y_pred, y_raw = evaluate(
        model, te_loader, criterion, device, last_epoch, "Test", use_amp)

    from sklearn.metrics import (
        f1_score as _f1, precision_score as _prec, recall_score as _rec,
    )
    te_f1_macro  = _f1(y_true,   y_pred, average="macro",  zero_division=0)
    te_f1_micro  = _f1(y_true,   y_pred, average="micro",  zero_division=0)
    te_prec      = _prec(y_true, y_pred, average="macro",  zero_division=0)
    te_sens      = _rec(y_true,  y_pred, average="macro",  zero_division=0)
    te_specs     = _specificity_per_class(y_true, y_pred, n_classes=5)
    te_spec_mac  = float(te_specs.mean())

    print("\n  PRIMARY — 5-class DR Grading (Bounded Ordinal Regression)")
    print("=" * 72)
    print(f"  Accuracy              : {te_acc:.4f}")
    print(f"  QWK                   : {te_qwk:.4f}")
    print(f"  F1-Score  (Macro)     : {te_f1_macro:.4f}")
    print(f"  F1-Score  (Micro)     : {te_f1_micro:.4f}")
    print(f"  Precision (Macro)     : {te_prec:.4f}")
    print(f"  Sensitivity (Macro)   : {te_sens:.4f}")
    print(f"  Specificity (Macro)   : {te_spec_mac:.4f}")
    print(f"  Test Loss (OrdSmoMSE) : {te_loss:.4f}")
    print()
    print(classification_report(y_true, y_pred,
          target_names=CLASS_NAMES, digits=4, zero_division=0))

    # ── Export 1: metrics_summary.csv ────────────────────────────────────────
    print("\nSaving thesis evaluation assets ...")
    summary_path = out_dir / "metrics_summary.csv"
    save_metrics_summary(y_true, y_pred, te_loss, te_qwk, summary_path)

    # ── Export 2: per_class_performance.csv ──────────────────────────────────
    per_class_path = out_dir / "per_class_performance.csv"
    save_per_class_performance(y_true, y_pred, CLASS_NAMES, per_class_path)

    # ── Export 3: confusion_matrix_multiclass.csv + PNG ──────────────────────
    cm_csv_path = out_dir / "confusion_matrix_multiclass.csv"
    cm_png_path = out_dir / "confusion_matrix_multiclass.png"
    save_multiclass_confusion_csv(y_true, y_pred, CLASS_NAMES, cm_csv_path)
    save_confusion_matrix(y_true, y_pred, cm_png_path)

    # ── Export 4: confusion_matrix_referable_binary.csv + PNG ────────────────
    bin_csv_path = out_dir / "confusion_matrix_referable_binary.csv"
    bin_png_path = out_dir / "confusion_matrix_referable_binary.png"
    print("\n  Binary Referable-DR evaluation  (Non-Referable: 0–1 | Referable: 2–4)")
    save_binary_confusion(y_true, y_pred, bin_csv_path, bin_png_path)

    # ── Export 5: ROC curves + training history ───────────────────────────────
    roc_path     = out_dir / "roc_curves_exp6_2.png"
    history_path = out_dir / "training_history_exp6_2.png"
    save_roc_curves(y_true, y_raw, roc_path)
    save_training_history(history, history_path)

    # ── Final summary banner ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  FINAL SUMMARY  (Exp 6.2 — Definitive Benchmark Build)")
    print("=" * 72)
    print(f"  QWK                   : {te_qwk:.4f}")
    print(f"  F1-Score  (Macro)     : {te_f1_macro:.4f}")
    print(f"  F1-Score  (Micro)     : {te_f1_micro:.4f}")
    print(f"  Accuracy  (5-class)   : {te_acc:.4f}")
    print(f"  Sensitivity (Macro)   : {te_sens:.4f}")
    print(f"  Specificity (Macro)   : {te_spec_mac:.4f}")
    print("=" * 72)

    print("\nAll Drive outputs:")
    print(f"  • {save_path}           (best model weights)")
    print(f"  • {log_path}               (training history JSON)")
    print(f"  • {summary_path}              (overall metrics CSV)")
    print(f"  • {per_class_path}       (per-class F1/Sens/Spec CSV)")
    print(f"  • {cm_csv_path}   (5×5 confusion matrix CSV)")
    print(f"  • {cm_png_path}   (5×5 confusion matrix PNG)")
    print(f"  • {bin_csv_path}   (binary referable CM CSV)")
    print(f"  • {bin_png_path}   (binary referable CM PNG)")
    print(f"  • {roc_path}              (ROC curves PNG)")
    print(f"  • {history_path}     (training history PNG)")
    print("\nDone.")


# ── Entry point ───────────────────────────────────────────────────────────────
main()
