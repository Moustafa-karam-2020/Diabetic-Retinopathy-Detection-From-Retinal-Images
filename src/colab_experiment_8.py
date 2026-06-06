"""
============================================================================
EXPERIMENT 9 — LHT-ViT  Ordinal-Aware Training  (Corrected Hyperparameters)
   EfficientNet-B0 + ViT-Small/16  ·  Bidirectional Co-Attention  ·  MFB
   OrdinalSmoothMSE Loss  ·  AdamW(lr=3e-4, wd=1e-3)
   5-ep Linear Warmup → CosineAnnealingLR(T=35)
   Dropout: drop_path=0.20, head=0.30, mfb=0.30
   40-Epoch Max  ·  Early Stopping on VAL QWK (patience=8)
   Google Colab Pro  (NVIDIA L4)
============================================================================

WHY EXPERIMENT 8 UNDERFITTED (accuracy stuck at 55%, QWK plateau at 0.71)
---------------------------------------------------------------------------
The loss curves were perfect — both train and val decreasing in parallel.
That problem is solved. The accuracy failure had three separate causes:

CAUSE 1 — LR too small for ordinal loss gradients.
  OrdinalSmoothMSE computes MSE between CDF vectors in [0, 1].
  Gradients are in the range ~0.01–0.05 — about 100× smaller than
  CrossEntropyLoss gradients (~1.0). The lr=5e-5 used in Exp 8 was
  inherited from the CE experiments and is far too small for ordinal MSE.
  FIX → lr = 3e-4 (6× higher), which gives the same effective step size.

CAUSE 2 — Total dropout too aggressive for small gradients.
  drop_path=0.40 + head=0.35 + mfb=0.40 combined stochastically zero out
  a large fraction of the network every forward pass. With CE's strong
  gradients this is fine. With ordinal's weak gradients, the training
  signal disappears into the stochastic noise.
  FIX → drop_path=0.20, head=0.30, mfb=0.30. Still regularises enough to
  keep the parallel curves, but doesn't drown the ordinal signal.

CAUSE 3 — OneCycleLR's aggressive peak destabilises ordinal training.
  OneCycleLR ramps to its peak LR rapidly then crashes. At the LR peak
  (~epoch 13) the ordinal CDF targets shift faster than the model can
  track, causing QWK to oscillate (visible in the Exp 8 table: 0.7072 →
  0.7094 → 0.7149 → 0.7062). Ordinal training needs a stable,
  monotonically-decreasing LR after warmup.
  FIX → 5-epoch linear warmup then CosineAnnealingLR(T_max=35, eta_min=1e-6).
  Smooth ramp-up + smooth decay = stable CDF learning throughout.

ARCHITECTURE: UNCHANGED from Exp 7/8.
LOSS FUNCTION: OrdinalSmoothMSE — UNCHANGED from Exp 8 (perfect curves).
EARLY STOPPING: now on VAL QWK (not val loss) — QWK is the thesis metric.

TARGET: QWK ≥ 0.88, Accuracy ≥ 0.84, F1-macro ≥ 0.80
============================================================================
"""

import gc, json, math, os, random, sys, warnings
from collections import Counter
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
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import timm

from sklearn.metrics import (
    accuracy_score, classification_report, cohen_kappa_score,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_curve, auc,
)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import label_binarize
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

# =============================================================================
# CONFIGURATION
# =============================================================================

CFG = dict(
    data_dir   = "/content/dataset/augmented_resized_V2/train",
    output_dir = "/content/drive/MyDrive/DR_Experiment_9_Outputs",
    cache_dir  = "/content/cache/preproc",   # reuse Exp 7/8 cache

    num_samples  = 80_000,
    image_size   = 224,
    batch_size   = 64,
    num_workers  = 4,
    pin_memory   = True,
    seed         = SEED,

    epochs       = 40,

    # FIX 1: lr raised from 5e-5 → 3e-4 for ordinal gradient scale
    lr            = 3e-4,
    weight_decay  = 1e-3,
    eta_min       = 1e-6,

    # FIX 3: warmup + cosine instead of OneCycleLR
    warmup_epochs = 5,    # linear warmup from lr/10 → lr over 5 epochs
    # CosineAnnealingLR T_max = epochs - warmup_epochs = 35

    # Early stopping on VAL QWK (thesis metric), patience=8
    patience     = 8,

    # FIX 2: dropout reduced from 0.40/0.35/0.40 → 0.20/0.30/0.30
    drop_path_rate = 0.20,   # was 0.40
    attn_drop_rate = 0.10,   # was 0.15
    head_dropout   = 0.30,   # was 0.35
    mfb_dropout    = 0.30,   # was 0.40

    # Architecture (unchanged)
    attn_dim    = 512,
    mfb_k       = 5,
    mfb_out     = 1024,
    num_classes = 5,

    # Ordinal loss smoothing
    ordinal_smooth = 0.05,

    # Gradient clipping
    grad_clip = 1.0,
)

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
TRAIN_COLOR = "#1a3a6b"   # dark navy blue
VAL_COLOR   = "#c0392b"   # deep red

out_dir = Path(CFG["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
Path(CFG["cache_dir"]).mkdir(parents=True, exist_ok=True)

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available()

print("=" * 72)
print("  EXPERIMENT 9 — LHT-ViT  Ordinal-Aware  (Corrected LR + Dropout + Schedule)")
print("=" * 72)
print(f"  Device        : {device}  |  AMP: {use_amp}")
print(f"  Loss          : OrdinalSmoothMSE (smooth={CFG['ordinal_smooth']})")
print(f"  LR            : {CFG['lr']} (raised from 5e-5)  wd={CFG['weight_decay']}")
print(f"  Schedule      : {CFG['warmup_epochs']}-ep linear warmup → "
      f"CosineAnnealingLR(T={CFG['epochs']-CFG['warmup_epochs']}, eta_min={CFG['eta_min']})")
print(f"  Dropout       : drop_path={CFG['drop_path_rate']}  "
      f"head={CFG['head_dropout']}  mfb={CFG['mfb_dropout']}"
      f"  (all reduced from Exp 8)")
print(f"  Early stop    : VAL QWK (patience={CFG['patience']})")
print(f"  Samples       : {CFG['num_samples']:,}  batch={CFG['batch_size']}")
print()


# =============================================================================
# ORDINAL LOSS + DECODER
# =============================================================================

class OrdinalSmoothMSE(nn.Module):
    """
    Ordinal CDF regression loss.

    Converts both predictions and labels into cumulative distribution
    functions over the K=5 grades, then minimises MSE between them.

    Why this produces parallel loss curves:
    Both train and val sets share the same ordinal difficulty distribution.
    A model that learns the CDF boundary correctly shows both losses
    decreasing at the same rate.

    The loss values are in [0, ~0.1] — much smaller than CrossEntropyLoss
    values in [0.4, 1.5]. This is WHY lr must be ~6× higher than for CE.
    """

    def __init__(self, num_classes=5, smooth=0.05):
        super().__init__()
        self.K      = num_classes
        self.smooth = smooth

    def forward(self, logits, labels):
        probs    = F.softmax(logits, dim=1)
        cdf_pred = torch.cumsum(probs, dim=1)[:, :-1]           # [B, K-1]
        k_idx    = torch.arange(self.K-1, device=logits.device).unsqueeze(0)
        cdf_true = (labels.unsqueeze(1) > k_idx).float()        # [B, K-1]
        cdf_true = cdf_true * (1.0 - self.smooth) + self.smooth * 0.5
        return F.mse_loss(cdf_pred, cdf_true)


def ordinal_predict(logits):
    """Decode CDF predictions to integer grades."""
    probs = F.softmax(logits, dim=1)
    cdf   = torch.cumsum(probs, dim=1)[:, :-1]
    return (cdf > 0.5).sum(dim=1)


# =============================================================================
# MODEL  (UNCHANGED from Exp 7/8)
# =============================================================================

class CoAttention(nn.Module):
    def __init__(self, cnn_dim, vit_dim, attn_dim=512):
        super().__init__()
        self.proj_cnn = nn.Linear(cnn_dim, attn_dim)
        self.proj_vit = nn.Linear(vit_dim, attn_dim)
        self.norm_cnn = nn.LayerNorm(attn_dim)
        self.norm_vit = nn.LayerNorm(attn_dim)

    def forward(self, cnn_tokens, vit_tokens):
        cnn_p = self.proj_cnn(cnn_tokens)
        vit_p = self.proj_vit(vit_tokens)
        scale = cnn_p.size(-1) ** 0.5
        attn_c2v = torch.softmax(torch.bmm(cnn_p, vit_p.transpose(1,2)) / scale, dim=-1)
        cnn_att  = self.norm_cnn(cnn_p + torch.bmm(attn_c2v, vit_p))
        attn_v2c = torch.softmax(torch.bmm(vit_p, cnn_p.transpose(1,2)) / scale, dim=-1)
        vit_att  = self.norm_vit(vit_p + torch.bmm(attn_v2c, cnn_p))
        return cnn_att.mean(dim=1), vit_att.mean(dim=1)


class MFBPooling(nn.Module):
    def __init__(self, dim_q=512, dim_v=512, mfb_k=5, mfb_out=1024, mfb_dropout=0.30):
        super().__init__()
        self.K = mfb_k; self.mfb_out = mfb_out
        self.proj_q = nn.Linear(dim_q, mfb_out * mfb_k)
        self.proj_v = nn.Linear(dim_v, mfb_out * mfb_k)
        self.dropout = nn.Dropout(p=mfb_dropout)

    def forward(self, q, v):
        B   = q.size(0)
        z_q = self.proj_q(q).view(B, self.mfb_out, self.K)
        z_v = self.proj_v(v).view(B, self.mfb_out, self.K)
        z   = (z_q * z_v).sum(dim=-1)
        z   = self.dropout(z)
        # NaN-safe power normalisation
        z_f = z.float()
        z_f = torch.sign(z_f) * torch.sqrt(torch.abs(z_f) + 1e-8)
        z   = z_f.to(z.dtype)
        return z / z.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)


class HybridLHT_ViT(nn.Module):
    def __init__(self, attn_dim=512, mfb_k=5, mfb_out=1024, num_classes=5,
                 mfb_dropout=0.30, head_dropout=0.30,
                 drop_path_rate=0.20, attn_drop_rate=0.10, pretrained=True):
        super().__init__()
        self.cnn = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=0)
        self.vit = timm.create_model(
            "vit_small_patch16_224", pretrained=pretrained, num_classes=0,
            drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop_rate)
        cnn_dim = self.cnn.num_features   # 1280
        vit_dim = self.vit.embed_dim      # 384
        self.coattn = CoAttention(cnn_dim, vit_dim, attn_dim)
        self.mfb    = MFBPooling(attn_dim, attn_dim, mfb_k, mfb_out, mfb_dropout)
        self.head   = nn.Sequential(
            nn.LayerNorm(mfb_out),
            nn.Linear(mfb_out, 512),
            nn.GELU(),
            nn.Dropout(p=head_dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        cnn_tok = self.cnn.forward_features(x).flatten(2).transpose(1, 2)
        vit_tok = self.vit.forward_features(x)[:, 1:, :]
        cv, vv  = self.coattn(cnn_tok, vit_tok)
        return self.head(self.mfb(cv, vv))


# =============================================================================
# LR SCHEDULE  — linear warmup + cosine annealing
# =============================================================================

def build_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs,
                                   base_lr, eta_min):
    """
    Phase 1 (epochs 0..warmup_epochs-1): linear ramp from base_lr/10 to base_lr.
    Phase 2 (epochs warmup_epochs..total_epochs-1): cosine decay to eta_min.

    Returns a LambdaLR that steps once per EPOCH (not per batch).
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup: factor goes from 0.1 → 1.0
            return 0.1 + 0.9 * epoch / max(warmup_epochs - 1, 1)
        # Cosine decay
        t = epoch - warmup_epochs
        T = max(total_epochs - warmup_epochs, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * t / T))
        ratio  = eta_min / max(base_lr, 1e-12)
        return ratio + (1.0 - ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# =============================================================================
# PREPROCESSING / DATASET  (unchanged — reuses Exp 7/8 .npy cache)
# =============================================================================

def ben_graham_preprocess(image, sigma_x=10):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        image = image[y:y+h, x:x+w]
    return cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), sigma_x), -4, 128)


def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    out = np.empty_like(image)
    for c in range(3): out[:,:,c] = clahe.apply(image[:,:,c])
    return out


def _preprocess_single(img_path, target_size=224):
    img = cv2.imread(img_path)
    if img is None: return np.zeros((target_size,target_size,3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size,target_size), interpolation=cv2.INTER_AREA)
    img = ben_graham_preprocess(img)
    img = apply_clahe(img)
    img = cv2.resize(img, (target_size,target_size), interpolation=cv2.INTER_AREA)
    return img.astype(np.uint8)


def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.4, border_mode=cv2.BORDER_CONSTANT),
        A.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02, p=0.3),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])


class LazyDRDataset(Dataset):
    def __init__(self, records, cache_dir, transforms, image_size=224):
        self.records   = records
        self.cache_dir = Path(cache_dir)
        self.transforms = transforms
        self.image_size = image_size

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec  = self.records[idx]
        cp   = self.cache_dir / f"{Path(rec['path']).stem}.npy"
        img  = np.load(str(cp)) if cp.exists() else _preprocess_single(rec["path"], self.image_size)
        if not cp.exists(): np.save(str(cp), img)
        return self.transforms(image=img)["image"], int(rec["label"])


def build_dataloaders(cfg):
    data_dir = Path(cfg["data_dir"]); records = []
    for label in range(5):
        cd = data_dir / str(label)
        if not cd.exists(): continue
        for ext in ("*.jpeg","*.jpg","*.png"):
            for p in cd.glob(ext): records.append({"path":str(p),"label":label})
    if not records:
        raise RuntimeError(f"No images found under {data_dir}")

    labels_all = [r["label"] for r in records]
    if len(records) > cfg["num_samples"]:
        sss = StratifiedShuffleSplit(1, test_size=1.0-cfg["num_samples"]/len(records),
                                      random_state=cfg["seed"])
        keep, _ = next(sss.split(records, labels_all))
        records = [records[i] for i in keep]; labels_all = [labels_all[i] for i in keep]

    idx = list(range(len(records)))
    ix_tr, ix_tmp = train_test_split(idx, test_size=0.20, stratify=labels_all, random_state=cfg["seed"])
    lbl_tmp = [labels_all[i] for i in ix_tmp]
    ix_va, ix_te = train_test_split(ix_tmp, test_size=0.50, stratify=lbl_tmp, random_state=cfg["seed"])

    tr_recs = [records[i] for i in ix_tr]
    va_recs = [records[i] for i in ix_va]
    te_recs = [records[i] for i in ix_te]

    cache_dir = Path(cfg["cache_dir"])
    missing = [r for r in records if not (cache_dir/f"{Path(r['path']).stem}.npy").exists()]
    if missing:
        print(f"  Pre-caching {len(missing):,} images ...")
        for rec in tqdm(missing, desc="Cache", leave=False):
            img = _preprocess_single(rec["path"], cfg["image_size"])
            np.save(str(cache_dir/f"{Path(rec['path']).stem}.npy"), img)
    else:
        print(f"  Cache complete ({len(records):,} entries).")

    tr_ds = LazyDRDataset(tr_recs, cfg["cache_dir"], get_train_transforms(), cfg["image_size"])
    va_ds = LazyDRDataset(va_recs, cfg["cache_dir"], get_val_transforms(),   cfg["image_size"])
    te_ds = LazyDRDataset(te_recs, cfg["cache_dir"], get_val_transforms(),   cfg["image_size"])

    # WeightedRandomSampler: grade-2 (Moderate) boost ×1.5
    tr_labels = [r["label"] for r in tr_recs]
    cnt = Counter(tr_labels); N = len(tr_labels); C = len(cnt)
    bw  = {c: N/(C*v) for c,v in cnt.items()}; bw[2] *= 1.5
    sw  = torch.tensor([bw[l] for l in tr_labels], dtype=torch.float)
    sampler = WeightedRandomSampler(sw, N, replacement=True)

    kw = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
              pin_memory=cfg["pin_memory"],
              persistent_workers=cfg["num_workers"]>0)
    tr_loader = DataLoader(tr_ds, sampler=sampler, **kw)
    va_loader = DataLoader(va_ds, shuffle=False, **kw)
    te_loader = DataLoader(te_ds, shuffle=False, **kw)
    print(f"  Train: {len(tr_ds):,}  |  Val: {len(va_ds):,}  |  Test: {len(te_ds):,}")
    return tr_loader, va_loader, te_loader


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer,
                    scaler, device, epoch, use_amp, grad_clip):
    model.train()
    total_loss = 0.0; all_true = []; all_pred = []
    pbar = tqdm(loader, desc=f"Ep{epoch:02d}[Train]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, labels)
        if not torch.isfinite(loss):
            print(f"\n  [NaN] epoch={epoch}, batch skipped")
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer); scaler.update()
        total_loss += loss.item() * images.size(0)
        preds = ordinal_predict(logits.detach())
        all_pred.extend(preds.cpu().tolist()); all_true.extend(labels.cpu().tolist())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    n = len(all_true)
    acc = accuracy_score(all_true, all_pred)
    qwk = cohen_kappa_score(all_true, all_pred, weights="quadratic")
    return total_loss / max(n,1), acc, qwk


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch, desc, use_amp):
    model.eval()
    total_loss = 0.0; all_true = []; all_pred = []; all_probs = []
    for images, labels in tqdm(loader, desc=f"Ep{epoch:02d}[{desc}]", leave=False):
        images  = images.to(device, non_blocking=True)
        targets = labels.long().to(device, non_blocking=True)
        with autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, targets)
        total_loss += loss.item() * images.size(0)
        preds = ordinal_predict(logits)
        probs = F.softmax(logits, dim=1)
        all_pred.extend(preds.cpu().tolist()); all_true.extend(targets.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
    n = len(all_true)
    acc = accuracy_score(all_true, all_pred)
    qwk = cohen_kappa_score(all_true, all_pred, weights="quadratic")
    return total_loss/max(n,1), acc, qwk, np.array(all_true), np.array(all_pred), np.array(all_probs)


# =============================================================================
# EARLY STOPPING  (on VAL QWK this time)
# =============================================================================

class EarlyStopping:
    def __init__(self, patience=8, mode="max"):
        """mode='max' for QWK (higher=better), 'min' for loss."""
        self.patience = patience
        self.mode     = mode
        self.best     = -float("inf") if mode == "max" else float("inf")
        self.counter  = 0
        self.stop     = False

    def step(self, value):
        improved = (value > self.best) if self.mode == "max" else (value < self.best)
        if improved:
            self.best = value; self.counter = 0; return True
        self.counter += 1
        if self.counter >= self.patience: self.stop = True
        return False


# =============================================================================
# PLOTTING  —  BLUE/RED THESIS THEME
# =============================================================================

def plot_training_history(history, path):
    if not history: return
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="white")
    fig.suptitle("Experiment 9 — Training History  (LHT-ViT Ordinal-Aware, corrected)",
                 fontsize=14, fontweight="bold", y=1.02)

    panels = [
        ("train_loss", "val_loss", "OrdinalSmoothMSE Loss", "Loss"),
        ("train_acc",  "val_acc",  "Accuracy",              "Accuracy"),
        ("train_qwk",  "val_qwk",  "Quadratic Weighted Kappa", "QWK"),
    ]
    for ax, (tk, vk, title, ylabel) in zip(axes, panels):
        ax.set_facecolor("white")
        ax.plot(ep, [h[tk] for h in history], color=TRAIN_COLOR, lw=2.0, label="Train")
        ax.plot(ep, [h[vk] for h in history], color=VAL_COLOR,   lw=2.0, label="Validation")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Epoch", fontsize=11); ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(frameon=False, fontsize=10); ax.grid(alpha=0.20)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6); spine.set_color("black")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white"); plt.close()
    print(f"  Saved: {path}")


def plot_loss_only(history, path):
    """Single-panel Fig.3 style: blue=train, red=val, white background."""
    if not history: return
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(ep, [h["train_loss"] for h in history], color=TRAIN_COLOR, lw=2.2, label="Training loss")
    ax.plot(ep, [h["val_loss"]   for h in history], color=VAL_COLOR,   lw=2.2, label="Validation loss")
    ax.set_xlabel("Epochs", fontsize=12); ax.set_ylabel("OrdinalSmoothMSE Loss", fontsize=12)
    ax.set_title("Training and Validation Loss", fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=11); ax.grid(alpha=0.20)
    for spine in ax.spines.values(): spine.set_linewidth(0.6); spine.set_color("black")
    fig.text(0.5, -0.04, "Fig. 3.  Training and validation loss curves.", ha="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white"); plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(y_true, y_pred, path, class_names=None):
    if class_names is None: class_names = CLASS_NAMES
    cm   = confusion_matrix(y_true, y_pred)
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(8, 6.5), facecolor="white")
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=11); ax.set_ylabel("True Label", fontsize=11)
    ax.set_title("Normalised Confusion Matrix — Experiment 9", fontsize=13, fontweight="bold")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{norm[i,j]:.2f}\n({cm[i,j]})", ha="center", va="center", fontsize=8,
                    color="white" if norm[i,j] > 0.5 else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white"); plt.close()
    print(f"  Saved: {path}")


def plot_binary_confusion(y_true, y_pred, csv_path, png_path):
    y_tb = (y_true>=2).astype(int); y_pb = (y_pred>=2).astype(int)
    cm   = confusion_matrix(y_tb, y_pb, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    pd.DataFrame({"Metric":["TN","FP","FN","TP","Sensitivity","Specificity"],
                  "Value":[int(tn),int(fp),int(fn),int(tp),
                           round(tp/max(tp+fn,1),4), round(tn/max(tn+fp,1),4)]}).to_csv(csv_path, index=False)
    norm   = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    labels = ["Non-Referable (0–1)", "Referable (2–4)"]
    fig, ax = plt.subplots(figsize=(5, 4.5), facecolor="white")
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1); plt.colorbar(im, ax=ax)
    ax.set_xticks([0,1]); ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([0,1]); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11); ax.set_ylabel("True Label", fontsize=11)
    ax.set_title("Binary Referable-DR CM — Experiment 9", fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{norm[i,j]:.2f}\n({cm[i,j]})", ha="center", va="center", fontsize=11,
                    color="white" if norm[i,j] > 0.5 else "black")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white"); plt.close()
    print(f"  Saved: {png_path}")


def plot_roc_curves(y_true, y_probs, path, n_classes=5):
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    palette = [TRAIN_COLOR, "#2874a6", "#229954", "#d35400", VAL_COLOR]
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")
    ax.plot([0,1],[0,1], "k--", lw=0.8, label="Chance")
    for c in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:,c], y_probs[:,c])
        ax.plot(fpr, tpr, lw=2, color=palette[c],
                label=f"{CLASS_NAMES[c]}  (AUC={auc(fpr,tpr):.3f})")
    ax.set_xlabel("FPR", fontsize=12); ax.set_ylabel("TPR", fontsize=12)
    ax.set_title("ROC Curves (One-vs-Rest) — Exp 9", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10, frameon=False)
    for sp in ax.spines.values(): sp.set_linewidth(0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white"); plt.close()
    print(f"  Saved: {path}")


def _specificity_per_class(y_true, y_pred, n=5):
    specs = []
    for c in range(n):
        yt_b = (y_true==c).astype(int); yp_b = (y_pred==c).astype(int)
        tn = int(np.sum((yt_b==0)&(yp_b==0))); fp = int(np.sum((yt_b==0)&(yp_b==1)))
        specs.append(tn/max(tn+fp,1))
    return np.array(specs)


def save_metrics_csv(y_true, y_pred, te_loss, te_qwk, path):
    specs = _specificity_per_class(y_true, y_pred)
    rows = [
        {"Metric":"Accuracy",               "Value":round(accuracy_score(y_true,y_pred),4)},
        {"Metric":"F1-Score (Macro)",        "Value":round(f1_score(y_true,y_pred,average="macro",zero_division=0),4)},
        {"Metric":"F1-Score (Micro)",        "Value":round(f1_score(y_true,y_pred,average="micro",zero_division=0),4)},
        {"Metric":"Precision (Macro)",       "Value":round(precision_score(y_true,y_pred,average="macro",zero_division=0),4)},
        {"Metric":"Sensitivity (Macro)",     "Value":round(recall_score(y_true,y_pred,average="macro",zero_division=0),4)},
        {"Metric":"Specificity (Macro)",     "Value":round(float(specs.mean()),4)},
        {"Metric":"QWK",                     "Value":round(te_qwk,4)},
        {"Metric":"Test Loss (OrdinalMSE)",  "Value":round(te_loss,4)},
    ]
    pd.DataFrame(rows).to_csv(path, index=False); print(f"  Saved: {path}")


def save_per_class_csv(y_true, y_pred, path):
    n = len(CLASS_NAMES)
    f1s  = f1_score(y_true,y_pred,average=None,zero_division=0,labels=list(range(n)))
    sens = recall_score(y_true,y_pred,average=None,zero_division=0,labels=list(range(n)))
    spec = _specificity_per_class(y_true,y_pred,n)
    rows = [{"Class":CLASS_NAMES[c],"F1":round(float(f1s[c]),4),
             "Sensitivity":round(float(sens[c]),4),"Specificity":round(float(spec[c]),4)} for c in range(n)]
    pd.DataFrame(rows).to_csv(path, index=False); print(f"  Saved: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    tr_loader, va_loader, te_loader = build_dataloaders(CFG)

    print("\nBuilding HybridLHT_ViT (Experiment 9) ...")
    model = HybridLHT_ViT(
        attn_dim=CFG["attn_dim"], mfb_k=CFG["mfb_k"], mfb_out=CFG["mfb_out"],
        num_classes=CFG["num_classes"], mfb_dropout=CFG["mfb_dropout"],
        head_dropout=CFG["head_dropout"], drop_path_rate=CFG["drop_path_rate"],
        attn_drop_rate=CFG["attn_drop_rate"], pretrained=True,
    ).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    criterion = OrdinalSmoothMSE(num_classes=CFG["num_classes"], smooth=CFG["ordinal_smooth"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                                   weight_decay=CFG["weight_decay"], eps=1e-6)

    # Warmup + cosine — steps per EPOCH
    scheduler = build_warmup_cosine_scheduler(
        optimizer, CFG["warmup_epochs"], CFG["epochs"], CFG["lr"], CFG["eta_min"],
    )
    scaler    = GradScaler("cuda", enabled=use_amp)
    save_path = out_dir / "best_model_exp9.pth"

    # Early stop on VAL QWK (mode='max')
    stopper        = EarlyStopping(patience=CFG["patience"], mode="max")
    best_val_qwk   = -1.0
    best_qwk_epoch = 0
    history        = []
    stopped_early  = False
    last_epoch     = 0

    print("\n" + "="*72)
    print(f"  TRAINING — early-stop on VAL QWK (patience={CFG['patience']})")
    print(f"             checkpoint on best VAL QWK")
    print("="*72)
    hdr = (f"{'Ep':>4}  {'LR':>9}  "
           f"{'TrLoss':>8}  {'TrAcc':>7}  {'TrQWK':>7}  "
           f"{'VaLoss':>8}  {'VaAcc':>7}  {'VaQWK':>7}  "
           f"{'Save':>5}  {'Pat':>4}")
    print(hdr); print("-"*len(hdr))

    for epoch in range(1, CFG["epochs"]+1):
        last_epoch = epoch
        gc.collect(); torch.cuda.empty_cache()
        current_lr = optimizer.param_groups[0]["lr"]

        tr_loss, tr_acc, tr_qwk = train_one_epoch(
            model, tr_loader, criterion, optimizer, scaler,
            device, epoch, use_amp, CFG["grad_clip"],
        )
        va_loss, va_acc, va_qwk, _, _, _ = evaluate(
            model, va_loader, criterion, device, epoch, "Val", use_amp,
        )
        scheduler.step()   # one step per epoch

        stopper.step(va_qwk)
        saved = ""
        if va_qwk > best_val_qwk:
            best_val_qwk = va_qwk; best_qwk_epoch = epoch
            torch.save({"epoch":epoch, "model_state_dict":model.state_dict(),
                        "val_qwk":va_qwk, "val_loss":va_loss, "cfg":CFG}, save_path)
            saved = "✓"

        history.append({"epoch":epoch,"lr":current_lr,
                        "train_loss":tr_loss,"train_acc":tr_acc,"train_qwk":tr_qwk,
                        "val_loss":va_loss,  "val_acc":va_acc,  "val_qwk":va_qwk})
        print(f"{epoch:>4}  {current_lr:>9.2e}  "
              f"{tr_loss:>8.4f}  {tr_acc:>7.4f}  {tr_qwk:>7.4f}  "
              f"{va_loss:>8.4f}  {va_acc:>7.4f}  {va_qwk:>7.4f}  "
              f"{saved:>5}  {stopper.counter:>4}")

        if stopper.stop:
            print(f"\n  >>> EARLY STOPPING at epoch {epoch} "
                  f"(QWK no improvement for {CFG['patience']} epochs)")
            stopped_early = True; break

    print(f"\n  Best val QWK: {best_val_qwk:.4f}  (epoch {best_qwk_epoch})")
    (out_dir/"train_log_exp9.json").write_text(
        json.dumps({"best_val_qwk":best_val_qwk,"epochs_run":last_epoch,
                    "stopped_early":stopped_early,"cfg":CFG,"history":history}, indent=2),
        encoding="utf-8",
    )

    # TEST EVALUATION
    print("\n" + "="*72); print("  TEST EVALUATION"); print("="*72)
    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded epoch={ckpt['epoch']}  val_QWK={ckpt['val_qwk']:.4f}")

    te_loss, te_acc, te_qwk, y_true, y_pred, y_probs = evaluate(
        model, te_loader, criterion, device, last_epoch, "Test", use_amp,
    )
    te_f1_mac = f1_score(y_true,y_pred,average="macro",zero_division=0)
    te_f1_mic = f1_score(y_true,y_pred,average="micro",zero_division=0)
    te_prec   = precision_score(y_true,y_pred,average="macro",zero_division=0)
    te_sens   = recall_score(y_true,y_pred,average="macro",zero_division=0)
    te_specs  = _specificity_per_class(y_true,y_pred)

    print("\n  PRIMARY METRICS — 5-Class DR Grading"); print("="*72)
    print(f"  Accuracy              : {te_acc:.4f}")
    print(f"  QWK                   : {te_qwk:.4f}")
    print(f"  F1-Score  (Macro)     : {te_f1_mac:.4f}")
    print(f"  F1-Score  (Micro)     : {te_f1_mic:.4f}")
    print(f"  Precision (Macro)     : {te_prec:.4f}")
    print(f"  Sensitivity (Macro)   : {te_sens:.4f}")
    print(f"  Specificity (Macro)   : {float(te_specs.mean()):.4f}")
    print(f"  Test Loss (OrdinalMSE): {te_loss:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0))

    y_tb = (y_true>=2).astype(int); y_pb = (y_pred>=2).astype(int)
    print("  BINARY (Referable DR = grade ≥ 2)")
    print(f"  Sensitivity : {recall_score(y_tb,y_pb,pos_label=1,zero_division=0):.4f}")
    print(f"  Specificity : {recall_score(y_tb,y_pb,pos_label=0,zero_division=0):.4f}")
    print(f"  F1-Score    : {f1_score(y_tb,y_pb,pos_label=1,zero_division=0):.4f}")

    print("\nSaving thesis assets ...")
    save_metrics_csv(y_true, y_pred, te_loss, te_qwk, out_dir/"metrics_summary_exp9.csv")
    save_per_class_csv(y_true, y_pred, out_dir/"per_class_performance_exp9.csv")
    pd.DataFrame(confusion_matrix(y_true,y_pred,labels=list(range(5))),
                 index=CLASS_NAMES,columns=CLASS_NAMES).to_csv(
                 out_dir/"confusion_matrix_multiclass_exp9.csv")
    plot_confusion_matrix(y_true, y_pred, out_dir/"confusion_matrix_multiclass_exp9.png")
    plot_binary_confusion(y_true, y_pred,
                          out_dir/"confusion_matrix_binary_exp9.csv",
                          out_dir/"confusion_matrix_binary_exp9.png")
    plot_roc_curves(y_true, y_probs, out_dir/"roc_curves_exp9.png")
    plot_training_history(history, out_dir/"training_history_exp9.png")
    plot_loss_only(history, out_dir/"loss_curves_exp9.png")

    print("\n" + "="*72)
    print("  FINAL SUMMARY — Experiment 9 vs Experiment 7")
    print("="*72)
    print(f"  QWK (5-class)         : {te_qwk:.4f}   (Exp 7 was 0.8637)")
    print(f"  Accuracy (5-class)    : {te_acc:.4f}   (Exp 7 was 0.8223)")
    print(f"  F1-Score  (Macro)     : {te_f1_mac:.4f}   (Exp 7 was 0.7849)")
    print(f"  Sensitivity (Macro)   : {te_sens:.4f}   (Exp 7 was 0.7772)")
    print(f"  Specificity (Macro)   : {float(te_specs.mean()):.4f}   (Exp 7 was 0.9477)")
    print("="*72)
    print(f"\nAll outputs: {CFG['output_dir']}")
    print("Done.")


main()