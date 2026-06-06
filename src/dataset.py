import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Ben Graham preprocessing
# ---------------------------------------------------------------------------

def ben_graham_preprocess(image: np.ndarray, sigmaX: int = 10) -> np.ndarray:
    """
    Apply Ben Graham's fundus preprocessing entirely in memory (no disk I/O).

    Steps
    -----
    1. Crop to the circular fundus boundary by finding the largest inscribed
       circle inside the non-black region, then pad to a square with the
       circle's diameter as the side length.
    2. Resize to a fixed intermediate size so the Gaussian sigma is scale-
       independent.
    3. Blend with a large-radius Gaussian blur to remove slow illumination
       gradients and enhance local lesion contrast:

           out = clip( 4·img  −  4·GaussianBlur(img)  +  128 )

       This is exactly the formula from Ben Graham's 2015 Kaggle write-up.

    Args:
        image  : uint8 RGB image loaded by cv2 (after BGR→RGB conversion).
        sigmaX : Gaussian blur standard deviation (default 10 for 380-px images).

    Returns:
        Preprocessed uint8 RGB image of the same spatial size as the input.
        All operations happen on in-memory numpy arrays — nothing is written
        to disk.
    """
    # ── Step 1: crop to circular fundus boundary ─────────────────────────────
    # Convert to grayscale to find the illuminated fundus region
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Threshold out the dark background (fundus cameras produce near-black corners)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Find contours of the fundus region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Use the largest contour to fit the enclosing circle
        largest = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(largest)
        cx, cy, radius = int(cx), int(cy), int(radius)

        # Clamp crop bounds to valid image dimensions
        h, w = image.shape[:2]
        x1 = max(cx - radius, 0)
        y1 = max(cy - radius, 0)
        x2 = min(cx + radius, w)
        y2 = min(cy + radius, h)
        image = image[y1:y2, x1:x2]

    # ── Step 2: resize (ensures sigmaX is scale-independent) ─────────────────
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)

    # ── Step 3: Ben Graham local contrast enhancement ────────────────────────
    # Blend = 4·image − 4·GaussianBlur(image) + 128
    # ksize=(0,0) tells OpenCV to derive the kernel size from sigmaX
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigmaX)
    image   = cv2.addWeighted(image, 4, blurred, -4, 128)

    return image


# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------

def get_train_transforms(image_size: int = 384) -> A.Compose:
    return A.Compose([
        # CLAHE applied first — before resize — to boost lesion contrast on the
        # full-resolution preprocessed image.
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 384) -> A.Compose:
    return A.Compose([
        # CLAHE kept for val/test so inference conditions match training.
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DRDataset(Dataset):
    """
    Diabetic Retinopathy classification dataset.

    Reads images directly into memory via OpenCV — no temporary files
    or disk caching. Labels are derived from parent folder names (0-4).
    """

    def __init__(self, dataframe: pd.DataFrame, transform: A.Compose | None = None):
        """
        Args:
            dataframe: DataFrame with columns ['image_path', 'label'].
            transform:  Albumentations Compose pipeline.
        """
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image_path: str = row["image_path"]
        label: int = int(row["label"])

        # ── 1. Load: BGR → RGB, fully in RAM — no temp files written ─────────
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ── 2. Ben Graham preprocessing (crop, contrast enhancement) ─────────
        image = ben_graham_preprocess(image)

        # ── 3. Albumentations pipeline (CLAHE → resize → augment → normalise) ─
        if self.transform:
            image = self.transform(image=image)["image"]  # → torch.Tensor

        return image, label


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
RANDOM_STATE = 42


# Kaggle-aware default root: the `diabetic-retinopathy-resized` dataset on
# Kaggle is mounted read-only under /kaggle/input/. Locally we fall back to
# the E:-drive archive used during development.
if os.path.isdir("/kaggle/input/diabetic-retinopathy-resized"):
    DEFAULT_ROOT_DIR = "/kaggle/input/diabetic-retinopathy-resized"
else:
    DEFAULT_ROOT_DIR = r"E:\Master Thesis\Dataset preprocessing\archive (1)"


def prepare_data(
    root_dir: str = DEFAULT_ROOT_DIR,
    num_samples: int | None = None,
) -> pd.DataFrame:
    """
    Scans *root_dir* recursively, collects every image file, and assigns
    a label from the parent folder name (expected: '0', '1', '2', '3', '4').

    Args
    ----
    root_dir    : path to a directory whose children are per-class folders
                  named '0', '1', '2', '3', '4'.
    num_samples : if an int, return a *stratified* sample of that many rows;
                  if None, return every discovered image (recommended on
                  Kaggle with the full 143K-image dataset).

    Returns a pandas DataFrame with columns ['image_path', 'label'].
    Nothing is written to disk.
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    records: list[dict] = []
    for img_path in root.rglob("*"):
        if img_path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        label_str = img_path.parent.name
        if label_str not in {"0", "1", "2", "3", "4"}:
            continue  # skip images not directly inside a class folder
        records.append({"image_path": str(img_path), "label": int(label_str)})

    if not records:
        raise ValueError(f"No valid images found under {root_dir}")

    df = pd.DataFrame(records)

    # No sub-sampling requested → use every discovered image
    if num_samples is None:
        return df.reset_index(drop=True)

    # Stratified sample — preserves class distribution
    if len(df) < num_samples:
        raise ValueError(
            f"Found only {len(df)} images, but {num_samples} are required."
        )

    _, df_sampled = train_test_split(
        df,
        test_size=num_samples,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )

    return df_sampled.reset_index(drop=True)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    root_dir: str = DEFAULT_ROOT_DIR,
    batch_size: int = 16,
    image_size: int = 384,
    num_samples: int | None = None,
    num_workers: int | None = None,
    pin_memory:  bool | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds train / validation / test DataLoaders.

    Split:  80 % train  |  10 % validation  |  10 % test  (stratified).
    All splitting is done in memory — no files written anywhere.

    Args
    ----
    root_dir    : dataset root containing per-class subfolders '0'..'4'.
    batch_size  : samples per mini-batch.
    image_size  : spatial size after the albumentations Resize step (default 384).
    num_samples : if None (default), use ALL discovered images (e.g. the
                  full 143K Kaggle dataset). Pass an int for a stratified
                  sub-sample (useful for quick sanity runs on a laptop).
    num_workers : worker processes per loader. If None, auto-selects:
                    - 0 on Windows (multiprocessing with cv2/albumentations
                      is unstable and freezes the machine)
                    - 4 on Linux / Kaggle (safe, big I/O speed-up)

    Returns:
        (train_loader, val_loader, test_loader)
    """
    df = prepare_data(root_dir, num_samples=num_samples)

    # ── First split: 80 % train  vs  20 % temp ──────────────────────────────
    df_train, df_temp = train_test_split(
        df,
        test_size=0.20,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )

    # ── Second split: 50 % of temp → val, 50 % → test  (each = 10 % total) ──
    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.50,
        stratify=df_temp["label"],
        random_state=RANDOM_STATE,
    )

    train_dataset = DRDataset(df_train, transform=get_train_transforms(image_size))
    val_dataset   = DRDataset(df_val,   transform=get_val_transforms(image_size))
    test_dataset  = DRDataset(df_test,  transform=get_val_transforms(image_size))

    # Auto-select worker count if not explicitly supplied
    if num_workers is None:
        num_workers = 0 if os.name == "nt" else 4

    # Auto-select pin_memory if not explicitly supplied
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    # persistent_workers only valid when num_workers > 0
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **loader_kwargs)

    print(
        f"Dataset split -- "
        f"Train: {len(train_dataset)}  |  "
        f"Val: {len(val_dataset)}  |  "
        f"Test: {len(test_dataset)}  "
        f"(workers={num_workers}, batch={batch_size})"
    )
    return train_loader, val_loader, test_loader
