import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Cache redirection (must be set BEFORE torch / timm import)
# ---------------------------------------------------------------------------
IS_KAGGLE: bool = os.path.isdir("/kaggle/working")

if IS_KAGGLE:
    os.environ.setdefault("TORCH_HOME", "/kaggle/working/cache")
    os.environ.setdefault("HF_HOME",    "/kaggle/working/cache")
else:
    os.environ.setdefault("TORCH_HOME", r"E:\Master Thesis\DR_Thesis_Project\cache")
    os.environ.setdefault("HF_HOME",    r"E:\Master Thesis\DR_Thesis_Project\cache")

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)

# Allow running either from the src/ directory or the project root
sys.path.insert(0, str(Path(__file__).parent))

from dataset import get_dataloaders
from models  import LHT_CNN


# ---------------------------------------------------------------------------
# Configuration (Kaggle-aware)
# ---------------------------------------------------------------------------

# Hardcoded dataset path — per the Kaggle mount point for the EyePACS +
# APTOS + Messidor combined DR dataset. This path is fixed regardless of
# any CLI arguments or environment variables.
DATA_DIR = "/kaggle/input/datasets/ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy/augmented_resized_V2"

if IS_KAGGLE:
    OUTPUT_DIR = Path("/kaggle/working")
else:
    OUTPUT_DIR = Path(r"E:\Master Thesis\DR_Thesis_Project\weights")

WEIGHTS_PATH = OUTPUT_DIR / "best_model_mfb.pth"
CM_PATH      = OUTPUT_DIR / "confusion_matrix.png"
BAR_PATH     = OUTPUT_DIR / "per_class_metrics.png"

CLASS_NAMES = ["Healthy", "Mild", "Moderate", "Severe", "Proliferative"]
NUM_CLASSES = 5
BATCH_SIZE  = 32


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model:  torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) as NumPy arrays over the full loader."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds  = logits.argmax(dim=1).cpu().numpy()

        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().tolist())

    return np.array(y_true), np.array(y_pred)


# ---------------------------------------------------------------------------
# Visualization 1 — Normalized Confusion Matrix (percentages)
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    class_names: list[str],
    save_path:   Path,
) -> None:
    """
    Row-normalized confusion matrix: each row sums to 100 %.
    Diagonal cells = per-class recall (%).
    """
    cm      = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None) * 100.0

    plt.figure(figsize=(8, 6.5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Percentage (%)"},
        linewidths=0.6,
        linecolor="white",
        square=True,
        vmin=0,
        vmax=100,
        annot_kws={"fontsize": 11},
    )
    plt.title("Normalized Confusion Matrix (Row %)", fontsize=14, pad=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label",      fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Per-class metrics
# ---------------------------------------------------------------------------

def compute_per_class_metrics(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    num_classes: int,
) -> dict[str, list[float]]:
    """
    One-vs-Rest per-class Accuracy, Precision, Recall, F1-Score.

    * Accuracy  = (TP + TN) / N         (treats the class as positive, the rest as negative)
    * Precision = TP / (TP + FP)
    * Recall    = TP / (TP + FN)
    * F1        = 2·P·R / (P + R)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {"Accuracy": [], "Precision": [], "Recall": [], "F1-Score": []}

    for c in range(num_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        tn = int(np.sum((y_pred != c) & (y_true != c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))

        n    = tp + tn + fp + fn
        acc  = (tp + tn) / n                     if n              > 0 else 0.0
        prec = tp / (tp + fp)                    if (tp + fp)      > 0 else 0.0
        rec  = tp / (tp + fn)                    if (tp + fn)      > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec)     if (prec + rec)   > 0 else 0.0

        metrics["Accuracy"].append(acc)
        metrics["Precision"].append(prec)
        metrics["Recall"].append(rec)
        metrics["F1-Score"].append(f1)

    return metrics


# ---------------------------------------------------------------------------
# Visualization 2 — Grouped bar chart
# ---------------------------------------------------------------------------

def plot_per_class_bar(
    metrics:     dict[str, list[float]],
    class_names: list[str],
    save_path:   Path,
) -> None:
    """Grouped bar chart: 4 metrics × 5 classes."""
    metric_names = list(metrics.keys())
    n_metrics    = len(metric_names)
    n_classes    = len(class_names)

    x     = np.arange(n_classes)
    width = 0.8 / n_metrics

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # blue / orange / green / red

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, mname in enumerate(metric_names):
        values = metrics[mname]
        offset = (i - (n_metrics - 1) / 2) * width
        bars   = ax.bar(
            x + offset, values, width,
            label=mname,
            color=colors[i],
            edgecolor="white",
            linewidth=0.8,
        )
        for b, v in zip(bars, values):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.012,
                f"{v:.2f}",
                ha="center", va="bottom",
                fontsize=8, color="#222222",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_xlabel("Class",  fontsize=12)
    ax.set_ylabel("Score",  fontsize=12)
    ax.set_ylim(0.0, 1.10)
    ax.set_title(
        "Per-Class Metrics — Accuracy, Precision, Recall, F1-Score",
        fontsize=14, pad=12, fontweight="bold",
    )
    ax.legend(title="Metric", loc="lower right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Ensure Unicode output works on Windows consoles too
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device : {device}")
    if device.type == "cuda":
        print(f"  GPU       : {torch.cuda.get_device_name(0)}")
    print(f"Environment  : {'Kaggle' if IS_KAGGLE else 'Local'}")
    print(f"Data dir     : {DATA_DIR}")
    print(f"Output dir   : {OUTPUT_DIR}")
    print(f"Weights file : {WEIGHTS_PATH}")

    # ── Data: we only need the test loader ───────────────────────────────────
    print("\nLoading dataset ...")
    _, _, test_loader = get_dataloaders(
        root_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
    )
    print(f"Test set size: {len(test_loader.dataset)}")

    # ── Model ────────────────────────────────────────────────────────────────
    print(f"\nLoading weights from: {WEIGHTS_PATH}")
    ckpt = torch.load(WEIGHTS_PATH, map_location=device)
    use_mfb = bool(ckpt.get("use_mfb", True))  # default True — this is the MFB checkpoint

    model = LHT_CNN(
        num_classes=NUM_CLASSES,
        pretrained=False,
        use_mfb=use_mfb,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Checkpoint   : epoch {ckpt.get('epoch', '?')} | "
          f"val_qwk = {ckpt.get('val_qwk', float('nan')):.4f} | "
          f"use_mfb = {use_mfb}")

    # ── Inference ────────────────────────────────────────────────────────────
    y_true, y_pred = run_inference(model, test_loader, device)

    # ── Overall metrics ──────────────────────────────────────────────────────
    acc    = accuracy_score(y_true, y_pred)
    prec_m = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_m  = recall_score   (y_true, y_pred, average="macro", zero_division=0)
    f1_m   = f1_score       (y_true, y_pred, average="macro", zero_division=0)
    qwk    = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    print("\n" + "=" * 55)
    print("  OVERALL TEST METRICS")
    print("=" * 55)
    print(f"  Accuracy               : {acc:.4f}")
    print(f"  Precision (macro)      : {prec_m:.4f}")
    print(f"  Recall    (macro)      : {rec_m:.4f}")
    print(f"  F1-Score  (macro)      : {f1_m:.4f}")
    print(f"  Quadratic Weighted Kappa: {qwk:.4f}")

    print("\n" + "=" * 55)
    print("  CLASSIFICATION REPORT")
    print("=" * 55)
    print(classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    ))

    # ── Visualization 1: Confusion Matrix ────────────────────────────────────
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, CM_PATH)

    # ── Per-class metrics + Visualization 2: Bar Chart ───────────────────────
    per_class = compute_per_class_metrics(y_true, y_pred, NUM_CLASSES)

    df_pc = pd.DataFrame(per_class, index=CLASS_NAMES).round(4)
    print("\n" + "=" * 55)
    print("  PER-CLASS METRICS  (one-vs-rest)")
    print("=" * 55)
    print(df_pc.to_string())

    plot_per_class_bar(per_class, CLASS_NAMES, BAR_PATH)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
