import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Environment-aware paths (resolved BEFORE importing torch / timm so the
# cache redirection takes effect on the very first import).
# ---------------------------------------------------------------------------

IS_KAGGLE: bool = os.path.isdir("/kaggle/working")

if IS_KAGGLE:
    DEFAULT_DATA_DIR   = "/kaggle/input/diabetic-retinopathy-resized"
    DEFAULT_OUTPUT_DIR = "/kaggle/working"
    _CACHE_DIR         = "/kaggle/working/cache"
else:
    DEFAULT_DATA_DIR   = r"E:\Master Thesis\Dataset preprocessing\archive (1)"
    DEFAULT_OUTPUT_DIR = r"E:\Master Thesis\DR_Thesis_Project\weights"
    _CACHE_DIR         = r"E:\Master Thesis\DR_Thesis_Project\cache"

# Must be set before torch / timm import
os.environ.setdefault("TORCH_HOME", _CACHE_DIR)
os.environ.setdefault("HF_HOME",    _CACHE_DIR)

import torch
import torch.nn as nn
from tqdm import tqdm

# Allow running directly from the src/ directory or the project root
sys.path.insert(0, str(Path(__file__).parent))

from dataset import get_dataloaders
from models  import LHT_CNN
from utils   import evaluate, get_cost_sensitive_weights


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train LHT_CNN on Diabetic Retinopathy (with optional MFB ablation)."
    )

    # Paths
    p.add_argument("--data_dir",   type=str, default=DEFAULT_DATA_DIR,
                   help="Root dir containing per-class subfolders '0'..'4'.")
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                   help="Directory to save weights, logs, and images.")

    # Hyperparameters (Kaggle-friendly defaults)
    p.add_argument("--batch_size", type=int, default=32,
                   help="Mini-batch size (default 32 for Kaggle's 16 GB GPUs).")
    p.add_argument("--epochs",     type=int, default=30,
                   help="Number of training epochs.")
    p.add_argument("--lr",         type=float, default=1e-4,
                   help="AdamW learning rate.")
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="AdamW weight decay (L2 regularization, default 1e-4).")
    p.add_argument("--image_size", type=int, default=384,
                   help="Image size after albumentations Resize (ViT needs multiples of 16).")
    p.add_argument("--num_classes", type=int, default=5)

    # Data sub-sampling
    p.add_argument("--num_samples", type=int, default=None,
                   help="Stratified sub-sample size. None = use all images (default).")
    p.add_argument("--num_workers", type=int, default=None,
                   help="DataLoader workers. None = auto (0 on Windows, 4 on Linux/Kaggle).")

    # Ablation switch
    p.add_argument("--use_mfb",   dest="use_mfb", action="store_true",
                   help="Use Multimodal Fusion Block + Co-Attention (default).")
    p.add_argument("--no_mfb",    dest="use_mfb", action="store_false",
                   help="Ablation: disable MFB — use plain projection + concat baseline.")
    p.set_defaults(use_mfb=True)

    # Optional tag for the saved filename (handy for ablation pairs)
    p.add_argument("--tag", type=str, default=None,
                   help="Optional suffix for saved files, e.g. 'mfb' or 'baseline'.")

    # Early stopping
    p.add_argument("--patience", type=int, default=5,
                   help="Early-stopping patience (epochs without val-loss improvement). "
                        "Set to 0 to disable early stopping.")

    # Multi-GPU
    p.add_argument("--data_parallel", dest="data_parallel", action="store_true",
                   help="Wrap model in nn.DataParallel across all visible CUDA devices "
                        "(default: auto-enabled when >1 GPU is available).")
    p.add_argument("--no_data_parallel", dest="data_parallel", action="store_false",
                   help="Force single-GPU training even if multiple GPUs are visible.")
    p.set_defaults(data_parallel=None)  # None → auto-detect in main()

    return p.parse_args()


# ---------------------------------------------------------------------------
# Training / validation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    epoch:     int,
) -> float:
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    epoch:     int,
) -> tuple[float, dict[str, float]]:
    model.eval()
    running_loss = 0.0
    all_preds:  list[int] = []
    all_labels: list[int] = []

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Val]  ", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = running_loss / len(loader.dataset)
    metrics  = evaluate(all_labels, all_preds)
    return avg_loss, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Resolve filename tag (auto if user didn't supply one) ─────────────────
    tag = args.tag if args.tag is not None else ("mfb" if args.use_mfb else "baseline")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # If the user explicitly passed --tag, save with the suffix (useful for ablation
    # pairs); otherwise save as plain `best_model.pth`.
    if args.tag is not None:
        save_path = output_dir / f"best_model_{tag}.pth"
        log_path  = output_dir / f"train_log_{tag}.json"
    else:
        save_path = output_dir / "best_model.pth"
        log_path  = output_dir / "train_log.json"

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0

    # Auto-enable DataParallel if user didn't specify and >1 GPU is visible
    if args.data_parallel is None:
        args.data_parallel = n_gpus > 1

    print(f"Using device: {device}")
    if device.type == "cuda":
        for i in range(n_gpus):
            print(f"  GPU[{i}]: {torch.cuda.get_device_name(i)}")
    print(f"Environment : {'Kaggle' if IS_KAGGLE else 'Local'}")
    print(f"Data dir    : {args.data_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Variant     : {'MFB + Co-Attention' if args.use_mfb else 'Baseline (no MFB)'}  [tag='{tag}']")
    print(f"DataParallel: {args.data_parallel}  (n_gpus visible = {n_gpus})")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\nLoading data ...")
    train_loader, val_loader, _ = get_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
    )

    # Pull training labels directly from the underlying DataFrame
    train_labels = train_loader.dataset.df["label"].tolist()

    # ── Cost-sensitive loss (inverse-frequency weights on TRAIN split only) ──
    class_weights = get_cost_sensitive_weights(train_labels).to(device)
    print(f"\nClass weights: {class_weights.cpu().numpy().round(4)}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding model ...")
    model = LHT_CNN(
        num_classes=args.num_classes,
        pretrained=True,
        use_mfb=args.use_mfb,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params / 1e6:.2f} M")

    # ── Multi-GPU wrapping ────────────────────────────────────────────────────
    # nn.DataParallel replicates the model on every visible CUDA device, splits
    # each mini-batch along dim 0, runs the forward/backward pass on each GPU
    # in parallel, and gathers gradients on GPU 0. On Kaggle's dual-T4 setup,
    # this effectively doubles throughput.
    #
    # Note: when the module is wrapped, `model.module` is the underlying
    # LHT_CNN. We always checkpoint `model.module.state_dict()` so the saved
    # file is loadable into a plain LHT_CNN (no DataParallel required at
    # inference time).
    if args.data_parallel and n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Wrapped model in nn.DataParallel across {n_gpus} GPUs.")

    # Helper: always get at the real LHT_CNN, whether wrapped or not
    def unwrap(m: nn.Module) -> nn.Module:
        return m.module if isinstance(m, nn.DataParallel) else m

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # AdamW (decoupled weight decay) is the modern default — applies the L2
    # penalty as true weight decay rather than mixing it into the gradient,
    # which interacts more cleanly with adaptive learning rates.
    # weight_decay=1e-4 is the standard regularization strength for vision
    # transformer / CNN hybrids and is our primary defence against overfitting.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    print(f"Optimizer    : AdamW(lr={args.lr}, weight_decay={args.weight_decay})")

    # ── Training loop + Early Stopping ────────────────────────────────────────
    # Single source of truth: best_val_loss.
    #   * The same condition (val_loss < best_val_loss) both
    #     1) overwrites best_model.pth  AND
    #     2) resets the early-stopping patience counter.
    #   * Strict inequality — a tie does NOT count as an improvement, so a
    #     stagnating run will eventually trip early stopping.
    best_val_loss:     float = float("inf")
    best_qwk_at_best:  float = -1.0   # tracked for logging only
    epochs_no_improve: int   = 0
    history: list[dict] = []

    es_enabled = args.patience > 0
    print(f"\nStarting training for up to {args.epochs} epochs "
          f"(early stopping: {'on, patience=' + str(args.patience) if es_enabled else 'off'})\n")
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>8}  {'Acc':>6}  {'QWK':>6}  {'Saved':>5}  {'NoImp':>5}")
    print("-" * 62)

    stopped_early = False
    last_epoch    = 0

    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, epoch
        )

        acc = val_metrics["accuracy"]
        qwk = val_metrics["qwk"]

        # ── Checkpoint + early-stop counter — both driven by val_loss ─────────
        saved = ""
        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            best_qwk_at_best  = qwk
            epochs_no_improve = 0

            # Save ONLY on strict val_loss improvement.
            # We persist the unwrapped state_dict so a plain LHT_CNN can load
            # the checkpoint without needing to know about DataParallel.
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     unwrap(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":             val_loss,
                    "val_acc":              acc,
                    "val_qwk":              qwk,
                    "use_mfb":              args.use_mfb,
                    "args":                 vars(args),
                },
                save_path,
            )
            saved = "*"
        else:
            epochs_no_improve += 1

        print(
            f"{epoch:>5}  {train_loss:>10.4f}  {val_loss:>8.4f}  "
            f"{acc:>6.4f}  {qwk:>6.4f}  {saved:>5}  {epochs_no_improve:>5}"
        )

        history.append({
            "epoch":             epoch,
            "train_loss":        train_loss,
            "val_loss":          val_loss,
            "val_acc":           acc,
            "val_qwk":           qwk,
            "saved":             bool(saved),
            "epochs_no_improve": epochs_no_improve,
        })

        # ── Trigger early stop ────────────────────────────────────────────────
        if es_enabled and epochs_no_improve >= args.patience:
            stopped_early = True
            print(
                f"\nEarly stopping triggered at epoch {epoch}: "
                f"val_loss has not improved for {args.patience} consecutive epochs "
                f"(best val_loss = {best_val_loss:.4f})."
            )
            break

    # ── Persist a JSON training log next to the weights ──────────────────────
    log_payload = {
        "timestamp":         datetime.now().isoformat(timespec="seconds"),
        "tag":               tag,
        "use_mfb":           args.use_mfb,
        "best_val_loss":     best_val_loss,
        "best_qwk_at_best":  best_qwk_at_best,
        "stopped_early":     stopped_early,
        "epochs_run":        last_epoch,
        "args":              vars(args),
        "history":           history,
    }
    log_path.write_text(json.dumps(log_payload, indent=2), encoding="utf-8")

    print(f"\nTraining complete.  Epochs run: {last_epoch}/{args.epochs}"
          f"{'  (early stopped)' if stopped_early else ''}")
    print(f"Best Val Loss      : {best_val_loss:.4f}")
    print(f"Val QWK at best    : {best_qwk_at_best:.4f}")
    print(f"Best weights       : {save_path}")
    print(f"Training log       : {log_path}")


if __name__ == "__main__":
    main()
