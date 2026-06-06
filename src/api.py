import os
import sys
import io
from pathlib import Path
from contextlib import asynccontextmanager

# Must be set BEFORE importing torch or timm — keeps weights off the C: drive.
os.environ["TORCH_HOME"] = r"E:\Master Thesis\DR_Thesis_Project\cache"
os.environ["HF_HOME"]    = r"E:\Master Thesis\DR_Thesis_Project\cache"

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# Allow imports from the same src/ directory
sys.path.insert(0, str(Path(__file__).parent))
from models import LHT_CNN


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEIGHTS_PATH = r"E:\Master Thesis\DR_Thesis_Project\weights\best_model.pth"
IMAGE_SIZE   = 384
NUM_CLASSES  = 5

CLASS_NAMES: dict[int, str] = {
    0: "Healthy",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative",
}

SEVERITY_DESCRIPTIONS: dict[int, str] = {
    0: "No apparent diabetic retinopathy.",
    1: "Mild non-proliferative DR — microaneurysms only.",
    2: "Moderate non-proliferative DR — more than just microaneurysms.",
    3: "Severe non-proliferative DR — extensive abnormalities present.",
    4: "Proliferative DR — neovascularisation or vitreous/preretinal haemorrhage.",
}


# ---------------------------------------------------------------------------
# Preprocessing — identical to dataset.py val/test pipeline
# ---------------------------------------------------------------------------

def ben_graham_preprocess(image: np.ndarray, sigmaX: int = 10) -> np.ndarray:
    """
    Ben Graham fundus preprocessing (same as dataset.py):
    1. Crop to circular fundus boundary.
    2. Resize to 512 × 512.
    3. Subtract large-radius Gaussian blur to enhance local contrast.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(largest)
        cx, cy, radius = int(cx), int(cy), int(radius)
        h, w = image.shape[:2]
        image = image[
            max(cy - radius, 0) : min(cy + radius, h),
            max(cx - radius, 0) : min(cx + radius, w),
        ]

    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigmaX)
    image = cv2.addWeighted(image, 4, blurred, -4, 128)
    return image


# Albumentations val-set pipeline (CLAHE → resize → normalise → tensor)
_TRANSFORM = A.Compose([
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def preprocess(image_bytes: bytes) -> torch.Tensor:
    """
    Full preprocessing pipeline for an uploaded fundus image.

    Steps
    -----
    1. Decode bytes → PIL → NumPy RGB uint8.
    2. Ben Graham crop + contrast enhancement.
    3. CLAHE → resize 384 → ImageNet normalise → CHW float32 tensor.
    4. Add batch dimension → [1, 3, 384, 384].
    """
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np  = np.array(pil_img, dtype=np.uint8)               # HWC RGB

    img_np  = ben_graham_preprocess(img_np)                    # fundus crop + contrast
    tensor  = _TRANSFORM(image=img_np)["image"]                # CHW float32
    return tensor.unsqueeze(0)                                  # [1, C, H, W]


# ---------------------------------------------------------------------------
# Model — loaded once at startup, shared across all requests
# ---------------------------------------------------------------------------

_model: LHT_CNN | None = None
_device: torch.device  | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the server starts; release on shutdown."""
    global _model, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[startup] Device: {_device}")

    _model = LHT_CNN(num_classes=NUM_CLASSES, pretrained=False).to(_device)
    checkpoint = torch.load(WEIGHTS_PATH, map_location=_device)
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.eval()

    saved_epoch = checkpoint.get("epoch", "?")
    saved_qwk   = checkpoint.get("val_qwk", float("nan"))
    print(f"[startup] Loaded weights — epoch {saved_epoch}, val QWK = {saved_qwk:.4f}")
    print(f"[startup] Server ready at http://0.0.0.0:8000")

    yield   # server runs here

    print("[shutdown] Releasing model.")
    del _model


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LHT-CNN Diabetic Retinopathy API",
    description=(
        "Upload a fundus photograph and receive an automated DR severity "
        "grade (0 = Healthy → 4 = Proliferative) with a softmax confidence score."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def health_check():
    """Quick liveness check — returns model status."""
    return {
        "status": "ok",
        "model":  "LHT_CNN",
        "device": str(_device),
        "classes": CLASS_NAMES,
    }


@app.post("/predict/", tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    """
    **POST** a fundus image (JPEG / PNG) and receive a DR severity prediction.

    ### Response fields
    | Field | Type | Description |
    |---|---|---|
    | `class_name` | str | DR severity label |
    | `severity_level` | int | Grade 0 – 4 |
    | `confidence` | float | Softmax score for the predicted class (0–1) |
    | `all_probabilities` | dict | Softmax scores for all 5 classes |
    | `description` | str | Plain-English clinical description |
    """
    # ── Validate MIME type ────────────────────────────────────────────────────
    if file.content_type not in {"image/jpeg", "image/jpg", "image/png",
                                  "image/bmp", "image/tiff"}:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. "
                   f"Please upload a JPEG or PNG fundus image.",
        )

    # ── Read and preprocess ───────────────────────────────────────────────────
    try:
        image_bytes = await file.read()
        tensor      = preprocess(image_bytes).to(_device)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Image preprocessing failed: {exc}",
        )

    # ── Inference ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = _model(tensor)                          # [1, 5]
        probs  = torch.softmax(logits, dim=1)[0]         # [5]

    severity_level = int(probs.argmax().item())
    confidence     = round(float(probs[severity_level].item()), 3)

    all_probs = {
        CLASS_NAMES[i]: round(float(probs[i].item()), 3)
        for i in range(NUM_CLASSES)
    }

    return JSONResponse(content={
        "class_name":       CLASS_NAMES[severity_level],
        "severity_level":   severity_level,
        "confidence":       confidence,
        "all_probabilities": all_probs,
        "description":      SEVERITY_DESCRIPTIONS[severity_level],
        "model":            "LHT_CNN",
        "image_size_used":  f"{IMAGE_SIZE}x{IMAGE_SIZE}",
    })


# ---------------------------------------------------------------------------
# Entry point (for running directly: python src/api.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
