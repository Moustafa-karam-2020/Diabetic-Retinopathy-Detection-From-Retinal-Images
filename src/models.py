import os

# Must be set BEFORE importing torch or timm to redirect all weight downloads
# away from the nearly-full C: drive on local Windows runs.
# On Kaggle, /kaggle/working/ is always writable, so we send caches there.
if os.path.isdir("/kaggle/working"):
    _CACHE_DIR = "/kaggle/working/cache"
else:
    _CACHE_DIR = r"E:\Master Thesis\DR_Thesis_Project\cache"

os.environ.setdefault("TORCH_HOME", _CACHE_DIR)
os.environ.setdefault("HF_HOME",    _CACHE_DIR)

import torch
import torch.nn as nn
import timm


# ---------------------------------------------------------------------------
# Multimodal Fusion Block (MFB) with Co-Attention
# ---------------------------------------------------------------------------

class MultimodalFusionBlock(nn.Module):
    """
    Co-attention fusion of a CNN feature vector and a ViT feature vector.

    Mechanism
    ---------
    1. Both feature vectors are independently projected to a shared *proj_dim*.
    2. Co-attention gates:
         - CNN projection  → sigmoid gate  applied to ViT projection
         - ViT projection  → sigmoid gate  applied to CNN projection
       This lets each modality suppress or amplify dimensions of the other.
    3. The two attended vectors are layer-normalised and concatenated.
    """

    def __init__(self, cnn_dim: int, vit_dim: int, proj_dim: int = 512) -> None:
        super().__init__()

        # Projection heads — bring both streams to the same dimension
        self.cnn_proj = nn.Sequential(nn.Linear(cnn_dim, proj_dim), nn.GELU())
        self.vit_proj = nn.Sequential(nn.Linear(vit_dim, proj_dim), nn.GELU())

        # Co-attention gates
        # CNN features generate an element-wise gate for the ViT stream
        self.cnn_gate = nn.Linear(proj_dim, proj_dim)
        # ViT features generate an element-wise gate for the CNN stream
        self.vit_gate = nn.Linear(proj_dim, proj_dim)

        self.norm_cnn = nn.LayerNorm(proj_dim)
        self.norm_vit = nn.LayerNorm(proj_dim)

    def forward(
        self, cnn_feat: torch.Tensor, vit_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            cnn_feat : [B, cnn_dim]
            vit_feat : [B, vit_dim]
        Returns:
            fused    : [B, 2 * proj_dim]
        """
        cnn_p = self.cnn_proj(cnn_feat)   # [B, proj_dim]
        vit_p = self.vit_proj(vit_feat)   # [B, proj_dim]

        # Each modality attends to the other
        gate_for_vit = torch.sigmoid(self.cnn_gate(cnn_p))   # [B, proj_dim]
        gate_for_cnn = torch.sigmoid(self.vit_gate(vit_p))   # [B, proj_dim]

        attended_cnn = self.norm_cnn(cnn_p * gate_for_cnn)   # [B, proj_dim]
        attended_vit = self.norm_vit(vit_p * gate_for_vit)   # [B, proj_dim]

        return torch.cat([attended_cnn, attended_vit], dim=-1)  # [B, 2*proj_dim]


# ---------------------------------------------------------------------------
# LHT_CNN — Hybrid CNN-ViT model
# ---------------------------------------------------------------------------

class BaselineFusionBlock(nn.Module):
    """
    Ablation baseline — fusion WITHOUT co-attention.

    Projects both streams to *proj_dim* and concatenates them.
    Produces an output of identical shape to MultimodalFusionBlock
    ([B, 2 * proj_dim]) so the downstream classifier head is unchanged,
    making the comparison a clean ablation of the co-attention mechanism only.
    """

    def __init__(self, cnn_dim: int, vit_dim: int, proj_dim: int = 512) -> None:
        super().__init__()
        self.cnn_proj = nn.Sequential(nn.Linear(cnn_dim, proj_dim), nn.GELU())
        self.vit_proj = nn.Sequential(nn.Linear(vit_dim, proj_dim), nn.GELU())

    def forward(
        self, cnn_feat: torch.Tensor, vit_feat: torch.Tensor
    ) -> torch.Tensor:
        cnn_p = self.cnn_proj(cnn_feat)   # [B, proj_dim]
        vit_p = self.vit_proj(vit_feat)   # [B, proj_dim]
        return torch.cat([cnn_p, vit_p], dim=-1)  # [B, 2*proj_dim]


class LHT_CNN(nn.Module):
    """
    Hybrid CNN-ViT model for 5-class Diabetic Retinopathy classification.

    Backbone 1 : EfficientNet-B0      (CNN  — local texture / fine-grained)
    Backbone 2 : ViT-Tiny patch16     (ViT  — global context / long-range)
    Fusion     : Multimodal Fusion Block with Co-Attention (MFB)  — if use_mfb=True
                 Plain projection + concatenation                 — if use_mfb=False

    The ``use_mfb`` switch exists for the ablation study: both configurations
    share identical backbones, projection dimensions, and classifier head, so
    any performance difference is attributable to the co-attention block.

    Input  : (B, 3, 384, 384)
    Output : (B, num_classes)  — raw logits
    """

    def __init__(
        self,
        num_classes: int = 5,
        proj_dim: int = 512,
        pretrained: bool = True,
        dropout: float = 0.5,   # heavier dropout on the FC head to combat overfitting
        use_mfb: bool = True,
    ) -> None:
        super().__init__()
        self.use_mfb = use_mfb

        # ── CNN backbone ─────────────────────────────────────────────────────
        self.cnn_backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        # ── ViT backbone ─────────────────────────────────────────────────────
        self.vit_backbone = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True,
        )

        cnn_dim: int = self.cnn_backbone.num_features   # 1 280
        vit_dim: int = self.vit_backbone.num_features   #   192

        # ── Fusion block: MFB with Co-Attention  OR  plain concat ─────────────
        if use_mfb:
            self.fusion = MultimodalFusionBlock(cnn_dim, vit_dim, proj_dim)
        else:
            self.fusion = BaselineFusionBlock(cnn_dim, vit_dim, proj_dim)

        # ── Classifier head (identical for both variants) ─────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : image tensor  (B, 3, 384, 384)
        Returns:
            logits            (B, num_classes)
        """
        cnn_feat = self.cnn_backbone(x)              # [B, 1280]
        vit_feat = self.vit_backbone(x)              # [B,  192]
        fused    = self.fusion(cnn_feat, vit_feat)   # [B, 1024]
        return self.classifier(fused)                # [B,    5]
