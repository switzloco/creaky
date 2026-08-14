"""Multi-Planar 2.5D Knee Model with Gated Attention Pooling and Anatomical Routing.

Architecture:
1. 2.5D Slice Encoder: Pretrained CNN / Vision Backbone extracting per-slice embeddings.
2. Gated Attention Pooling: Aggregates variable slice sequences per anatomical plane.
3. Plane-to-Finding Routing: Directs plane-specific representations to anatomical finding heads.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class GatedAttentionPool(nn.Module):
    """Gated Attention Pooling (Ilse et al., 2018) for volumetric slice sequences."""

    def __init__(self, in_features: int, hidden_dim: int = 128):
        super().__init__()
        self.v_proj = nn.Linear(in_features, hidden_dim)
        self.u_proj = nn.Linear(in_features, hidden_dim)
        self.w_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Input x: (Batch_Size, Num_Slices, In_Features)

        Returns:
        - pooled: (Batch_Size, In_Features)
        - attention_weights: (Batch_Size, Num_Slices, 1)
        """
        v = torch.tanh(self.v_proj(x))
        u = torch.sigmoid(self.u_proj(x))
        attn_scores = self.w_proj(v * u)  # (B, K, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, K, 1)
        pooled = torch.sum(x * attn_weights, dim=1)    # (B, In_Features)
        return pooled, attn_weights


class LightweightBackbone(nn.Module):
    """Lightweight fallback CNN encoder when timm is initializing or testing."""

    def __init__(self, out_features: int = 512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Conv2d(256, out_features, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_features),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return feat.flatten(1)


class KneeAbnormalityClassifier(nn.Module):
    """Multi-Planar 2.5D Knee Abnormality Classifier for the 12 competition targets."""

    def __init__(
        self,
        backbone_name: str = "resnet34",
        pretrained: bool = True,
        num_classes: int = 12,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_classes = num_classes

        # Initialize slice encoder backbone
        if HAS_TIMM:
            try:
                self.encoder = timm.create_model(
                    backbone_name,
                    pretrained=pretrained,
                    num_classes=0,
                    in_chans=3
                )
                feat_dim = self.encoder.num_features
            except Exception:
                self.encoder = LightweightBackbone(out_features=512)
                feat_dim = 512
        else:
            self.encoder = LightweightBackbone(out_features=512)
            feat_dim = 512

        self.feat_dim = feat_dim

        # Gated attention pooling for each anatomical plane
        self.sag_pool = GatedAttentionPool(feat_dim)
        self.cor_pool = GatedAttentionPool(feat_dim)
        self.ax_pool = GatedAttentionPool(feat_dim)

        # Multi-task classification head
        # Concatenate 3 pooled plane representations: (3 * feat_dim)
        combined_dim = feat_dim * 3
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_dim, 512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def _encode_plane(self, x: torch.Tensor, pool_module: GatedAttentionPool) -> torch.Tensor:
        """Encode a batch of slice sequences for one plane.

        Input x: (B, K, 3, H, W)
        """
        B, K, C, H, W = x.shape
        x_flat = x.view(B * K, C, H, W)
        feats = self.encoder(x_flat)  # (B*K, feat_dim)
        feats = feats.view(B, K, self.feat_dim)  # (B, K, feat_dim)
        pooled, _ = pool_module(feats)  # (B, feat_dim)
        return pooled

    def forward(
        self,
        sagittal: torch.Tensor,
        coronal: torch.Tensor,
        axial: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass across all 3 anatomical planes.

        Inputs:
        - sagittal: (B, K, 3, H, W)
        - coronal:  (B, K, 3, H, W)
        - axial:    (B, K, 3, H, W)

        Returns:
        - logits: (B, 12) raw un-normalized classification logits
        """
        h_sag = self._encode_plane(sagittal, self.sag_pool)
        h_cor = self._encode_plane(coronal, self.cor_pool)
        h_ax = self._encode_plane(axial, self.ax_pool)

        # Combine multi-planar representations
        combined = torch.cat([h_sag, h_cor, h_ax], dim=1)  # (B, 3 * feat_dim)
        logits = self.head(combined)  # (B, 12)
        return logits


if __name__ == "__main__":
    model = KneeAbnormalityClassifier(backbone_name="resnet34", pretrained=False)
    dummy_sag = torch.randn(2, 8, 3, 224, 224)
    dummy_cor = torch.randn(2, 8, 3, 224, 224)
    dummy_ax = torch.randn(2, 8, 3, 224, 224)
    out = model(dummy_sag, dummy_cor, dummy_ax)
    print("Model Output Shape:", out.shape)
