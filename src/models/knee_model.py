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


class KneeAnatomicalMoEClassifier(nn.Module):
    """Multi-Planar 2.5D Knee Model with Anatomical Plane-Specific Expert Routing.
    
    Routes anatomical representations according to clinical MR physics:
    - Sagittal plane -> ACL, Medial Meniscus, Lateral Meniscus heads
    - Coronal plane -> MCL head
    - Axial plane -> PF OA, Effusion, Synovitis heads
    - Combined multi-planar -> Medial OA, Lateral OA, Baker's, Contusion, Fracture heads
    """
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
        feat_dim = 512
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
                self.encoder = None

        if not hasattr(self, "encoder") or self.encoder is None:
            try:
                import torchvision.models as tv_models
                if backbone_name == "resnet34":
                    weights = tv_models.ResNet34_Weights.DEFAULT if pretrained else None
                    resnet = tv_models.resnet34(weights=weights)
                    feat_dim = resnet.fc.in_features
                    resnet.fc = nn.Identity()
                    self.encoder = resnet
                elif backbone_name == "resnet18":
                    weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
                    resnet = tv_models.resnet18(weights=weights)
                    feat_dim = resnet.fc.in_features
                    resnet.fc = nn.Identity()
                    self.encoder = resnet
                elif backbone_name == "resnet50":
                    weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
                    resnet = tv_models.resnet50(weights=weights)
                    feat_dim = resnet.fc.in_features
                    resnet.fc = nn.Identity()
                    self.encoder = resnet
                elif backbone_name in ["convnext_tiny", "convnext_small"]:
                    factory = tv_models.convnext_small if backbone_name == "convnext_small" else tv_models.convnext_tiny
                    w_enum = tv_models.ConvNeXt_Small_Weights.DEFAULT if backbone_name == "convnext_small" else tv_models.ConvNeXt_Tiny_Weights.DEFAULT
                    weights = w_enum if pretrained else None
                    model = factory(weights=weights)
                    feat_dim = model.classifier[2].in_features
                    model.classifier[2] = nn.Identity()
                    self.encoder = model
                elif backbone_name in ["efficientnet_v2_s", "effnet"]:
                    weights = tv_models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
                    model = tv_models.efficientnet_v2_s(weights=weights)
                    feat_dim = model.classifier[1].in_features
                    model.classifier[1] = nn.Identity()
                    self.encoder = model
                else:
                    self.encoder = LightweightBackbone(out_features=512)
                    feat_dim = 512
            except Exception:
                self.encoder = LightweightBackbone(out_features=512)
                feat_dim = 512

        self.feat_dim = feat_dim

        # Gated attention pooling per anatomical plane
        self.sag_pool = GatedAttentionPool(feat_dim)
        self.cor_pool = GatedAttentionPool(feat_dim)
        self.ax_pool = GatedAttentionPool(feat_dim)

        combined_dim = feat_dim * 3

        # Anatomical Expert Heads
        self.sag_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim + combined_dim, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3)  # [ACL, Medial Meniscus, Lateral Meniscus]
        )

        self.cor_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim + combined_dim, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # [MCL]
        )

        self.ax_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim + combined_dim, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3)  # [PF OA, Effusion, Synovitis]
        )

        self.joint_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_dim, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 5)  # [Medial OA, Lateral OA, Baker's, Contusion, Fracture]
        )

    def _encode_plane(self, x: torch.Tensor, pool_module: GatedAttentionPool) -> torch.Tensor:
        B, K, C, H, W = x.shape
        x_flat = x.view(B * K, C, H, W)
        feats = self.encoder(x_flat)
        if isinstance(feats, torch.Tensor) and feats.dim() > 2:
            feats = feats.flatten(1)
        feats = feats.view(B, K, self.feat_dim)
        pooled, _ = pool_module(feats)
        return pooled

    def forward(
        self,
        sagittal: torch.Tensor,
        coronal: torch.Tensor,
        axial: torch.Tensor
    ) -> torch.Tensor:
        h_sag = self._encode_plane(sagittal, self.sag_pool)
        h_cor = self._encode_plane(coronal, self.cor_pool)
        h_ax = self._encode_plane(axial, self.ax_pool)

        combined = torch.cat([h_sag, h_cor, h_ax], dim=1)

        out_sag = self.sag_head(torch.cat([h_sag, combined], dim=1))
        out_cor = self.cor_head(torch.cat([h_cor, combined], dim=1))
        out_ax = self.ax_head(torch.cat([h_ax, combined], dim=1))
        out_joint = self.joint_head(combined)

        logits = torch.stack([
            out_sag[:, 0],     # ACL
            out_cor[:, 0],     # MCL
            out_sag[:, 1],     # Medial Meniscus
            out_sag[:, 2],     # Lateral Meniscus
            out_joint[:, 0],   # Medial OA
            out_joint[:, 1],   # Lateral OA
            out_ax[:, 0],      # PF OA
            out_ax[:, 1],      # Effusion
            out_ax[:, 2],      # Synovitis
            out_joint[:, 2],   # Baker's
            out_joint[:, 3],   # Contusion
            out_joint[:, 4],   # Fracture
        ], dim=1)

        return logits


if __name__ == "__main__":
    model = KneeAnatomicalMoEClassifier(backbone_name="resnet34", pretrained=False)
    dummy_sag = torch.randn(2, 8, 3, 224, 224)
    dummy_cor = torch.randn(2, 8, 3, 224, 224)
    dummy_ax = torch.randn(2, 8, 3, 224, 224)
    out = model(dummy_sag, dummy_cor, dummy_ax)
    print("Model Output Shape:", out.shape)

