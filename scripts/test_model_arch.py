"""Test script for ResNet34 multi-planar model architecture with anatomical routing."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class GatedAttentionPool(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 128):
        super().__init__()
        self.v_proj = nn.Linear(in_features, hidden_dim)
        self.u_proj = nn.Linear(in_features, hidden_dim)
        self.w_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = torch.tanh(self.v_proj(x))
        u = torch.sigmoid(self.u_proj(x))
        attn_scores = self.w_proj(v * u)
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled = torch.sum(x * attn_weights, dim=1)
        return pooled

class KneeAnatomicalMoEClassifier(nn.Module):
    """Multi-Planar ResNet34 Classifier with Anatomical Plane-Specific Expert Routing."""
    def __init__(self, backbone_name: str = "resnet34", pretrained: bool = False, dropout: float = 0.2):
        super().__init__()
        # Backbone: torchvision ResNet34
        if backbone_name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            resnet = models.resnet34(weights=weights)
            self.feat_dim = resnet.fc.in_features  # 512
            resnet.fc = nn.Identity()
            self.encoder = resnet
        elif backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
            self.feat_dim = resnet.fc.in_features  # 512
            resnet.fc = nn.Identity()
            self.encoder = resnet
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Gated attention pooling per anatomical plane
        self.sag_pool = GatedAttentionPool(self.feat_dim)
        self.cor_pool = GatedAttentionPool(self.feat_dim)
        self.ax_pool = GatedAttentionPool(self.feat_dim)

        combined_dim = self.feat_dim * 3  # Sag + Cor + Ax = 1536

        # Anatomical Expert Heads
        # 1. Sagittal-specialized: ACL, Medial Meniscus, Lateral Meniscus (targets: 0, 2, 3)
        self.sag_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim + combined_dim, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3)  # [ACL, Medial Meniscus, Lateral Meniscus]
        )

        # 2. Coronal-specialized: MCL (target: 1)
        self.cor_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim + combined_dim, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # [MCL]
        )

        # 3. Axial-specialized: PF OA, Effusion, Synovitis (targets: 6, 7, 8)
        self.ax_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim + combined_dim, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3)  # [PF OA, Effusion, Synovitis]
        )

        # 4. Joint/Compartment-general: Medial OA, Lateral OA, Baker's, Contusion, Fracture (targets: 4, 5, 9, 10, 11)
        self.joint_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_dim, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 5)  # [Medial OA, Lateral OA, Baker's, Contusion, Fracture]
        )

    def _encode_plane(self, x: torch.Tensor, pool_module: GatedAttentionPool) -> torch.Tensor:
        # x: (B, K, 3, H, W)
        B, K, C, H, W = x.shape
        x_flat = x.view(B * K, C, H, W)
        feats = self.encoder(x_flat)  # (B*K, feat_dim)
        feats = feats.view(B, K, self.feat_dim)
        return pool_module(feats)  # (B, feat_dim)

    def forward(self, sag: torch.Tensor, cor: torch.Tensor, ax: torch.Tensor) -> torch.Tensor:
        # sag, cor, ax: (B, K, 3, H, W)
        h_sag = self._encode_plane(sag, self.sag_pool)  # (B, feat_dim)
        h_cor = self._encode_plane(cor, self.cor_pool)  # (B, feat_dim)
        h_ax = self._encode_plane(ax, self.ax_pool)    # (B, feat_dim)

        combined = torch.cat([h_sag, h_cor, h_ax], dim=1)  # (B, 3*feat_dim)

        # Route through expert heads
        out_sag = self.sag_head(torch.cat([h_sag, combined], dim=1))    # (B, 3): [ACL, Medial Meniscus, Lateral Meniscus]
        out_cor = self.cor_head(torch.cat([h_cor, combined], dim=1))    # (B, 1): [MCL]
        out_ax = self.ax_head(torch.cat([h_ax, combined], dim=1))      # (B, 3): [PF OA, Effusion, Synovitis]
        out_joint = self.joint_head(combined)                          # (B, 5): [Medial OA, Lateral OA, Baker's, Contusion, Fracture]

        # Reassemble into canonical 12-target order:
        # 0: ACL, 1: MCL, 2: Medial Meniscus, 3: Lateral Meniscus,
        # 4: Medial OA, 5: Lateral OA, 6: PF OA, 7: Effusion,
        # 8: Synovitis, 9: Baker's, 10: Contusion, 11: Fracture
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
    print("Initializing KneeAnatomicalMoEClassifier...")
    model = KneeAnatomicalMoEClassifier(backbone_name="resnet34", pretrained=False)
    B, K = 2, 8
    sag = torch.randn(B, K, 3, 256, 256)
    cor = torch.randn(B, K, 3, 256, 256)
    ax = torch.randn(B, K, 3, 256, 256)
    logits = model(sag, cor, ax)
    print("Output shape:", logits.shape)
    assert logits.shape == (B, 12), f"Expected (2, 12), got {logits.shape}"
    print("Test passed successfully!")
