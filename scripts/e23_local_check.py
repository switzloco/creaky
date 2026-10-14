import os
import sys
import torch
import torch.nn as nn
import timm

# Local sanity check for CoAtNet-384 MoE Classifier

class GatedAttentionSlicePooling(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, K, D)
        weights = self.attn(x)  # (B, K, 1)
        pooled = torch.sum(x * weights, dim=1)  # (B, D)
        return pooled


class KneeAnatomicalMoEClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
        pretrained: bool = False,
        num_classes: int = 12,
        dropout: float = 0.25
    ):
        super().__init__()
        self.num_classes = num_classes

        kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
        self.encoder = timm.create_model(backbone_name, **kw)
        if hasattr(self.encoder, "set_grad_checkpointing"):
            self.encoder.set_grad_checkpointing(True)
        self.feat_dim = self.encoder.num_features

        self.pool_sag = GatedAttentionSlicePooling(self.feat_dim)
        self.pool_cor = GatedAttentionSlicePooling(self.feat_dim)
        self.pool_ax = GatedAttentionSlicePooling(self.feat_dim)

        self.head_sag = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, 3)  # ACL, Medial Meniscus, Lateral Meniscus
        )
        self.head_cor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, 1)  # MCL
        )
        self.head_ax = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, 3)  # PF OA, Effusion, Synovitis
        )
        self.head_multi = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim * 3, 5)  # Medial OA, Lateral OA, Baker's, Contusion, Fracture
        )

    def forward(self, planes: dict) -> torch.Tensor:
        # planes: dict of "Sagittal", "Coronal", "Axial" -> (B, K, 3, H, W)
        sag = planes["Sagittal"]
        cor = planes["Coronal"]
        ax = planes["Axial"]

        b, k, c, h, w = sag.shape

        # Encode slices
        f_sag = self.encoder(sag.view(b * k, c, h, w)).view(b, k, -1)
        f_cor = self.encoder(cor.view(b * k, c, h, w)).view(b, k, -1)
        f_ax = self.encoder(ax.view(b * k, c, h, w)).view(b, k, -1)

        p_sag = self.pool_sag(f_sag)
        p_cor = self.pool_cor(f_cor)
        p_ax = self.pool_ax(f_ax)

        p_multi = torch.cat([p_sag, p_cor, p_ax], dim=-1)

        out_sag = self.head_sag(p_sag)
        out_cor = self.head_cor(p_cor)
        out_ax = self.head_ax(p_ax)
        out_multi = self.head_multi(p_multi)

        # Target order (12 competition classes):
        # 0: ACL, 1: MCL, 2: Medial Meniscus, 3: Lateral Meniscus,
        # 4: Medial OA, 5: Lateral OA, 6: PF OA, 7: Effusion, 8: Synovitis,
        # 9: Baker's, 10: Contusion, 11: Fracture
        logits = torch.empty(b, 12, device=sag.device, dtype=sag.dtype)
        logits[:, 0] = out_sag[:, 0]
        logits[:, 1] = out_cor[:, 0]
        logits[:, 2] = out_sag[:, 1]
        logits[:, 3] = out_sag[:, 2]
        logits[:, 4] = out_multi[:, 0]
        logits[:, 5] = out_multi[:, 1]
        logits[:, 6] = out_ax[:, 0]
        logits[:, 7] = out_ax[:, 1]
        logits[:, 8] = out_ax[:, 2]
        logits[:, 9] = out_multi[:, 2]
        logits[:, 10] = out_multi[:, 3]
        logits[:, 11] = out_multi[:, 4]

        return logits


def test_sanity():
    print("Initializing CoAtNet-384 MoE Classifier on CPU (pretrained=False)...")
    # For quick CPU test, we can use 2 slices per plane or 16 slices with tiny batch
    model = KneeAnatomicalMoEClassifier(
        backbone_name="coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
        pretrained=False
    )
    model.train()

    print(f"Model created. feat_dim = {model.feat_dim}")
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total model parameters: {total_params:.1f}M")

    # Mock batch: 1 study, 4 slices per plane (for fast CPU check), 3 channels, 384x384
    B, K, C, H, W = 1, 4, 3, 384, 384
    planes = {
        "Sagittal": torch.randn(B, K, C, H, W),
        "Coronal": torch.randn(B, K, C, H, W),
        "Axial": torch.randn(B, K, C, H, W)
    }

    print("Running forward pass...")
    logits = model(planes)
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (B, 12), f"Expected shape ({B}, 12), got {logits.shape}"

    print("Running backward pass...")
    target = torch.zeros(B, 12)
    loss = nn.BCEWithLogitsLoss()(logits, target)
    loss.backward()
    print(f"Backward pass successful! Loss = {loss.item():.4f}")

    print("\n[SUCCESS] Local sanity check passed 100%!")

if __name__ == "__main__":
    test_sanity()
