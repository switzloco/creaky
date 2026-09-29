import torch
import torch.nn as nn
import torch.nn.functional as F

class PathologyCooccurrenceTransformer(nn.Module):
    """Transformer block that models anatomical co-occurrence across the 12 knee pathologies.
    
    Allows pathologies (like ACL, MCL, and Meniscus tears) to attend to each other's 
    anatomical features before final logit projection.
    """
    def __init__(self, num_pathologies: int = 12, token_dim: int = 64, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_pathologies = num_pathologies
        self.token_dim = token_dim
        
        # Learnable pathology identity embedding
        self.pathology_embed = nn.Parameter(torch.randn(1, num_pathologies, token_dim) * 0.02)
        
        # Self-Attention across the 12 pathology tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=nhead,
            dim_feedforward=token_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.cross_attention = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Final logit projectors
        self.direct_proj = nn.Linear(token_dim, 1)
        self.refined_proj = nn.Linear(token_dim, 1)

    def forward(self, pathology_tokens: torch.Tensor) -> torch.Tensor:
        """
        Input:
            pathology_tokens: (B, 12, token_dim) extracted from plane MoE heads
        Output:
            logits: (B, 12) co-occurrence-refined class logits
        """
        B = pathology_tokens.shape[0]
        # Direct baseline logit projection
        base_logits = self.direct_proj(pathology_tokens).squeeze(-1)  # (B, 12)
        
        # Add pathology identity embedding & cross-attend
        x = pathology_tokens + self.pathology_embed
        refined_tokens = self.cross_attention(x)  # (B, 12, token_dim)
        
        # Residual co-occurrence delta
        delta_logits = self.refined_proj(refined_tokens).squeeze(-1)  # (B, 12)
        
        return base_logits + 0.5 * delta_logits


if __name__ == "__main__":
    print("Testing PathologyCooccurrenceTransformer...")
    model = PathologyCooccurrenceTransformer(num_pathologies=12, token_dim=64, nhead=4)
    dummy_input = torch.randn(4, 12, 64)
    out = model(dummy_input)
    print("  Input shape :", dummy_input.shape)
    print("  Output shape:", out.shape)
    assert out.shape == (4, 12), f"Expected (4, 12), got {out.shape}"
    
    # Check backward pass
    loss = out.sum()
    loss.backward()
    print("  Backward pass: OK (gradients verified)")
    print("Co-occurrence module test passed successfully!")
