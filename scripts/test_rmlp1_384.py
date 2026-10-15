import torch
import timm

print("Testing coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k at img_size=384...")
m = timm.create_model(
    "coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k",
    pretrained=False,
    num_classes=0,
    in_chans=3,
    img_size=384,
)
m.set_grad_checkpointing(True)
print(f"num_features={m.num_features}, params={sum(p.numel() for p in m.parameters())/1e6:.2f}M")
x = torch.randn(2, 3, 384, 384)
out = m(x)
print("out shape:", out.shape)
out.sum().backward()
print("Backward pass OK!")
