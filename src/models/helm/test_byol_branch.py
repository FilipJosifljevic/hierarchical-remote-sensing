"""
Standalone smoke test for the BYOL branch. Run from src/models/helm/:
    python3 test_byol_branch.py
"""
import torch

from encoder import HierarchyTokenViT
from byol_branch import BYOLBranch

torch.manual_seed(0)

M = 30
print("Building encoder (downloads pretrained weights on first run)...")
encoder = HierarchyTokenViT(num_hierarchy_tokens=M, pretrained=True)

byol = BYOLBranch(
    online_encoder=encoder,
    patch_embed_dim=encoder.embed_dim,
    projector_hidden_dim=256,
    projector_out_dim=64,
    predictor_hidden_dim=256,
)

B = 4
view1 = torch.randn(B, 3, 224, 224)
view2 = torch.randn(B, 3, 224, 224)

online_w = next(encoder.parameters()).clone()
target_w = next(byol.target_encoder.parameters()).clone()
assert torch.allclose(online_w, target_w), "target should start identical to online"
print("Target network initialized as exact copy of online: OK")

loss = byol(view1, view2)
print("L_b:", loss.item())
assert loss.item() >= 0

loss.backward()
assert encoder.hierarchy_tokens.grad is not None, "gradient should reach the SHARED encoder"
assert byol.predictor[0].weight.grad is not None, "gradient should reach the predictor"
assert byol.target_encoder.hierarchy_tokens.grad is None, "target network must NEVER get gradients"
print("Gradient flow into online encoder + predictor, NOT into target: OK")

with torch.no_grad():
    for p in encoder.parameters():
        p.add_(torch.randn_like(p) * 0.01)  # simulate one optimizer step

target_before = byol.target_encoder.hierarchy_tokens.clone()
byol.update_target_network()
target_after = byol.target_encoder.hierarchy_tokens.clone()

assert not torch.allclose(target_before, target_after), "target should move after EMA update"
dist_before = (target_before - encoder.hierarchy_tokens).abs().mean()
dist_after = (target_after - encoder.hierarchy_tokens).abs().mean()
assert dist_after < dist_before, "EMA update should move target closer to online"
print(f"EMA update moved target closer to online: {dist_before.item():.6f} -> {dist_after.item():.6f}")

print("\nALL CHECKS PASSED")