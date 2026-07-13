"""
Standalone smoke test for the fully assembled HELM model.
Run from src/models/helm/:
    python3 test_helm_model.py
"""
import torch

from helm_model import HELM
import sys
sys.path.insert(0, "../../..")
from src.utils.hierarchy import build_edge_index

torch.manual_seed(0)

# Small fake hierarchy for a fast test (swap in your real dataset.parent/node_names
# once this passes, same as we did for the graph branch earlier).
parent = {"trees": "Forest_top", "chaparral": "Shrub", "Shrub": "Forest_top"}
node_names = ["trees", "chaparral", "Shrub", "Forest_top", "Water_top"]
M = len(node_names)
edge_index = build_edge_index(parent, node_names)

print("Building HELM model (downloads pretrained weights on first run)...")
model = HELM(
    edge_index=edge_index,
    num_labels=M,
    pretrained=True,
    graph_hidden_dim=32,
    byol_projector_hidden_dim=64,
    byol_projector_out_dim=32,
    byol_predictor_hidden_dim=64,
)

B, Bl = 6, 4  # total batch / labeled rows (must be the first Bl rows)
x = torch.randn(B, 3, 224, 224)
targets = torch.randint(0, 2, (Bl, M)).float()
byol_view1 = torch.randn(B, 3, 224, 224)
byol_view2 = torch.randn(B, 3, 224, 224)

out = model(x, targets, num_labeled=Bl, byol_view1=byol_view1, byol_view2=byol_view2)
print("Losses -- L_s: {:.4f}, L_g: {:.4f}, L_b: {:.4f}, total: {:.4f}".format(
    out["L_s"].item(), out["L_g"].item(), out["L_b"].item(), out["loss"].item()
))
print("cls_logits shape:", out["cls_logits"].shape)
print("graph_logits shape:", out["graph_logits"].shape)
assert out["cls_logits"].shape == (Bl, M)
assert out["graph_logits"].shape == (B, M)

expected_total = out["L_s"] + out["L_g"] + out["L_b"]
assert torch.allclose(out["loss"], expected_total, atol=1e-5)
print("Composite loss = L_s + L_g + L_b: confirmed")

out["loss"].backward()
assert model.encoder.hierarchy_tokens.grad is not None, "encoder didn't get gradients"
assert model.classification_branch.fc.weight.grad is not None, "classification branch didn't get gradients"
assert model.graph_branch.sage1.lin_l.weight.grad is not None, "graph branch didn't get gradients"
assert model.byol_branch.predictor[0].weight.grad is not None, "byol predictor didn't get gradients"
assert model.byol_branch.target_encoder.hierarchy_tokens.grad is None, "target network must NOT get gradients"
print("Gradient flow into ALL branches (and NOT into BYOL's target network): OK")

# Simulate an actual optimizer step before checking the EMA update
with torch.no_grad():
    for p in model.encoder.parameters():
        p.add_(torch.randn_like(p) * 0.01)

target_before = model.byol_branch.target_encoder.hierarchy_tokens.clone()
model.update_target_network()
target_after = model.byol_branch.target_encoder.hierarchy_tokens.clone()
assert not torch.allclose(target_before, target_after)
dist_before = (target_before - model.encoder.hierarchy_tokens).abs().mean()
dist_after = (target_after - model.encoder.hierarchy_tokens).abs().mean()
assert dist_after < dist_before
print(f"model.update_target_network(): target moved closer to online: {dist_before.item():.6f} -> {dist_after.item():.6f}")

model.eval()
with torch.no_grad():
    probs = model.predict(x)
print("predict() output shape:", probs.shape)
assert probs.shape == (B, M)
assert (probs >= 0).all() and (probs <= 1).all()
print("predict() produces valid probabilities: OK")

print("\nALL CHECKS PASSED")