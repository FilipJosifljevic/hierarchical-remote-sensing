"""
Standalone smoke test for the encoder + classification branch together.
Run from src/models/helm/:

    python3 test_classification_branch.py
"""
import torch

from encoder import HierarchyTokenViT
from classification_branch import ClassificationBranch

torch.manual_seed(0)

NUM_LABELS = 30  # matches your UCM-HMLC hierarchy (17 leaf + 9 intermediate + 4 top)
BATCH_SIZE = 2

print("Building encoder (downloads pretrained weights on first run)...")
encoder = HierarchyTokenViT(num_hierarchy_tokens=NUM_LABELS, pretrained=True)
encoder.eval()

print("Building classification branch...")
branch = ClassificationBranch(embed_dim=encoder.embed_dim, num_labels=NUM_LABELS)
branch.eval()

print("Running a dummy batch through the encoder...")
x = torch.randn(BATCH_SIZE, 3, 224, 224)
with torch.no_grad():
    z_hier, z_patch = encoder(x)

print("z_hierarchy shape:", z_hier.shape)  # expect [2, 30, 768]
print("z_patch shape:    ", z_patch.shape)  # expect [2, 196, 768]
assert z_hier.shape == (BATCH_SIZE, NUM_LABELS, encoder.embed_dim)
assert z_patch.shape == (BATCH_SIZE, 196, encoder.embed_dim)

print("\nRunning the classification branch...")
logits = branch(z_hier)
print("logits shape:", logits.shape)  # expect [2, 30]
assert logits.shape == (BATCH_SIZE, NUM_LABELS)

# Dummy ground-truth multi-label vectors (in reality these come from your dataset)
targets = torch.randint(0, 2, (BATCH_SIZE, NUM_LABELS)).float()
print("dummy targets:", targets)

loss = branch.compute_loss(logits, targets)
print("\nL_s (classification loss):", loss.item())
assert loss.item() > 0  # BCE is always positive

# Confirm gradients actually flow end-to-end (encoder -> branch -> loss)
branch.train()
encoder.train()
z_hier, _ = encoder(x)
logits = branch(z_hier)
loss = branch.compute_loss(logits, targets)
loss.backward()
assert encoder.hierarchy_tokens.grad is not None, "gradient did not reach the encoder's hierarchy tokens!"
assert branch.fc.weight.grad is not None, "gradient did not reach the classification head!"
print("\nGradient flow encoder -> branch -> loss: OK")

print("\nALL CHECKS PASSED")