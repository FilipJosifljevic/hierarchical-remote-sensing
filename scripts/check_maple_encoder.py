
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from src.data.datasets.ucm import UCMHMLCDataset
from src.models.maple.encoder import MAPLEEncoder

print("Loading UCM dataset...")
dataset = UCMHMLCDataset(image_root="data/raw/UCMerced_LandUse/Images", transform=None)

print("Building MAPLEEncoder (downloads all-mpnet-base-v2 + ViT weights on first run)...")
encoder = MAPLEEncoder(
    node_names=dataset.node_names,
    parent=dataset.parent,
    backbone_name="vit_small_patch16_224.dino",
    pretrained=True,
)
print(f"embed_dim: {encoder.embed_dim}")

x = torch.randn(2, 3, 224, 224)
z_hierarchy, z_patch = encoder(x)
print(f"z_hierarchy shape: {z_hierarchy.shape}")
print(f"z_patch shape: {z_patch.shape}")
assert z_hierarchy.shape == (2, dataset.num_nodes, encoder.embed_dim)
assert z_patch.shape == (2, 196, encoder.embed_dim)

raw_tokens = encoder.hierarchy_tokens[0]
normed = F.normalize(raw_tokens, dim=-1)
sim = normed @ normed.T
mask = ~torch.eye(dataset.num_nodes, dtype=torch.bool)
avg_sim = sim[mask].mean().item()
print(f"\nAverage pairwise similarity of hierarchy_tokens at init: {avg_sim:.4f}")

loss = z_hierarchy.sum() + z_patch.sum()
loss.backward()
assert encoder.hierarchy_tokens.grad is not None
assert encoder.hierarchy_pos_embed.grad is not None
print("Gradient flow: OK")

print("\nALL CHECKS PASSED")