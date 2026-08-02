import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from src.data.datasets.ucm import UCMHMLCDataset
from src.models.maple.semantic_init import HierarchicalSemanticInitializer, build_hierarchy_descriptions

print("Loading UCM dataset (source of the hierarchy)...")
dataset = UCMHMLCDataset(image_root="data/raw/UCMerced_LandUse/Images", transform=None)

print("\nExample descriptions:")
descriptions = build_hierarchy_descriptions(dataset.node_names, dataset.parent)
for name in ["airplane", "trees", "Artificial Surfaces", "Urban Fabric"]:
    print(f"  {name}: {descriptions[name]}")

print("\nBuilding HierarchicalSemanticInitializer (downloads all-mpnet-base-v2 on first run)...")
initializer = HierarchicalSemanticInitializer(
    node_names=dataset.node_names,
    parent=dataset.parent,
    target_dim=384, 
)

embeddings = initializer()
print(f"\nOutput shape: {embeddings.shape}")
assert embeddings.shape == (dataset.num_nodes, 384)

normed = F.normalize(embeddings, dim=-1)
sim = normed @ normed.T
mask = ~torch.eye(dataset.num_nodes, dtype=torch.bool)
avg_sim = sim[mask].mean().item()
print(f"Average pairwise similarity across all 30 nodes: {avg_sim:.4f}")

idx = {name: i for i, name in enumerate(dataset.node_names)}
sim_related = sim[idx["trees"], idx["Forests"]].item()       
sim_unrelated = sim[idx["trees"], idx["water"]].item()        
print(f"\nSimilarity(trees, Forests) [same branch]: {sim_related:.4f}")
print(f"Similarity(trees, water) [different branches]: {sim_unrelated:.4f}")
if sim_related > sim_unrelated:
    print("Semantically related nodes are more similar than unrelated ones: matches expectation")
else:
    print("NOTE: related nodes are NOT more similar than unrelated ones here -- "
          "worth investigating before relying on this initialization strategy")

loss = embeddings.sum()
loss.backward()
assert initializer.projection[0].weight.grad is not None
print("\nGradient flows through the projection layer: OK")

print("\nALL CHECKS PASSED")