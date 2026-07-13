"""
Standalone smoke test for the encoder + graph branch together.
Run from the project ROOT (not src/models/helm/, since this needs `from src....` imports):

    python3 src/models/helm/test_graph_branch.py
"""
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3])) 
from encoder import HierarchyTokenViT
from graph_branch import GraphBranch
from src.utils.hierarchy import build_edge_index

torch.manual_seed(0)

from src.data.datasets.ucm import UCMHMLCDataset
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
dataset = UCMHMLCDataset(image_root="../../../data/raw/UCMerced_LandUse/Images", transform=transform)
edge_index = build_edge_index(dataset.parent, dataset.node_names)
M = dataset.num_nodes  # 30

encoder2 = HierarchyTokenViT(num_hierarchy_tokens=M, pretrained=True)
branch2 = GraphBranch(embed_dim=encoder2.embed_dim, num_labels=M, hidden_dim=64)

img, labels = dataset[0]
x = img.unsqueeze(0)  # [1, 3, 224, 224]
z_hier, _ = encoder2(x)
logits = branch2(z_hier, edge_index)
print("\nReal-hierarchy test:")
print("logits shape:", logits.shape)  # expect [1, 30]