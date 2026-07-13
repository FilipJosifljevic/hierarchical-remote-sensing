import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from src.utils.hierarchy import batch_edge_index

class GraphBranch(nn.Module):
    def __init__(self, embed_dim: int, num_labels: int, hidden_dim: int = 256):
        super().__init__()
        self.num_labels = num_labels
        self.sage1 = SAGEConv(embed_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, num_labels)

    def forward(self, z_hierarchy: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        B, M, d = z_hierarchy.shape
        assert M == self.num_labels, f"expected {self.num_labels} nodes, got {M}"

        x = z_hierarchy.reshape(B * M, d)
        batched_edges = batch_edge_index(edge_index, num_nodes=M, batch_size=B).to(x.device)

        h = self.sage1(x, batched_edges)
        h = F.relu(h)
        h = self.sage2(h, batched_edges)

        h = h.view(B, M, d)
        f_g = h.mean(dim=1)
        logits = self.fc(f_g)
        return logits
    
    @staticmethod
    def compute_loss(logits: torch.Tensor, targets: torch.Tensor, num_labeled: int) -> torch.Tensor:
        labeled_logits = logits[:num_labeled]
        return F.binary_cross_entropy_with_logits(labeled_logits, targets)