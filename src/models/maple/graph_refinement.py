from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from src.utils.hierarchy import batch_edge_index

class MAPLEGraphBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            aggr: str = "mean",
    ):
        super().__init__()
        self.conv = SAGEConv(dim, dim, aggr=aggr)
        self.norm = nn.LayerNorm(dim)

    def forward(
            self,
            h:torch.Tensor,
            edge_index: torch.Tensor
    ) -> torch.Tensor:
        m = self.conv(h, edge_index)
        h = self.norm(m + h)
        h = F.gelu(h)
        return h

class MAPLEGraphRefinement(nn.Module):
    def __init__(
            self,
            dim: int,
            num_layers: int = 2,
            aggr: str = "mean",
    ):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList([MAPLEGraphBlock(dim, aggr=aggr) for _ in range(num_layers)])

    def forward(
            self,
            z_hierarchy: torch.Tensor,
            edge_index: torch.Tensor
    ) -> torch.Tensor:
        B, M, d = z_hierarchy.shape
        assert d == self.dim, f"expected dim {self.dim}, got {d}"

        h = z_hierarchy.reshape(B * M, d)
        batched_edges = batch_edge_index(edge_index, num_nodes=M, batch_size=B).to(h.device)

        for block in self.blocks:
            h = block(h, batched_edges)

        return h.view(B, M, d)