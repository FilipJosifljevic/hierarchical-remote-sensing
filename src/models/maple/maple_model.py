from typing import Dict, List

import torch
import torch.nn as nn

from src.models.maple.encoder import MAPLEEncoder
from src.models.maple.graph_refinement import MAPLEGraphRefinement
from src.models.maple.fusion import AdaptiveMultimodalFusion
from src.models.maple.head import MAPLEPredictionHead, adaptive_level_aware_loss

class MAPLE(nn.Module):
    def __init__(
        self,
        node_names: List[str],
        parent: Dict[str, str],
        depth: Dict[str, int],
        edge_index: torch.Tensor,
        backbone_name: str = "vit_small_patch16_224.dino",
        pretrained: bool = True,
        sentence_model_name: str = "all-mpnet-base-v2",
        graph_num_layers: int = 2,
    ):
        super().__init__()
        self.node_names = node_names
        self.depth = depth
        self.num_labels = len(node_names)

        self.encoder = MAPLEEncoder(
            node_names=node_names,
            parent=parent,
            backbone_name=backbone_name,
            pretrained=pretrained,
            sentence_model_name=sentence_model_name,
        )
        self.graph_refinement = MAPLEGraphRefinement(
            dim=self.encoder.embed_dim, num_layers=graph_num_layers
        )
        self.fusion = AdaptiveMultimodalFusion(dim=self.encoder.embed_dim)
        self.head = MAPLEPredictionHead(dim=self.encoder.embed_dim)

        self.register_buffer("edge_index", edge_index)

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_hierarchy, z_patch = self.encoder(x)
        e = self.graph_refinement(z_hierarchy, self.edge_index)
        h_tilde, gamma = self.fusion(e, z_patch)
        logits = self.head(h_tilde)

        loss = adaptive_level_aware_loss(logits, targets, self.depth, self.node_names)

        return {
            "loss": loss,
            "logits": logits.detach(),
            "gamma": gamma.detach(),
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        z_hierarchy, z_patch = self.encoder(x)
        e = self.graph_refinement(z_hierarchy, self.edge_index)
        h_tilde, _ = self.fusion(e, z_patch)
        logits = self.head(h_tilde)
        return torch.sigmoid(logits)