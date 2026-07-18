from typing import Dict, Optional

import torch
import torch.nn as nn

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))  # project root, so `from src...` resolves
from src.models.helm.encoder import HierarchyTokenViT
from src.models.helm.classification_branch import ClassificationBranch
from src.models.helm.graph_branch import GraphBranch
from src.models.helm.byol_branch import BYOLBranch
from src.utils.attention_diversity import attention_diversity_loss


class HELM(nn.Module):
    def __init__(
        self,
        edge_index: torch.Tensor,
        num_labels: int,
        backbone_name: str = "vit_base_patch16_224.dino",
        pretrained: bool = True,
        graph_hidden_dim: int = 256,
        byol_projector_hidden_dim: int = 4096,
        byol_projector_out_dim: int = 256,
        byol_predictor_hidden_dim: int = 4096,
        byol_target_momentum: float = 0.996,
        lambda_diversity: float = 0.0,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.lambda_diversity = lambda_diversity

        self.encoder = HierarchyTokenViT(
            num_hierarchy_tokens=num_labels, backbone_name=backbone_name, pretrained=pretrained
        )
        self.classification_branch = ClassificationBranch(
            embed_dim=self.encoder.embed_dim, num_labels=num_labels
        )
        self.graph_branch = GraphBranch(
            embed_dim=self.encoder.embed_dim, num_labels=num_labels, hidden_dim=graph_hidden_dim
        )
        self.byol_branch = BYOLBranch(
            online_encoder=self.encoder,  # SAME encoder instance -- shared, not copied
            patch_embed_dim=self.encoder.embed_dim,
            projector_hidden_dim=byol_projector_hidden_dim,
            projector_out_dim=byol_projector_out_dim,
            predictor_hidden_dim=byol_predictor_hidden_dim,
            target_momentum=byol_target_momentum,
        )

        self.register_buffer("edge_index", edge_index)

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        num_labeled: int,
        byol_view1: torch.Tensor,
        byol_view2: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.lambda_diversity > 0:
            z_hierarchy, _, hierarchy_attn = self.encoder(x, return_hierarchy_attention=True)
            L_div = attention_diversity_loss(hierarchy_attn)
        else:
            z_hierarchy, _ = self.encoder(x)
            L_div = z_hierarchy.sum() * 0.0 


        zero = z_hierarchy.sum() * 0.0
        if num_labeled > 0:
            labeled_z_hierarchy = z_hierarchy[:num_labeled]
            cls_logits = self.classification_branch(labeled_z_hierarchy)
            L_s = self.classification_branch.compute_loss(cls_logits, targets)

            graph_logits = self.graph_branch(z_hierarchy, self.edge_index)
            L_g = self.graph_branch.compute_loss(graph_logits, targets, num_labeled=num_labeled)
        else:
            cls_logits = torch.empty((0, self.num_labels), device=x.device)
            graph_logits = self.graph_branch(z_hierarchy, self.edge_index)  # still run -- unlabeled rows still benefit from graph message-passing
            L_s = zero
            L_g = zero

        L_b = self.byol_branch(byol_view1, byol_view2)

        L = L_s + L_g + L_b + self.lambda_diversity * L_div

        return {
            "loss": L,
            "L_s": L_s.detach(),
            "L_g": L_g.detach(),
            "L_b": L_b.detach(),
            "L_div": L_div.detach(),
            "cls_logits": cls_logits.detach(),
            "graph_logits": graph_logits.detach(),
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        z_hierarchy, _ = self.encoder(x)
        cls_logits = self.classification_branch(z_hierarchy)
        graph_logits = self.graph_branch(z_hierarchy, self.edge_index)
        probs = (torch.sigmoid(cls_logits) + torch.sigmoid(graph_logits)) / 2
        return probs

    def update_target_network(self) -> None:
        self.byol_branch.update_target_network()