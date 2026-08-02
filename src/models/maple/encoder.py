from typing import Dict, List, Tuple

import timm
import torch
import torch.nn as nn

from src.models.maple.semantic_init import HierarchicalSemanticInitializer

class MAPLEEncoder(nn.Module):
    def __init__(
            self,
            node_names: List[str],
            parent: Dict[str, str],
            backbone_name: str = "vit_small_patch16_224.dino",
            pretrained: bool = True,
            sentence_model_name: str = "all-mpnet-base-v2",
            pos_embed_init_std: float = 0.02,
    ):
        super().__init__()
        self.node_names = node_names
        self.M = len(node_names)

        vit = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)

        if vit.reg_token is not None:
            raise NotImplementedError(f"{backbone_name} uses register tokens; unsupported here.")
        if vit.no_embed_class:
            raise NotImplementedError(f"{backbone_name} has no_embed_class=True; unsupported here.")
        if vit.cls_token is None:
            raise ValueError(f"{backbone_name} has no cls_token.")

        self.embed_dim: int = vit.embed_dim
        self.patch_embed = vit.patch_embed
        self.patch_drop = vit.patch_drop
        self.norm_pre = vit.norm_pre
        self.blocks = vit.blocks
        self.norm = vit.norm

        semantic_initializer = HierarchicalSemanticInitializer(
            node_names=node_names,
            parent=parent,
            target_dim=self.embed_dim,
            sentence_model_name=sentence_model_name,
        )

        with torch.no_grad():
            initial_values = semantic_initializer().clone()
        self.hierarchy_tokens = nn.Parameter(initial_values.unsqueeze(0))

        hierarchy_pos_embed_init = torch.randn(1, self.M, self.embed_dim) * pos_embed_init_std
        self.hierarchy_pos_embed = nn.Parameter(hierarchy_pos_embed_init)

        pos_embed = vit.pos_embed.detach().clone()
        patch_pos = pos_embed[:, 1:, :]
        self.patch_pos_embed = nn.Parameter(patch_pos)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        Tp = self.patch_embed(x)
        Tp = Tp + self.patch_pos_embed

        Tcls = self.hierarchy_tokens.expand(B, -1, -1) + self.hierarchy_pos_embed.expand(B, -1, -1)

        T = torch.cat([Tcls, Tp], dim=1)
        T = self.patch_drop(T)
        T = self.norm_pre(T)
        T = self.blocks(T)
        T = self.norm(T)

        z_hierarchy = T[:, : self.M, :]
        z_patch = T[:, self.M :, :]
        return z_hierarchy, z_patch