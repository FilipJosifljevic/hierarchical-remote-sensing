from typing import Tuple

import timm
import torch
import torch.nn as nn

class HierarchyTokenViT(nn.Module):
    def __init__(
            self,
            num_hierarchy_tokens: int,
            backbone_name: str = "vit_base_patch16_224.dino",
            pretrained: bool = True,
    ):
        super().__init__()
        self.M = num_hierarchy_tokens

        vit = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)

        if vit.reg_token is not None:
            raise NotImplementedError(
                f"{backbone_name} uses register tokens; this wrapper only handles the "
                f"standard single-[CLS]-token case. See the module docstring."
            )
        if vit.no_embed_class:
            raise NotImplementedError(
                f"{backbone_name} has no_embed_class=True (position embedding excludes "
                f"the class token); this wrapper assumes the standard "
                f"no_embed_class=False layout. See the module docstring."
            )
        if vit.cls_token is None:
            raise ValueError(f"{backbone_name} has no cls_token to build hierarchy tokens from.")
        
        self.embed_dim: int = vit.embed_dim
        self.patch_embed = vit.patch_embed
        self.patch_drop = vit.patch_drop
        self.norm_pre = vit.norm_pre
        self.blocks = vit.blocks
        self.norm = vit.norm

        cls_token = vit.cls_token.detach().clone()
        self.hierarchy_tokens = nn.Parameter(cls_token.repeat(1, self.M, 1))

        pos_embed = vit.pos_embed.detach().clone()
        cls_pos, patch_pos = pos_embed[:, :1, :], pos_embed[:, 1:, :]
        self.hierarchy_pos_embed = nn.Parameter(cls_pos.repeat(1, self.M, 1))
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