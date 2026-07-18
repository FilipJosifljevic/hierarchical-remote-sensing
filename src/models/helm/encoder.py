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

        cls_token = vit.cls_token.detach().clone()  # [1, 1, d]
        symmetry_breaking_noise_std = 0.02  # small relative to typical embedding norms; enough to break exact ties
        hierarchy_tokens_init = cls_token.repeat(1, self.M, 1) + torch.randn(1, self.M, self.embed_dim) * symmetry_breaking_noise_std
        self.hierarchy_tokens = nn.Parameter(hierarchy_tokens_init)  # [1, M, d]

        pos_embed = vit.pos_embed.detach().clone()  # [1, 1+Np, d]  (num_prefix_tokens=1)
        cls_pos, patch_pos = pos_embed[:, :1, :], pos_embed[:, 1:, :]
        hierarchy_pos_embed_init = cls_pos.repeat(1, self.M, 1) + torch.randn(1, self.M, self.embed_dim) * symmetry_breaking_noise_std
        self.hierarchy_pos_embed = nn.Parameter(hierarchy_pos_embed_init)  # [1, M, d]
        self.patch_pos_embed = nn.Parameter(patch_pos)  # [1, Np, d]

    def _last_block_forward_with_attention(self, block, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_module = block.attn
        normed = block.norm1(x)

        B, N, C = normed.shape
        qkv = attn_module.qkv(normed).reshape(B, N, 3, attn_module.num_heads, attn_module.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = attn_module.q_norm(q), attn_module.k_norm(k)
        q = q * attn_module.scale
        attn_weights = (q @ k.transpose(-2, -1)).softmax(dim=-1)  # [B, num_heads, N, N] -- NOT detached
        attn_dropped = attn_module.attn_drop(attn_weights)

        attn_out = (attn_dropped @ v).transpose(1, 2).reshape(B, N, attn_module.attn_dim)
        attn_out = attn_module.norm(attn_out)
        attn_out = attn_module.proj(attn_out)
        attn_out = attn_module.proj_drop(attn_out)

        x = x + block.drop_path1(block.ls1(attn_out))
        x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        return x, attn_weights

    def forward(
        self, x: torch.Tensor, return_hierarchy_attention: bool = False
    ):
        
        B = x.shape[0]
        Tp = self.patch_embed(x)  # [B, Np, d]
        Tp = Tp + self.patch_pos_embed

        Tcls = self.hierarchy_tokens.expand(B, -1, -1) + self.hierarchy_pos_embed.expand(B, -1, -1)  # [B, M, d]

        T = torch.cat([Tcls, Tp], dim=1)  # [B, M+Np, d]
        T = self.patch_drop(T)
        T = self.norm_pre(T)

        if not return_hierarchy_attention:
            T = self.blocks(T)
            T = self.norm(T)
            z_hierarchy = T[:, : self.M, :]
            z_patch = T[:, self.M :, :]
            return z_hierarchy, z_patch

        for block in self.blocks[:-1]:
            T = block(T)
        T, attn_weights = self._last_block_forward_with_attention(self.blocks[-1], T)
        T = self.norm(T)

        z_hierarchy = T[:, : self.M, :]
        z_patch = T[:, self.M :, :]

        hierarchy_attn = attn_weights[:, :, : self.M, self.M :].mean(dim=1)  # [B, M, Np], avg over heads
        return z_hierarchy, z_patch, hierarchy_attn