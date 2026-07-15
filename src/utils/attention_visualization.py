import types
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn.functional as F

def _unfused_attention_forward_with_capture(self, x, attn_mask=None, is_causal=False):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    self._captured_attn = attn.detach()  # [B, num_heads, N, N] -- stashed for the caller
    attn = self.attn_drop(attn)

    x = attn @ v
    x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
    x = self.norm(x)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

@contextmanager
def capture_attention(block):
    original_forward = block.attn.forward
    block.attn.forward = types.MethodType(_unfused_attention_forward_with_capture, block.attn)
    try:
        yield
    finally:
        block.attn.forward = original_forward

@torch.no_grad()
def get_hierarchy_token_attention(
    encoder,
    x: torch.Tensor,
    block_idx: int = -1,
) -> torch.Tensor:
    B = x.shape[0]
    M = encoder.M

    block = encoder.blocks[block_idx]
    with capture_attention(block):
        encoder(x)
        attn = block.attn._captured_attn

    hierarchy_to_patch = attn[:, :, :M, M:]  # [B, num_heads, M, Np]
    hierarchy_to_patch = hierarchy_to_patch.mean(dim=1)  # average over heads -> [B, M, Np]

    Np = hierarchy_to_patch.shape[-1]
    grid_size = int(Np ** 0.5)
    assert grid_size * grid_size == Np, f"Np={Np} is not a perfect square -- non-square patch grid?"

    attn_maps = hierarchy_to_patch.view(B, M, grid_size, grid_size)
    return attn_maps

def upsample_attention_map(attn_map: torch.Tensor, image_size: int = 224) -> torch.Tensor:
    attn_map = attn_map.unsqueeze(0).unsqueeze(0)  # [1, 1, Hh, Wp]
    upsampled = F.interpolate(attn_map, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return upsampled.squeeze(0).squeeze(0)  # [image_size, image_size]