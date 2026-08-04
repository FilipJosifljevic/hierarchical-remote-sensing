"""
MAPLE's Adaptive Multimodal Fusion (Sec 2.5):

    h~_v = gamma_v (*) e_v + (1 - gamma_v) (*) z

where e_v is node v's GNN-refined ("semantic prior") embedding, z is a global
visual representation, gamma_v in [0,1]^d comes from a small gating network,
and (*) is elementwise multiplication.

FIX (diagnosed via same-image-different-label vs same-label-different-image
gamma comparisons -- see conversation): the ORIGINAL gate used plain
concatenation [e_v ; z] followed by a single Linear layer. Concatenation +
single linear can only combine e_v and z ADDITIVELY -- expressing "how much
THIS label's embedding should modulate the shared visual signal" is really a
MULTIPLICATIVE interaction, which a single linear layer over a concatenation
can only approximate weakly and indirectly. Empirically this showed up as
gamma correlating more strongly across DIFFERENT LABELS within the same image
(0.88) than across DIFFERENT IMAGES for the SAME label (0.75) -- i.e. the gate
was behaving more like an image-level signal than a label-specific one.

FIX: add an explicit elementwise interaction term (e_v * z) to the gate's
input, and use a small 2-layer MLP (with GELU) instead of one Linear layer, so
the network has direct, explicit capacity to use per-label modulation of the
shared visual signal rather than needing to approximate it indirectly.

DESIGN DECISIONS (unchanged from before, still apply):
  1. z = mean-pooled patch tokens (global visual signal, shared across all M
     nodes within an image).
  2. Gate now takes [e_v ; z ; e_v * z] (3*dim) instead of just [e_v ; z] (2*dim).
"""
import torch
import torch.nn as nn


class AdaptiveMultimodalFusion(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or max(dim // 2, 8)
        self.gate = nn.Sequential(
            nn.Linear(3 * dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim),
            nn.Sigmoid(),
        )

    def forward(self, e: torch.Tensor, z_patch: torch.Tensor):
        B, M, d = e.shape
        z_global = z_patch.mean(dim=1)  # [B, d]
        z_broadcast = z_global.unsqueeze(1).expand(-1, M, -1)  # [B, M, d]

        interaction = e * z_broadcast  # [B, M, d] -- explicit multiplicative term
        gate_input = torch.cat([e, z_broadcast, interaction], dim=-1)  # [B, M, 3d]
        gamma = self.gate(gate_input)  # [B, M, d], bounded in [0,1]

        h_tilde = gamma * e + (1 - gamma) * z_broadcast
        return h_tilde, gamma