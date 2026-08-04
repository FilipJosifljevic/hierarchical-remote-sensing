import torch
import torch.nn as nn


class AdaptiveMultimodalFusion(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or dim
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