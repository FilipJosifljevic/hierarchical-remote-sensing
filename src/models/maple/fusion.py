import torch
import torch.nn as nn

class AdaptiveMultimodalFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.LayerNorm(dim),
            nn.Sigmoid(),
        )

    def forward(self, e: torch.Tensor, z_patch: torch.Tensor):
        B, M, d = e.shape
        z_global = z_patch.mean(dim=1)
        z_broadcast = z_global.unsqueeze(1).expand(-1, M, -1) 

        gate_input = torch.cat([e, z_broadcast], dim=-1)
        gamma = self.gate(gate_input)

        h_tilde = gamma * e + (1 - gamma) * z_broadcast
        return h_tilde, gamma