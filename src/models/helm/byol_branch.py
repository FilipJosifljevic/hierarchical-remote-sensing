import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )

class BYOLBranch(nn.Module):
    def __init__(
        self,
        online_encoder: nn.Module,
        patch_embed_dim: int,
        projector_hidden_dim: int = 4096,
        projector_out_dim: int = 256,
        predictor_hidden_dim: int = 4096,
        target_momentum: float = 0.996,      
    ):
        super().__init__()
        self.online_encoder = online_encoder
        self.target_encoder = copy.deepcopy(online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        
        self.online_projector = _mlp(patch_embed_dim, projector_hidden_dim, projector_out_dim)
        self.target_projector = copy.deepcopy(self.online_projector)
        for p in self.target_projector.parameters():
            p.requires_grad = False

        self.predictor = _mlp(projector_out_dim, predictor_hidden_dim, projector_out_dim)
        self.target_momentum = target_momentum

    @staticmethod
    def _pool(z_patch: torch.Tensor) -> torch.Tensor:
        return z_patch.mean(dim=1)
    
    @staticmethod
    def _negative_cosine_similarity(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        p = F.normalize(p, dim=-1, eps=1e-8)
        z = F.normalize(z, dim=-1, eps=1e-8)
        return 2 - 2 * (p * z).sum(dim=-1)
    
    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        _, z_patch1_online = self.online_encoder(view1)
        _, z_patch2_online = self.online_encoder(view2)
        pred1 = self.predictor(self.online_projector(self._pool(z_patch1_online)))
        pred2 = self.predictor(self.online_projector(self._pool(z_patch2_online)))

        with torch.no_grad():
            _, z_patch1_target = self.target_encoder(view1)
            _, z_patch2_target = self.target_encoder(view2)
            targ1 = self.target_projector(self._pool(z_patch1_target))
            targ2 = self.target_projector(self._pool(z_patch2_target))

        loss = (
            self._negative_cosine_similarity(pred1, targ2.detach())
            + self._negative_cosine_similarity(pred2, targ1.detach())
        )

        return loss.mean() / 2
    
    @torch.no_grad()
    def update_target_network(self) -> None:

        for online_params, target_params in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_params.data = (
                self.target_momentum * target_params.data
                + (1 - self.target_momentum) * online_params.data
            )

        for online_params, target_params in zip(
            self.online_projector.parameters(), self.target_projector.parameters()
        ):
            target_params.data = (
                self.target_momentum * target_params.data
                + (1 - self.target_momentum) * online_params.data
            )