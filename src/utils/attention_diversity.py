import torch
import torch.nn.functional as F


def attention_diversity_loss(hierarchy_attn: torch.Tensor) -> torch.Tensor:
    B, M, Np = hierarchy_attn.shape
    normed = F.normalize(hierarchy_attn, dim=-1, eps=1e-8)  # [B, M, Np]
    sim_matrix = normed @ normed.transpose(-2, -1)  # [B, M, M] -- pairwise cosine similarities

    mask = ~torch.eye(M, dtype=torch.bool, device=hierarchy_attn.device)  # exclude self-similarity (always 1.0)
    pairwise_sims = sim_matrix[:, mask]  # [B, M*(M-1)]
    return pairwise_sims.mean()