from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAPLEPredictionHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.classifier = nn.Linear(dim, 1)

    def forward(self, h_tilde: torch.Tensor) -> torch.Tensor:
        return self.classifier(h_tilde).squeeze(-1)


def adaptive_level_aware_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    depth: Dict[str, int],
    node_names: List[str],
) -> torch.Tensor:
    device = logits.device
    B, M = logits.shape

    levels: Dict[int, List[int]] = {}
    for i, name in enumerate(node_names):
        levels.setdefault(depth[name], []).append(i)

    total_loss = logits.sum() * 0.0
    num_levels_used = 0

    for level, indices in levels.items():
        idx = torch.tensor(indices, device=device)
        level_logits = logits[:, idx]   
        level_targets = targets[:, idx] 

        active_counts = level_targets.sum(dim=1)  
        single_mask = active_counts == 1
        multi_mask = ~single_mask

        level_loss_terms = []

        if single_mask.any():
            ce_logits = level_logits[single_mask]
            ce_targets = level_targets[single_mask].argmax(dim=1)
            level_loss_terms.append(F.cross_entropy(ce_logits, ce_targets))

        if multi_mask.any():
            bce_logits = level_logits[multi_mask]
            bce_targets = level_targets[multi_mask]
            level_loss_terms.append(F.binary_cross_entropy_with_logits(bce_logits, bce_targets))

        if level_loss_terms:
            total_loss = total_loss + sum(level_loss_terms) / len(level_loss_terms)
            num_levels_used += 1

    return total_loss / max(num_levels_used, 1)