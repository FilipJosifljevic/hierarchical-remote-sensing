import torch
import torch.nn as nn

class ClassificationBranch(nn.Module):
    def __init__(self, embed_dim: int, num_labels: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_labels)

    def forward(self, z_hierarchy: torch.Tensor) -> torch.Tensor:
        f_cls = z_hierarchy.mean(dim=1)
        logits = self.fc(f_cls)
        return logits
    
    @staticmethod
    def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.binary_cross_entropy_with_logits(logits, targets)