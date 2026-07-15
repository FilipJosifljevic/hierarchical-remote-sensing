from typing import Dict

import numpy as np
import torch
from sklearn.metrics import average_precision_score, label_ranking_loss


@torch.no_grad()
def compute_metrics(probs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    probs_np = probs.cpu().numpy()
    targets_np = targets.cpu().numpy()

    valid_cols = [i for i in range(targets_np.shape[1]) if targets_np[:, i].sum() > 0 and targets_np[:, i].sum() < len(targets_np)]
    dropped = targets_np.shape[1] - len(valid_cols)
    if dropped > 0:
        print(f"Note: {dropped} label column(s) had no positive/negative variation in this "
              f"evaluation set and were excluded from the AUPRC macro average.")

    if valid_cols:
        auprc = average_precision_score(targets_np[:, valid_cols], probs_np[:, valid_cols], average="macro")
    else:
        auprc = float("nan")

    ranking_loss = label_ranking_loss(targets_np, probs_np)

    return {"auprc": float(auprc), "ranking_loss": float(ranking_loss)}