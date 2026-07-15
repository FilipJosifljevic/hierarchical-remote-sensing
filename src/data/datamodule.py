import random
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset


def split_train_test(
        num_samples: int, n_train: int = 1667, n_test: int = 433, seed: int = 42
) -> Tuple[List[int], List[int]]:
    assert n_train + n_test <= num_samples, (
        f"n_train ({n_train}) + n_test ({n_test}) exceeds num_samples ({num_samples})"
    )
    rng = random.Random(seed)
    all_indices = list(range(num_samples))
    rng.shuffle(all_indices)
    train_indices = all_indices[:n_train]
    test_indices = all_indices[n_train : n_train + n_test]

    return train_indices, test_indices

def sample_labeled_subset(train_indices: List[int], fraction: float, seed: int) -> List[int]:
    assert 0 < fraction <= 1.0, f"fraction must be in (0,1], got {fraction}"
    rng = random.Random(seed)
    shuffled = train_indices.copy()
    rng.shuffle(shuffled)
    n_labeled = max(1, round(fraction * len(train_indices)))
    return shuffled[:n_labeled]

class SemiSupervisedUCM(Dataset):
    def __init__(
            self,
            base_dataset: Dataset,
            indices : List[int],
            labeled_indices: Optional[List[int]],
            plain_transform,
            byol_transform,
    ):
        self.base = base_dataset
        self.indices = indices
        self.labeled_set = set(labeled_indices) if labeled_indices is not None else set(indices)
        self.plain_transform = plain_transform
        self.byol_transform = byol_transform

    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, i: int):
        real_idx = self.indices[i]
        img, label = self.base[real_idx]
        is_labeled = real_idx in self.labeled_set

        plain = self.plain_transform(img)
        view1, view2 = self.byol_transform(img)

        return plain, view1, view2, (label if is_labeled else None), is_labeled
    

def make_semi_supervised_collate_fn(num_labels: int):
    def collate_fn(batch):
        plains, view1s, view2s, labels, is_labeled_flags = zip(*batch)

        order = sorted(range(len(batch)), key=lambda i: not is_labeled_flags[i])
        plains = [plains[i] for i in order]
        view1s = [view1s[i] for i in order]
        view2s = [view2s[i] for i in order]
        labels = [labels[i] for i in order]
        is_labeled_flags = [is_labeled_flags[i] for i in order]

        num_labeled = sum(is_labeled_flags)

        x = torch.stack(plains)
        byol_view1 = torch.stack(view1s)
        byol_view2 = torch.stack(view2s)

        labeled_targets = [l for l in labels if l is not None]
        if labeled_targets:
            targets = torch.stack(labeled_targets)
        else:
            targets = torch.empty((0, num_labels), dtype=torch.float32)

        return {
            "x": x,
            "targets": targets,
            "num_labeled": num_labeled,
            "byol_view1": byol_view1,
            "byol_view2": byol_view2,
        }

    return collate_fn