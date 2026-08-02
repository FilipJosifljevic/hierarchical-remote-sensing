import random
from typing import Dict, List, Tuple

from torch.utils.data import Dataset


def sample_kshot_indices(
    dataset: Dataset,
    train_indices: List[int],
    leaf_names: List[str],
    k: int,
    seed: int,
) -> Tuple[List[int], Dict[str, int]]:
    rng = random.Random(seed)
    name_to_idx = {name: i for i, name in enumerate(dataset.node_names)}
    train_set = set(train_indices)

    selected: set = set()
    per_class_counts: Dict[str, int] = {}

    for leaf in leaf_names:
        leaf_idx = name_to_idx[leaf]
        candidates = [
            i for i in train_indices
            if dataset.samples[i][1][leaf_idx] == 1
        ]
        rng.shuffle(candidates)
        chosen = candidates[:k]
        selected.update(chosen)
        per_class_counts[leaf] = len(chosen)

        if len(chosen) < k:
            print(f"Warning: only {len(chosen)} candidates available for class "
                  f"'{leaf}' in the train split (wanted {k})")

    return sorted(selected), per_class_counts


class KShotSubset(Dataset):
    def __init__(self, base_dataset: Dataset, indices: List[int], transform=None):
        self.base = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        real_idx = self.indices[i]
        img, label = self.base[real_idx]
        if self.transform:
            img = self.transform(img)
        return img, label