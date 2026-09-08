import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils.hierarchy import build_hierarchy

DFC15_LEAF_NAMES = ["impervious", "water", "clutter", "vegetation", "building", "tree", "boat", "car"]

DFC15_SUBSET_THRESHOLD = 0.95


class DFC15HMLCDataset(Dataset):
    def __init__(self, image_root: str, labels_csv: str, transform=None,
                 subset_threshold: float = DFC15_SUBSET_THRESHOLD):
        self.transform = transform

        df = pd.read_csv(labels_csv)
        id_col = df.columns[0] 
        leaf_cols = list(df.columns[1:])
        missing = set(DFC15_LEAF_NAMES) - set(leaf_cols)
        if missing:
            raise ValueError(
                f"Expected leaf columns {DFC15_LEAF_NAMES} not fully found in CSV "
                f"(missing: {missing}). Actual columns: {leaf_cols}."
            )

        label_matrix = df[DFC15_LEAF_NAMES].values
        self.node_names = list(DFC15_LEAF_NAMES)
        self.parent, self.depth = build_hierarchy(
            label_matrix, self.node_names, subset_threshold=subset_threshold
        )
        self.num_nodes = len(self.node_names)
        self.leaf_names = list(DFC15_LEAF_NAMES)
        name_to_idx = {name: i for i, name in enumerate(self.node_names)}

        self.samples = []
        not_found = []
        for _, row in df.iterrows():
            image_id = str(row[id_col]).strip()
            found_path = None
            for split_dir in ("images_tr", "images_test"):
                candidate = os.path.join(image_root, split_dir, f"{image_id}.png")
                if os.path.exists(candidate):
                    found_path = candidate
                    break
            if found_path is None:
                not_found.append(image_id)
                continue

            full_vec = torch.zeros(self.num_nodes, dtype=torch.float32)
            for leaf in DFC15_LEAF_NAMES:
                if row[leaf] == 1:
                    node = leaf
                    while True:
                        full_vec[name_to_idx[node]] = 1.0
                        if node in self.parent:
                            node = self.parent[node]
                        else:
                            break
            self.samples.append((found_path, full_vec))

        if not_found:
            print(f"WARNING: {len(not_found)} image IDs from the CSV had no matching file "
                  f"in images_tr/ or images_test/ (first few: {not_found[:5]}) -- "
                  f"this likely means the filename-matching assumption needs adjusting.")

        print(f"DFC15-HMLC: loaded {len(self.samples)} images, "
              f"inferred {self.num_nodes}-node hierarchy from {len(DFC15_LEAF_NAMES)} leaf labels "
              f"(subset_threshold={subset_threshold})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label