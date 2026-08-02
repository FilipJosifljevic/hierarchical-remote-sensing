import os
from typing import Callable, Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

class AIDHMLCDataset(Dataset):
    def __init__(
            self,
            image_root: str,
            labels_csv: str,
            ucm_node_names: List[str],
            ucm_parent: Dict[str, str],
            ucm_depth: Dict[str, int],
            transform: Optional[Callable] = None,
    ):
        self.image_root = image_root
        self.transform = transform

        df = pd.read_csv(labels_csv)
        AID_TO_UCM_NAME_MAP = {"tanks": "storage tanks"}
        df = df.rename(columns=AID_TO_UCM_NAME_MAP)
        leaf_names = list(df.columns[1:])

        ucm_leaf_names = [n for n in ucm_node_names if n not in ucm_parent.values()]
        missing = set(leaf_names) - set(ucm_node_names)

        if missing:
            raise ValueError(
                f"AID's label columns {missing} don't match any UCM hierarchy node -- "
                f"the 'reuse UCM's hierarchy' shortcut assumes identical leaf names. "
                f"Check for naming differences (e.g. 'storage tanks' vs 'storagetanks')."
            )

        self.node_names = ucm_node_names
        self.num_nodes = len(ucm_node_names)
        self.parent = ucm_parent
        self.depth = ucm_depth
        self.leaf_names = leaf_names

        self._filename_to_path: Dict[str, str] = {}
        for subdir in ["images_tr", "images_test"]:
            full_subdir = os.path.join(image_root, subdir)
            if not os.path.isdir(full_subdir):
                continue
            for cls_name in sorted(os.listdir(full_subdir)):
                cls_dir = os.path.join(full_subdir, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith((".tif", ".tiff", "jpg", ".png")):
                        key = os.path.splitext(fname)[0]
                        self._filename_to_path[key] = os.path.join(cls_dir, fname)

        self.samples: List[Tuple[str, np.ndarray]] = []
        missing_images = 0
        for _, row in df.iterrows():
            key = str(row[r"IMAGE\LABEL"])
            full_path = self._filename_to_path.get(key)
            if full_path is None:
                missing_images += 1
                continue

            leaf_vector = {name: int(row[name]) for name in leaf_names}
            full_vector = np.zeros(self.num_nodes, dtype=np.int64)
            name_to_idx = {name: i for i, name in enumerate(self.node_names)}

            for name, val in leaf_vector.items():
                full_vector[name_to_idx[name]] = val

            for _ in range(3):
                for child, par in self.parent.items():
                    if full_vector[name_to_idx[child]] == 1:
                        full_vector[name_to_idx[par]] = 1

            self.samples.append((full_path, full_vector))

        if missing_images:
            print(f"Warning: {missing_images}/{len(df)} labeled images not found under {image_root}")
        print(f"AID-HMLC: loaded {len(self.samples)} images, reusing UCM's {self.num_nodes}-node hierarchy")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, labels = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(labels, dtype=torch.float32)
        return image, labels