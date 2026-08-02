import os
import sys
from pathlib import Path
from typing import Callable, Optional, List, Tuple, Dict, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

sys.path.append(str(Path(__file__).resolve().parents[3]))  # project root, so `from src.utils...` resolves
from src.utils.hierarchy import build_hierarchy

UCM_HMLC_HF_URL = (
    "https://huggingface.co/datasets/marjandl/UCM-HMLC/resolve/main/UCM-HMLC.txt"
)

UCM_LEAF_NAMES = {
    "airplane", "bare-soil", "buildings", "cars", "chaparral", "court", "dock",
    "field", "grass", "mobile-home", "pavement", "sand", "sea", "ship",
    "storage tanks", "trees", "water",
}


def _download_label_file(dest_path: str) -> None:
    import urllib.request

    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    print(f"Downloading UCM-HMLC labels to {dest_path} ...")
    urllib.request.urlretrieve(UCM_HMLC_HF_URL, dest_path)


class UCMHMLCDataset(Dataset):
    def __init__(
        self,
        image_root: str,
        label_file: Optional[str] = None,
        transform: Optional[Callable] = None,
        download: bool = True,
    ):
        self.image_root = image_root
        self.transform = transform

        if label_file is None:
            label_file = os.path.join(os.path.dirname(os.path.normpath(image_root)), "UCM-HMLC.txt")
        if not os.path.exists(label_file):
            if not download:
                raise FileNotFoundError(
                    f"Label file not found at {label_file} and download=False. "
                    f"Download it manually from {UCM_HMLC_HF_URL}"
                )
            _download_label_file(label_file)

        image_paths, label_matrix, node_names = self._parse_label_file(label_file)

        self._index_images_on_disk()

        self.samples: List[Tuple[str, np.ndarray]] = []
        missing = 0
        for img_name, labels in zip(image_paths, label_matrix):
            full_path = self._filename_to_path.get(img_name)
            if full_path is None:
                missing += 1
                continue
            self.samples.append((full_path, labels))
        if missing:
            print(f"Warning: {missing}/{len(image_paths)} labeled images not found under {image_root}")

        self.node_names = node_names 
        self.num_nodes = len(node_names)
        unknown_leaves = UCM_LEAF_NAMES - set(node_names)
        if unknown_leaves:
            print(f"Warning: expected leaf names not found in data columns: {unknown_leaves}")
        category_names = set(node_names) - UCM_LEAF_NAMES

        manual_overrides = {"field": "Arable Land", "Arable Land": "Agricultural Areas"}

        self.parent, self.depth = build_hierarchy(
            label_matrix, node_names, category_names=category_names, manual_overrides=manual_overrides
        )
        roots = [n for n in node_names if n not in self.parent]
        print(f"Inferred hierarchy: {len(roots)} root node(s), {self.num_nodes} nodes total.")
        print(f"Roots: {roots}")

    def _parse_label_file(self, label_file: str) -> Tuple[List[str], np.ndarray, List[str]]:
        with open(label_file, "r") as f:
            lines = [line.rstrip("\n") for line in f]

        header = lines[0].split("\t")
        node_names = header[2:]

        image_paths = []
        rows = []
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.split("\t")
            image_paths.append(parts[1])
            rows.append([int(x) for x in parts[2:]])

        label_matrix = np.array(rows, dtype=np.int64)
        assert label_matrix.shape[1] == len(node_names), (
            f"Column mismatch: {label_matrix.shape[1]} label columns vs "
            f"{len(node_names)} header names"
        )
        return image_paths, label_matrix, node_names

    def _index_images_on_disk(self) -> None:
        """Build a lookup from bare filename (e.g. 'agricultural00.tif') -> full path."""
        self._filename_to_path: Dict[str, str] = {}
        for cls_name in sorted(os.listdir(self.image_root)):
            cls_dir = os.path.join(self.image_root, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".tif", ".tiff", ".jpg", ".png")):
                    self._filename_to_path[fname] = os.path.join(cls_dir, fname)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, labels = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(labels, dtype=torch.float32)
        return image, labels
