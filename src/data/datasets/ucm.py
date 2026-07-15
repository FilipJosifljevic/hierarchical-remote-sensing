"""
UCM-HMLC: hierarchical multi-label UC Merced Land Use dataset.

Images: original UC Merced Land Use images (21 folders x 100 images), e.g. from
    https://github.com/masakulaYOU/UCMerced_LandUse
Labels: hierarchical multi-label annotations released by Stoimchev et al., built on
    the CORINE Land Cover (CLC 2018) nomenclature, hosted at
    https://huggingface.co/datasets/marjandl/UCM-HMLC
    (raw file: UCM-HMLC.txt, tab-separated, 2101 rows: image_path + 30 binary label
    columns spanning leaf + intermediate + top levels of the hierarchy, no fixed
    column grouping by level -- see `infer_hierarchy` below.)

This dataset class:
  1. Loads the tsv label file (local cache, or downloads once from HF and caches it).
  2. Matches each row's image_path to the corresponding file under `image_root`
     (searching the flat UCMerced folder-of-folders layout).
  3. Exposes the FULL label vector (all 30 nodes: leaf + intermediate + top) per
     sample, since HELM-style approaches supervise every level of the hierarchy,
     not just the leaves.
  4. Infers parent/depth for every label node directly from the label matrix
     (see src/utils/hierarchy.py) instead of hand-encoding the tree, and exposes
     it via `self.parent` / `self.depth` / `self.node_names`.
"""
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

# The 17 canonical leaf-level multi-label attributes for UCM (Chaudhuri et al.),
# used here (rather than positional column slicing) to tell the hierarchy inference
# which columns are leaves vs. category (intermediate/top) nodes -- this matters
# because a sibling leaf can look like a false "parent" of another leaf if they
# happen to always co-occur in this small a dataset (see src/utils/hierarchy.py).
UCM_LEAF_NAMES = {
    "airplane", "bare-soil", "buildings", "cars", "chaparral", "court", "dock",
    "field", "grass", "mobile-home", "pavement", "sand", "sea", "ship",
    "storage tanks", "trees", "water",
}


def _download_label_file(dest_path: str) -> None:
    """Download the UCM-HMLC.txt annotation file from HuggingFace and cache it locally."""
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
        """
        Args:
            image_root: path to the folder containing the 21 UCM class subfolders
                        (e.g. ".../UCMerced_LandUse/Images").
            label_file: path to a locally cached UCM-HMLC.txt. If None, defaults to
                        `<image_root>/../UCM-HMLC.txt` and downloads it there if
                        `download=True` and it doesn't exist yet.
            transform: torchvision-style image transform.
            download: whether to auto-download the label file if missing.
        """
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

        # Match each row to an actual file on disk (UCM images are grouped in
        # per-class folders, e.g. "agricultural/agricultural00.tif").
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

        self.node_names = node_names  # all 30 label columns, in file order
        self.num_nodes = len(node_names)

        # Infer hierarchy (parent + depth per node) directly from the label matrix.
        # Restrict valid "ancestor" candidates to non-leaf (category) columns so that
        # two sibling leaves that happen to always co-occur (e.g. sand+sea in beach
        # images) can't be mistaken for a parent-child pair.
        unknown_leaves = UCM_LEAF_NAMES - set(node_names)
        if unknown_leaves:
            print(f"Warning: expected leaf names not found in data columns: {unknown_leaves}")
        category_names = set(node_names) - UCM_LEAF_NAMES

        # One confirmed manual override: 'field' is the ONLY leaf contributing to
        # 'Arable Land' / 'Agricultural Areas' in this dataset, so all three have
        # IDENTICAL positive-row sets (verified with scripts/diagnose_hierarchy.py --
        # 103/103 rows, zero difference either direction) and are indistinguishable
        # from data alone. We assert the intended chain from the CORINE nomenclature
        # instead of leaving all three stranded as separate roots.
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
        # header[0] is empty (row-index column), header[1] is "image_path", rest are labels
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
        labels = torch.tensor(labels, dtype=torch.float32)  # multi-label -> float for BCE
        return image, labels
