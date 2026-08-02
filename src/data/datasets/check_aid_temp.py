import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torchvision import transforms

from ucm import UCMHMLCDataset
from aid import AIDHMLCDataset

print("Loading UCM dataset (source of the hierarchy AID will reuse)...")
ucm_dataset = UCMHMLCDataset(image_root="../../../data/raw/UCMerced_LandUse/Images", transform=None)
print(f"UCM: {len(ucm_dataset)} images, {ucm_dataset.num_nodes} hierarchy nodes\n")

print("Loading AID dataset...")
aid_dataset = AIDHMLCDataset(
    image_root="../../../data/raw/AID_Dataset/images",
    labels_csv="../../../data/raw/AID_Dataset/multilabel.csv",
    ucm_node_names=ucm_dataset.node_names,
    ucm_parent=ucm_dataset.parent,
    ucm_depth=ucm_dataset.depth,
    transform=None,
)

print(f"\nAID: {len(aid_dataset)} images loaded")
assert len(aid_dataset) > 0, "No images loaded -- check image_root/labels_csv paths"
assert aid_dataset.num_nodes == 30, f"Expected 30 hierarchy nodes (reused from UCM), got {aid_dataset.num_nodes}"

roots = [n for n in aid_dataset.node_names if n not in aid_dataset.parent]
expected_roots = {"Artificial Surfaces", "Forest and Semi-Natural Areas", "Agricultural Areas", "Water Bodies"}
print(f"Roots: {roots}")
assert set(roots) == expected_roots, f"Roots don't match UCM's expected 4 -- hierarchy reuse may be broken: {roots}"
print("Hierarchy reuse matches UCM's structure exactly: OK")

name_to_idx = {name: i for i, name in enumerate(aid_dataset.node_names)}
propagation_errors = 0
samples_checked = 0
for img_path, label_vec in aid_dataset.samples[:500]: 
    samples_checked += 1
    for name in aid_dataset.leaf_names:
        idx = name_to_idx[name]
        if label_vec[idx] == 1:
            node = name
            while node in aid_dataset.parent:
                node = aid_dataset.parent[node]
                if label_vec[name_to_idx[node]] != 1:
                    propagation_errors += 1
                    print(f"  PROPAGATION ERROR: leaf '{name}' active but ancestor '{node}' is not, "
                          f"in sample {img_path}")

print(f"\nChecked propagation correctness on {samples_checked} samples")
assert propagation_errors == 0, f"{propagation_errors} propagation errors found -- label vectors are inconsistent"
print("Label propagation (leaf -> intermediate -> top) is consistent: OK")

print("\nPer-leaf positive-sample counts:")
label_matrix = np.array([vec for _, vec in aid_dataset.samples])
for name in aid_dataset.leaf_names:
    idx = name_to_idx[name]
    count = int(label_matrix[:, idx].sum())
    flag = "  <-- suspiciously low/zero, check this label" if count < 5 else ""
    print(f"  {name}: {count}{flag}")

print("\nLoading one image with a real transform...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
aid_dataset.transform = transform
img, label = aid_dataset[0]
print(f"Image shape: {img.shape}")
print(f"Label shape: {label.shape}, dtype: {label.dtype}")
assert img.shape == (3, 224, 224)
assert label.shape == (30,)

print("\nALL CHECKS PASSED")