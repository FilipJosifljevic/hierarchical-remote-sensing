import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data.datasets.ucm import UCMHMLCDataset
from src.data.datasets.aid import AIDHMLCDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["ucm", "aid"])
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--labels_csv", type=str, default=None)
    parser.add_argument("--ucm_image_root", type=str, default="data/raw/UCMerced_LandUse/Images")
    parser.add_argument("--high_violation_leaves", type=str, required=True,
                         help="comma-separated leaf names with high violation rates, "
                              "from check_hierarchy_consistency.py's output FOR THIS DATASET "
                              "(e.g. 'buildings,court,grass'). Do not reuse another dataset's list.")
    parser.add_argument("--zero_violation_leaves", type=str, required=True,
                         help="comma-separated leaf names with ~0%% violation rate, "
                              "from check_hierarchy_consistency.py's output FOR THIS DATASET")
    args = parser.parse_args()

    if args.dataset == "ucm":
        image_root = args.image_root or "data/raw/UCMerced_LandUse/Images"
        dataset = UCMHMLCDataset(image_root=image_root, transform=None)
    else:
        image_root = args.image_root or "data/raw/AID_Dataset"
        labels_csv = args.labels_csv or "data/raw/AID_Dataset/multilabel.csv"
        ucm_dataset = UCMHMLCDataset(image_root=args.ucm_image_root, transform=None)
        dataset = AIDHMLCDataset(
            image_root=image_root, labels_csv=labels_csv,
            ucm_node_names=ucm_dataset.node_names, ucm_parent=ucm_dataset.parent,
            ucm_depth=ucm_dataset.depth, transform=None,
        )

    name_to_idx = {name: i for i, name in enumerate(dataset.node_names)}
    roots = [n for n in dataset.node_names if n not in dataset.parent]
    root_indices = [name_to_idx[r] for r in roots]
    print(f"Root categories ({len(roots)}): {roots}\n")

    label_matrix = np.array([vec for _, vec in dataset.samples])  # [N, M]
    n_active_roots_per_image = label_matrix[:, root_indices].sum(axis=1)  # [N]

    leaf_names = [n for n in dataset.node_names if n not in dataset.parent.values()]

    results = []
    for leaf in leaf_names:
        leaf_idx = name_to_idx[leaf]
        mask = label_matrix[:, leaf_idx] == 1
        n_images = mask.sum()
        if n_images == 0:
            continue
        avg_active_roots = n_active_roots_per_image[mask].mean()
        results.append((leaf, n_images, avg_active_roots))

    overall_avg = n_active_roots_per_image.mean()
    print(f"Overall average active root categories per image (all {len(label_matrix)} images): {overall_avg:.3f}\n")

    print(f"{'Leaf':<20} {'N images':>10} {'Avg active roots (given leaf=1)':>35}")
    for leaf, n_images, avg_roots in sorted(results, key=lambda r: -r[2]):
        diff = avg_roots - overall_avg
        flag = "  <-- above average" if diff > 0.1 else ("  <-- below average" if diff < -0.1 else "")
        print(f"{leaf:<20} {n_images:>10} {avg_roots:>35.3f}{flag}")

    print("\n--- Direct comparison: flagged high-violation leaves vs. zero-violation leaves ---")
    high_violation_leaves = [x.strip() for x in args.high_violation_leaves.split(",")]
    zero_violation_leaves = [x.strip() for x in args.zero_violation_leaves.split(",")]

    results_dict = {leaf: avg for leaf, _, avg in results}
    high_vals = [results_dict[l] for l in high_violation_leaves if l in results_dict]
    zero_vals = [results_dict[l] for l in zero_violation_leaves if l in results_dict]

    print(f"High-violation leaves {high_violation_leaves}:")
    print(f"  avg active roots (given leaf=1): {np.mean(high_vals):.3f}")
    print(f"Zero-violation leaves (sample): {zero_violation_leaves}")
    print(f"  avg active roots (given leaf=1): {np.mean(zero_vals):.3f}")
    print(f"\nOverall dataset average: {overall_avg:.3f}")

    if np.mean(high_vals) > np.mean(zero_vals) + 0.05:
        print("\nSUPPORTS the ambiguity hypothesis: high-violation leaves appear in "
              "MORE categorically complex (multi-root) scenes than zero-violation leaves")
    elif np.mean(high_vals) < np.mean(zero_vals) - 0.05:
        print("\nCONTRADICTS the ambiguity hypothesis: high-violation leaves actually appear "
              "in LESS complex scenes -- the violations likely have a different cause")
    else:
        print("\nNO CLEAR DIFFERENCE: scene complexity doesn't obviously explain the violation pattern")


if __name__ == "__main__":
    main()