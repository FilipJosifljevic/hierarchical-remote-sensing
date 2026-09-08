import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))
from dfc15 import DFC15HMLCDataset, DFC15_LEAF_NAMES
from src.utils.hierarchy import build_edge_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, required=True)
    parser.add_argument("--subset_threshold", type=float, default=0.95)
    args = parser.parse_args()

    print("=" * 70)
    print("CHECK 1: Loading dataset, file matching")
    print("=" * 70)
    dataset = DFC15HMLCDataset(
        image_root=args.image_root, labels_csv=args.labels_csv,
        subset_threshold=args.subset_threshold, transform=None,
    )
    print(f"\nLoaded {len(dataset)} samples successfully.")
    if len(dataset) == 0:
        print("FATAL: zero samples loaded -- stop here and fix file matching before continuing.")
        return

    print("\n" + "=" * 70)
    print("CHECK 2: Hierarchy structure (inspect by eye)")
    print("=" * 70)
    print(f"Node names ({dataset.num_nodes} total): {dataset.node_names}")
    print(f"Parent map: {dataset.parent}")
    print(f"Depth: {dataset.depth}")
    roots = [n for n in dataset.node_names if n not in dataset.parent]
    print(f"Roots ({len(roots)}): {roots}")

    print("\n" + "=" * 70)
    print("CHECK 3: Propagation correctness (full dataset)")
    print("=" * 70)
    name_to_idx = {n: i for i, n in enumerate(dataset.node_names)}
    violations = 0
    checked = 0
    for _, label_vec in dataset.samples:
        for child, par in dataset.parent.items():
            checked += 1
            if label_vec[name_to_idx[child]] == 1 and label_vec[name_to_idx[par]] == 0:
                violations += 1
    print(f"Checked {checked} (sample, edge) pairs across {len(dataset)} samples.")
    if violations == 0:
        print("PASSED: zero propagation violations -- every active leaf's parent is also active, "
              "exactly as expected (propagation is done at load time, this should ALWAYS be 0).")
    else:
        print(f"FAILED: {violations} propagation violations found -- this indicates a real bug "
              f"in the loader's propagation logic, not expected under any circumstances.")

    print("\n" + "=" * 70)
    print("CHECK 4: Per-node positive-sample counts")
    print("=" * 70)
    counts = {name: 0 for name in dataset.node_names}
    for _, label_vec in dataset.samples:
        for i, name in enumerate(dataset.node_names):
            if label_vec[i] == 1:
                counts[name] += 1
    for name, count in sorted(counts.items(), key=lambda x: x[1]):
        pct = 100 * count / len(dataset)
        flag = "  <-- LOW, check before using for K-shot or stratified splits" if count < 50 else ""
        print(f"  {name:<15} {count:>6} ({pct:>5.1f}%){flag}")

    print("\n" + "=" * 70)
    print("CHECK 5: build_edge_index sanity check")
    print("=" * 70)
    edge_index = build_edge_index(dataset.parent, dataset.node_names)
    print(f"edge_index shape: {tuple(edge_index.shape)} (expected [2, {len(dataset.parent)}])")
    assert edge_index.shape == (2, len(dataset.parent)), "edge_index shape mismatch!"
    print("PASSED: build_edge_index runs correctly on DFC-15's real parent/node_names")

    print("\n" + "=" * 70)
    print("CHECK 6: Full __getitem__ path on a real sample")
    print("=" * 70)
    img, label = dataset[0]
    print(f"Image: {img.size}, mode={img.mode}")
    print(f"Label shape: {label.shape}, dtype={label.dtype}")
    print(f"Label values: {label.tolist()}")

    print("\n" + "=" * 70)
    print("ALL CHECKS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()