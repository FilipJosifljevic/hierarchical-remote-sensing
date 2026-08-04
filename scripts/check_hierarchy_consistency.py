import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data.datasets.ucm import UCMHMLCDataset
from src.data.datasets.aid import AIDHMLCDataset
from src.data.datamodule import split_train_test
from src.utils.hierarchy import build_edge_index
from src.models.maple.maple_model import MAPLE

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["ucm", "aid"])
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--labels_csv", type=str, default=None)
    parser.add_argument("--ucm_image_root", type=str, default="data/raw/UCMerced_LandUse/Images")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--backbone_name", type=str, default="vit_small_patch16_224.dino")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n_test", type=int, default=None, help="defaults: 433 ucm, 600 aid")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.dataset == "ucm":
        image_root = args.image_root or "data/raw/UCMerced_LandUse/Images"
        dataset = UCMHMLCDataset(image_root=image_root, transform=None)
        n_train_default, n_test_default = 1667, 433
    else:
        image_root = args.image_root or "data/raw/AID_Dataset"
        labels_csv = args.labels_csv or "data/raw/AID_Dataset/multilabel.csv"
        ucm_dataset = UCMHMLCDataset(image_root=args.ucm_image_root, transform=None)
        dataset = AIDHMLCDataset(
            image_root=image_root, labels_csv=labels_csv,
            ucm_node_names=ucm_dataset.node_names, ucm_parent=ucm_dataset.parent,
            ucm_depth=ucm_dataset.depth, transform=None,
        )
        n_train_default, n_test_default = 2400, 600
    n_test = args.n_test or n_test_default

    edge_index = build_edge_index(dataset.parent, dataset.node_names)
    name_to_idx = {name: i for i, name in enumerate(dataset.node_names)}

    _, test_indices = split_train_test(
        num_samples=len(dataset), n_train=n_train_default, n_test=n_test, seed=args.split_seed
    )

    model = MAPLE(
        node_names=dataset.node_names, parent=dataset.parent, depth=dataset.depth,
        edge_index=edge_index, backbone_name=args.backbone_name, pretrained=True,
    ).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Evaluating on {len(test_indices)} test images, threshold={args.threshold}\n")

    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    class TestSubset(torch.utils.data.Dataset):
        def __init__(self, base, indices, tfm):
            self.base, self.indices, self.tfm = base, indices, tfm
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            img, label = self.base[self.indices[i]]
            return self.tfm(img), label

    loader = DataLoader(TestSubset(dataset, test_indices, transform), batch_size=args.batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(args.device)
            probs = model.predict(x)
            all_preds.append((probs > args.threshold).float().cpu())
    all_preds = torch.cat(all_preds, dim=0)  # [N_test, M]
    N = all_preds.shape[0]

    edges = list(dataset.parent.items())  # (child, parent) pairs
    edge_results = []
    total_child_active = 0
    total_violations = 0

    for child, parent in edges:
        c_idx, p_idx = name_to_idx[child], name_to_idx[parent]
        child_active = all_preds[:, c_idx] == 1
        parent_inactive = all_preds[:, p_idx] == 0
        violation = child_active & parent_inactive

        n_child_active = child_active.sum().item()
        n_violations = violation.sum().item()
        edge_results.append((child, parent, n_child_active, n_violations))

        total_child_active += n_child_active
        total_violations += n_violations

    print(f"{'Child':<45} {'Parent':<45} {'Child Active':>12} {'Violations':>10} {'Rate':>8}")
    for child, parent, n_active, n_viol in sorted(edge_results, key=lambda r: -r[3]):
        rate = n_viol / n_active if n_active > 0 else 0.0
        flag = "  <-- high" if rate > 0.1 else ""
        print(f"{child:<45} {parent:<45} {n_active:>12} {n_viol:>10} {rate:>7.1%}{flag}")

    overall_rate = total_violations / total_child_active if total_child_active > 0 else 0.0
    print(f"\nOVERALL: {total_violations}/{total_child_active} "
          f"child-active instances violate parent consistency ({overall_rate:.1%})")

    samples_with_any_violation = 0
    for i in range(N):
        sample_pred = all_preds[i]
        has_violation = False
        for child, parent in edges:
            c_idx, p_idx = name_to_idx[child], name_to_idx[parent]
            if sample_pred[c_idx] == 1 and sample_pred[p_idx] == 0:
                has_violation = True
                break
        if has_violation:
            samples_with_any_violation += 1

    pct = samples_with_any_violation / N if N > 0 else 0.0
    print(f"\n{samples_with_any_violation}/{N} test images ({pct:.1%}) have AT LEAST ONE "
          f"hierarchy-consistency violation somewhere in their predicted label vector")
    print(f"(for reference: ground-truth label vectors have 0% violations, by construction)")


if __name__ == "__main__":
    main()