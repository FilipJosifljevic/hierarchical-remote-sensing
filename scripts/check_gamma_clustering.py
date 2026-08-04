import argparse
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data.datasets.ucm import UCMHMLCDataset
from src.data.datasets.aid import AIDHMLCDataset
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
    parser.add_argument("--label", type=str, required=True, help="e.g. 'cars'")
    parser.add_argument("--n_examples", type=int, default=8, help="per group (positive/negative)")
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

    edge_index = build_edge_index(dataset.parent, dataset.node_names)
    name_to_idx = {name: i for i, name in enumerate(dataset.node_names)}
    if args.label not in name_to_idx:
        raise ValueError(f"'{args.label}' not in node_names: {dataset.node_names}")
    label_idx = name_to_idx[args.label]

    model = MAPLE(
        node_names=dataset.node_names, parent=dataset.parent, depth=dataset.depth,
        edge_index=edge_index, backbone_name=args.backbone_name, pretrained=True,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    gate_shapes = [tuple(p.shape) for p in model.fusion.gate.parameters()]
    print(f"Fusion gate parameter shapes: {gate_shapes}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    def get_gamma_for_label(img_idx):
        raw_img, _ = dataset[img_idx]
        x = transform(raw_img).unsqueeze(0)
        with torch.no_grad():
            z_hierarchy, z_patch = model.encoder(x)
            e = model.graph_refinement(z_hierarchy, model.edge_index)
            _, gamma = model.fusion(e, z_patch)
        return gamma[0, label_idx]

    positive_indices, negative_indices = [], []
    for i, (_, label_vec) in enumerate(dataset.samples):
        if label_vec[label_idx] == 1 and len(positive_indices) < args.n_examples:
            positive_indices.append(i)
        elif label_vec[label_idx] == 0 and len(negative_indices) < args.n_examples:
            negative_indices.append(i)
        if len(positive_indices) >= args.n_examples and len(negative_indices) >= args.n_examples:
            break

    print(f"Positive ({args.label} present) indices: {positive_indices}")
    print(f"Negative ({args.label} absent) indices: {negative_indices}")

    pos_gammas = [get_gamma_for_label(i) for i in positive_indices]
    neg_gammas = [get_gamma_for_label(i) for i in negative_indices]

    def cos_sim(a, b):
        a_n = F.normalize(a - a.mean(), dim=0)
        b_n = F.normalize(b - b.mean(), dim=0)
        return (a_n * b_n).sum().item()

    pos_pos = [cos_sim(pos_gammas[i], pos_gammas[j])
               for i in range(len(pos_gammas)) for j in range(i + 1, len(pos_gammas))]
    neg_neg = [cos_sim(neg_gammas[i], neg_gammas[j])
               for i in range(len(neg_gammas)) for j in range(i + 1, len(neg_gammas))]
    pos_neg = [cos_sim(pos_gammas[i], neg_gammas[j])
               for i in range(len(pos_gammas)) for j in range(len(neg_gammas))]

    within_avg = (statistics.mean(pos_pos) + statistics.mean(neg_neg)) / 2
    between_avg = statistics.mean(pos_neg)
    print(f"\nWithin positive group: mean sim = {statistics.mean(pos_pos):.4f}")
    print(f"Within negative group: mean sim = {statistics.mean(neg_neg):.4f}")
    print(f"Between groups: mean sim = {statistics.mean(pos_neg):.4f}")
    print(f"\nWithin-group average: {within_avg:.4f}")
    print(f"Between-group average: {between_avg:.4f}")
    if within_avg > between_avg + 0.02:
        print("Gamma clusters meaningfully by actual label presence -- genuine instance-level adaptivity")
    else:
        print("No meaningful clustering by label presence")


if __name__ == "__main__":
    main()