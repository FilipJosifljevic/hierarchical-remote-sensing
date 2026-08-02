import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data.datasets.ucm import UCMHMLCDataset, UCM_LEAF_NAMES
from src.data.datasets.aid import AIDHMLCDataset
from src.utils.hierarchy import build_edge_index
from src.models.maple.maple_model import MAPLE

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ucm", choices=["ucm", "aid"])
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--labels_csv", type=str, default=None, help="only for --dataset aid")
    parser.add_argument("--ucm_image_root", type=str, default="data/raw/UCMerced_LandUse/Images")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--backbone_name", type=str, default="vit_small_patch16_224.dino")
    parser.add_argument("--image_index", type=int, default=0)
    parser.add_argument("--output", type=str, default="outputs/maple_gamma_visualization.png")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.dataset == "ucm":
        image_root = args.image_root or "data/raw/UCMerced_LandUse/Images"
        dataset = UCMHMLCDataset(image_root=image_root, transform=None)
    else:
        image_root = args.image_root or "data/raw/AID_Dataset/images"
        labels_csv = args.labels_csv or "data/raw/AID_Dataset/multilabel.csv"
        ucm_dataset = UCMHMLCDataset(image_root=args.ucm_image_root, transform=None)
        dataset = AIDHMLCDataset(
            image_root=image_root, labels_csv=labels_csv,
            ucm_node_names=ucm_dataset.node_names, ucm_parent=ucm_dataset.parent,
            ucm_depth=ucm_dataset.depth, transform=None,
        )

    edge_index = build_edge_index(dataset.parent, dataset.node_names)

    model = MAPLE(
        node_names=dataset.node_names, parent=dataset.parent, depth=dataset.depth,
        edge_index=edge_index, backbone_name=args.backbone_name, pretrained=True,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    model_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    display_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    raw_img, label_vector = dataset[args.image_index]
    x = model_transform(raw_img).unsqueeze(0)
    display_img = display_transform(raw_img).permute(1, 2, 0).numpy()

    with torch.no_grad():
        z_hierarchy, z_patch = model.encoder(x)
        e = model.graph_refinement(z_hierarchy, model.edge_index)
        h_tilde, gamma = model.fusion(e, z_patch)
        logits = model.head(h_tilde)
        probs = torch.sigmoid(logits)

    gamma_per_node = gamma[0].mean(dim=-1).numpy()
    probs_np = probs[0].numpy()
    labels_np = label_vector.numpy() if hasattr(label_vector, "numpy") else np.array(label_vector)

    node_names = dataset.node_names
    order = np.argsort(gamma_per_node) 

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [1, 2]})

    axes[0].imshow(display_img)
    axes[0].set_title(f"Image {args.image_index}")
    axes[0].axis("off")

    colors = []
    ytick_labels = []
    for idx in order:
        name = node_names[idx]
        is_true = bool(labels_np[idx])
        ytick_labels.append(f"{name}{'  *' if is_true else ''}")
        colors.append("tab:red" if is_true else "tab:blue")

    y_pos = np.arange(len(node_names))
    axes[1].barh(y_pos, gamma_per_node[order], color=colors)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(ytick_labels, fontsize=8)
    axes[1].axvline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("gamma (0 = trusts vision, 1 = trusts hierarchy/semantic prior)")
    axes[1].set_title("Fusion gate value per label\n(red = ground-truth positive, marked with *)")
    axes[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {args.output}")

    print("\nPredicted vs ground truth (threshold 0.5):")
    for idx in range(len(node_names)):
        pred = "1" if probs_np[idx] > 0.5 else "0"
        true = "1" if labels_np[idx] else "0"
        flag = "" if pred == true else "  <-- MISMATCH"
        print(f"  {node_names[idx]:<45} pred={pred} (p={probs_np[idx]:.3f})  true={true}{flag}")


if __name__ == "__main__":
    main()