import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- avoids crashing in headless/SSH environments with no display
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data.datasets.ucm import UCMHMLCDataset
from src.models.helm.helm_model import HELM
from src.utils.attention_visualization import get_hierarchy_token_attention, upsample_attention_map
from src.utils.hierarchy import build_edge_index

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, default="data/raw/UCMerced_LandUse/Images")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="path to a trained HELM checkpoint (.pt). If omitted, "
                              "uses the encoder's pretrained (not fine-tuned) weights.")
    parser.add_argument("--backbone_name", type=str, default="vit_small_patch16_224.dino")
    parser.add_argument("--image_index", type=int, default=0, help="index into the full 2100-image dataset")
    parser.add_argument("--labels", type=str, nargs="*", default=None,
                         help="which hierarchy node names to visualize; defaults to "
                              "whichever labels are POSITIVE for the chosen image")
    parser.add_argument("--block_idx", type=int, default=-1, help="-1 = last transformer block")
    parser.add_argument("--output", type=str, default="outputs/attention_visualization.png")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    dataset = UCMHMLCDataset(image_root=args.image_root, transform=None)
    num_labels = dataset.num_nodes
    edge_index = build_edge_index(dataset.parent, dataset.node_names)

    raw_img, label_vector = dataset[args.image_index]

    model_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    display_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # NO normalize -- this copy is just for human viewing
    ])

    x = model_transform(raw_img).unsqueeze(0)  # [1, 3, 224, 224]
    display_img = display_transform(raw_img).permute(1, 2, 0).numpy()  # [224, 224, 3], displayable

    model = HELM(edge_index=edge_index, num_labels=num_labels, backbone_name=args.backbone_name, pretrained=True)
    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint given -- using pretrained (not HELM-fine-tuned) encoder weights.")
    model.eval()

    if args.labels is None:
        positive_indices = [i for i, v in enumerate(label_vector.tolist()) if v == 1]
        label_names_to_show = [dataset.node_names[i] for i in positive_indices]
        if not label_names_to_show:
            print("This image has no positive labels(?); falling back to first 4 hierarchy nodes.")
            label_names_to_show = dataset.node_names[:4]
    else:
        label_names_to_show = args.labels

    name_to_idx = {name: i for i, name in enumerate(dataset.node_names)}
    for name in label_names_to_show:
        assert name in name_to_idx, f"'{name}' is not a valid label name. Valid names: {dataset.node_names}"

    attn_maps = get_hierarchy_token_attention(model.encoder, x, block_idx=args.block_idx)  # [1, M, 14, 14]

    n = len(label_names_to_show)
    fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))
    axes[0].imshow(display_img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i, name in enumerate(label_names_to_show):
        idx = name_to_idx[name]
        attn_map = upsample_attention_map(attn_maps[0, idx], image_size=224).numpy()

        ax = axes[i + 1]
        ax.imshow(display_img)
        ax.imshow(attn_map, cmap="jet", alpha=0.5)
        is_positive = bool(label_vector[idx].item())
        ax.set_title(f"{name}\n({'present' if is_positive else 'absent'} in ground truth)")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()