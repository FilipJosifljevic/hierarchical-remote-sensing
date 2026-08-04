import argparse
import os
import sys
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data.datasets.ucm import UCMHMLCDataset, UCM_LEAF_NAMES
from src.data.datasets.aid import AIDHMLCDataset
from src.data.datamodule import split_train_test
from src.data.kshot_sampler import sample_kshot_indices, KShotSubset
from src.utils.hierarchy import build_edge_index
from src.metrics.multilabel_metrics import compute_metrics
from src.models.maple.maple_model import MAPLE

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_dataset(args):
    if args.dataset == "ucm":
        dataset = UCMHMLCDataset(image_root=args.image_root, transform=None)
        leaf_names = list(UCM_LEAF_NAMES)
    elif args.dataset == "aid":
        ucm_dataset = UCMHMLCDataset(image_root=args.ucm_image_root, transform=None)
        dataset = AIDHMLCDataset(
            image_root=args.image_root,
            labels_csv=args.labels_csv,
            ucm_node_names=ucm_dataset.node_names,
            ucm_parent=ucm_dataset.parent,
            ucm_depth=ucm_dataset.depth,
            transform=None,
        )
        leaf_names = list(UCM_LEAF_NAMES)
    else:
        raise ValueError(f"Unknown dataset '{args.dataset}' -- expected 'ucm' or 'aid'")
    return dataset, leaf_names


def evaluate(model: MAPLE, test_loader: DataLoader, device: str) -> dict:
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for x, targets in test_loader:
            x = x.to(device)
            probs = model.predict(x)
            all_probs.append(probs.cpu())
            all_targets.append(targets)
    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return compute_metrics(all_probs, all_targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["ucm", "aid"])
    parser.add_argument("--image_root", type=str, default=None,
                         help="for ucm: path to UCMerced Images folder. for aid: path to AID_Dataset/images")
    parser.add_argument("--labels_csv", type=str, default=None, help="only for --dataset aid")
    parser.add_argument("--ucm_image_root", type=str, default="data/raw/UCMerced_LandUse/Images",
                         help="only for --dataset aid -- needed to build the hierarchy AID reuses")
    parser.add_argument("--backbone_name", type=str, default="vit_small_patch16_224.dino")
    parser.add_argument("--sentence_model_name", type=str, default="all-mpnet-base-v2")
    parser.add_argument("--graph_num_layers", type=int, default=2)

    parser.add_argument("--k", type=int, default=8, choices=[4, 8, 12, 16],
                         help="shots per leaf class, per the paper's few-shot protocol")
    parser.add_argument("--split_seed", type=int, default=42,
                         help="train/test split seed -- keep FIXED across all runs/variants you compare")
    parser.add_argument("--run_seed", type=int, default=0,
                         help="K-shot sampling seed -- vary this across the paper's '3 runs per K'")
    parser.add_argument("--n_train", type=int, default=None,
                         help="override train split size (defaults: 1667 for ucm, 2400 for aid, per HELM's Table 3)")
    parser.add_argument("--n_test", type=int, default=None,
                         help="override test split size (defaults: 433 for ucm, 600 for aid)")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/checkpoints_maple")
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.image_root is None:
        args.image_root = "data/raw/UCMerced_LandUse/Images" if args.dataset == "ucm" else "data/raw/AID_Dataset/images"
    if args.labels_csv is None and args.dataset == "aid":
        args.labels_csv = "data/raw/AID_Dataset/multilabel.csv"
    if args.n_train is None:
        args.n_train = 1667 if args.dataset == "ucm" else 2400
    if args.n_test is None:
        args.n_test = 433 if args.dataset == "ucm" else 600

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"Device: {args.device} | Dataset: {args.dataset} | K={args.k} | run_seed={args.run_seed}")

    dataset, leaf_names = build_dataset(args)
    edge_index = build_edge_index(dataset.parent, dataset.node_names)

    train_indices, test_indices = split_train_test(
        num_samples=len(dataset), n_train=args.n_train, n_test=args.n_test, seed=args.split_seed
    )

    kshot_indices, per_class_counts = sample_kshot_indices(
        dataset, train_indices, leaf_names, k=args.k, seed=args.run_seed
    )
    print(f"K-shot training set: {len(kshot_indices)} images (union across {len(leaf_names)} leaf classes)")
    print(f"Test set: {len(test_indices)} images (fixed, fully labeled)")

    transform = build_transform()
    train_dataset = KShotSubset(dataset, kshot_indices, transform=transform)
    test_dataset = KShotSubset(dataset, test_indices, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = MAPLE(
        node_names=dataset.node_names,
        parent=dataset.parent,
        depth=dataset.depth,
        edge_index=edge_index,
        backbone_name=args.backbone_name,
        pretrained=True,
        sentence_model_name=args.sentence_model_name,
        graph_num_layers=args.graph_num_layers,
    ).to(args.device)

    if args.resume_from:
        model.load_state_dict(torch.load(args.resume_from, map_location=args.device))
        print(f"Resumed model weights from {args.resume_from} (optimizer state and epoch count NOT restored)")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    best_auprc = -1.0
    best_epoch = -1
    best_metrics = None
    final_metrics = None

    for epoch in range(args.start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            x = x.to(args.device)
            targets = targets.to(args.device)

            optimizer.zero_grad()
            out = model(x, targets)
            out["loss"].backward()
            optimizer.step()

            epoch_loss += out["loss"].item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch}/{args.epochs} -- loss: {avg_loss:.4f}")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate(model, test_loader, args.device)
            print(f"  [eval @ epoch {epoch}] AUPRC: {metrics['auprc']:.4f}, "
                  f"Ranking Loss: {metrics['ranking_loss']:.4f}")
            final_metrics = {"epoch": epoch, **metrics}
            if metrics["auprc"] > best_auprc:
                best_auprc = metrics["auprc"]
                best_epoch = epoch
                best_metrics = {"epoch": epoch, **metrics}

        if epoch % args.checkpoint_every == 0 and epoch != args.epochs:
            interim_path = os.path.join(
                args.checkpoint_dir,
                f"maple_{args.dataset}_k{args.k}_seed{args.run_seed}_epoch{epoch}.pt",
            )
            torch.save(model.state_dict(), interim_path)
            print(f"  [checkpoint] saved to {interim_path}")

    ckpt_path = os.path.join(
        args.checkpoint_dir,
        f"maple_{args.dataset}_k{args.k}_seed{args.run_seed}_epoch{args.epochs}.pt",
    )
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nSaved checkpoint to {ckpt_path}")

    results = {
        "dataset": args.dataset,
        "k": args.k,
        "run_seed": args.run_seed,
        "epochs": args.epochs,
        "checkpoint_path": ckpt_path,
        "final_metrics": final_metrics,
        "best_metrics": best_metrics,
    }
    results_path = os.path.join(
        args.checkpoint_dir,
        f"results_{args.dataset}_k{args.k}_seed{args.run_seed}.json",
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results summary to {results_path}")


if __name__ == "__main__":
    main()