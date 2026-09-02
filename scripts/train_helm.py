import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))  # project root, so `from src...` resolves
from src.data.datasets.ucm import UCMHMLCDataset
from src.data.datamodule import split_train_test, sample_labeled_subset, SemiSupervisedUCM, make_semi_supervised_collate_fn
from src.data.transforms.byol_augmentation import TwoViewTransform
from src.utils.hierarchy import build_edge_index
from src.metrics.multilabel_metrics import compute_metrics
from src.models.helm.helm_model import HELM

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_plain_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def evaluate(model: HELM, test_loader: DataLoader, device: str) -> dict:
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            probs = model.predict(x)
            all_probs.append(probs.cpu())
            all_targets.append(batch["targets"])  # test set is fully labeled -- see main()
    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return compute_metrics(all_probs, all_targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, default="data/raw/UCMerced_LandUse/Images")
    parser.add_argument("--backbone_name", type=str, default="vit_small_patch16_224.dino",
                         help="timm ViT variant -- vit_base_patch16_224.dino (paper's likely choice, "
                              "slow on CPU) or vit_small_patch16_224.dino (~4x less compute, still DINO-pretrained)")
    parser.add_argument("--labeled_fraction", type=float, default=0.10,
                         help="1.0 = fully supervised; paper uses 0.01/0.05/0.10/0.25")
    parser.add_argument("--split_seed", type=int, default=42, help="train/test split seed -- keep FIXED across all runs/variants you compare")
    parser.add_argument("--run_seed", type=int, default=0, help="labeled-subset seed -- vary this across the paper's '3 runs per fraction'")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_diversity", type=float, default=0.0,
                         help="weight for the attention diversity auxiliary loss (0.0 = disabled, "
                              "matching all prior training runs). Start small, e.g. 0.05-0.1 -- this "
                              "loss is more sensitive to learning rate than the others.")
    parser.add_argument("--eval_every", type=int, default=5, help="evaluate on test set every N epochs")
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--checkpoint_every", type=int, default=5,
                         help="save a checkpoint every N epochs, IN ADDITION to the final one -- "
                              "protects against losing progress if training is interrupted "
                              "(e.g. a Colab session disconnecting)")
    parser.add_argument("--resume_from", type=str, default=None,
                         help="path to a checkpoint to resume from (e.g. after a Colab disconnect). "
                              "NOTE: this only restores MODEL weights, not optimizer momentum state -- "
                              "training will resume with a 'cold' optimizer, which is not perfectly "
                              "identical to an uninterrupted run but is far better than restarting "
                              "from scratch.")
    parser.add_argument("--start_epoch", type=int, default=1,
                         help="which epoch number to resume FROM -- set this to (last saved epoch + 1) "
                              "when using --resume_from, so epoch numbering/checkpoint filenames stay consistent")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Labeled fraction: {args.labeled_fraction} | split_seed={args.split_seed} | run_seed={args.run_seed}")

    base_dataset = UCMHMLCDataset(image_root=args.image_root, transform=None)
    num_labels = base_dataset.num_nodes
    edge_index = build_edge_index(base_dataset.parent, base_dataset.node_names)

    train_indices, test_indices = split_train_test(
        num_samples=len(base_dataset), n_train=1667, n_test=433, seed=args.split_seed
    )

    if args.labeled_fraction < 1.0:
        labeled_indices = sample_labeled_subset(train_indices, fraction=args.labeled_fraction, seed=args.run_seed)
    else:
        labeled_indices = train_indices  # fully supervised

    print(f"Train: {len(train_indices)} total, {len(labeled_indices)} labeled "
          f"({100 * len(labeled_indices) / len(train_indices):.1f}%)")
    print(f"Test: {len(test_indices)} (fully labeled, fixed)")

    plain_transform = build_plain_transform()
    byol_transform = TwoViewTransform(image_size=224)

    train_dataset = SemiSupervisedUCM(
        base_dataset, train_indices, labeled_indices, plain_transform, byol_transform
    )
    # Test set: labeled_indices=None -> ALL of test_indices treated as labeled
    # (we need ground truth for every test sample to compute metrics).
    test_dataset = SemiSupervisedUCM(
        base_dataset, test_indices, None, plain_transform, byol_transform
    )

    collate_fn = make_semi_supervised_collate_fn(num_labels)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # --- Model ---
    model = HELM(edge_index=edge_index, num_labels=num_labels, backbone_name=args.backbone_name,
                 pretrained=True, lambda_diversity=args.lambda_diversity).to(args.device)
    if args.resume_from:
        state_dict = torch.load(args.resume_from, map_location=args.device)
        model.load_state_dict(state_dict)
        print(f"Resumed model weights from {args.resume_from} (optimizer state and epoch count NOT restored -- see --resume_from help)")
    # Filter to trainable params only: model.parameters() also includes the BYOL
    # target network's FROZEN parameters (requires_grad=False). Most optimizers
    # silently skip params with no gradient, so including them wouldn't crash --
    # but being explicit here avoids relying on that implicit behavior.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    # --- Training loop ---
    for epoch in range(args.start_epoch, args.epochs + 1):
        model.train()
        epoch_losses = {"loss": 0.0, "L_s": 0.0, "L_g": 0.0, "L_b": 0.0, "L_div": 0.0}
        n_batches = 0
        n_batches_with_labels = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            x = batch["x"].to(args.device)
            targets = batch["targets"].to(args.device)
            byol_view1 = batch["byol_view1"].to(args.device)
            byol_view2 = batch["byol_view2"].to(args.device)
            num_labeled = batch["num_labeled"]

            optimizer.zero_grad()
            out = model(x, targets, num_labeled=num_labeled, byol_view1=byol_view1, byol_view2=byol_view2)
            out["loss"].backward()
            optimizer.step()
            model.update_target_network()  # EMA update -- AFTER optimizer.step(), never before

            epoch_losses["loss"] += out["loss"].item()
            epoch_losses["L_s"] += out["L_s"].item()
            epoch_losses["L_g"] += out["L_g"].item()
            epoch_losses["L_b"] += out["L_b"].item()
            epoch_losses["L_div"] += out["L_div"].item()
            n_batches += 1
            if num_labeled > 0:
                n_batches_with_labels += 1

        avg = {k: v / n_batches for k, v in epoch_losses.items()}
        print(f"Epoch {epoch}/{args.epochs} -- loss: {avg['loss']:.4f} "
              f"(L_s: {avg['L_s']:.4f}, L_g: {avg['L_g']:.4f}, L_b: {avg['L_b']:.4f}, L_div: {avg['L_div']:.4f}) "
              f"[{n_batches_with_labels}/{n_batches} batches had labeled samples]")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate(model, test_loader, args.device)
            print(f"  [eval @ epoch {epoch}] AUPRC: {metrics['auprc']:.4f}, "
                  f"Ranking Loss: {metrics['ranking_loss']:.4f}")

        if epoch % args.checkpoint_every == 0 and epoch != args.epochs:
            interim_path = os.path.join(
                args.checkpoint_dir,
                f"helm_frac{args.labeled_fraction}_seed{args.run_seed}_epoch{epoch}.pt",
            )
            torch.save(model.state_dict(), interim_path)
            print(f"  [checkpoint] saved to {interim_path}")

    ckpt_path = os.path.join(
        args.checkpoint_dir,
        f"helm_frac{args.labeled_fraction}_seed{args.run_seed}_epoch{args.epochs}.pt",
    )
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nSaved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()