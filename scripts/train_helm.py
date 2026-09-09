import argparse
import glob
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1])) 
from src.data.datasets.ucm import UCMHMLCDataset
from src.data.datasets.aid import AIDHMLCDataset
from src.data.datasets.dfc15 import DFC15HMLCDataset
from src.data.datamodule import split_train_test, sample_labeled_subset, SemiSupervisedUCM, make_semi_supervised_collate_fn
from src.data.transforms.byol_augmentation import TwoViewTransform
from src.utils.hierarchy import build_edge_index
from src.metrics.multilabel_metrics import compute_metrics
from src.models.helm.helm_model import HELM

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DATASET_SPLIT_DEFAULTS = {
    "ucm": (1667, 433),
    "aid": (2400, 600),
    "dfc15": (2674, 668),
}


def build_plain_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_dataset(args):
    if args.dataset == "ucm":
        image_root = args.image_root or "data/raw/UCMerced_LandUse/Images"
        return UCMHMLCDataset(image_root=image_root, transform=None)

    elif args.dataset == "aid":
        image_root = args.image_root or "data/raw/AID_Dataset"
        labels_csv = args.labels_csv or "data/raw/AID_Dataset/multilabel.csv"
        ucm_dataset = UCMHMLCDataset(image_root=args.ucm_image_root, transform=None)
        return AIDHMLCDataset(
            image_root=image_root, labels_csv=labels_csv,
            ucm_node_names=ucm_dataset.node_names, ucm_parent=ucm_dataset.parent,
            ucm_depth=ucm_dataset.depth, transform=None,
        )

    elif args.dataset == "dfc15":
        image_root = args.image_root or "data/raw/DFC15_multilabel"
        labels_csv = args.labels_csv or "data/raw/DFC15_multilabel/multilabel.csv"
        return DFC15HMLCDataset(
            image_root=image_root, labels_csv=labels_csv, transform=None,
            subset_threshold=args.subset_threshold,
        )

    else:
        raise ValueError(f"Unknown dataset '{args.dataset}'")


def strip_byol_for_inference(state_dict: dict) -> dict:
    return {k: v for k, v in state_dict.items() if not k.startswith("byol_branch.")}


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
    parser.add_argument("--dataset", type=str, default="ucm", choices=["ucm", "aid", "dfc15"])
    parser.add_argument("--image_root", type=str, default=None,
                         help="defaults per dataset if not set -- see build_dataset()")
    parser.add_argument("--labels_csv", type=str, default=None, help="for --dataset aid or dfc15")
    parser.add_argument("--ucm_image_root", type=str, default="data/raw/UCMerced_LandUse/Images",
                         help="only for --dataset aid -- needed to build the hierarchy AID reuses")
    parser.add_argument("--subset_threshold", type=float, default=0.95,
                         help="only for --dataset dfc15 -- see DFC15HMLCDataset docstring for "
                              "why 0.95 is the chosen default, not 1.0")
    parser.add_argument("--n_train", type=int, default=None, help="overrides the dataset's default split")
    parser.add_argument("--n_test", type=int, default=None, help="overrides the dataset's default split")
    parser.add_argument("--backbone_name", type=str, default="vit_small_patch16_224.dino",
                         help="timm ViT variant. DINO-pretrained variants (e.g. "
                              "vit_base_patch16_224.dino) were used for the attention "
                              "investigation (Sec 4.3-4.5) -- keep those checkpoints separate "
                              "from any run using a plain, supervised-pretrained variant "
                              "(e.g. vit_base_patch16_224, no .dino suffix), which is what the "
                              "original paper's appendix specifies.")
    parser.add_argument("--labeled_fraction", type=float, default=0.10,
                         help="1.0 = fully supervised; paper uses 0.01/0.05/0.10/0.25")
    parser.add_argument("--split_seed", type=int, default=42, help="train/test split seed -- keep FIXED across all runs/variants you compare")
    parser.add_argument("--run_seed", type=int, default=0, help="labeled-subset seed -- vary this across the paper's '3 runs per fraction'")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use_cosine_schedule", action="store_true",
                         help="enable cosine annealing LR schedule (T_max=epochs, no warmup), "
                              "matching the original paper's stated setup. Off by default so "
                              "existing reproduction runs (which used a constant LR) are unaffected.")
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
    parser.add_argument("--run_tag", type=str, default=None,
                         help="optional extra label included in checkpoint/results filenames, e.g. "
                              "'matched' for a paper-matching-config run, to keep it clearly "
                              "separate from your existing DINO-based checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    n_train_default, n_test_default = DATASET_SPLIT_DEFAULTS[args.dataset]
    n_train = args.n_train or n_train_default
    n_test = args.n_test or n_test_default

    print(f"Device: {args.device} | Dataset: {args.dataset}")
    print(f"Labeled fraction: {args.labeled_fraction} | split_seed={args.split_seed} | run_seed={args.run_seed}")
    print(f"Backbone: {args.backbone_name} | cosine_schedule={args.use_cosine_schedule} | "
          f"lambda_diversity={args.lambda_diversity}")

    base_dataset = build_dataset(args)
    num_labels = base_dataset.num_nodes
    edge_index = build_edge_index(base_dataset.parent, base_dataset.node_names)

    train_indices, test_indices = split_train_test(
        num_samples=len(base_dataset), n_train=n_train, n_test=n_test, seed=args.split_seed
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
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    scheduler = None
    if args.use_cosine_schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training loop ---
    final_metrics = None
    best_auprc = -1.0
    best_metrics = None
    best_ckpt_path = None

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

        if scheduler is not None:
            scheduler.step()

        avg = {k: v / n_batches for k, v in epoch_losses.items()}
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{args.epochs} -- loss: {avg['loss']:.4f} "
              f"(L_s: {avg['L_s']:.4f}, L_g: {avg['L_g']:.4f}, L_b: {avg['L_b']:.4f}, L_div: {avg['L_div']:.4f}) "
              f"[{n_batches_with_labels}/{n_batches} batches had labeled samples] [lr: {current_lr:.2e}]")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate(model, test_loader, args.device)
            print(f"  [eval @ epoch {epoch}] AUPRC: {metrics['auprc']:.4f}, "
                  f"Ranking Loss: {metrics['ranking_loss']:.4f}")
            final_metrics = {"epoch": epoch, **metrics}
            if metrics["auprc"] > best_auprc:
                best_auprc = metrics["auprc"]
                best_metrics = {"epoch": epoch, **metrics}
                # Overwrite a SINGLE best-checkpoint file each time -- we don't
                # want to accumulate one file per "new best so far" milestone.
                tag = f"_{args.run_tag}" if args.run_tag else ""
                best_ckpt_path = os.path.join(
                    args.checkpoint_dir,
                    f"helm_{args.dataset}_frac{args.labeled_fraction}_seed{args.run_seed}{tag}_best.pt",
                )
                torch.save(strip_byol_for_inference(model.state_dict()), best_ckpt_path)

        if epoch % args.checkpoint_every == 0 and epoch != args.epochs:
            tag = f"_{args.run_tag}" if args.run_tag else ""
            interim_path = os.path.join(
                args.checkpoint_dir,
                f"helm_{args.dataset}_frac{args.labeled_fraction}_seed{args.run_seed}{tag}_epoch{epoch}.pt",
            )
            torch.save(model.state_dict(), interim_path)
            print(f"  [checkpoint] saved to {interim_path}")

    tag = f"_{args.run_tag}" if args.run_tag else ""
    ckpt_path = os.path.join(
        args.checkpoint_dir,
        f"helm_{args.dataset}_frac{args.labeled_fraction}_seed{args.run_seed}{tag}_epoch{args.epochs}_final.pt",
    )
    torch.save(strip_byol_for_inference(model.state_dict()), ckpt_path)
    print(f"\nSaved final checkpoint to {ckpt_path}")
    print("  NOTE: this checkpoint is BYOL-stripped (no target network/projector/predictor). "
          "Load it with model.load_state_dict(state_dict, strict=False) -- NOT the default "
          "strict=True, which will raise a missing-keys error. Do not use this checkpoint "
          "with --resume_from.")

    interim_pattern = os.path.join(
        args.checkpoint_dir,
        f"helm_{args.dataset}_frac{args.labeled_fraction}_seed{args.run_seed}{tag}_epoch*.pt",
    )
    removed = 0
    for f in glob.glob(interim_pattern):
        if f != ckpt_path and f != best_ckpt_path:
            os.remove(f)
            removed += 1
    if removed:
        print(f"Removed {removed} interim checkpoint(s), kept final"
              f"{' and best' if best_ckpt_path and best_ckpt_path != ckpt_path else ''} only.")

    results = {
        "dataset": args.dataset,
        "labeled_fraction": args.labeled_fraction,
        "run_seed": args.run_seed,
        "backbone_name": args.backbone_name,
        "use_cosine_schedule": args.use_cosine_schedule,
        "lambda_diversity": args.lambda_diversity,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "n_train": n_train,
        "n_test": n_test,
        "checkpoint_path_final": ckpt_path,
        "checkpoint_path_best": best_ckpt_path,
        "final_metrics": final_metrics,
        "best_metrics": best_metrics,
    }
    results_path = os.path.join(
        args.checkpoint_dir,
        f"results_{args.dataset}_frac{args.labeled_fraction}_seed{args.run_seed}{tag}.json",
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results summary to {results_path}")


if __name__ == "__main__":
    main()