# HELM: Hierarchical Multi-Label Classification for Remote Sensing

Implementation of **HELM** (Hierarchical and Explicit Label Modeling with Graph Learning), a semi-supervised architecture for hierarchical multi-label classification of remote sensing imagery. Labels are organized into a tree, where the presence of a specific category (e.g., "cars") implies the presence of every category above it in the hierarchy (e.g., "Industrial, Commercial, and Transport Units" → "Artificial Surfaces"). HELM is designed to work with a small labeled fraction alongside a larger pool of unlabeled imagery.

## Architecture

- **Encoder**: ViT backbone with *M* hierarchy-specific tokens (one per node in the label hierarchy) replacing the standard single CLS token, each with its own learned positional embedding.
- **Classification branch**: pools all hierarchy tokens and predicts each category via a shared linear layer, trained with BCE on labeled examples.
- **Graph branch**: applies a two-layer GraphSAGE operator directly over the hierarchy token representations, propagating information along the hierarchy's parent-child edges.
- **BYOL branch**: a self-supervised branch operating on patch tokens, using an online/target network pair to extract training signal from unlabeled imagery.
- **Training objective**: `L = L_s + L_g + L_b + λ·L_div`, where `L_div` is an optional auxiliary attention-diversity loss encouraging hierarchy tokens to attend to different spatial regions of the image.

## Datasets

| Dataset | Images | Leaf Labels | Hierarchy | Split |
|---|---|---|---|---|
| **UCM-HMLC** | 2,100 | 17 | 30 nodes, 4 roots, 9 intermediate, 17 leaves (CORINE-derived) | 1667 / 433 |
| **AID** | 3,000 | 17 (identical to UCM) | Reuses UCM-HMLC's hierarchy directly | 2400 / 600 |
| **DFC-15** | 3,342 | 8 | 8 nodes, 2 levels (independently inferred — see note below) | 2673 / 669 |

- UCM-HMLC: [masakulaYOU/UCMerced_LandUse](https://github.com/masakulaYOU/UCMerced_LandUse); hierarchical labels auto-download on first use.
- AID (multi-label): [Hua-YS/AID-Multilabel-Dataset](https://github.com/Hua-YS/AID-Multilabel-Dataset)
- DFC-15 (multi-label): [Hua-YS/DFC15-Multilabel-Dataset](https://github.com/Hua-YS/DFC15-Multilabel-Dataset)

**Note on DFC-15's hierarchy**: the original paper constructs a 17-node, 3-level hierarchy for DFC-15 via CORINE Land Cover mapping (supplemented with LLM-assisted mapping for categories without a direct correspondence), but does not disclose the exact category names or parent-child structure used. This repository instead infers DFC-15's hierarchy directly from label co-occurrence (`subset_threshold=0.95`, see `src/utils/hierarchy.py`), yielding a shallower 8-node, 2-level structure. Results on DFC-15 are not directly comparable to the original paper's.

## Repository Structure

```
├── src/
│   ├── data/
│   │   ├── datasets/          # UCMHMLCDataset, AIDHMLCDataset, DFC15HMLCDataset
│   │   ├── datamodule.py      # train/test split, fractional labeled-subset sampling, collate fns
│   │   └── transforms/        # BYOL two-view augmentation
│   ├── models/
│   │   └── helm/               # encoder, classification/graph/BYOL branches, full model
│   ├── utils/
│   │   ├── hierarchy.py         # hierarchy inference (build_hierarchy) and graph utilities
│   │   └── attention_diversity.py  # auxiliary diversity loss
│   └── metrics/
│       └── multilabel_metrics.py  # AUPRC, Ranking Loss
├── scripts/
│   ├── train_helm.py            # HELM training entry point (multi-dataset)
│   ├── run_sweep_helm.py        # sweep driver (labeled fraction x seeds)
│   ├── merge_results.py         # combines per-run results JSON into one file
│   ├── run_ttest.py             # Welch's t-test between conditions
│   ├── check_dfc15.py           # DFC-15 loading/hierarchy verification
│   └── check_attention_routing.py  # per-token attention mass diagnostic
└── outputs/                     # checkpoints and results (gitignored)
```

## Setup

```bash
git clone https://github.com/FilipJosifljevic/hierarchical-remote-sensing.git
cd hierarchical-remote-sensing
pip install timm torch_geometric scikit-learn tqdm torchvision pandas pillow numpy
```

Datasets are expected under `data/raw/` — see [Datasets](#datasets) for sources and the folder layout each loader expects. UCM-HMLC's hierarchical label file auto-downloads on first use; no manual step needed.

## Usage

**Train HELM** (`--dataset` selects `ucm`, `aid`, or `dfc15`):

```bash
python3 scripts/train_helm.py --dataset aid \
    --backbone_name vit_base_patch16_224 --use_cosine_schedule \
    --epochs 100 --batch_size 16 --labeled_fraction 0.10
```

**Run a full sweep** (multiple seeds, one or more labeled fractions):

```bash
python3 scripts/run_sweep_helm.py --dataset aid --fractions 0.10 --seeds 0,1,2 \
    --backbone_name vit_base_patch16_224 --use_cosine_schedule --epochs 100 --batch_size 16
```

**Verify a dataset before training on it:**

```bash
python3 scripts/check_dfc15.py --image_root data/raw/DFC15_Dataset --labels_csv data/raw/DFC15_Dataset/multilabel.csv
```

**Check per-token attention routing on a trained checkpoint:**

```bash
python3 scripts/check_attention_routing.py --dataset aid \
    --checkpoint outputs/checkpoints/helm_aid_frac0.1_seed0_epoch100_final.pt --image_index 0
```

## Citing the Original Work

```bibtex
@article{helm2026,
  title={HELM: Hierarchical and Explicit Label Modeling with Graph Learning for Multi-Label Image Classification},
  note={REO Workshop, NeurIPS 2025},
  eprint={2603.11783}
}
```