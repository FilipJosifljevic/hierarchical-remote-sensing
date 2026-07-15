from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


def infer_parents(
    label_matrix: np.ndarray,
    label_names: Sequence[str],
    candidate_ancestor_names: "set[str] | None" = None,
) -> Dict[str, List[str]]:
    n_labels = label_matrix.shape[1]
    positive_sets = [set(np.nonzero(label_matrix[:, i])[0].tolist()) for i in range(n_labels)]

    ancestors: Dict[str, List[str]] = {name: [] for name in label_names}
    for i, j in itertools.permutations(range(n_labels), 2):
        if not positive_sets[i]:
            continue  # label never appears; skip
        if candidate_ancestor_names is not None and label_names[j] not in candidate_ancestor_names:
            continue  # j is not allowed to be anyone's ancestor (e.g. it's a sibling leaf)
        # i is a child of j if every row with i=1 also has j=1, and j is strictly broader
        if positive_sets[i] <= positive_sets[j] and positive_sets[j] != positive_sets[i]:
            ancestors[label_names[i]].append(label_names[j])
    return ancestors


def direct_parent(
    ancestors: Dict[str, List[str]], positive_counts: Dict[str, int]
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    parent: Dict[str, str] = {}
    ties: Dict[str, List[str]] = {}
    for label, ancs in ancestors.items():
        if not ancs:
            continue
        min_count = min(positive_counts[a] for a in ancs)
        closest = sorted(a for a in ancs if positive_counts[a] == min_count)
        if len(closest) == 1:
            parent[label] = closest[0]
        else:
            ties[label] = closest
            # deterministic fallback so depth/tree-building can still proceed:
            # pick alphabetically first candidate, but flag it via `ties`.
            parent[label] = closest[0]
    return parent, ties


def compute_depth(parent: Dict[str, str], label_names: Sequence[str]) -> Dict[str, int]:
    """Depth 0 = root (no parent). Depth increases by 1 per level down."""
    depth: Dict[str, int] = {}

    def _depth(label: str, seen: Tuple[str, ...] = ()) -> int:
        if label in depth:
            return depth[label]
        if label not in parent:
            depth[label] = 0
            return 0
        if label in seen:
            raise ValueError(f"Cycle detected in inferred hierarchy at '{label}': {seen}")
        d = 1 + _depth(parent[label], seen + (label,))
        depth[label] = d
        return d

    for name in label_names:
        _depth(name)
    return depth


def build_hierarchy(
    label_matrix: np.ndarray,
    label_names: Sequence[str],
    category_names: "set[str] | None" = None,
    manual_overrides: "Dict[str, str] | None" = None,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    ancestors = infer_parents(label_matrix, label_names, candidate_ancestor_names=category_names)
    positive_counts = {name: int(label_matrix[:, i].sum()) for i, name in enumerate(label_names)}
    parent, ties = direct_parent(ancestors, positive_counts)
    if ties:
        print(f"Warning: {len(ties)} label(s) had an ambiguous immediate parent "
              f"(tied candidates, picked alphabetically first): {ties}")
    if manual_overrides:
        for label, forced_parent in manual_overrides.items():
            parent[label] = forced_parent
    depth = compute_depth(parent, label_names)
    return parent, depth


def build_edge_index(parent: Dict[str, str], node_names: Sequence[str]) -> torch.Tensor:
    idx = {name: i for i, name in enumerate(node_names)}
    src, dst = [], []
    for child, par in parent.items():
        src.append(idx[child])
        dst.append(idx[par])
    return torch.tensor([src, dst], dtype=torch.long)


def batch_edge_index(edge_index: "torch.Tensor", num_nodes: int, batch_size: int) -> torch.Tensor:
    offsets = (torch.arange(batch_size, device=edge_index.device) * num_nodes).view(-1, 1, 1)  # [B,1,1]
    tiled = edge_index.unsqueeze(0) + offsets  # [B, 2, E]
    return tiled.permute(1, 0, 2).reshape(2, -1)  # [2, B*E]