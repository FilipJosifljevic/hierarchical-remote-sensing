from typing import Dict, List

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

def build_hierarchy_descriptions(node_names: List[str], parent: Dict[str, str]) -> Dict[str, str]:
    descriptions = {}
    for name in node_names:
        if name in parent:
            descriptions[name] = f"The category '{name}' which is a subcategory of {parent[name]}"
        else:
            descriptions[name] = f"The category '{name}', a top-level category"
    return descriptions

class HierarchicalSemanticInitializer(nn.Module):
    def __init__(
            self,
            node_names: List[str],
            parent: Dict[str, str],
            target_dim: int,
            sentence_model_name: str = "all-mpnet-base-v2"
    ):
        super().__init__()
        self.node_names = node_names
        self.M = len(node_names)

        descriptions = build_hierarchy_descriptions(node_names, parent)
        sentences = [descriptions[name] for name in node_names]

        sentence_model = SentenceTransformer(sentence_model_name)
        with torch.no_grad():
            raw_embeddings = sentence_model.encode(
                sentences, convert_to_tensor=True, show_progress_bar=False
            ).cpu()

        sentence_dim = raw_embeddings.shape[1]
        self.register_buffer("raw_embeddings", raw_embeddings.clone().float())

        self.projection = nn.Sequential(
            nn.Linear(sentence_dim, target_dim),
            nn.LayerNorm(target_dim)
        )

    def forward(self) -> torch.Tensor:
        return self.projection(self.raw_embeddings)

