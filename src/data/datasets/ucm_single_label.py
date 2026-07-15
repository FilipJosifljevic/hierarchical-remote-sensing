import os
from PIL import Image
from typing import Callable, Optional, List, Tuple, Dict

import torch
from torch.utils.data import Dataset

class UCMDataset(Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = root
        self.transform = transform

        self.classes = sorted([d for d in os.listdir(root) if os.listdir(os.path.join(root, d))])

        self.class_to_idx: Dict[str, int] = {
            cls_name: idx for idx, cls_name in enumerate(self.classes)
        }

        self.samples: List[Tuple[str, int]] = []

        for cls_name in self.classes:
            class_dir = os.path.join(root, cls_name)

            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith((".tif", ".tiff", ".jpg", ".png")):
                    path = os.path.join(class_dir, file_name)
                    label = self.class_to_idx[cls_name]
                    self.samples.append((path, label))

    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label
