from torch.utils.data import Dataset
from torchvision.io import read_image
from typing import Any
import torch

class InferDataset(Dataset):
    def __init__(self, img_dir: str, transform: None|Any = None) -> None:
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_dir)
    
    def __getitem__(self, idx) -> torch.Tensor:
        img_path = self.img_dir[idx]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image
    