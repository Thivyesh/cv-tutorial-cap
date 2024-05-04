import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from glob import glob
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from PIL import Image

class InferDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        # super(InferDataset, self,).__init__()

        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_dir)
    
    def __getitem__(self, idx):
        img_path = self.img_dir[idx]
        image = read_image(img_path)
        # image = image.numpy()
        if self.transform:
            image = self.transform(image)
        return image
    