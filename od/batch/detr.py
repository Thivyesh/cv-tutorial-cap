
import glob
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import ImageDraw
from transformers import pipeline
from model import InferDataset


class ObjectDetectionPipeline:
    def __init__(self, img_dir: str, batch_size: int = 64) -> None:
        """
        Initializes the object detection pipeline.

        Args:
            img_dir (str): Directory containing images.
            batch_size (int): Batch size for DataLoader (default is 64).
        """
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((512, 512))
        ])
        self.infer_data = InferDataset(img_dir=self.img_dir, transform=self.transform)
        self.infer_loader = DataLoader(
            self.infer_data,
            batch_size=self.batch_size,
            shuffle=False,
            # num_workers=4,
            # pin_memory=True
        )
        self.pipe = pipeline(model="facebook/detr-resnet-50", task="object-detection")

    def infer_and_display(self) -> None:
        """
        Performs object detection on images from the directory and displays the results.
        """
        imgs = next(iter(self.infer_loader))
        img = imgs[4].squeeze()

        with torch.no_grad():
            T = transforms.ToPILImage()
            pil_img = T(img)
            predictions = self.pipe(pil_img)

        draw = ImageDraw.Draw(pil_img)
        for prediction in predictions:
            box = prediction["box"]
            label = prediction["label"]
            score = prediction["score"]

            xmin, ymin, xmax, ymax = box.values()
            draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
            draw.text((xmin, ymin), f"{label}: {round(score, 2)}", fill="white")

        plt.imshow(pil_img)
        plt.show()

# Usage example:
img_dir = glob.glob("./imgs/*jpg")
obj_detection = ObjectDetectionPipeline(img_dir=img_dir)
obj_detection.infer_and_display()
