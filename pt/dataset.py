import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import random

class RemoteDataset(Dataset):
    def __init__(self, json_path, root_dir, transforms=None):
        """
        Args:
            json_path (str): path to JSON file with annotations.
            root_dir (str): project root directory.
            transforms (callable, optional): transformations applied to images and annotations.
        """
        self.root_dir = root_dir
        self.transforms = transforms

        with open(json_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        image_path = os.path.join(self.root_dir, entry['image'])
        image = Image.open(image_path).convert("RGB")

        # Собираем аннотации
        boxes = []
        labels = []

        for ann in entry.get('annotations', []):
            coords = ann['coordinates']
            x_center = coords['x']
            y_center = coords['y']
            width = coords['width']
            height = coords['height']

            boxes.append([x_center, y_center, width, height])
            labels.append(1)  # "controller" = class 1

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target

if __name__ == "__main__":

    test_transform = transforms.Compose([
        transforms.Resize((2016, 1512)), # decrease size 2x. save proportions
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    dataset = RemoteDataset(
        json_path="../dataset/train.json",
        root_dir="../",
        transforms=test_transform
    )

    dataloader = DataLoader(dataset, batch_size=3, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # Визуализация одного случайного батча из 3 изображений
    def denormalize(tensor):
        return tensor * 0.5 + 0.5

    images, targets = next(iter(dataloader))

    plt.figure(figsize=(18, 6))
    for i in range(3):
        img = denormalize(images[i]).permute(1, 2, 0).numpy()
        boxes = targets[i]['boxes'].numpy()

        ax = plt.subplot(1, 3, i + 1)
        ax.imshow(img)
        for box in boxes:
            box /= 2    # scale rectangle to resized photos size
            cx, cy, w, h = box.tolist()
            x_min = cx - w / 2
            y_min = cy - h / 2
            rect = plt.Rectangle((x_min, y_min), w, h, edgecolor='red', facecolor='none', linewidth=2)
            ax.add_patch(rect)

        ax.set_title(f"Sample {i}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()