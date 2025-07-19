import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from torchvision.ops import nms
import math
import matplotlib.pyplot as plt
import json
import random
from PIL import Image
from torchvision import models, transforms
import matplotlib.patches as patches
import os
from pt.utils import generate_anchors, decode_boxes

# -----------------------------
# SSD Detection Head
# -----------------------------
class SSDHead(nn.Module):
    def __init__(self, in_channels=512, num_anchors=3, num_classes=2):
        super().__init__()
        self.cls_head = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1) # 512, 6 , (3,3), 1, 1
        self.reg_head = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)           # 512, 12, (3,3), 1, 1
        self.num_anchors = num_anchors
        self.num_classes = num_classes

    def forward(self, x):
        # 2, 512, 16, 12
        cls_logits = self.cls_head(x)  # (2, 6, 16, 12)
        bbox_reg = self.reg_head(x)    # (2, 12, 16, 12)

        B, _, H, W = cls_logits.shape
        cls = cls_logits.permute(0, 2, 3, 1)            # (2, 6, 16, 12) -> (2, 16, 12, 6)
        cls = cls.reshape(B, -1, self.num_classes)      # (2, 16, 12, 6) -> (2, 576, 2)
        bbox = bbox_reg.permute(0, 2, 3, 1)             # (2, 12, 16, 12) -> (2, 16, 12, 12)
        bbox = bbox.reshape(B, -1, 4)                   # (2, 16, 12, 12) -> (2, 576, 4)
        return cls, bbox

# -----------------------------
# SSD Model
# -----------------------------
class SSDMobileNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = mobilenet_v2(weights='IMAGENET1K_V1').features
        self.extra = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1), # (2, 1280, 32, 24) -> (2, 256, 32, 24)
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # (2, 256, 32, 24)  -> (2, 512, 16, 12)
            nn.ReLU()
        )

        self.num_anchors = 3
        self.ratios = [1.5, 0.75, 3.0]
        self.ssd_head = SSDHead(in_channels=512, num_anchors=self.num_anchors, num_classes=num_classes)

    def forward(self, x):
        B = x.size(0)  # x - (2, 3, 2016, 1512)  B=2
        features = self.backbone(x)   # Tensor[2, 1280, 32, 24]
        features = self.extra(features)   # Tensor[2, 512, 16, 12]

        # Вычисляем размерность feature map и генерируем anchors
        _, _, H, W = features.shape # 16, 12
        # Вычисляем stride и scale динамически
        stride_h = x.shape[2] // H
        stride_w = x.shape[3] // W
        # assert stride_h == stride_w, "Stride по высоте и ширине не совпадают"
        stride = int((stride_h + stride_w) / 2)
        base_size = stride * 5  # 315

        # Вычисляем размерность feature map и генерируем anchors
        anchors = generate_anchors((H, W), base_size, self.ratios, stride)  # 16*12*1*3 = 576 -> Tensor[576, 4]
        anchors = anchors.to(x.device)     # Tensor[576, 4] -> перенесён на устройство
        anchors = anchors.unsqueeze(0)     # Tensor[1, 576, 4] -> добавляем ось батча
        anchors = anchors.expand(B, -1, -1)  # Tensor[2, 576, 4] -> один набор anchor'ов для всех примеров

        cls_logits, bbox_deltas = self.ssd_head(features)  # features: Tensor[2, 512, 16, 12] -> cls_logits: Tensor[2, 576, 2], bbox_reg: Tensor[2, 576, 4]
        boxes = decode_boxes(bbox_deltas, anchors)      # Tensor[B, 576, 4] — предсказанные боксы

        return cls_logits, boxes, bbox_deltas, anchors

# -----------------------------
# Test section
# -----------------------------
if __name__ == "__main__":
    model = SSDMobileNet(num_classes=2)
    dummy = torch.randn(2, 3, 2016, 1512)  # Tensor[2, 3, 2016, 1512]
    cls_logits, boxes, _, anchors = model(dummy)
    print("cls_logits:", cls_logits.shape)   # -> [2, 576, 2]
    print("boxes:", boxes.shape)             # -> [2, 576, 4]
    print("anchors:", anchors.shape)         # -> [2, 576, 4]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # === Step 1: Load JSON and select random photo ===
    json_path = "../dataset/val.json"
    with open(json_path, 'r') as f:
        data = json.load(f)

    sample = random.choice(data)  # Random entry from validation json
    rel_img_path = sample["image"]  # relative path to photo
    abs_img_path = os.path.join("../", rel_img_path)  # absolute path

    # === Step 2: Load image and annotations ===
    image = Image.open(abs_img_path).convert("RGB")
    annotation = sample["annotations"][0]
    cx, cy = annotation["coordinates"]["x"], annotation["coordinates"]["y"]
    w, h = annotation["coordinates"]["width"], annotation["coordinates"]["height"]

    # === Step 3: Image transformation ===
    transform = transforms.Compose([
        transforms.Resize((2016, 1512)),  # (height, width)
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)  # Tensor[1, 3, 2016, 1512]
    resized_image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Tensor[2016, 1512, 3]

    # === Step 4: Create model and get anchors ===
    model = SSDMobileNet().to(device)
    model.eval()
    with torch.no_grad():
        _, _, _, anchors = model(image_tensor)  # anchors: Tensor[1, NumAnchors, 4]
        anchors = anchors.squeeze(0).cpu()  # Tensor[NumAnchors, 4]

    # === Шаг 5: Визуализация ===
    fig, ax = plt.subplots(1)
    ax.imshow(resized_image_np)

    # --- Отрисовать ground-truth бокс (уменьшен в 4 раза) ---
    x1 = (cx - w / 2) / 2
    y1 = (cy - h / 2) / 2
    w4 = w / 2
    h4 = h / 2
    rect = patches.Rectangle((x1, y1), w4, h4, linewidth=2, edgecolor='green', facecolor='none')
    ax.add_patch(rect)
    ax.text(x1, y1 - 10, "GT box", color='green', fontsize=10)

    # --- Отрисовать анкоры (первые 30) ---
    for a in anchors[:10]:
        ax.add_patch(patches.Rectangle((a[0], a[1]), a[2], a[3], linewidth=1.0, edgecolor='red', facecolor='none'))

    plt.title("Anchors and GT box")
    plt.axis('off')
    plt.show()
