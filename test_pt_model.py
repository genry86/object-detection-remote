import torch
import json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision import transforms
from torchvision.ops import nms
from pt.model import SSDMobileNet  # Импортируем свою модель

def load_val_sample(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    sample = random.choice(data)
    image_path = sample["image"]
    image = Image.open(image_path).convert("RGB")

    ann = sample["annotations"][0]
    label = ann["label"]
    x = ann["coordinates"]["x"]
    y = ann["coordinates"]["y"]
    w = ann["coordinates"]["width"]
    h = ann["coordinates"]["height"]
    box = [x, y, w, h]  # cx, cy, w, h

    return image, torch.tensor(box, dtype=torch.float32), label

def prepare_image(image):
    transform = transforms.Compose([
        transforms.Resize((2016, 1512)),  # (H, W)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])
    image_tensor = transform(image)  # Tensor[3, 2016, 1512]
    image_tensor = image_tensor.unsqueeze(0)  # Tensor[1, 3, 2016, 1512]
    return image_tensor

def denormalize(tensor):
    # Обратное к Normalize(mean=0.5, std=0.5):  x * std + mean
    return tensor * 0.5 + 0.5

if __name__ == "__main__":
    device = torch.device("cpu")

    # Загружаем одно фото и аннотацию
    image_pil, gt_box, _ = load_val_sample("dataset/val.json")
    image_tensor = prepare_image(image_pil).to(device)

    # Инициализация модели
    model = SSDMobileNet()
    model.to(device)
    model.eval()

    with torch.no_grad():
        cls_logits, pred_box, _, _ = model(image_tensor)  # M=576 Output shapes: [1, M, 2], [1, M, 4], ...

    # Извлекаем предсказания
    logits = cls_logits.squeeze(0)            # Tensor[M, 2]
    boxes = pred_box.squeeze(0)               # Tensor[M, 4]

    # Преобразуем логиты в вероятности
    probs = torch.softmax(logits, dim=1)      # Tensor[M, 2]
    scores = probs[:, 1]                      # Вероятности класса "controller"

    # Конвертируем боксы из [cx, cy, w, h] → [x1, y1, x2, y2]
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)  # Tensor[M, 4]

    # Применяем NMS
    keep = nms(boxes_xyxy, scores, iou_threshold=0.5)

    # Визуализация
    image_tensor = denormalize(image_tensor)
    image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    plt.imshow(image_np)

    # GT бокс (уменьшенный в 4 раза, т.к. в тренировке был ресайз 4x)
    cx_gt, cy_gt, w_gt, h_gt = gt_box / 2.0
    x1_gt = cx_gt - w_gt / 2
    y1_gt = cy_gt - h_gt / 2
    rect_gt = patches.Rectangle((x1_gt, y1_gt), w_gt, h_gt, linewidth=2, edgecolor='red', facecolor='none')
    plt.gca().add_patch(rect_gt)
    plt.text(x1_gt, y1_gt - 5, "GT", color='red', fontsize=10)

    # Предсказанные боксы
    for idx in keep[:1]:  # покажем максимум 3 бокса
        score = scores[idx].item()
        x1, y1, x2, y2 = boxes_xyxy[idx]
        w_pred = x2 - x1
        h_pred = y2 - y1
        rect = patches.Rectangle((x1, y1), w_pred, h_pred, linewidth=2, edgecolor='blue', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(x1, y1 - 10, f"Pred: {score:.2f}", color='blue', fontsize=12)

    plt.axis("off")
    plt.title("Red: Ground truth, Blue: Predictions")
    plt.show()