import tensorflow as tf
import json
import random
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (320, 320))
    img = img / 255.0
    return img

# === Model testing ===
def test_model(model, json_path="../dataset/val.json", base_dir="../"):
    with open(json_path, 'r') as f:
        data = json.load(f)

    sample = random.choice(data)
    image_path = os.path.join(base_dir, sample['image'])
    ann = sample['annotations'][0]['coordinates']

    # Преобразуем аннотацию к [cx, cy, w, h] и нормализуем
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    gt_box = [
        ann['x'] / W,
        ann['y'] / H,
        ann['width'] / W,
        ann['height'] / H,
    ]

    input_tensor = preprocess_image(image_path)
    input_tensor = tf.expand_dims(input_tensor, axis=0)  # [1, 320, 320, 3]

    preds = model(input_tensor)
    class_pred = preds["labels"]
    bbox_pred = preds["boxes"].numpy()[0]  # [cx, cy, w, h]

    # Переводим bbox_pred в [x1, y1, x2, y2] в координатах исходного изображения
    cx, cy, bw, bh = bbox_pred
    x1 = int((cx - bw / 2) * W)
    y1 = int((cy - bh / 2) * H)
    x2 = int((cx + bw / 2) * W)
    y2 = int((cy + bh / 2) * H)

    # Аналогично для gt
    gcx, gcy, gw, gh = gt_box
    gx1 = int((gcx - gw / 2) * W)
    gy1 = int((gcy - gh / 2) * H)
    gx2 = int((gcx + gw / 2) * W)
    gy2 = int((gcy + gh / 2) * H)

    # Визуализация
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.gca().add_patch(plt.Rectangle((gx1, gy1), gx2 - gx1, gy2 - gy1,
                                      edgecolor='red', linewidth=2, fill=False, label="GT"))
    plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                      edgecolor='blue', linewidth=2, fill=False, label="Pred"))
    plt.legend(["GT", "Pred"])
    plt.title(f"Predicted prob: {class_pred.numpy()[0][0]:.2f}")
    plt.axis('off')
    plt.show()