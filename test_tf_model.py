import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_hub as hub
import numpy as np
from tf.utils import test_model
from tf.model import build_model
import os
import json
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

IMAGE_SIZE = (320, 320)

def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (320, 320))
    img = img / 255.0
    return img

if __name__ == "__main__":
    model_dir = "training_model"
    best_model_path = os.path.join(model_dir, "best.keras")
    image_size = (320, 320)
    json_path = "dataset/val.json"

    model = build_model()
    model.load_weights(best_model_path)

    with open(json_path, 'r') as f:
        data = json.load(f)

    sample = random.choice(data)
    image_path = os.path.join(sample['image'])
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
    gcx, gcy, gw, gh = gt_box
    gx1 = int((gcx - gw / 2) * W)
    gy1 = int((gcy - gh / 2) * H)
    gx2 = int((gcx + gw / 2) * W)
    gy2 = int((gcy + gh / 2) * H)

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.gca().add_patch(plt.Rectangle((gx1, gy1), gx2 - gx1, gy2 - gy1,
                                      edgecolor='red', linewidth=2, fill=False, label="GT"))

    input_tensor = preprocess_image(image_path)
    input_tensor = tf.expand_dims(input_tensor, axis=0)

    preds = model(input_tensor)
    if isinstance(preds, dict):
        cls_preds = preds["labels"]  # [B, H, W, A]
        box_preds = preds["final_boxes"]  # [B, N, 4] in normalized coords
    else:
        raise ValueError("Unexpected model output structure")

    boxes = box_preds[0].numpy()  # (N, 4)
    scores = cls_preds[0].numpy().reshape(-1)  # (H*W*A,)
    best_idx = np.argmax(scores)

    x, y, w, h = boxes[best_idx]
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)
    w = np.clip(w, 0, 1)
    h = np.clip(h, 0, 1)

    # H, W = IMAGE_SIZE
    x1 = (x - w / 2) * W
    y1 = (y - h / 2) * H
    w_pixel = w * W
    h_pixel = h * H

    rect = patches.Rectangle((x1, y1), w_pixel, h_pixel, linewidth=2, edgecolor='lime', facecolor='none')
    plt.gca().add_patch(rect)

    plt.legend(["GT", "Pred"])
    # plt.axis('off')
    plt.show()