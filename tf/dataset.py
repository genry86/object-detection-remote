import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from model import generate_anchors, IMAGE_SIZE, STRIDE, ANCHOR_SCALES, ANCHOR_RATIOS
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class RemoteDatasetTF(Sequence):
    def __init__(self, json_path, root_dir, batch_size=4, image_size=IMAGE_SIZE, shuffle=True, augmentation=None, **kwargs):
        super().__init__(**kwargs)
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augmentation = augmentation

        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

        self.stride = STRIDE
        self.fm_size = (image_size[0] // self.stride, image_size[1] // self.stride)
        self.num_anchors = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)

        self.anchors = generate_anchors(self.fm_size, ANCHOR_SCALES, ANCHOR_RATIOS, self.stride)
        self.anchors = self.anchors.numpy()
        self.anchors = self.anchors.reshape(self.fm_size[0], self.fm_size[1], self.num_anchors, 4)

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = []
        batch_cls = []
        batch_boxes = []

        for i in batch_indexes:
            entry = self.data[i]
            img_path = os.path.join(self.root_dir, entry["image"])
            image = Image.open(img_path).convert("RGB")
            image = image.resize(self.image_size)
            image_np = np.array(image).astype(np.float32) / 255.0
            if self.augmentation:
                image_np = self.augmentation(tf.convert_to_tensor(image_np)).numpy()

            H, W = self.image_size
            cls_map = np.zeros(self.fm_size + (self.num_anchors,), dtype=np.float32)
            box_map = np.zeros(self.fm_size + (self.num_anchors * 4,), dtype=np.float32)

            for ann in entry.get("annotations", []):
                x_gt = ann["coordinates"]["x"] / W
                y_gt = ann["coordinates"]["y"] / H
                w_gt = ann["coordinates"]["width"] / W
                h_gt = ann["coordinates"]["height"] / H

                cy = min(int(y_gt * self.fm_size[0]), self.fm_size[0] - 1)
                cx = min(int(x_gt * self.fm_size[1]), self.fm_size[1] - 1)

                for a in range(self.num_anchors):
                    xa, ya, wa, ha = self.anchors[cy, cx, a]

                    tx = (x_gt - xa) / wa
                    ty = (y_gt - ya) / ha
                    tw = np.log(w_gt / wa + 1e-6)
                    th = np.log(h_gt / ha + 1e-6)

                    cls_map[cy, cx, a] = 1.0
                    box_map[cy, cx, a * 4:(a + 1) * 4] = [tx, ty, tw, th]

                    break  # только один anchor для простоты

            batch_images.append(image_np)
            batch_cls.append(cls_map)
            batch_boxes.append(box_map)

        return tf.convert_to_tensor(batch_images), {
            "labels": tf.convert_to_tensor(batch_cls, dtype=tf.float32),
            "boxes": tf.convert_to_tensor(batch_boxes, dtype=tf.float32)
        }

if __name__ == "__main__":
    ds = RemoteDatasetTF(
        json_path="../dataset/train.json",
        root_dir="../",
        batch_size=1,
        shuffle=False
    )

    images, targets = ds[0]
    img = images[0].numpy()
    label_map = targets["labels"][0].numpy()
    box_map = targets["boxes"][0].numpy()

    print("Image shape:", img.shape)
    print("Label map shape:", label_map.shape)
    print("Box map shape:", box_map.shape)
