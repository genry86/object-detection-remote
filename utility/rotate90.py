import os
from PIL import Image

# Rotating photos from common iOS-camera landscape mode to portrait

train_folder_path = "../dataset/train"
val_folder_path = "../dataset/train"

for filename in os.listdir(train_folder_path):
    if filename.lower().endswith(('.png', '.jpeg')):
        path = os.path.join(train_folder_path, filename)

        img = Image.open(path)
        img = img.rotate(90, expand=True)
        img.save(os.path.join(train_folder_path, f"{filename}"))