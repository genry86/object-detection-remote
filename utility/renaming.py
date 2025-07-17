import os
import uuid
from PIL import Image, ImageOps

# 3024×4032
target_width = 3024
target_height = 4032

folder_path = "../dataset/train/"

files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# 1 - set random names for images
for index, filename in enumerate(files, start=1):
    ext = os.path.splitext(filename)[1]

    if ext == ".png" or ext == ".jpg" or ext == ".jpeg" or ext == ".JPG":
        new_name = f"guild-{uuid.uuid4()}{ext}"
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)

# 2 - convert png to jpeg
for filename in os.listdir(folder_path):
    if filename.lower().endswith(".png"):
        png_path = os.path.join(folder_path, filename)
        jpg_name = os.path.splitext(filename)[0] + ".jpeg"
        jpg_path = os.path.join(folder_path, jpg_name)

        with Image.open(png_path) as img:
            rgb_img = img.convert("RGB")
            rgb_img.save(jpg_path, "JPEG")
            os.remove(png_path)

files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

prefix = "image"
ind = 1

# 3 - set ordered names for images and '.jpeg' extension
for index, filename in enumerate(files, start=1):
    ext = os.path.splitext(filename)[1]
    nex_ext = ext

    if ext == ".png" or ext == ".jpg" or ext == ".JPG" or ext == ".jpeg":
        if nex_ext == ".jpg" or nex_ext == ".jpeg" or nex_ext == ".JPG":
            nex_ext = ".jpeg"

        new_name = f"{prefix}{ind}{nex_ext}"
        ind += 1

        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        # print(f"{filename} → {new_name}")
 
# 4 - rotate images
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpeg')):
        path = os.path.join(folder_path, filename)

        img = Image.open(path)
        pil = ImageOps.exif_transpose(img)
        w, h = pil.size

        if w != target_width and h != target_height:
            print(f"filename: {filename} -- width: {w}, height: {h}")  # 3024×4032
            pil = pil.rotate(90, expand=True)

        pil.save(os.path.join(folder_path, f"{filename}"))

print("Completed")