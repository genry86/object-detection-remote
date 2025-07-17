from PIL import Image, ExifTags
import os
import glob

train_folder_path = "../dataset/train"
val_folder_path = "../dataset/train"

image_paths = glob.glob(os.path.join(train_folder_path, '*.jpeg'))

for img_path in image_paths:
    img = Image.open(img_path)

    # Removing Exif tag and rotating photos properly
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = img._getexif()

        if exif is not None:
            orientation_value = exif.get(orientation, None)

            if orientation_value == 3:
                img = img.rotate(180, expand=True)
            elif orientation_value == 6:
                img = img.rotate(270, expand=True)
            elif orientation_value == 8:
                img = img.rotate(90, expand=True)

    except Exception as e:
        print(f"⚠️ Error EXIF handling {img_path}: {e}")

    img.save(img_path)
    print(f"✅ Proper photo saved: {img_path}")

print("\n🎯Finished")