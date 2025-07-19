import os
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split

from dataset import RemoteDatasetTF
from model import build_model
from EpochTracker import EpochTracker

# === Losses ===
def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        return tf.reduce_mean(weight * cross_entropy)
    return loss_fn

def smooth_l1_loss(y_true, y_pred, delta=1.0):
    diff = tf.abs(y_true - y_pred)
    less_than_delta = tf.cast(diff < delta, tf.float32)
    loss = less_than_delta * 0.5 * tf.pow(diff, 2) + (1 - less_than_delta) * (diff - 0.5)
    return tf.reduce_mean(loss)

# === Parameters ===
model_dir = "../training_model"
os.makedirs(model_dir, exist_ok=True)

best_model_path = os.path.join(model_dir, "best.keras")
last_model_path = os.path.join(model_dir, "last.keras")
epoch_path = os.path.join("epoch.txt")

epochs = 500
batch_size = 32
image_size = (320, 320)
json_path = "../dataset/train.json"
root_dir = "../"

# === Augmentations ===
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.0417),
    tf.keras.layers.RandomContrast(0.3)
])

# === Загрузка и сплит ===
with open(json_path, "r") as f:
    full_data = json.load(f)

train_data, val_data = train_test_split(full_data, test_size=0.1, random_state=42)

with open("train_split.json", "w") as f:
    json.dump(train_data, f)
with open("val_split.json", "w") as f:
    json.dump(val_data, f)

# === Датасеты ===
train_dataset = RemoteDatasetTF(
    json_path="train_split.json",
    root_dir=root_dir,
    batch_size=batch_size,
    image_size=image_size,
    shuffle=True,
    augmentation=augmentation
)

val_dataset = RemoteDatasetTF(
    json_path="val_split.json",
    root_dir=root_dir,
    batch_size=batch_size,
    image_size=image_size,
    shuffle=False
)

# === Модель ===
model = build_model(input_shape=image_size + (3,))



# === Восстановление модели ===
initial_epoch = 0
if os.path.exists(last_model_path):
    print("Restoring last model...")
    model.load_weights(last_model_path)

    if os.path.exists(epoch_path):
        with open(epoch_path, "r") as f:
            initial_epoch = int(f.read())
        print(f"Resuming from epoch {initial_epoch}")

# === Компиляция ===
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss={
        "labels": focal_loss(),
        "boxes": smooth_l1_loss
    }
)

# === Колбэки ===
callbacks = [
    EpochTracker(epoch_path),
    tf.keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(best_model_path, monitor="val_loss", save_best_only=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(last_model_path, save_best_only=False)
]

# === Тренировка ===
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    initial_epoch=initial_epoch,
    callbacks=callbacks
)