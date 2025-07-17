import tensorflow as tf
from tensorflow.keras import layers, models
from keras.saving import register_keras_serializable

# Константы
IMAGE_SIZE = (320, 320)
STRIDE = 32
ANCHOR_SCALES = [32, 64, 128]
ANCHOR_RATIOS = [0.5, 1.0, 2.0]


def generate_anchors(fm_size, scales, ratios, stride):
    anchors = []
    for y in range(fm_size[0]):
        for x in range(fm_size[1]):
            cy = (y + 0.5) * stride / IMAGE_SIZE[0]
            cx = (x + 0.5) * stride / IMAGE_SIZE[1]
            for scale in scales:
                for ratio in ratios:
                    h = scale * (ratio ** 0.5) / IMAGE_SIZE[0]
                    w = scale / (ratio ** 0.5) / IMAGE_SIZE[1]
                    anchors.append([cx, cy, w, h])
    anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)
    anchors = tf.reshape(anchors, [fm_size[0], fm_size[1], len(scales) * len(ratios), 4])
    return anchors

@register_keras_serializable()
class DecodeLayer(layers.Layer):
    def __init__(self, anchors, **kwargs):
        super().__init__(**kwargs)
        self.anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)

    def call(self, box_outputs):
        B = tf.shape(box_outputs)[0]
        H = tf.shape(box_outputs)[1]
        W = tf.shape(box_outputs)[2]
        box_outputs = tf.reshape(box_outputs, [B, H, W, -1, 4])

        xa, ya, wa, ha = tf.split(self.anchors, 4, axis=-1)
        tx, ty, tw, th = tf.split(box_outputs, 4, axis=-1)

        x = tx * wa + xa
        y = ty * ha + ya
        w = wa * tf.exp(tw)
        h = ha * tf.exp(th)

        boxes = tf.concat([x, y, w, h], axis=-1)
        boxes = tf.reshape(boxes, [B, -1, 4])
        return boxes

    def get_config(self):
        config = super().get_config()
        config.update({
            "anchors": self.anchors.numpy().tolist()
        })
        return config

    @classmethod
    def from_config(cls, config):
        anchors = tf.convert_to_tensor(config.pop("anchors"), dtype=tf.float32)
        return cls(anchors=anchors, **config)


def build_model(input_shape=(320, 320, 3)):
    inputs = layers.Input(shape=input_shape, name="image")
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, input_tensor=inputs, weights="imagenet"
    )

    x = base_model.output  # [B, 10, 10, 1280]
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)

    num_anchors = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)

    # Классификатор
    cls_output = layers.Conv2D(num_anchors, 1, padding="same", activation="sigmoid", name="labels")(x)

    # Смещения
    box_output = layers.Conv2D(num_anchors * 4, 1, padding="same", activation="linear", name="boxes")(x)

    fm_size = (input_shape[0] // STRIDE, input_shape[1] // STRIDE)
    anchors = generate_anchors(fm_size, ANCHOR_SCALES, ANCHOR_RATIOS, STRIDE)

    decode = DecodeLayer(anchors, name="final_boxes")
    decoded_boxes = decode(box_output)

    model = models.Model(inputs=inputs, outputs={
        "labels": cls_output,
        "boxes": box_output,
        "final_boxes": decoded_boxes
    })
    return model