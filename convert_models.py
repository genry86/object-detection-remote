import os
import coremltools as ct

import torch
from pt.model import SSDMobileNet

import tensorflow as tf
from tf.model import build_model, DecodeLayer

MODEL_PATH = os.path.join("training_model", "best.pth")
dst = os.path.join("training_model", "TVRemoteDetectionPytorch.mlpackage")

# PyTorch
model = SSDMobileNet(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Trace
input_tensor = torch.rand(2, 3, 2016, 1512)
traced_model = torch.jit.trace(model, input_tensor)

coreml_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="input", shape=input_tensor.shape)],
    convert_to="mlprogram"
)
coreml_model.save(dst)
#----------------------------

# TFLite
MODEL_PATH = os.path.join("training_model", "best.keras")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"DecodeLayer": DecodeLayer},
    compile=False
)
dst = os.path.join("training_model", "TVRemoteDetectionTFlite.tflite")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(dst, "wb") as f:
    f.write(tflite_model)