"""
PROGRAM: TensorFlow GPU Check and Minimal Inference Demo
AUTHOR:  Dr. Aliasghar Khavasi
DATE:    2026 - January

DESCRIPTION:
  A short TensorFlow script that:
    1) Detects available GPU devices and prints the result.
    2) Builds and compiles a minimal Keras Sequential model for a 20-feature input:
       - Dense(32, ReLU) → Dense(2, Softmax)
    3) Prints the model architecture summary.
    4) Generates a small dummy dataset (10 samples × 20 features) and runs a forward pass.
    5) Prints the predicted class probability vectors for the dummy samples.
"""

import tensorflow as tf
import numpy as np

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
print("GPUs:", gpus)

# Simple model definition
from tensorflow.keras import layers, models
model = models.Sequential([
    layers.Dense(32, activation='relu', input_shape=(20,)),
    layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.summary()

# Dummy data
data = np.random.rand(10, 20)
predictions = model.predict(data)
print("Predictions:", predictions)
