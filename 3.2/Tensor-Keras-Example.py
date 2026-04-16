import tensorflow as tf
import numpy as np
import os

# Force TensorFlow to use CPU only due to older VGA 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Verify environment isolation and device usage
print("TensorFlow version:", tf.__version__)
print("Available devices:", tf.config.list_physical_devices())

# 1. Improved model definition using Input layer 
from tensorflow.keras import layers, models

model = models.Sequential(
    [
        layers.Input(shape=(20,)),  # Explicit input layer 
        layers.Dense(32, activation="relu"),  # Hidden layer 
        layers.Dense(2, activation="softmax"),  # Output layer
    ]
)

# 2. Compile the model with recommended settings 
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# 3. Model Summary for verification 
model.summary()

# 4. Dummy data for testing (10 samples, 20 features) 
data = np.random.rand(10, 20)

# 5. Execute prediction 
predictions = model.predict(data)
print("\nPredictions shape:", predictions.shape)
print("Predictions:\n", predictions)
