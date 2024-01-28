import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Your data (x, y)
# ...

# Define your model
model = Sequential([
    Dense(128, activation='relu', input_shape=(input_size,)),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Specify the CPU cores to use (6 and 8)
os.environ["CUDA_VISIBLE_DEVICES"] = "6,8"

# Create a distribution strategy for using 2 CPU cores
strategy = tf.distribute.experimental.CentralStorageStrategy()

# Wrap the model with the strategy
with strategy.scope():
    # Train the model
    model.fit(x, y, epochs=10, batch_size=32)

    # Evaluate the model
    model.evaluate(test_x, test_y)

