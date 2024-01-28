import tensorflow as tf

# Check available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set GPU memory growth to avoid allocating all memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Your TensorFlow code here
        print("GPU is available and configured.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found.")

