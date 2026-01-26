import os
import time
import psutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

def get_optimal_device():
    """Checks for GPU availability and returns the device string."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"--- GPU Detected: {gpus[0].name} ---")
        # Prevent TF from pre-allocating all VRAM; only use what is needed
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
        return "/GPU:0"
    else:
        print("--- No GPU detected. Falling back to CPU. ---")
        return "/CPU:0"

def generate_dummy_images(num_images, shape=(224, 224, 3)):
    """Generates a batch of random images (Requirement #2)."""
    # Note: To use the 'Same Images' requirement, uncomment the seed in main.
    images = np.random.randint(0, 256, (num_images, *shape), dtype=np.uint8)
    return preprocess_input(images.astype(np.float32))

def run_performance_task(num_images=100):
    device_type = get_optimal_device()
    
    print("--- Initializing MobileNetV2 (Requirement #1) ---")
    
    # Force execution on the selected device
    with tf.device(device_type):
        model = MobileNetV2(weights='imagenet')
        images = generate_dummy_images(num_images)
        
        # GPU Warm-up: The first prediction is always slow; we don't want to time it.
        _ = model.predict(images[:1], verbose=0)
        
        print(f"--- Starting Inference on {num_images} images (Requirement #3) ---")
        
        # Resource Tracking (Requirement #4)
        process = psutil.Process(os.getpid())
        start_time = time.time()
        start_mem = process.memory_info().rss / (1024 * 1024) # MB

        # Prediction Loop
        for i in range(num_images):
            single_img = np.expand_dims(images[i], axis=0)
            _ = model.predict(single_img, verbose=0)

        end_time = time.time()
        end_mem = process.memory_info().rss / (1024 * 1024) # MB
        cpu_load = psutil.cpu_percent(interval=None)

    # Performance Calculations
    total_time = end_time - start_time
    avg_fps = num_images / total_time
    
    print("\n--- PERFORMANCE RESULTS ---")
    print(f"Hardware Used: {device_type}")
    print(f"Total Prediction Time: {total_time:.4f} seconds")
    print(f"Inference Speed: {avg_fps:.2f} FPS")
    print(f"Memory Delta: {end_mem - start_mem:.2f} MB")
    print(f"CPU Utilization: {cpu_load}%")
    print("---------------------------\n")

if __name__ == "__main__":
    # TASK REQUIREMENT:
    # Uncomment the seed below for the 'Same Images' test. 
    # Leave it commented for the 'Different Images' test.
    # np.random.seed(42)
    run_performance_task(num_images=100)