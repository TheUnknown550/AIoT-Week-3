import os
import time
import psutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

def generate_dummy_images(num_images, shape=(224, 224, 3)):
    """Generates a batch of random images."""
    # Using random uniform data between 0 and 255 to simulate RGB pixels
    images = np.random.randint(0, 256, (num_images, *shape), dtype=np.uint8)
    return preprocess_input(images.astype(np.float32))

def run_inference_test(num_images=100):
    print("--- Initializing MobileNetV2 ---")
    model = MobileNetV2(weights='imagenet')
    
    # Generate dummy data
    images = generate_dummy_images(num_images)
    
    print(f"--- Starting Inference on {num_images} images ---")
    
    # Measure Resource Usage Start
    process = psutil.Process(os.getpid())
    start_cpu_percent = psutil.cpu_percent(interval=None)
    start_mem = process.memory_info().rss / (1024 * 1024) # MB
    start_time = time.time()

    # The Prediction Loop
    # We predict one by one to better simulate edge device processing overhead
    for i in range(num_images):
        single_img = np.expand_dims(images[i], axis=0)
        _ = model.predict(single_img, verbose=0)

    # Measure Resource Usage End
    end_time = time.time()
    end_mem = process.memory_info().rss / (1024 * 1024) # MB
    end_cpu_percent = psutil.cpu_percent(interval=None)
    
    # Calculations
    total_time = end_time - start_time
    avg_fps = num_images / total_time
    
    print("\n--- RESULTS ---")
    print(f"Total Time: {total_time:.4f} seconds")
    print(f"Inference Speed: {avg_fps:.2f} FPS")
    print(f"Memory Used: {end_mem - start_mem:.2f} MB (Delta)")
    print(f"Final Memory Footprint: {end_mem:.2f} MB")
    print(f"CPU Load Change: {end_cpu_percent - start_cpu_percent}%")
    print("----------------\n")

if __name__ == "__main__":
    # Note: Run this twice as per your task instructions:
    # 1. In 'Power Saver' mode (unplugged)
    # 2. In 'High Performance' mode
    run_inference_test(num_images=100)