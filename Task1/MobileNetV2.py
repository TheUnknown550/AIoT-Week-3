import os
import time
import psutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# Force TensorFlow to log device placement to see if it's actually using the GPU
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 

def get_optimal_device():
    """Checks for NVIDIA GPU and configures it."""
    print("--- Checking for Hardware Acceleration ---")
    
    # List all physical devices for debugging
    physical_devices = tf.config.list_physical_devices()
    print(f"System sees devices: {physical_devices}")

    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # RTX 4050 specific optimization: Memory growth is vital for laptop GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Identify which GPU (should be the RTX 4050)
            print(f"--- SUCCESS: GPU Detected: {gpus[0].name} ---")
            return "/GPU:0"
        except Exception as e:
            print(f"--- GPU found but initialization failed: {e} ---")
            return "/CPU:0"
    else:
        print("--- No GPU detected by TensorFlow. Check CUDA/cuDNN installation. ---")
        return "/CPU:0"

def generate_dummy_images(num_images, shape=(224, 224, 3)):
    """Generates a batch of random images (Requirement #2)."""
    # Using random NumPy array as requested
    images = np.random.randint(0, 256, (num_images, *shape), dtype=np.uint8)
    return preprocess_input(images.astype(np.float32))

def run_performance_task(num_images=100):
    device_type = get_optimal_device()
    
    # Requirement #1: Initialize MobileNetV2
    print(f"--- Initializing MobileNetV2 on {device_type} ---")
    
    with tf.device(device_type):
        model = MobileNetV2(weights='imagenet')
        images = generate_dummy_images(num_images)
        
        # GPU Warm-up (Important for RTX 4000 series kernels)
        _ = model.predict(images[:1], verbose=0)
        
        print(f"--- Starting Inference on {num_images} images (Requirement #3) ---")
        
        # Resource Tracking (Requirement #4)
        process = psutil.Process(os.getpid())
        start_time = time.time()
        start_mem = process.memory_info().rss / (1024 * 1024)

        # Prediction Loop
        for i in range(num_images):
            # Process one by one to simulate Edge/AIoT stream
            single_img = np.expand_dims(images[i], axis=0)
            _ = model.predict(single_img, verbose=0)

        end_time = time.time()
        end_mem = process.memory_info().rss / (1024 * 1024)
        cpu_load = psutil.cpu_percent(interval=None)

    total_time = end_time - start_time
    avg_fps = num_images / total_time
    
    print("\n--- PERFORMANCE RESULTS ---")
    print(f"Hardware Used: {device_type}")
    print(f"Total Prediction Time: {total_time:.4f} seconds")
    print(f"Inference Speed: {avg_fps:.2f} FPS")
    print(f"Memory Delta: {end_mem - start_mem:.2f} MB")
    print(f"Average CPU Utilization: {cpu_load}%")
    print("---------------------------\n")

if __name__ == "__main__":
    # For Requirement #3: First run with seed for 'Same Images', 
    # then comment out for 'Different Images'.
    # np.random.seed(42)
    run_performance_task(num_images=100)