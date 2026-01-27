import requests
import time
import numpy as np
import psutil
import os

# CONFIGURATION
SERVER_IP = "127.0.0.1"
URL = f"http://{SERVER_IP}:5000/predict_batch"
NUM_IMAGES = 100
RANDOM_SEED = 4  # Set this to any integer to get consistent images

def run_batch_test(seed=None):
    if seed is not None:
        np.random.seed(seed)
        print(f"--- Seed set to: {seed} ---")

    print(f"--- Preparing Batch of {NUM_IMAGES} images ---")
    # Images will now be identical across runs if seed is the same
    images = np.random.randint(0, 256, (NUM_IMAGES, 224, 224, 3), dtype=np.uint8)
    
    print("Converting batch to JSON-ready list...")
    batch_list = images.tolist()
    
    process = psutil.Process(os.getpid())
    start_time = time.time()
    
    try:
        print(f"--- Sending Batch to {SERVER_IP} ---")
        # Standard timeout for 100 images over Hotspot
        response = requests.post(URL, json={"images": batch_list}, timeout=120) 
        
        end_time = time.time()
        
        if response.status_code == 200:
            total_time = end_time - start_time
            print("\n--- PERFORMANCE RESULTS (WITH SEED) ---")
            print(f"Total Batch Loopback Time: {total_time:.4f} seconds")
            print(f"Throughput: {NUM_IMAGES / total_time:.2f} images/sec")
    except Exception as e:
        print(f"Batch Request Failed: {e}")

if __name__ == "__main__":
    run_batch_test(seed=RANDOM_SEED)