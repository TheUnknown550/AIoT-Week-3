import requests
import time
import numpy as np
import psutil
import os

SERVER_IP = "127.0.0.1"
URL = f"http://{SERVER_IP}:5000/predict_batch"
NUM_IMAGES = 100

def run_batch_test():
    print(f"--- Preparing Batch of {NUM_IMAGES} images ---")
    # Generate 100 random images
    images = np.random.randint(0, 256, (NUM_IMAGES, 224, 224, 3), dtype=np.uint8)
    
    # Convert entire batch to list (Warning: This will use significant RAM)
    print("Converting batch to JSON-ready list...")
    batch_list = images.tolist()
    
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / (1024 * 1024)
    
    print(f"--- Sending Batch Request (Total Size: ~{len(str(batch_list))/1024/1024:.2f} MB) ---")
    
    # Start Total Timing
    start_time = time.time()
    
    try:
        # Extended timeout (60s) because sending 100 images over Wi-Fi is a huge payload
        response = requests.post(URL, json={"images": batch_list}, timeout=60)
        
        end_time = time.time()
        
        if response.status_code == 200:
            total_time = end_time - start_time
            res_data = response.json()
            
            print("\n--- BATCH PERFORMANCE RESULTS ---")
            print(f"Total Batch Loopback Time: {total_time:.4f} seconds")
            print(f"Throughput: {NUM_IMAGES / total_time:.2f} images/sec")
            print(f"Server returned {len(res_data['results'])} predictions.")
            
    except Exception as e:
        print(f"Batch Request Failed: {e}")

    end_mem = process.memory_info().rss / (1024 * 1024)
    print(f"Client Memory Usage Delta: {end_mem - start_mem:.2f} MB")

if __name__ == "__main__":
    run_batch_test()