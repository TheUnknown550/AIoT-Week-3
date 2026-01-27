import requests
import time
import numpy as np
import psutil
import os

# CONFIGURATION
SERVER_IP = "172.20.10.3"  # Your Linux IP
URL = f"http://{SERVER_IP}:5000/predict"
NUM_IMAGES = 10  # Reduced for Wi-Fi testing (sending 100 over Wi-Fi is slow!)

def run_loopback_test():
    print(f"--- Starting Loopback Task: Client -> {SERVER_IP} ---")
    
    # Prepare dummy data
    images = np.random.randint(0, 256, (NUM_IMAGES, 224, 224, 3), dtype=np.uint8)
    
    process = psutil.Process(os.getpid())
    loopback_times = []
    
    start_mem = process.memory_info().rss / (1024 * 1024)

    for i in range(NUM_IMAGES):
        # 1. Prepare image as list (as per task requirement)
        img_list = images[i].tolist()
        
        print(f"[{i+1}/{NUM_IMAGES}] Sending Request...")
        
        # 2. Start timing the Loopback
        iter_start = time.time()
        
        try:
            # Note: 30s timeout because Wi-Fi can be slow for large JSON
            response = requests.post(URL, json={"image": img_list}, timeout=30)
            
            # 3. Stop timing upon Receiving Response
            iter_end = time.time()
            
            if response.status_code == 200:
                loopback_times.append(iter_end - iter_start)
                print(f"   Success! Confidence: {response.json()['confidence']:.2f}")
            
        except Exception as e:
            print(f"   Error on iteration {i}: {e}")

    end_mem = process.memory_info().rss / (1024 * 1024)
    
    # Calculate Results
    total_loopback_time = sum(loopback_times)
    avg_loopback = total_loopback_time / len(loopback_times) if loopback_times else 0

    print("\n--- LOOPBACK PERFORMANCE RESULTS ---")
    print(f"Total Loopback Time (Sum): {total_loopback_time:.4f} seconds")
    print(f"Average Loopback Time per Image: {avg_loopback:.4f} seconds")
    print(f"Client Memory Delta: {end_mem - start_mem:.2f} MB")
    print(f"Client CPU Load: {psutil.cpu_percent()}%")
    print("------------------------------------\n")

if __name__ == "__main__":
    run_loopback_test()