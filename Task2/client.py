import requests
import time
import numpy as np
import psutil

# 1. Define endpoint
URL = "http://127.0.0.1:5000/predict"

def run_client():
    # 2. Loop that generates an image (placeholder 224x224x3 array)
    for i in range(5):
        sample_img = np.random.random((224, 224, 3)).tolist() # Convert to list for JSON
        
        # 3. Record resources and timing
        start_time = time.time()
        cpu_before = psutil.cpu_percent()
        
        print(f"Iteration {i}: Sending Request...")
        response = requests.post(URL, json={"image": sample_img})
        
        if response.status_code == 200:
            print(f"Receiving Response: {response.json()}")
        
        end_time = time.time()
        print(f"Round Trip Time: {end_time - start_time:.4f}s")
        print(f"Edge CPU Usage: {psutil.cpu_percent()}% (Reference: {cpu_before}%)\n")

run_client()