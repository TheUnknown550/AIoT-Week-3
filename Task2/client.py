import requests
import time
import numpy as np

URL = "http://172.20.10.3:5000/predict"

def send_image():
    # Simulate a 224x224x3 image and convert to list
    dummy_img = np.random.randint(0, 255, (224, 224, 3)).tolist()
    
    print("Sending Request...")
    start = time.time()
    
    response = requests.post(URL, json={"image": dummy_img})
    
    end = time.time()
    if response.status_code == 200:
        print(f"Receiving Response: {response.json()}")
        print(f"Time Taken: {end - start:.4f}s")
    else:
        print(f"Error: {response.status_code}")

if __name__ == "__main__":
    np.random.seed(42)
    send_image()