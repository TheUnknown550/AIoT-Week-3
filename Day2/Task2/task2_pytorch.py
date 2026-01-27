import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import time

# 1. Load Pretrained Model (ResNet50 is a non-transformer CNN)
weights = models.ResNet50_Weights.IMAGENET1K_V1
model = models.resnet50(weights=weights).cuda().eval()

# 2. Setup Preprocessing
preprocess = weights.transforms()

def benchmark_pytorch(image_path, iterations=50):
    img = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0).cuda()
    
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
            
    print(f"Running {iterations} iterations...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_tensor)
            
    torch.cuda.synchronize()
    end_time = time.time()
    
    latency = ((end_time - start_time) / iterations) * 1000
    fps = 1000 / latency
    print(f"PyTorch Latency: {latency:.2f} ms | FPS: {fps:.2f}")

if __name__ == "__main__":
    # Ensure you have a 'test.jpg' in your directory
    benchmark_pytorch("test.jpg")