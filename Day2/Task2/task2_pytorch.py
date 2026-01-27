import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import time

# 1. Load Pretrained Model (ResNet50 IMAGENET1K_V1)
weights = models.ResNet50_Weights.IMAGENET1K_V1
model = models.resnet50(weights=weights).cuda().eval()

# 2. Define Preprocessing (Standard for ImageNet)
preprocess = weights.transforms()

def run_inference(image_path):
    img = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0).cuda()
    
    # Warm up GPU
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_tensor)
            
    # Measure Latency
    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor)
    latency = (time.time() - start_time) * 1000 # ms
    
    print(f"PyTorch Inference Latency: {latency:.2f} ms")
    return output

if __name__ == "__main__":
    # Replace with a local image path
    run_inference("test_image.jpg")