import torch
from torch2trt import torch2trt
import torchvision.models as models
import time
from PIL import Image

# 1. Load Model
weights = models.ResNet50_Weights.IMAGENET1K_V1
model = models.resnet50(weights=weights).cuda().eval()

# 2. Convert to TensorRT
# We use fp16_mode=True to significantly boost performance on Jetson hardware
x = torch.ones((1, 3, 224, 224)).cuda()
print("Converting to TensorRT (this may take a few minutes)...")
model_trt = torch2trt(model, [x], fp16_mode=True)

def benchmark_trt(image_path, iterations=50):
    preprocess = weights.transforms()
    img = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0).cuda()

    print("Warming up...")
    for _ in range(10):
        _ = model_trt(input_tensor)

    print(f"Running {iterations} iterations...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(iterations):
        _ = model_trt(input_tensor)
        
    torch.cuda.synchronize()
    end_time = time.time()
    
    latency = ((end_time - start_time) / iterations) * 1000
    fps = 1000 / latency
    print(f"TensorRT Latency: {latency:.2f} ms | FPS: {fps:.2f}")

if __name__ == "__main__":
    benchmark_trt("test.jpg")