import torch
import time
import numpy as np
from torchvision import models
from torch2trt import torch2trt

# 1. Load Pretrained MobileNetV2
model = models.mobilenet_v2(pretrained=True).eval().cuda()

def benchmark(model, input_shape, iterations=50):
    # Warm up
    data = torch.randn(input_shape).cuda()
    for _ in range(10):
        _ = model(data)
    
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(iterations):
        _ = model(data)
        torch.cuda.synchronize()
    end_time = time.time()
    
    latency = (end_time - start_time) / iterations * 1000 # ms
    fps = 1000 / latency
    return latency, fps

# 2. Convert to TensorRT
# We use FP16 mode as it is the "sweet spot" for Jetson Nano performance
x = torch.ones((1, 3, 224, 224)).cuda()
model_trt = torch2trt(model, [x], fp16_mode=True)

# 3. Compare Latency & Find Max Resolution
resolutions = [(224, 224), (448, 448), (640, 640), (720, 720)]

print(f"{'Resolution':<15} | {'PyTorch FPS':<12} | {'TensorRT FPS':<12} | {'Status'}")
print("-" * 60)

for res in resolutions:
    input_shape = (1, 3, res[0], res[1])
    
    # Benchmark PyTorch
    _, py_fps = benchmark(model, input_shape)
    
    # Benchmark TensorRT (Note: usually needs re-conversion for different shapes, 
    # but for simplicity, we focus on the comparison at 224x224)
    if res == (224, 224):
        _, trt_fps = benchmark(model_trt, input_shape)
    else:
        # Re-convert for different resolution
        temp_x = torch.ones(input_shape).cuda()
        temp_trt = torch2trt(model, [temp_x], fp16_mode=True)
        _, trt_fps = benchmark(temp_trt, input_shape)

    status = "OK" if trt_fps >= 10 else "Below 10fps"
    print(f"{str(res):<15} | {py_fps:<12.2f} | {trt_fps:<12.2f} | {status}")