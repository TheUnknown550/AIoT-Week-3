import torch
from torch2trt import torch2trt
import torchvision.models as models
import time
from PIL import Image

# 1. Load Model
weights = models.ResNet50_Weights.IMAGENET1K_V1
model = models.resnet50(weights=weights).cuda().eval()

# 2. Create dummy input for the converter (batch_size, channels, height, width)
x = torch.ones((1, 3, 224, 224)).cuda()

# 3. Convert to TensorRT (FP16 mode is highly recommended for Jetson Nano)
print("Converting model to TensorRT... (this may take a few minutes)")
model_trt = torch2trt(model, [x], fp16_mode=True)

def run_trt_inference(image_path):
    preprocess = weights.transforms()
    img = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0).cuda()

    # Warm up
    for _ in range(5):
        _ = model_trt(input_tensor)

    # Measure Latency
    start_time = time.time()
    output = model_trt(input_tensor)
    latency = (time.time() - start_time) * 1000 # ms
    
    print(f"TensorRT Inference Latency: {latency:.2f} ms")
    print(f"Estimated FPS: {1000/latency:.2f}")
    return output

if __name__ == "__main__":
    run_trt_inference("test_image.jpg")