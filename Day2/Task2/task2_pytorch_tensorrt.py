import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch2trt import torch2trt  # The Accelerator Library
from PIL import Image
import time
import os
import random
import json
import urllib.request

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_DIR = "my_datasets"
DEVICE = torch.device('cuda') # TRT requires GPU

def load_labels():
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    try:
        with urllib.request.urlopen(url) as f: return json.load(f)
    except: return None

def main():
    print(f"--- Running TensorRT Optimized Benchmark ---")

    # 1. Load Standard Model First
    print("[1] Loading GoogLeNet...")
    try:
        weights = models.GoogLeNet_Weights.IMAGENET1K_V2
        model = models.googlenet(weights=weights).to(DEVICE).eval()
        preprocess = weights.transforms()
    except:
        model = models.googlenet(pretrained=True).to(DEVICE).eval()
        preprocess = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # 2. Optimize with TensorRT
    print("[2] Compiling to TensorRT (FP16)... (This takes ~1-2 mins)")
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    
    # FP16 Mode = Half Precision (Massive speedup on Jetson)
    model_trt = torch2trt(model, [dummy_input], fp16_mode=True)
    print("    > Optimization Complete!")

    # 3. Latency Test (Using Optimized Model)
    print("\n[3] Benchmarking Inference Speed (TensorRT)...")
    
    # Warmup
    for _ in range(10): _ = model_trt(dummy_input)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(50):
        _ = model_trt(dummy_input)
    torch.cuda.synchronize()
    
    fps = 50 / (time.time() - start)
    print(f"    > FPS (TensorRT): {fps:.2f}")

    # 4. Resolution Stress Test
    print("\n[4] Testing 1080p Pipeline (Resize + TRT Inference)...")
    high_res_input = torch.randn(1, 3, 1080, 1920).to(DEVICE)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(30):
        # Resize is still PyTorch (GPU), Inference is TensorRT
        resized = torch.nn.functional.interpolate(high_res_input, size=(224, 224), mode='bilinear', align_corners=False)
        _ = model_trt(resized)
    torch.cuda.synchronize()
    
    res_fps = 30 / (time.time() - start)
    print(f"    > Pipeline FPS (1080p): {res_fps:.2f}")

    # 5. Predict Real Images
    print(f"\n[5] Predicting Images from '{DATASET_DIR}'...")
    labels = load_labels()
    image_files = []
    for root, _, files in os.walk(DATASET_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, f))
    
    if image_files:
        samples = random.sample(image_files, min(len(image_files), 3))
        for img_path in samples:
            try:
                img = Image.open(img_path).convert('RGB')
                input_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
                
                # Use TensorRT model for prediction
                output = model_trt(input_tensor)
                
                probs = torch.nn.functional.softmax(output[0], dim=0)
                top_prob, top_id = torch.topk(probs, 1)
                
                label = labels[top_id] if labels else str(top_id.item())
                print(f"    > {os.path.basename(img_path)}: {label} ({top_prob.item()*100:.1f}%)")
            except Exception as e:
                print(f"    > Error reading {img_path}: {e}")
    else:
        print("    > No images found.")

if __name__ == "__main__":
    main()