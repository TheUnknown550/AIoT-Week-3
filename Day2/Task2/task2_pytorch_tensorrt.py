import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch2trt import torch2trt 
from PIL import Image
import time
import os
import random
import json
import urllib.request
import gc  # Garbage Collector

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_DIR = "my_datasets"
DEVICE = torch.device('cuda')

def load_labels():
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    try:
        with urllib.request.urlopen(url) as f: return json.load(f)
    except: return None

def main():
    print(f"--- Running TensorRT Optimized Benchmark (MobileNetV2) ---")

    # 1. Load Standard Model
    print("[1] Loading MobileNetV2...")
    try:
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        model = models.mobilenet_v2(weights=weights).to(DEVICE).eval()
        preprocess = weights.transforms()
    except:
        print("    > Using legacy load method...")
        model = models.mobilenet_v2(pretrained=True).to(DEVICE).eval()
        preprocess = transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # 2. Optimize with TensorRT (Low Memory Mode)
    print("[2] Compiling to TensorRT (FP16)...")
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    
    try:
        # CRITICAL FIX: Limit workspace to ~64MB (1 << 26) to prevent crashes
        model_trt = torch2trt(model, [dummy_input], fp16_mode=True, max_workspace_size=1<<26)
        print("    > Optimization Complete!")
        
        # CRITICAL FIX: Delete the heavy original model immediately!
        del model
        torch.cuda.empty_cache()
        gc.collect()
        print("    > Original model deleted to free RAM.")
        
    except Exception as e:
        print(f"    > Error during optimization: {e}")
        return

    # 3. Latency Test
    print("\n[3] Benchmarking Inference Speed (TensorRT)...")
    for _ in range(10): _ = model_trt(dummy_input) # Warmup
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(50):
        _ = model_trt(dummy_input)
    torch.cuda.synchronize()
    
    fps = 50 / (time.time() - start)
    print(f"    > FPS (TensorRT): {fps:.2f}")

    # 4. Resolution Stress Test
    print("\n[4] Testing 1080p Pipeline...")
    # Create input, but keep it minimal
    high_res_input = torch.randn(1, 3, 1080, 1920).to(DEVICE)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(30):
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
                # Load image fresh to save memory
                img = Image.open(img_path).convert('RGB')
                input_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
                output = model_trt(input_tensor)
                
                probs = torch.nn.functional.softmax(output[0], dim=0)
                top_prob, top_id = torch.topk(probs, 1)
                label = labels[top_id] if labels else str(top_id.item())
                
                print(f"    > {os.path.basename(img_path)}: {label} ({top_prob.item()*100:.1f}%)")
                
                # Free memory after each image
                del input_tensor, output
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"    > Error: {e}")
    else:
        print("    > No images found.")

if __name__ == "__main__":
    main()