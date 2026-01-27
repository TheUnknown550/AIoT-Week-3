import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import time
import os
import random
import json
import urllib.request

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_DIR = "my_datasets"  # Folder with your images
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_labels():
    """Downloads ImageNet class names for readable predictions."""
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    try:
        with urllib.request.urlopen(url) as f:
            return json.load(f)
    except:
        return None

def main():
    print(f"--- Running Standard PyTorch Benchmark (MobileNetV2) on {DEVICE} ---")

    # 1. Load Model (MobileNetV2)
    print("[1] Loading MobileNetV2...")
    try:
        # Modern PyTorch Syntax (For your Laptop)
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        model = models.mobilenet_v2(weights=weights).to(DEVICE).eval()
        preprocess = weights.transforms()
    except:
        # Legacy Syntax (For Jetson Nano / Older PyTorch)
        print("    > Using legacy load method (standard for Jetson)...")
        model = models.mobilenet_v2(pretrained=True).to(DEVICE).eval()
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # 2. Latency Test (Speed)
    print("\n[2] Benchmarking Inference Speed (Batch Size 1)...")
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    
    # Warmup
    for _ in range(10): _ = model(dummy_input)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(50):
        with torch.no_grad(): _ = model(dummy_input)
    torch.cuda.synchronize()
    
    fps = 50 / (time.time() - start)
    print(f"    > FPS (Standard): {fps:.2f}")

    # 3. Resolution Stress Test (1080p Pipeline)
    print("\n[3] Testing 1080p Pipeline (Resize + Inference)...")
    high_res_input = torch.randn(1, 3, 1080, 1920).to(DEVICE)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(30):
        # Resize High-Res -> 224x224
        resized = torch.nn.functional.interpolate(high_res_input, size=(224, 224), mode='bilinear', align_corners=False)
        # Infer
        with torch.no_grad(): _ = model(resized)
    torch.cuda.synchronize()
    
    res_fps = 30 / (time.time() - start)
    print(f"    > Pipeline FPS (1080p): {res_fps:.2f}")

    # 4. Predict Real Images
    print(f"\n[4] Predicting Images from '{DATASET_DIR}'...")
    labels = load_labels()
    image_files = []
    for root, _, files in os.walk(DATASET_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, f))
    
    if image_files:
        # Test 3 random images
        samples = random.sample(image_files, min(len(image_files), 3))
        for img_path in samples:
            try:
                img = Image.open(img_path).convert('RGB')
                input_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    output = model(input_tensor)
                
                probs = torch.nn.functional.softmax(output[0], dim=0)
                top_prob, top_id = torch.topk(probs, 1)
                
                label = labels[top_id] if labels else str(top_id.item())
                print(f"    > {os.path.basename(img_path)}: {label} ({top_prob.item()*100:.1f}%)")
            except Exception as e:
                print(f"    > Error reading {img_path}")
    else:
        print("    > No images found.")

if __name__ == "__main__":
    main()