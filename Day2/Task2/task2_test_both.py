import os
import subprocess
import sys
import re

# ==========================================
# 1. DEFINE THE PYTORCH SCRIPT (Speed + Prediction)
# ==========================================
pytorch_code = """
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import time
import sys
import os
import glob
import random
import json
import urllib.request

# Configuration
DATASET_DIR = "my_datasets"

# Try to import torch2trt
try:
    from torch2trt import torch2trt
    HAS_TRT = True
except ImportError:
    HAS_TRT = False

def load_imagenet_labels():
    # Helper to get class names (PyTorch doesn't have this built-in like Keras)
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    try:
        with urllib.request.urlopen(url) as f:
            return json.load(f)
    except:
        return None

def run_pytorch_full_suite():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"P: Device: {device}")

    # --- 1. Load Model ---
    print("P: Loading GoogLeNet...")
    try:
        weights = models.GoogLeNet_Weights.IMAGENET1K_V2
        model = models.googlenet(weights=weights).to(device).eval()
        preprocess = weights.transforms()
    except:
        model = models.googlenet(pretrained=True).to(device).eval()
        # Fallback transform
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # --- 2. Optimization ---
    model_final = model
    if HAS_TRT:
        print("P: Optimizing with TensorRT (FP16)...")
        try:
            model_trt = torch2trt(model, [dummy_input], fp16_mode=True)
            model_final = model_trt
        except Exception as e:
            print(f"P: Opt failed: {e}")

    # --- 3. Latency Benchmark ---
    print("P: Running Latency Test...")
    for _ in range(10): _ = model_final(dummy_input) # Warmup
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(50): _ = model_final(dummy_input)
    torch.cuda.synchronize()
    fps = 50 / (time.time() - start)
    print(f"RESULT_LATENCY:{fps:.2f}")

    # --- 4. Resolution Test ---
    print("P: Running Resolution Stress Test (1080p)...")
    high_res = torch.randn(1, 3, 1080, 1920).to(device)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(30):
        # Measure Resize + Inference
        resized = torch.nn.functional.interpolate(high_res, size=(224, 224), mode='bilinear', align_corners=False)
        _ = model_final(resized)
    torch.cuda.synchronize()
    res_fps = 30 / (time.time() - start)
    print(f"RESULT_RES_FPS:{res_fps:.2f}")

    # --- 5. Real Image Prediction ---
    print(f"P: Predicting images from '{DATASET_DIR}'...")
    labels = load_imagenet_labels()
    
    # Find images
    image_files = []
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    
    if image_files:
        # Pick 3 random images
        samples = random.sample(image_files, min(len(image_files), 3))
        
        for img_path in samples:
            try:
                img = Image.open(img_path).convert('RGB')
                input_tensor = preprocess(img).unsqueeze(0).to(device)
                
                # Inference
                with torch.no_grad():
                    output = model_final(input_tensor)
                
                # Decode
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                top_prob, top_catid = torch.topk(probabilities, 1)
                
                label_name = labels[top_catid] if labels else str(top_catid.item())
                filename = os.path.basename(img_path)
                
                print(f"PRED_LOG:[{filename}] -> {label_name} ({top_prob.item()*100:.1f}%)")
            except Exception as e:
                print(f"P: Failed to predict {img_path}: {e}")
    else:
        print("P: No images found to predict.")

if __name__ == "__main__":
    try:
        run_pytorch_full_suite()
    except Exception as e:
        print(f"P: Critical Error: {e}")
"""

# ==========================================
# 2. DEFINE THE TENSORFLOW SCRIPT (Speed + Prediction)
# ==========================================
tensorflow_code = """
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import time
import os
import shutil
import numpy as np
import random

# Configuration
DATASET_DIR = "my_datasets"

def run_tf_full_suite():
    # GPU Setup
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

    print("TF: Loading MobileNetV2...")
    model = tf.keras.applications.MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))
    
    # --- Optimization (TF-TRT) with Fallback ---
    infer_func = None
    try:
        print("TF: Attempting TF-TRT Conversion...")
        tmp_dir = 'tmp_mobilenet'
        if os.path.exists(tmp_dir): shutil.rmtree(tmp_dir)
        model.save(tmp_dir)
        
        conv_params = trt.TrtConversionParams(precision_mode='FP16')
        converter = trt.TrtGraphConverterV2(input_saved_model_dir=tmp_dir, conversion_params=conv_params)
        converter.convert()
        def input_fn(): yield [tf.random.normal((1, 224, 224, 3))]
        converter.build(input_fn=input_fn)
        
        saved_model_dir = 'mobilenet_trt'
        if os.path.exists(saved_model_dir): shutil.rmtree(saved_model_dir)
        converter.save(saved_model_dir)
        
        imported = tf.saved_model.load(saved_model_dir)
        infer_func = imported.signatures['serving_default']
        print("TF: Optimization Successful.")
    except Exception as e:
        print(f"TF: Optimization Skipped (Native Mode). Reason: {e}")
        infer_func = None # Fallback to native model object

    # Helper to run inference (handles native vs TRT differences)
    def run_inference(data):
        if infer_func:
            return infer_func(data)
        else:
            return model(data) # Native Keras call

    dummy_input = tf.random.normal((1, 224, 224, 3))

    # --- Latency Benchmark ---
    print("TF: Running Latency Test...")
    for _ in range(10): _ = run_inference(dummy_input)
    start = time.time()
    for _ in range(50): _ = run_inference(dummy_input)
    fps = 50 / (time.time() - start)
    print(f"RESULT_LATENCY:{fps:.2f}")

    # --- Resolution Benchmark ---
    print("TF: Running Resolution Stress Test (1080p)...")
    high_res = tf.random.normal((1, 1080, 1920, 3))
    for _ in range(5): _ = tf.image.resize(high_res, (224, 224))
    start = time.time()
    for _ in range(30):
        resized = tf.image.resize(high_res, (224, 224))
        _ = run_inference(resized)
    res_fps = 30 / (time.time() - start)
    print(f"RESULT_RES_FPS:{res_fps:.2f}")

    # --- Real Image Prediction ---
    print(f"TF: Predicting images from '{DATASET_DIR}'...")
    image_files = []
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    
    if image_files:
        samples = random.sample(image_files, min(len(image_files), 3))
        
        for img_path in samples:
            try:
                # Load and Preprocess
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
                img_arr = tf.keras.preprocessing.image.img_to_array(img)
                img_arr = tf.keras.applications.mobilenet_v2.preprocess_input(img_arr)
                img_batch = np.expand_dims(img_arr, axis=0)
                
                # Predict
                if infer_func:
                    # TRT returns a dict, we need the tensor
                    preds = infer_func(tf.constant(img_batch))
                    # usually key is 'predictions' or 'logits', finding first key
                    output_tensor = list(preds.values())[0]
                    output_numpy = output_tensor.numpy()
                else:
                    output_numpy = model.predict(img_batch)
                
                # Decode
                decoded = tf.keras.applications.mobilenet_v2.decode_predictions(output_numpy, top=1)[0]
                label_name = decoded[0][1]
                confidence = decoded[0][2] * 100
                
                filename = os.path.basename(img_path)
                print(f"PRED_LOG:[{filename}] -> {label_name} ({confidence:.1f}%)")
            except Exception as e:
                print(f"TF: Prediction failed for {img_path}: {e}")
    else:
        print("TF: No images found.")

if __name__ == "__main__":
    try:
        run_tf_full_suite()
    except Exception as e:
        print(f"TF: Critical Error: {e}")
"""

# ==========================================
# 3. MASTER RUNNER
# ==========================================
def create_files():
    print("[Master] Generating scripts...")
    with open("benchmark_pt_full.py", "w") as f: f.write(pytorch_code)
    with open("benchmark_tf_full.py", "w") as f: f.write(tensorflow_code)

def run_framework(script_name, label):
    print(f"\n{'-'*60}")
    print(f"Running {label}...")
    print(f"{'-'*60}")
    
    metrics = {"fps": 0.0, "res_fps": 0.0, "predictions": []}
    
    try:
        # Run process and capture output in real-time
        process = subprocess.Popen([sys.executable, script_name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.strip()) # Echo output to console
                
                # Parse data from the stream
                if "RESULT_LATENCY:" in line:
                    metrics["fps"] = float(line.split(":")[1])
                elif "RESULT_RES_FPS:" in line:
                    metrics["res_fps"] = float(line.split(":")[1])
                elif "PRED_LOG:" in line:
                    metrics["predictions"].append(line.split("PRED_LOG:")[1].strip())
                    
    except Exception as e:
        print(f"Error running {label}: {e}")
        
    return metrics

def main():
    create_files()
    
    # Run PyTorch
    pt_data = run_framework("benchmark_pt_full.py", "PyTorch (GoogLeNet)")
    
    # Run TensorFlow
    tf_data = run_framework("benchmark_tf_full.py", "TensorFlow (MobileNetV2)")
    
    # Final Report
    print("\n\n")
    print("="*60)
    print("               FINAL ASSIGNMENT REPORT")
    print("="*60)
    
    # 1. Speed Comparison
    print(f"{'Metric':<25} | {'PyTorch':<10} | {'TensorFlow':<10}")
    print("-" * 60)
    print(f"{'Latency (FPS)':<25} | {pt_data['fps']:<10.2f} | {tf_data['fps']:<10.2f}")
    print(f"{'1080p Pipeline (FPS)':<25} | {pt_data['res_fps']:<10.2f} | {tf_data['res_fps']:<10.2f}")
    print("-" * 60)
    
    # 2. Prediction Samples
    print("\n[PyTorch Prediction Samples]")
    for p in pt_data["predictions"]:
        print(f"  > {p}")
        
    print("\n[TensorFlow Prediction Samples]")
    for p in tf_data["predictions"]:
        print(f"  > {p}")
        
    # Cleanup
    if os.path.exists("benchmark_pt_full.py"): os.remove("benchmark_pt_full.py")
    if os.path.exists("benchmark_tf_full.py"): os.remove("benchmark_tf_full.py")

if __name__ == "__main__":
    main()