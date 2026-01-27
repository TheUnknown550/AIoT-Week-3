
import torch
import torchvision.models as models
from torch2trt import torch2trt
import time
import numpy as np

def benchmark_model(model, input_tensor, model_name=""):
    """
    Benchmarks the inference speed of a given model.

    Args:
        model: The model to benchmark.
        input_tensor: A dummy input tensor for the model.
        model_name (str): The name of the model for display purposes.
    
    Returns:
        A tuple containing average inference time (ms) and FPS.
    """
    print(f"--- Benchmarking {model_name} ---")
    
    # Warm-up runs
    for _ in range(10):
        _ = model(input_tensor)
    
    torch.cuda.synchronize()
    
    timings = []
    for _ in range(50):
        start_time = time.time()
        _ = model(input_tensor)
        torch.cuda.synchronize()
        end_time = time.time()
        timings.append((end_time - start_time) * 1000) # Convert to ms
        
    avg_inference_time = np.mean(timings)
    fps = 1000 / avg_inference_time
    
    print(f"Average Inference Time: {avg_inference_time:.3f} ms")
    print(f"Frames Per Second (FPS): {fps:.2f}")
    print("-" * (20 + len(model_name)))
    return avg_inference_time, fps

def resolution_stress_test(model, resolutions):
    """
    Tests the model's performance with various input resolutions.
    MEASURES: Pre-processing (Resize) + Inference
    """
    print("\n--- Resolution Stress Test (Target: 20 FPS) ---")
    
    for width, height in resolutions:
        # Create the high-res input outside the loop (simulating a camera frame coming in)
        dummy_high_res_input = torch.randn(1, 3, height, width).cuda()
        
        # Warmup
        for _ in range(5):
             _ = torch.nn.functional.interpolate(dummy_high_res_input, size=(224, 224), mode='bilinear')
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        # Run 30 iterations to get an average "Pipeline" speed
        iterations = 30
        for _ in range(iterations):
            # 1. Resize (This is what slows down high resolutions!)
            resized_input = torch.nn.functional.interpolate(
                dummy_high_res_input, 
                size=(224, 224), 
                mode='bilinear', 
                align_corners=False
            )
            # 2. Inference
            _ = model(resized_input)
            
        torch.cuda.synchronize()
        end_time = time.time()
        
        total_time_ms = (end_time - start_time) * 1000
        avg_time_per_frame = total_time_ms / iterations
        fps = 1000 / avg_time_per_frame
        
        status = "Pass" if fps >= 20 else "Fail"
        print(f"Input {width}x{height} -> Pipeline FPS: {fps:.2f} ({status})")
    
    print("-" * 50)

if __name__ == '__main__':
    # 1. Set device to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("CUDA not available. Exiting. This script requires a GPU.")
        exit()

    print(f"Using device: {device}")

    # 2. Load GoogLeNet Model
    try:
        # Recommended modern approach
        weights = models.GoogLeNet_Weights.IMAGENET1K_V2
        model = models.googlenet(weights=weights)
        print("Successfully loaded GoogLeNet with IMAGENET1K_V2 weights.")
    except AttributeError:
        # Fallback for older torchvision versions
        print("Could not load IMAGENET1K_V2 weights. Falling back to `pretrained=True`.")
        print("Consider upgrading torchvision for newer weights: `pip install --upgrade torchvision`")
        model = models.googlenet(pretrained=True)

    model = model.to(device).eval()

    # 3. Create dummy input tensor
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # 4. Benchmark Native PyTorch Model
    pytorch_time, pytorch_fps = benchmark_model(model, dummy_input, "PyTorch (Native)")

    # 5. Convert to TensorRT
    print("\nConverting model to TensorRT with FP16 optimization...")
    try:
        model_trt = torch2trt(model, [dummy_input], fp16_mode=True)
        print("TensorRT conversion successful.")
    except Exception as e:
        print(f"Error during TensorRT conversion: {e}")
        print("Please ensure torch2trt is correctly installed.")
        exit()

    # 6. Benchmark TensorRT Model
    trt_time, trt_fps = benchmark_model(model_trt, dummy_input, "TensorRT (Optimized)")

    # 7. Compare Results
    print("\n--- Performance Comparison ---")
    print(f"PyTorch (Native) FPS: {pytorch_fps:.2f}")
    print(f"TensorRT (FP16) FPS: {trt_fps:.2f}")
    if pytorch_fps > 0:
        speedup = trt_fps / pytorch_fps
        print(f"Speedup: {speedup:.2f}x")
    print("-" * 28)
    
    # 8. Resolution Stress Test
    resolutions_to_test = [(640, 480), (1280, 720), (1920, 1080)]
    resolution_stress_test(model_trt, resolutions_to_test)
