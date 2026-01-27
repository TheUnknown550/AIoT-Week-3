import time
import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import shutil
import os

def configure_gpu_memory_growth():
    """Configures TensorFlow to allow memory growth (prevents OOM on Jetson)."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Configured memory growth for {len(gpus)} GPU(s).")
        except RuntimeError as e:
            print(f"Error configuring GPU memory growth: {e}")
    else:
        print("No GPU found. This script requires a GPU.")

def benchmark_model(model_func, input_data, model_name):
    """Measures inference latency and FPS."""
    print(f"\n--- Benchmarking {model_name} ---")
    
    # Warm-up (Critical for TF-TRT to trigger optimization kernels)
    print("Warming up...")
    for _ in range(10):
        _ = model_func(input_data)
        
    print("Benchmarking...")
    latencies = []
    # Run 50 times for stability
    for _ in range(50):
        start_time = time.time()
        _ = model_func(input_data)
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000) # ms
        
    avg_latency = np.mean(latencies)
    fps = 1000 / avg_latency
    
    print(f"Average Latency: {avg_latency:.3f} ms")
    print(f"Frames Per Second (FPS): {fps:.2f}")
    print("-" * (20 + len(model_name)))
    return fps

def convert_to_tftrt(keras_model, precision_mode='FP16'):
    """
    Converts a Keras model to TF-TRT by saving it to disk first.
    """
    print(f"\n--- Converting to TF-TRT ({precision_mode}) ---")
    
    # 1. Save Keras model to a temporary folder (REQUIRED for TF-TRT)
    tmp_saved_model_dir = 'tmp_mobilenet_v2'
    if os.path.exists(tmp_saved_model_dir):
        shutil.rmtree(tmp_saved_model_dir) # Cleanup previous runs
    keras_model.save(tmp_saved_model_dir)
    print("Intermediate SavedModel created on disk.")

    # 2. Define Conversion Parameters
    conversion_params = trt.TrtConversionParams(
        precision_mode=precision_mode
    )
    
    # 3. Initialize Converter pointing to the saved directory
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=tmp_saved_model_dir,
        conversion_params=conversion_params
    )

    # 4. Convert (This handles the graph optimization)
    # Note: explicit conversion function is usually not needed for FP16, 
    # but 'convert()' call is required.
    converter.convert()
    
    # 5. Build (Optional but recommended to pre-build engines)
    def input_fn():
        # Yields a dummy input for calibration/building
        yield [tf.random.normal((1, 224, 224, 3), dtype=tf.float32)]
    
    converter.build(input_fn=input_fn)
    
    # 6. Save the TRT-optimized model
    output_saved_model_dir = f"mobilenetv2_tftrt_{precision_mode.lower()}"
    if os.path.exists(output_saved_model_dir):
        shutil.rmtree(output_saved_model_dir)
        
    converter.save(output_saved_model_dir)
    print(f"TF-TRT Optimized model saved to: {output_saved_model_dir}")
    
    # 7. Reload the optimized model
    saved_model_loaded = tf.saved_model.load(output_saved_model_dir)
    
    return saved_model_loaded

def resolution_loop_test(model_func, resolutions):
    """
    Tests Pipeline Performance: Resize + Inference
    """
    print("\n--- Resolution Loop Test (Target: 10 FPS) ---")
    
    for width, height in resolutions:
        # Create High-Res Input (Simulating camera)
        high_res_input = tf.random.normal((1, height, width, 3), dtype=tf.float32)
        
        # Warmup
        for _ in range(5):
             _ = tf.image.resize(high_res_input, (224, 224))
        
        # Run loop for accurate average
        iterations = 30
        start_time = time.time()
        
        for _ in range(iterations):
            # 1. Resize (The costly part for high res)
            resized_input = tf.image.resize(high_res_input, (224, 224))
            
            # 2. Inference
            # Note: SavedModels usually require input as a list or keyword arg if not using default sig
            _ = model_func(resized_input)
        
        total_time = time.time() - start_time
        avg_time_per_frame = total_time / iterations
        fps = 1.0 / avg_time_per_frame
        
        status = "Pass" if fps >= 10 else "Fail"
        print(f"Resolution {width}x{height} -> Pipeline FPS: {fps:.2f} ({status})")

    print("-" * 50)

if __name__ == '__main__':
    configure_gpu_memory_growth()
    
    # 1. Load Keras Model
    print("\n[1] Loading Keras MobileNetV2...")
    native_keras_model = tf.keras.applications.MobileNetV2(
        weights='imagenet', input_shape=(224, 224, 3)
    )
    
    # 2. Benchmark Native Keras (Slow)
    # We wrap .predict() or simply call the model to benchmark
    # Using model(data) is standard for raw speed in TF2
    print("Benchmarking Native Keras...")
    dummy_data = tf.random.normal((1, 224, 224, 3))
    benchmark_model(native_keras_model, dummy_data, "Keras (Native)")
    
    # 3. Convert to TF-TRT
    tftrt_loaded = convert_to_tftrt(native_keras_model, precision_mode='FP16')
    
    # Get the inference function (Signature: serving_default)
    infer_func = tftrt_loaded.signatures['serving_default']
    
    # Note: TF-TRT concrete functions usually expect the input tensor 
    # to be passed as a keyword argument or flattened list. 
    # We grab the name of the first input argument to be safe.
    # However, passing positional args often works for single-input models.
    
    # 4. Benchmark TF-TRT
    benchmark_model(infer_func, dummy_data, "TF-TRT (FP16)")
    
    # 5. Resolution Test
    res_list = [(1280, 720), (1920, 1080)]
    resolution_loop_test(infer_func, res_list)