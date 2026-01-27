import time
import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import shutil
import os

def configure_gpu_memory_growth():
    """Configures TensorFlow to allow memory growth."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Configured memory growth for {len(gpus)} GPU(s).")
        except RuntimeError as e:
            print(f"Error configuring GPU memory growth: {e}")
    else:
        print("No GPU found. This script is designed for GPU execution.")

def benchmark_model(model_func, input_data, model_name):
    """Measures inference latency and FPS."""
    print(f"\n--- Benchmarking {model_name} ---")
    
    # Warm-up 
    print("Warming up...")
    try:
        for _ in range(10): _ = model_func(input_data)
    except Exception as e:
        print(f"Benchmark failed during warmup: {e}")
        return 0

    print("Benchmarking...")
    latencies = []
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
    Converts a Keras model to TF-TRT. 
    Returns None if TF-TRT is not supported (e.g., on Windows).
    """
    print(f"\n--- Converting to TF-TRT ({precision_mode}) ---")
    
    tmp_saved_model_dir = 'tmp_mobilenet_v2'
    if os.path.exists(tmp_saved_model_dir):
        shutil.rmtree(tmp_saved_model_dir) 
    keras_model.save(tmp_saved_model_dir)
    print("Intermediate SavedModel created on disk.")

    try:
        # Define Conversion Parameters
        conversion_params = trt.TrtConversionParams(precision_mode=precision_mode)
        
        # Initialize Converter
        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=tmp_saved_model_dir,
            conversion_params=conversion_params
        )

        # This line triggers the error if TRT is missing
        converter.convert()
        
        def input_fn():
            yield [tf.random.normal((1, 224, 224, 3), dtype=tf.float32)]
        
        converter.build(input_fn=input_fn)
        
        output_saved_model_dir = f"mobilenetv2_tftrt_{precision_mode.lower()}"
        if os.path.exists(output_saved_model_dir):
            shutil.rmtree(output_saved_model_dir)
            
        converter.save(output_saved_model_dir)
        print(f"TF-TRT Optimized model saved to: {output_saved_model_dir}")
        
        return tf.saved_model.load(output_saved_model_dir)

    except RuntimeError as e:
        print(f"\n[WARNING] TensorRT Conversion Failed: {e}")
        print("If you are on Windows, this is expected (TRT is Linux only).")
        print("Skipping optimization step...")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during conversion: {e}")
        return None

def resolution_loop_test(model_func, resolutions):
    """Tests Pipeline Performance: Resize + Inference"""
    print("\n--- Resolution Loop Test (Target: 10 FPS) ---")
    
    for width, height in resolutions:
        high_res_input = tf.random.normal((1, height, width, 3), dtype=tf.float32)
        
        # Warmup
        for _ in range(5): _ = tf.image.resize(high_res_input, (224, 224))
        
        iterations = 30
        start_time = time.time()
        
        for _ in range(iterations):
            resized_input = tf.image.resize(high_res_input, (224, 224))
            _ = model_func(resized_input)
        
        total_time = time.time() - start_time
        fps = iterations / total_time
        
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
    
    # 2. Benchmark Native Keras
    dummy_data = tf.random.normal((1, 224, 224, 3))
    benchmark_model(native_keras_model, dummy_data, "Keras (Native)")
    
    # 3. Try to Convert to TF-TRT
    tftrt_loaded = convert_to_tftrt(native_keras_model, precision_mode='FP16')
    
    if tftrt_loaded:
        # If conversion worked (Jetson), benchmark the optimized model
        infer_func = tftrt_loaded.signatures['serving_default']
        benchmark_model(infer_func, dummy_data, "TF-TRT (FP16)")
        
        # Run resolution test on Optimized Model
        res_list = [(1280, 720), (1920, 1080)]
        resolution_loop_test(infer_func, res_list)
    else:
        # If conversion failed (Windows), run resolution test on Native Model just to check logic
        print("\n[INFO] Running resolution test on NATIVE model (since TRT failed)...")
        # Keras models are callable directly, so we pass the model object itself
        resolution_loop_test(native_keras_model, [(1280, 720), (1920, 1080)])