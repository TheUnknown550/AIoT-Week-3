import os
import time
import numpy as np

# Use the specific path for your CUDA 12.4 installation
try:
    os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin")
except FileNotFoundError:
    pass

import cupy as cp

def test_gpu():
    try:
        # 1. Hardware Detection
        dev = cp.cuda.Device(0)
        props = cp.cuda.runtime.getDeviceProperties(0)
        print(f"✅ GPU Detected: {props['name'].decode()}")
        
        # 2. Modern Memory Check
        free_mem, total_mem = dev.mem_info
        print(f"📊 Memory: {free_mem / 1024**3:.2f}GB Free / {total_mem / 1024**3:.2f}GB Total")

        # 3. Performance Benchmark
        size = 5000
        print(f"\n🚀 Benchmarking {size}x{size} Matrix Multiplication...")

        # CPU (NumPy)
        a_cpu = np.random.randn(size, size).astype(np.float32)
        start = time.time()
        _ = np.dot(a_cpu, a_cpu)
        print(f"💻 CPU Time: {time.time() - start:.4f} seconds")

        # GPU (CuPy)
        a_gpu = cp.random.randn(size, size).astype(cp.float32)
        start = time.time()
        _ = cp.dot(a_gpu, a_gpu)
        cp.cuda.Stream.null.synchronize() # Wait for GPU to finish!
        print(f"🏎️  GPU Time: {time.time() - start:.4f} seconds")

    except Exception as e:
        print(f"❌ Error during test: {e}")

if __name__ == "__main__":
    test_gpu()