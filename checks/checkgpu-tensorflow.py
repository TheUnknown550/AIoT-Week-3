import tensorflow as tf

# List the devices TensorFlow can see
print("Checking for GPU...")
devices = tf.config.list_physical_devices('GPU')

if devices:
    print(f"✅ Success! TensorFlow detected: {devices}")
else:
    print("❌ GPU still not detected. Checking dependencies...")