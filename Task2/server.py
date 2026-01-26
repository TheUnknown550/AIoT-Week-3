from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# 1. Initialize Model once at startup
model = tf.keras.applications.MobileNetV2(weights='imagenet')

@app.route('/predict', methods=['POST'])
def predict():
    # 2. Extract JSON data
    data = request.json
    image_list = data['image']
    
    # Convert list back to numpy array and preprocess
    img_array = np.array(image_list, dtype=np.float32)
    # MobileNetV2 expects shape (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Perform prediction
    predictions = model.predict(img_array)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
    
    # Return confidence score (index 2 in the decoded tuple)
    return jsonify({"label": decoded[0][1], "confidence": float(decoded[0][2])})

if __name__ == '__main__':
    app.run(port=5000)