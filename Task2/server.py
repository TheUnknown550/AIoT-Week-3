from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import tensorflow as tf
import numpy as np
import uvicorn

# 1. Initialize model only once using lifespan context
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    app.state.model = tf.keras.applications.MobileNetV2(weights='imagenet')
    yield
    # Clean up resources on shutdown (optional)
    del app.state.model

app = FastAPI(lifespan=lifespan)

# 2. Create the /predict endpoint
@app.post("/predict")
async def predict(request: Request):
    # Extract JSON data
    data = await request.json()
    image_list = data['image']
    
    # Preprocess the Python list back into a Tensor
    img_array = np.array(image_list, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Perform prediction using the model stored in app.state
    predictions = app.state.model.predict(img_array)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
    
    return {"label": decoded[0][1], "confidence": float(decoded[0][2])}

if __name__ == '__main__':
    # host='0.0.0.0' allows external connections if you need it
    uvicorn.run(app, host="127.0.0.1", port=5000)