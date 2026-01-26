from fastapi import FastAPI, Request
import tensorflow as tf
import numpy as np
import uvicorn
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Initializing MobileNetV2 for Batch Processing ---")
    app.state.model = tf.keras.applications.MobileNetV2(weights='imagenet')
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/predict_batch")
async def predict_batch(request: Request):
    data = await request.json()
    # data['images'] will now be a list of 100 image lists
    batch_array = np.array(data['images'], dtype=np.float32)
    
    # MobileNetV2 preprocessing (handles the whole batch at once)
    batch_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(batch_array)

    # Single Inference call for all 100 images
    predictions = app.state.model.predict(batch_preprocessed, verbose=0)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)
    
    # Format results for all images
    results = [{"label": d[0][1], "confidence": float(d[0][2])} for d in decoded]
    
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)