from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import time
import logging
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import mlflow
import mlflow.sklearn
from typing import List, Dict, Any
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('predict_requests_total', 'Total prediction requests')
REQUEST_LATENCY = Histogram('predict_latency_seconds', 'Prediction request latency')
MODEL_ACCURACY = Counter('model_predictions_total', 'Total model predictions', ['result'])

app = FastAPI(
    title="ModelServeAI",
    description="AI Model Serving with MLflow tracking and Prometheus monitoring",
    version="1.0.0"
)

# Load model
model = None
try:
    with open('app/model.pkl', 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
except FileNotFoundError:
    logger.warning("Model file not found, using dummy model")

class UserInput(BaseModel):
    user_id: int
    age: int
    gender: str
    category_preference: str
    price_range: str

class PredictionResponse(BaseModel):
    user_id: int
    recommendations: List[Dict[str, Any]]
    confidence: float
    timestamp: str

@app.get("/")
def root():
    return {"message": "ModelServeAI is running!", "status": "healthy"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": time.time()
    }

@app.post("/predict", response_model=PredictionResponse)
@REQUEST_LATENCY.time()
def predict(data: UserInput):
    REQUEST_COUNT.inc()
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([data.dict()])
        
        # Make prediction
        if model is not None:
            prediction = model.predict(input_data)
            confidence = 0.85  # Mock confidence score
        else:
            # Dummy prediction for demo
            prediction = [f"product_{i}" for i in range(1, 4)]
            confidence = 0.75
        
        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_param("user_id", data.user_id)
            mlflow.log_metric("confidence", confidence)
        
        # Update Prometheus metrics
        MODEL_ACCURACY.labels(result='success').inc()
        
        response = PredictionResponse(
            user_id=data.user_id,
            recommendations=[
                {"product_id": pred, "score": confidence} 
                for pred in (prediction if isinstance(prediction, list) else [prediction])
            ],
            confidence=confidence,
            timestamp=str(time.time())
        )
        
        logger.info(f"Prediction made for user {data.user_id}")
        return response
        
    except Exception as e:
        MODEL_ACCURACY.labels(result='error').inc()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
