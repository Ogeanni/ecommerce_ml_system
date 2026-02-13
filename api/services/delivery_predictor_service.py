"""
Delivery Predictor Service (Regression)
Independent microservice - Port 8002
Only loads delivery time prediction model
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from datetime import datetime
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ONLY import what this service needs
from src.regression.predict import DeliveryTimePredictor

app = FastAPI(
    title="Delivery Predictor Service",
    description="Regression model for delivery time estimation",
    version="1.0.0"
)
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])

# Load model
print("Loading Delivery Time Prediction Model...")
try:
    predictor = DeliveryTimePredictor(
        model_path="models/regression/saved_models/best_model_xgboost.pkl",
        preprocessing_path="models/regression/preprocessing"
    )
    print(" Delivery Predictor loaded successfully")
except Exception as e:
    print(f" Failed to load model: {e}")
    predictor = None

class PredictionRequest(BaseModel):
    order_data: Dict[str, Any]

class BatchPredictionRequest(BaseModel):
    orders: Dict[str, Any]

# ENDPOINTS
@app.get("/")
def root():
    return {
        "service": "Delivery Predictor",
        "model": "Regression",
        "port": 8002
    }

@app.get("/health")
def health():
    return {
        "status": "healthy" if predictor else "unhealthy",
        "model_loaded": predictor is not None}

@app.post("/predict")
async def predict(request: PredictionRequest):
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = predictor.predict_single_order(request.order_data)

        return {
            "predicted_days": result["predicted_days"],
            "predicted_exact_days": result["predicted_exact_days"],
            "delivery_category": result["delivery_category"],
            "confidence_interval": result["confidence_interval"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        print(f"\n Batch prediction: {len(request.orders)} orders")
        
        # Convert orders list to DataFrame
        import pandas as pd
        orders_df = pd.DataFrame(request.orders)
        
        batch_results = predictor.batch_predict(orders_df)
        
        # Convert DataFrame results back to list of dicts
        results = []
        for idx, row in batch_results.iterrows():
            results.append({
                "predicted_days": row.get("predicted_days", 0),
                "predicted_days_exact": row.get("predicted_days_exact", 0.0),
                "delivery_category": row.get("delivery_category", "unknown"),
                "confidence_interval": {
                    "lower": row.get("ci_lower", 0),
                    "upper": row.get("ci_upper", 0)
                },
                "order_id": row.get("order_id", f"order_{idx}")
            })
        
        print(f" Batch complete: {len(results)} results")
        
        return {
            "success": True,
            "total": len(request.orders),
            "processed": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        print(f"✗ Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    print("Starting Delivery Predictor Service on port 8002...")
    uvicorn.run("api.services.delivery_predictor_service:app",
                host="0.0.0.0",
                port=8002,
                reload=True)