"""
Quality Predictor Service (Tabular Classification)
Independent microservice - Port 8001
Only loads quality prediction model and dependencies
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
from src.tabular_classification.predict import QualityPredictor

app = FastAPI(
    title="Quality Predictor Service",
    description="Tabular classification for order quality assessment",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# LOAD MODEL (only this service's model)
# ============================================================================
print("Loading Quality Prediction Model...")
try:
    predictor = QualityPredictor(
        model_path="models/tabular_classification/saved_models/best_model_random_forest.plk",
        preprocessing_path="models/tabular_classification/preprocessing"
    )
    print(" Quality Predictor loaded successfully")
except Exception as e:
    print(f" Failed to load model: {e}")
    predictor = None

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class PredictionRequest(BaseModel):
    order_data: Dict[str, Any]

class BatchPredictionRequest(BaseModel):
    orders: Dict[str, Any]

# ============================================================================
# ENDPOINTS
# ============================================================================
@app.get("/")
def root():
    return {
        "service": "Quality Predictor",
        "model": "Tabular Classification",
        "port": 8001,
        "status": "ready" if predictor else "model not loaded"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy" if predictor else "unhealthy",
        "model_loaded": predictor is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Predict order quality
    
    Returns:
    - risk_score: float (0-1)
    - risk_level: str (LOW/MEDIUM/HIGH)
    - prediction: str (quality label)
    - should_flag: bool
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = predictor.predict_single_order(request.order_data, threshold=0.5)

        return {
            "risk_score": result["risk_score"],
            "risk_level": result["risk_level"],
            "prediction": result["quality_label"],
            "should_flag": result["should_flag"],
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

        results = predictor.batch_predict(orders_df, threshold=0.35)

        # Convert DataFrame results back to list of dicts
        results_list = []
        for idx, row in results.iterrows():
            results_list.append({
                "risk_score": row.get("risk_score", 0),
                "risk_level": row.get("risk_level", "UNKNOWN"),
                "prediction": row.get("quality_label", "unknown"),
                "should_flag": row.get("should_flag", False),
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
        print(f" Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("Starting Quality Predictor Service on port 8001...")
    uvicorn.run(
        "api.services.quality_predictor_service:app",
        host="0.0.0.0",
        port=8001,
        reload=True)