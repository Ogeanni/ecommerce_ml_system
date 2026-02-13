"""
Image Classifier Service
Independent microservice - Port 8003
Only loads image classification model (TensorFlow/Keras)
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tempfile
from datetime import datetime
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ONLY import what this service needs
from src.image_classification.predict import ImageClassificationPredictor

app = FastAPI(
    title="Image Classifier Service",
    version="1.0.0"
)

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])

# Load model
print("Loading Image Classification Model...")
try:
    predictor = ImageClassificationPredictor(
        model_path="models/image_classification/saved_models/fashion_mnist_transfer_final"
    )
    print(" Image Classifier loaded successfully")
except Exception as e:
    print(f" Failed to load model: {e}")
    predictor = None

@app.get("/")
def root():
    return {"service": "Image Classifier",
            "model": "CNN/Transfer Learning",
            "port": 8003}

@app.get("/health")
def health():
    return {"status": "healthy" if predictor else "unhealthy",
            "model_loaded": predictor is not None}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        # Save image temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix="png")
        content = await image.read()
        temp_file.write(content)
        temp_file.close()

        # Predict
        result = predictor.predict_single(temp_file.name)

        # Cleanup
        os.unlink(temp_file.name)

        return {
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "top_predictions": result["top_predictions"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict/batch")
async def predict_batch(images: UploadFile = File(...)):
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        print(f"\n→ Batch image prediction: {len(images)} images")
        
        # Save all images temporarily
        temp_files = []
        for idx, image in enumerate(images):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix="png")
            content = await images.read()
            temp_file.write(content)
            temp_file.close()
            temp_files.append(temp_file.name)

        results = predictor.predict_batch(temp_files)

        # Convert results to response format
        results_list = []
        for idx, result in enumerate(results):
            results_list.append({
                "predicted_class": result["predicted_class"],
                "confidence": result["confidence"],
                "top_predictions": result.get("top_predictions", []),
                "image_index": idx
            })
        # Cleanup temp files
        for temp_file in temp_files:
            os.unlink(temp_file)
        
        print(f" Batch complete: {len(results)} predictions")

        return {
            "success": True,
            "total": len(images),
            "processed": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f" Batch prediction failed: {e}")
        # Cleanup on error
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))




if __name__ == "__main__":
    import uvicorn
    print("Starting Image Classifier Service on port 8003...")
    uvicorn.run("api.services.image_classifier_service:app",
                host="0.0.0.0",
                port=8003)

