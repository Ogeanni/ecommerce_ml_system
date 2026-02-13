"""
Object Detector Service
Independent microservice - Port 8004
Only loads object detection model (YOLO)
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os
from pathlib import Path
import tempfile
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ONLY import what this service needs
from src.object_detection.detect import ObjectDetector

app = FastAPI(
    title="Object Detector Service",
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
    predictor = ObjectDetector(
        weights_path="models/object_detection/runs/object_detection/train/weights/best.pt",
        data_yaml="data/object_detection/dataset.yaml"
    )
    print(" Object Detector loaded successfully")
except Exception as e:
    print(f" Failed to load model: {e}")
    predictor = None

@app.get("/")
def root():
    return {
        "service": "Object Detector",
        "model": "YOLO",
        "port": 8004
    }

@app.get("/health")
def health():
    return {
        "status": "healthy" if predictor else "unhealthy",
        "model_loaded": predictor is not None
    }

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        # Save image temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix="png")
        content = await image.read()
        temp_file.write(content)
        temp_file.close()

        # Detect
        result = predictor.detect(temp_file.name)

        # Cleanup
        os.unlink(temp_file.name)

        return {
            "num_detections": result["num_detections"],
            "detections": [
            {
                "class": det["class"],
                "confidence": det["confidence"],
                "bbox": det["bbox"]
            }
            for det in result["detections"]
            ],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict/batch")
async def predict_batch(images: UploadFile = File(...)):
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        print(f"\n Batch object detection: {len(images)} images")
        
        # Save all images temporarily
        temp_files = []
        for idx, image in enumerate(images):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            content = await image.read()
            temp_file.write(content)
            temp_file.close()
            temp_files.append(temp_file.name)
 
        batch_results = predictor.batch_detect(temp_files)
        
        # Convert results to response format
        results = []
        for idx, result in enumerate(batch_results):
            results.append({
                "num_detections": result["num_detections"],
                "detections": [
                    {
                        "class": det["class"],
                        "confidence": det["confidence"],
                        "bbox": det["bbox"]
                    }
                    for det in result["detections"]
                ],
                "image_index": idx
            })
        
        # Cleanup temp files
        for temp_file in temp_files:
            os.unlink(temp_file)
        
        print(f" Batch complete: {len(results)} detections")
        
        return {
            "success": True,
            "total": len(images),
            "processed": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        print(f" Batch detection failed: {e}")
        # Cleanup on error
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("Starting Object Detector Service on port 8004...")
    uvicorn.run("api.services.object_detector_service:app",
                host="0.0.0.0",
                port=8004)