
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Body
from pydantic import BaseModel, Field
import pandas as pd
import json

from typing import Optional, Dict, List, Any
from datetime import datetime
import os
import base64
import io
from PIL import Image
import tempfile

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Import Predictor
from src.integrated_system.unified_predictor import EcommerceQualityControlSystem

router = APIRouter()

# Initialize system
print("Initializing E-commerce Quality Control System...")
system = EcommerceQualityControlSystem()
print(" System ready!")

# pydantic models
class OrderAnalyzeRequest(BaseModel):
    order_data: Dict[str, Any] = Field(
        ...,
        description="Dictionary containing all order features - any fields accepted"),
    product_image: Optional[str] = None

class BatchAnalyzeRequest(BaseModel):
    orders: List[Dict[str, Any]]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def save_base64_image(base64_string: str)-> str:
    """Save base64 image to temporary file and return path"""
    try:
        # Decode base64
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix="png")
        image.save(temp_file)
        temp_file.close()

        return temp_file.name

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
    
# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.get("/")
def home():
    """API root endpoint"""
    return {
        "message": "E-commerce Quality Control API",
        "version": "1.0.0",
        "system": "Unified QC System with 4 ML Models",
        "endpoints": {
            "order_analysis": "/api/analyze/order",
            "image_analysis": "/api/analyze/image",
            "combined_analysis": "/api/analyze/combined",
            "batch_analysis": "/api/analyze/batch",
            "health": "/health"
        },
        "models": {
            "1": "Quality Prediction (Tabular Classification)",
            "2": "Delivery Time Prediction (Regression)",
            "3": "Product Image Classification",
            "4": "Object Detection"
        }
    }

@router.get("/health")
def health_check():
    """Health check endpoint"""
    active_models = list(system.models.keys())
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_models": active_models,
        "total_models": len(active_models)
    }

# ============================================================================
# ORDER ANALYSIS
# ============================================================================
@router.post("api/analyze/order")
async def analyze_order(request: OrderAnalyzeRequest):
    """
    Analyze order quality and delivery time
    
    Uses:
    - Quality Prediction (Tabular Classification)
    - Delivery Time Prediction (Regression)
    
    Returns comprehensive order analysis with risk assessment
    """
    try:
        # Convert request to dict
        order_data = request.dict()
    
        # Analyze order (no image)
        result = system.analyze_order(order_data, product_image=None)
        return {
            "success": True,
            "analysis": result,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                'error': str(e),
                'message': 'Analysis failed'
            }
        )
# ============================================================================
# IMAGE ANALYSIS
# ============================================================================
@router.post("/api/analyze/image")
async def analyze_image(
    image: UploadFile = File(...),
    include_object_detection: bool = Form(True)
):
    """
    Analyze product image
    
    Uses:
    - Image Classification (product category)
    - Object Detection (optional - detects items in image)
    
    Returns image classification and object detection results
    """
    try:
        # Save uploaded image to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix="png")
        content = await image.read()
        temp_file.write(content)
        temp_file.close()

        # Create minimal order data for image analysis
        order_data = {
            "order_id": f"IMG_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        # Analyze with image
        result = system.analyze_order(order_data, product_image=temp_file.name)

         # Clean up temp file
        os.unlink(temp_file.name)

        # Extract only image-related predictions
        image_result = {
            "image_classification": result["predictions"].get("image_classification"),
            "object_detection": result["predictions"].get("object_detection") if include_object_detection else None
        }
        return {
            "success": True,
            "analysis": image_result,
            "flags": result.get("flags", []),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# COMBINED ANALYSIS (Order + Image)
# ============================================================================
@router.post("/api/analyze/combined")
async def analyze_combine(
    order_data: str = Body(...), # JSON string of order data
    image: Optional[UploadFile] = File(None)):
    """
    Complete order analysis with optional product image
    
    Uses ALL 4 models:
    - Quality Prediction
    - Delivery Time Prediction
    - Image Classification (if image provided)
    - Object Detection (if image provided)
    
    Returns comprehensive analysis combining all models
    """
    try:
        # Parse order data JSON
        try:
            order_dict = json.loads(order_data)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="order_data must be valid JSON string"
            )
        # Validate required field
        if "order_id" not in order_dict:
            raise HTTPException(
                status_code=400,
                detail="order_id is required in order_data"
            )
        # Handle image if provided
        image_path = None
        if image:
            image_path = save_base64_image(image)

        # Run comprehensive analysis
        result = system.analyze_order(order_dict, product_image=image_path)

        # Generate report
        report = system.generate_report(result)

        # Clean up temp file
        if image_path:
            os.unlink(image_path)
        
        return {
            "sucess": True,
            "analysis": result,
            "report": report,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# ============================================================================
# BATCH ANALYSIS
# ============================================================================

# Batch analyze endpoint
@router.post("/api/analyze/batch")
async def batch_analyze(request: BatchAnalyzeRequest):
    """
    Analyze multiple orders in batch
    
    Example:
    {
        "orders": [
            {"order_id": "001", "price": 100, "feature_x": "A", ...},
            {"order_id": "002", "price": 200, "feature_y": "B", ...},
            ... different features per order are OK!
        ]
    }
    
    """
    try:
        # Convert to DataFrame
        orders_df = pd.DataFrame(request.order)

        # Run batch analysis
        results = system.batch_analyze_order(orders_df)

        # Convert results to JSON-serializable format
        results_list = results.to_dict("records")

        # Calculate summary statistics
        summary = {
            "total_orders": len(results_list),
            "risk_distribution": {},
            "flag_summary": {}
        }

        if results_list:
            # Risk distribution
            risk_counts = {}
            for result in results_list:
                assessment = result.get("overall_assessment", "unkown")
                risk_counts["assessment"] = risk_counts.get(assessment, 0) + 1
            summary["risk_distribution"] = risk_counts

            # Flags summary
            all_flags = []
            for result in results_list:
                all_flags.extend(results.get("flags", []))

            flag_counts = {}
            for flag in all_flags:
                flag_counts["flag"] = flag_counts.get("flag", 0) + 1
            summary["flag_summary"] = flag_counts
            
        return {
            "success": True,
            "summary": summary,
            "results": results_list,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@router.post("/api/report/generate")
async def generate_report(analysis_result: Dict[str, Any] = Body(...)):
    """
    Generate human-readable report from analysis result
    
    Pass the complete analysis result from any analysis endpoint
    """
    try:
        report = system.generate_report(analysis_result)

        return {
            "success": True,
            "report": report,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# ============================================================================
# SYSTEM INFO
# ============================================================================
@router.get("/api/system/info")
async def get_system_info():
    """Get information about the QC system and loaded models"""

    model_status = {}
    for model_name, model in system.models.items():
        model_status[model_name] = {
            "loaded": True,
            "type": type(model).__name__
        }

    return {
        "system": "E-commerce Quality Control System",
        "description": "Accepts any order features dynamically",
        "models": model_status,
        "total_models": len(system.models),
        "config": system.config,
        "timestamp": datetime.now().isoformat()
    }

@router.get("/api/examples/order")
async def get_example_order():
    """
    Get an example order structure
    
    This is just an example - you can send ANY fields!
    """
    return {
        "message": "Example order structure (you can add/remove any fields)",
        "example": {
            "order_id": "ORDER_12345",
            "total_price": 150.00,
            "total_freight": 25.00,
            "num_items": 2,
            "total_weight_kg": 3.5,
            "primary_category": "electronics",
            "customer_state": "SP",
            "seller_state": "RJ",
            "is_same_state": 0,
            "promised_delivery_days": 10,
            "primary_payment_type": "credit_card",
            "max_installments": 3,
            "purchase_month": 6,
            "is_weekend": 0,
            "is_holiday_season": 0,
            "# ADD ANY CUSTOM FIELDS HERE": "...",
            "custom_feature_1": "any_value",
            "custom_feature_2": 123,
            "custom_feature_n": "works!"
        },
        "note": "The API is flexible - add or remove any fields based on your model's needs!"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.model_api:router",
        host="0.0.0.0",
        port=8000,
        reload=True)