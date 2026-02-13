"""
Main Orchestrator API - Microservices Architecture
Coordinates independent model services without loading any models itself
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import httpx
import asyncio
from datetime import datetime
import json

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


app = FastAPI(
    title="E-commerce QC Orchestrator",
    description="Coordinates independent model services",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SERVICE REGISTRY - Configure your model service URLs
SERVICES = {
    "quality_predictor": {
        "url": "http://localhost:8001",
        "enabled": True,
        "timeout": 30,
        "supports_batch": True
    },
    "delivery_predictor": {
        "url": "http://localhost:8002",
        "enabled": True,
        "timeout": 30,
        "supports_batch": True
    },
    "image_classifier": {
        "url": "http://localhost:8003",
        "enabled": True,
        "timeout": 60,                   # Longer for image processing
        "supports_batch": True                   
    },
    "object_detector": {
        "url": "http://localhost:8004",
        "enabled": True,
        "timeout": 60,
        "supports_batch": True
    }
}

# PYDANTIC MODELS
class OrderRequest(BaseModel):
    order_data: Dict[str, Any]

class BatchOrderRequest(BaseModel):
    orders: List[Dict[str, Any]]

    class Config:
        json_schema_extra = {
            "example": {
                "orders": [
                    {"order_id": "001", "total_price": 150, "category": "electronics"},
                    {"order_id": "002", "total_price": 89, "category": "clothing"},
                    {"order_id": "003", "total_price": 200, "category": "furniture"}
                ]
            }
        }

# HELPER FUNCTIONS
async def call_service(service_name: str, endpoint: str, data: dict = None, files: dict = None):
    """
    Call a model service asynchronously
    Returns: (success: bool, result: dict, error: str)
    """
    service_config = SERVICES.get(service_name)

    if not service_config and not service_config["enabled"]:
        return False, None, f"Service {service_name} not enabled"
    
    url = f"{service_config['url']}--{endpoint}"
    timeout = service_config["timeout"]

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if files:
                response = await client.post(url, data=data, files=files)
            else:
                response = await client.post(url, json=data)
            
            if response.status_code == 200:
                return True, response.json(), None
            else:
                return False, None, f"Service returned {response.status_code}: {response.text}"
    
    except httpx.TimeoutException:
        return False, None, f"Service timeout after {timeout}s"
    except httpx.ConnectError:
        return False, None, f"Cannot connect to service at {url}"
    except Exception as e:
        return False, None, str(e)
    

async def call_batch_service(service_name: str, orders: List[Dict[str, Any]]):
    """
    Call a service's batch endpoint
    Returns: (success, results_list, error)
    """
    service_config = SERVICES.get(service_name)
    
    if not service_config or not service_config["enabled"]:
        return False, None, f"Service {service_name} not enabled"
    
    url = f"{service_config['url']}/predict/batch"
    timeout = service_config["timeout"] * 2  # Double timeout for batch
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json={"orders": orders})
            
            if response.status_code == 200:
                return True, response.json(), None
            else:
                return False, None, f"Batch failed: {response.status_code}"
    
    except Exception as e:
        return False, None, str(e)
    
async def check_service_health(service_name: str):
    """Check if a service is healthy"""
    service_config = SERVICES.get(service_name)

    if not service_config:
        return False
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{service_config['url']}/health")
            return response.status_code == 200
    except:
        return False
    

def aggregate_batch_results(quality_results, delivery_results, orders):
    """
    Aggregate results from multiple services for batch processing
    
    Args:
        quality_results: List of quality predictions
        delivery_results: List of delivery predictions
        orders: Original order list
    
    Returns:
        List of aggregated results
    """
    aggregated = []
    
    for idx, order in enumerate(orders):
        order_id = order.get("order_id", f"order_{idx}")
        
        result = {
            "order_id": order_id,
            "predictions": {},
            "flags": [],
            "recommendations": [],
            "overall_risk_score": 0.0,
            "overall_assessment": "UNKNOWN"
        }
        
        risk_scores = []
        
        # Quality prediction
        if quality_results and idx < len(quality_results):
            q = quality_results[idx]
            if "error" not in q:
                result["predictions"]["quality"] = q
                
                if q.get("should_flag"):
                    result["flags"].append("HIGH_QUALITY_RISK")
                    result["recommendations"].append(
                        f"High quality risk ({q.get('risk_score', 0):.0%}) - Additional QC needed"
                    )
                risk_scores.append(q.get("risk_score", 0))
        
        # Delivery prediction
        if delivery_results and idx < len(delivery_results):
            d = delivery_results[idx]
            if "error" not in d:
                result["predictions"]["delivery"] = d
                
                predicted_days = d.get("predicted_days", 0)
                if predicted_days > 15:
                    result["flags"].append("LONG_DELIVERY_TIME")
                    result["recommendations"].append(
                        f"Estimated delivery: {predicted_days} days - Consider expedited shipping"
                    )
                # Normalize to 0-1 risk score
                risk_scores.append(min(predicted_days / 30, 1.0))
        
        # Calculate overall risk
        if risk_scores:
            result["overall_risk_score"] = sum(risk_scores) / len(risk_scores)
            
            if result["overall_risk_score"] > 0.7:
                result["overall_assessment"] = "HIGH_RISK"
                result["recommendations"].insert(0, " HIGH RISK ORDER - Thorough review recommended")
            elif result["overall_risk_score"] > 0.4:
                result["overall_assessment"] = "MEDIUM_RISK"
                result["recommendations"].insert(0, " MEDIUM RISK ORDER - Standard QC applies")
            else:
                result["overall_assessment"] = "LOW_RISK"
                result["recommendations"].insert(0, " LOW RISK ORDER - Standard processing")
        
        aggregated.append(result)
    
    return aggregated

    
# API ENDPOINTS
@app.get("/")
def root():
    return {
        "service": "E-commerce QC Orchestrator",
        "architecture": "microservices",
        "description": "Coordinates 4 independent model services",
        "services": {
            "quality_predictor": "http://localhost:8001 - Tabular Classification",
            "delivery_predictor": "http://localhost:8002 - Regression",
            "image_classifier": "http://localhost:8003 - Image Classification",
            "object_detector": "http://localhost:8004 - Object Detection"
        },
        "endpoints": {
            "health": "GET /health - Check all services",
            "analyze_order": "POST /api/analyze/order",
            "analyze_image": "POST /api/analyze/image",
            "analyze_combined": "POST /api/analyze/combined"
        }
    }

@app.get("/health")
async def health_check():
    """Check health of all services"""
    service_health = {}
    
    # Check all services concurrently
    tasks = [
        check_service_health(service_name)
        for service_name in SERVICES.keys()
    ]
    
    results = await asyncio.gather(*tasks)
    
    for service_name, is_healthy in zip(SERVICES.keys(), results):
        service_health[service_name] = "healthy" if is_healthy else "unavailable"
    
    all_healthy = all(results)
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.now().isoformat(),
        "services": service_health
    }

@app.post("/api/analyze/order")
async def analyze_order(request: OrderRequest):
    """
    Analyze order using quality and delivery prediction services
    Calls services in parallel for better performance
    """
    order_data = request.order_data

    if "order_id" not in order_data:
        raise HTTPException(status_code=400, detail="order_id required")
    
    # Call both services concurrently
    tasks = []
    service_names = []

    if SERVICES["quality_predictor"]["enabled"]:
        tasks.append(call_service("quality_predictor", "/predict", {"order_data": order_data}))
        service_names.append("quality_predictor")

    if SERVICES["delivery_predictor"]["enabled"]:
        tasks.append(call_service("delivery_predictor", "/predict", {"order_data":order_data}))
        service_names.append("delivery_predictor")

    results = await asyncio.gather(*tasks)

    # Aggregate results
    predictions = {}
    service_status = {}
    flags = []
    recommendations = []
    risk_scores = []

    for service_name, (success, result, error) in zip(service_names, results):
        if success:
            service_status[service_name] = "success"

            if service_name == "quality_predictor":
                predictions["quality"] = result
                if result.get("should_flag"):
                    flags.append("HIGH_QUALITY_RISK")
                    recommendations.append(f"High quality risk - Additional QC needed")
                risk_scores.append(result.get("risk_score"))
            elif service_name == "delivery_predictor":
                predictions["delivery"] = result
                if result.get("predicted_days", 0) > 15:
                    flags.append("LONG_DELIVERY_TIME")
                risk_scores.append(min(result.get("predicted_days", 0)/30, 1.0))
        else:
            service_status[service_name] = f"error: {error}"
            predictions[service_name] = {"error": error}

    # Overall assessment
    overall_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0

    if overall_risk_score > 0.7:
        overall_assessment = "HIGH_RISK"
    elif overall_risk_score > 0.4:
        overall_assessment = "MEDIUM_RISK"
    else:
        overall_assessment = "LOW_RISK"
    
    return {
        "order_id": order_data["order_id"],
        "timestamp": datetime.now().isoformat(),
        "predictions": predictions,
        "flags": flags,
        "recommendations": recommendations,
        "overall_risk_score": overall_risk_score,
        "overall_assessment": overall_assessment,
        "service_status": service_status
    }

@app.post("/api/analyze/batch")
async def analyze_batch(request: BatchOrderRequest):
    """
    Batch order analysis - Optimized for performance
    
    Strategy:
    1. Services that support batch → Send entire batch
    2. Services without batch → Process individually in parallel
    3. Aggregate all results
    """
    orders = request.orders

    if not orders:
        raise HTTPException(status_code=400, detail="No orders provided")
    
    # Validate all orders have order_id
    for idx, order in enumerate(orders):
        if "order_id" not in order:
            raise HTTPException(status_code=400,detail=f"Order at index {idx} missing order_id")
        
    print(f"\n{'='*80}")
    print(f"BATCH ANALYSIS: {len(orders)} orders")
    print(f"{'='*80}\n")

    # Track results
    quality_results = None
    delivery_results = None
    service_status = {}

    # STRATEGY 1: Call batch endpoints for services that support it
    batch_tasks = []
    batch_services = []

    if SERVICES["quality_predictor"]["enabled"]:
        print(f" Sending batch to Quality Predictor ({len(orders)} orders)")
        batch_tasks.append(call_batch_service("quality_predictor", orders))
        batch_services.append("quality_predictor")

    if SERVICES["delivery_predictor"]["enabled"]:
        print(f" Sending batch to Delivery Predictor ({len(orders)} orders)")
        batch_tasks.append(call_batch_service("delivery_predictor", orders))
        batch_services.append("delivery_predictor")

    # Execute batch calls in parallel
    if batch_tasks:
        batch_results = await asyncio.gather(*batch_tasks)

        for service_name, (success, result, error) in zip(batch_services, batch_results):
            if success:
                service_status[service_name] = "success"
                print(f" {service_name} batch completed")
                
                if service_name == "quality_predictor":
                    quality_results = result.get("results", [])
                elif service_name == "delivery_predictor":
                    delivery_results = result.get("results", [])
            else:
                service_status[service_name] = f"error: {error}"
                print(f" {service_name} batch failed: {error}")
    
    # Aggregate all results
    aggregated_results = aggregate_batch_results(
        quality_results,
        delivery_results,
        orders
    )

    # Calculate summary statistics
    summary = {
        "total_orders": len(orders),
        "processed": len(aggregated_results),
        "risk_distribution": {},
        "flags_summary": {},
        "average_risk_score": 0.0
    }

    # Risk distribution
    risk_counts = {}
    total_risk = 0
    for result in aggregated_results:
        assessment = result["overall_assessment"]
        risk_counts[assessment] = risk_counts.get(assessment, 0) + 1
        total_risk += result["overall_risk_score"]

    summary["risk_distribution"] = risk_counts
    summary["average_risk_score"] = total_risk / len(aggregated_results) if aggregated_results else 0
    
    # Flags summary
    all_flags = []
    for result in aggregated_results:
        all_flags.extend(result["flags"])
    
    flag_counts = {}
    for flag in all_flags:
        flag_counts[flag] = flag_counts.get(flag, 0) + 1
    summary["flags_summary"] = flag_counts

    print(f"\n{'='*80}")
    print(f"BATCH COMPLETE:")
    print(f"  Total: {summary['total_orders']}")
    print(f"  High Risk: {risk_counts.get('HIGH_RISK', 0)}")
    print(f"  Medium Risk: {risk_counts.get('MEDIUM_RISK', 0)}")
    print(f"  Low Risk: {risk_counts.get('LOW_RISK', 0)}")
    print(f"  Avg Risk Score: {summary['average_risk_score']:.2%}")
    print(f"{'='*80}\n")
    
    return {
        "success": True,
        "summary": summary,
        "results": aggregated_results,
        "service_status": service_status,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/analyze/image")
async def analyze_image(image: UploadFile = File(...)):
    image_bytes = await image.read()

    tasks = []
    service_names = []

    if SERVICES["image_classifier"]["enabled"]:
        files = {"image": ("image.jpg", image_bytes, "image/jpeg")}
        tasks.append(call_service("image_classifier", "/predict", files=files))
        service_names.append("image_classifier")

    if SERVICES["object_detector"]["enabled"]:
        files = {"image": ("image.jpg", image_bytes, "image/jpeg")}
        tasks.append(call_service("object_detector", "/detect", files=files))
        service_names.append("object_detector")

    results = await asyncio.gather(*tasks)

    predictions = {}
    service_status = {}
    flags = []
    for service_name, (success, result, error) in zip(service_names, results):
        if success:
            service_status[service_name] = "success"
            predictions[service_name] = result
        else:
            service_status[service_name] = f"error: {error}"
            predictions[service_name] = {"error": error}

    return {
        "timestamp": datetime.now().isoformat(),
        "predictions": predictions,
        "flags": flags,
        "service_status": service_status
    }

@app.post("/api/analyze/combined")
async def analyze_combined(
    order_data: str = Body(...),
    image: Optional[UploadFile] = File(None)
):
    """
    Complete analysis with order data + optional image
    
    Uses all 4 services:
    - Quality Prediction
    - Delivery Prediction
    - Image Classification (if image provided)
    - Object Detection (if image provided)
    """
    # Parse order data
    try:
        order_dict = json.load(order_data)
    except:
        raise HTTPException(status_code=400, detail="Invalid order_data JSON")
    
    if "order_id" not in order_dict:
        raise HTTPException(status_code=400, detail="order_id required")
    
    tasks = []
    service_names = []

    # Order-based services
    if SERVICES["quality_predictor"]["enabled"]:
        tasks.append(call_service("quality_predictor", "/predict", order_dict))
        service_names.append("quality_predictor")

    if SERVICES["delivery_predictor"]["enabled"]:
        tasks.append(call_service("delivery_predictor", "/predict", order_dict))
        service_names.append("delivery_predictor")

    # Image-based services
    if image:
        image_bytes = await image.read()
        if SERVICES["image_classifier"]["enabled"]:
            files = {"image": ("image.jpg", image_bytes, "image/jpeg")}
            tasks.append(call_service("image_classifier", "/predict", files=files))
            service_names.append("image_classifier")

        if SERVICES["object_detector"]["enabled"]:
            files = {"image": ("image.jpg", image_bytes, "image/jpeg")}
            tasks.append(call_service("object_detector", "/detect", files=files))
            service_names.append("object_detector")

    # Execute all calls in parallel
    results = await asyncio.gather(*tasks)

    predictions = {}
    service_status = {}
    flags = []
    risk_scores = []
    for service_name, (success, result, error) in zip(service_names, results):
        if success:
            service_status[service_name] = "success"
            predictions[service_name] = result

            if service_name == "quality_predictor" and result.get("should_flag"):
                flags.append("HIGH_QUALITY_RISK")
                risk_scores.append(result.get("risk_score", 0))
            elif service_name == "delivery_predictor":
                risk_scores.append(min(result.get("predicted_days", 0)/30, 1.0))
        else:
            service_status[service_name] = f"error: {error}"
            predictions[service_name] = {"error": error}
        
    overall_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0

    if overall_risk_score > 0.7:
        overall_assessment = "HIGH_RISK"
    elif overall_risk_score > 0.4:
        overall_assessment = "MEDIUM_RISK"
    else:
        overall_assessment = "LOW_RISK"

    return {
        "order_id": order_dict["order_id"],
        "timestamp": datetime.now().isoformat(),
        "predictions": predictions,
        "flags": flags,
        "overall_risk_score": overall_risk_score,
        "overall_assessment": overall_assessment,
        "service_status": service_status
    }


@app.get("/api/services/status")
async def services_status():
    """Get detailed status of all services"""
    service_health = {}

    for service_name, config in SERVICES.items():
        is_healthy = await check_service_health(service_name)
        service_health[service_name] = {
            "url": config["url"],
            "enabled": config["enabled"],
            "status": "healthy" if is_healthy else "unavailable",
            "supports_batch": config.get("supports_batch", False),
            "timeout": config["timeout"]
        }
    return {
        "timestamp": datetime.now().isoformat(),
        "services": service_health
    }


if __name__ == "__main__":
    import uvicorn
    
    print("="*80)
    print("E-COMMERCE QC ORCHESTRATOR")
    print("="*80)
    print("\nService Registry:")
    for name, config in SERVICES.items():
        batch_support = " Batch" if config.get("supports_batch") else " No Batch"
        enabled = " Enabled" if config["enabled"] else " Disabled"
        print(f"  - {name:20} {config['url']:30} {batch_support:10} {enabled}")
    print("\nStarting orchestrator on port 8000...")
    print("API Docs: http://localhost:8000/docs")
    print("="*80)
    uvicorn.run(
        "api.orchestrator_api:app",
        host="0.0.0.0",
        port=8000)