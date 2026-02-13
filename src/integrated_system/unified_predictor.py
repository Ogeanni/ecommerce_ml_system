"""
Unified prediction system combining all four ML models
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Union
import json
import warnings
warnings.filterwarnings('ignore')

# Import all model predictors
from src.tabular_classification.predict import QualityPredictor
from src.regression.predict import DeliveryTimePredictor
from src.image_classification.predict import ImageClassificationPredictor
from src.object_detection.detect import ObjectDetector

class EcommerceQualityControlSystem:
    """
    Unified E-commerce Quality Control System
    Combines all four ML models for comprehensive analysis
    """
    def __init__(self, config_path: str = "config/system_config.json"):
        """
        Initialize unified system
        
        Args:
            config_path: Path to system configuration
        """
        print("="*80)
        print("INITIALIZING E-COMMERCE QUALITY CONTROL SYSTEM")
        print("="*80)
        
        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize models
        self.models = {}
        self._initialize_models()

        print("\n System initialized successfully!")
        print(f"   Active models: {len(self.models)}")

    def _load_config(self, config_path: str)-> dict:
        """Load system configuration"""
        if Path(config_path).exists():
            with open(config_path, "r") as f:
                return json.load(f)

        else:
            # Default configuration
            return {
                "models": {
                    "tabular_classification": {
                        "enabled": True,
                        "model_path": "models/tabular_classification/saved_models/best_model_random_forest.pkl",
                        "preprocessing_path": "models/tabular_classification/preprocessing"
                    },
                    "regression": {
                        "enabled": True,
                        "model_path": "models/regression/saved_models/best_model_xgboost.pkl",
                        "preprocessing_path": "models/regression/preprocessing"
                    },
                    "image_classification": {
                        "enabled": True,
                        "model_path": "models/image_classification/saved_models/fashion_mnist_transfer_final.h5"
                    },
                    "object_detection": {
                        "enabled": True,
                        "weights_path": "models/object_detection/runs/object_detection/train/weights/best.pt",
                        "data_yaml": "daa/object_detection/dataset.yaml"
                    }
                },
                "thresholds": {
                    "quality_risk": 0.35,
                    "delivery_delay_warning": 15
                }
            }
        
    
    def _initialize_models(self):
        """Initialize all enabled models"""
        print("\nInitializing models...")
        
        # 1. Tabular Classification (Quality Prediction)
        if self.config["models"]["tabular_classification"]["enabled"]:
            try:
                print("\n[1/4] Loading Tabular Classification model...")
                self.models["quality_predictor"] = QualityPredictor(
                    model_path=self.config["models"]["tabular_classification"]["model_path"],
                    preprocessing_path=self.config["models"]["tabular_classification"]["preprocessing"])
                print("   Quality prediction model loaded")
            except Exception as e:
                print(f"   Failed to load quality predictor: {e}")

        # 2. Regression (Delivery Time Prediction)
        if self.config["models"]["regression"]["enabled"]:
            try:
                print("\n[2/4] Loading Regression model...")
                self.models["delivery_predictor"] = DeliveryTimePredictor(
                    model_path=self.config["models"]["regression"]["model_path"],
                    preprocessing_path=self.config["models"]["regression"]["preprocessing"]
                )
                print("   Delivery time prediction model loaded")
            except Exception as e:
                print(f"   Failed to load delivery predictor: {e}")

        # 3. Image Classification
        if self.config["models"]["image_classification"]["enabled"]:
            try:
                print("\n[3/4] Loading Image Classification model...")
                self.models["image_classifier"] = ImageClassificationPredictor(
                    model_path=self.config["models"]["image_classification"]["model_path"]
                )
                print("   Image classification model loaded")
            except Exception as e:
                print(f"   Failed to load image classifier: {e}")

        # 4. Object Detection
        if self.config["models"]["object_detection"]["enabled"]:
            try:
                print("\n[4/4] Loading Object Detection model...")
                self.models["object_detector"] = ObjectDetector(
                    weights_path=self.config["models"]["object_detection"]["weights_path"],
                    data_yaml=self.config["data"]["object_detection"]["data_yaml"]
                )
                print("   Object detection model loaded")
            except Exception as e:
                print(f"   Failed to load object detector: {e}")

    def analyze_order(self,
                      order_data: Dict,
                      product_image: str = None)-> Dict:
        """
        Comprehensive order analysis using all models
        
        Args:
            order_data: Dictionary with order features
            product_image: Path to product image (optional)
            
        Returns:
            Complete analysis results
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE ORDER ANALYSIS")
        print("="*80)

        results = {
            "order_id": order_data.get("order_id", "unknown"),
            "timestamp": pd.Timestamp.now().isoformat(),
            "predictions": {},
            "recommendations": [],
            "overall_risk_score": 0.0,
            "flags": []
        }
        # 1. Quality Prediction (Tabular Classification)
        if "quality_predictor" in self.models:
            print("\n[1/4] Predicting order quality...")
            try:
                quality_result = self.models["quality_predictor"].predict_single_order(
                    order_data,
                    threshold=self.config["thresholds"]["quality_risk"])
                
                results["predictions"]["quality"] = {
                    "risk_score": quality_result["risk_score"],
                    "risk_level": quality_result["risk_level"],
                    "prediction": quality_result["quality_label"],
                    "should_flag": quality_result["should_flag"]
                }

                if quality_result["should_flag"]:
                    results["flags"].append("HIGH_QUALITY_RISK")
                    results["recommendations"].append(f"High quality risk ({quality_result['risk_score']:.0%}) - "
                        "Consider additional QC checks before shipping")
                print(f"  Quality Risk: {quality_result['risk_level']} ({quality_result['risk_score']:.0%})")

            except Exception as e:
                print(f"   Quality prediction failed: {e}")
                results['predictions']['quality'] = {'error': str(e)}
        
        # 2. Delivery Time Prediction (Regression)
        if "delivery_predictor" in self.models:
            print("\n[2/4] Predicting delivery time...")
            try:
                delivery_result = self.models["delivery_predictor"].predict_single_order(order_data)

                results["predictions"]["delivery"] = {
                    "predicted_days": delivery_result["predicted_days"],
                    "exact_days": delivery_result["predicetd_days_exact"],
                    "category": delivery_result["delivery_category"],
                    "confidence_interval": delivery_result["confidence_interval"]
                }

                if delivery_result["predicted_days"] > self.config["thresholds"]["delivery_delay_warning"]:
                    results["flags"].append("LONG_DELIVERY_TIME")
                    results["recommendations"].append(
                        f"Estimated delivery: {delivery_result['predicted_days']} days - "
                        "Consider expedited shipping or customer communication"
                    )
                print(f"  Predicted Delivery: {delivery_result['predicted_days']} days "
                      f"({delivery_result['delivery_category']})")
            except Exception as e:
                print(f"   Delivery prediction failed: {e}")
                results['predictions']['delivery'] = {'error': str(e)}

        # 3. Image Classification (if image provided)
        if product_image and "image_classifier" in self.models:
            print("\n[3/4] Classifying product image...")
            try:
                image_result = self.models["image_classifier"].predict_single(product_image)

                results["predictions"]["image_classification"] = {
                    "predicted_class": image_result["predicted_class"],
                    "confidence": image_result["confidence"],
                    "top_predictions": image_result["top_predictions"]
                }
                if "primary_category" in order_data:
                    if image_result["predicted_class"].lower() not in order_data["primary_category"].lower():
                        results["flags"].append("CATEGORY_MISMATCH")
                        results["recommendations"].append(f"Image category ({image_result['predicted_class']}) may not match "
                            f"listed category ({order_data['primary_category']})")
                        
                print(f"  Image Category: {image_result['predicted_class']} "
                    f"({image_result['confidence']:.0%} confidence)")
                
            except Exception as e:
                print(f"   Image classification failed: {e}")
                results['predictions']['image_classification'] = {'error': str(e)}

        # 4. Object Detection (if image provided)
        if product_image and "image_detector" in self.models:
            print("\n[4/4] Detecting objects in image...")
            try:
                detection_result = self.models["object_detector"].detect(product_image)

                results["predictions"]["object_detection"] = {
                    "num_objects": detection_result["num_detections"],
                    "detected_objects": [
                        {
                            'class': det['class'],
                            'confidence': det['confidence'],
                            'bbox': det['bbox']
                        }
                        for det in detection_result["detections"]
                    ]
                }

                # Check if expected number of items detected
                if "num_items" in order_data:
                    if detection_result["num_detections"] != order_data["num_items"]:
                        results["flags"].append("ITEM_COUNT_MISMATCH")
                        results["recommendations"].append(
                            f"Detected {detection_result['num_detections']} items in image, "
                            f"but order contains {order_data['num_items']} items"
                        )
                print(f"  Objects Detected: {detection_result['num_detections']}")
                for det in detection_result['detections']:
                    print(f"    - {det['class']} ({det['confidence']:.0%})")
                
            except Exception as e:
                print(f"  ✗ Object detection failed: {e}")
                results['predictions']['object_detection'] = {'error': str(e)}

        # Calculate overall risk score
        risk_components = []

        if "quality" in results["predictions"] and "risk_score" in results["predictions"]["quality"]:
            risk_components.append(results["predictions"]["quality"]["risk_score"])

        if "delivery" in results["predictions"] and "predicted_days" in results["predictions"]["delivery"]:
            # Normalize delivery days to 0-1 scale (higher days = higher risk)
            delivery_risk = min(results["predictions"]["delivery"]["predicted_days"]/30, 1.0)
            risk_components.append(delivery_risk)

        if risk_components:
            results["overall_risk_score"] = np.mean(risk_components)

        # Generate overall assessment
        if results['overall_risk_score'] > 0.7:
            results['overall_assessment'] = 'HIGH_RISK'
            results['recommendations'].insert(0, "⚠️  HIGH RISK ORDER - Recommend thorough review before processing")
        elif results['overall_risk_score'] > 0.4:
            results['overall_assessment'] = 'MEDIUM_RISK'
            results['recommendations'].insert(0, "⚡ MEDIUM RISK ORDER - Standard QC procedures apply")
        else:
            results['overall_assessment'] = 'LOW_RISK'
            results['recommendations'].insert(0, "✓ LOW RISK ORDER - Proceed with standard processing")
        
        return results
    
    def batch_analyze_order(self,
                            orders_df: pd.DataFrame,
                            image_column: str = None,
                            output_path: str = None)-> pd.DataFrame:
        """
        Analyze multiple orders in batch
        
        Args:
            orders_df: DataFrame with order data
            image_column: Column name containing image paths (optional)
            output_path: Path to save results
            
        Returns:
            DataFrame with analysis results
        """
        print("\n" + "="*80)
        print(f"BATCH ORDER ANALYSIS - {len(orders_df)} ORDERS")
        print("="*80)

        results_list = []

        for idx, row in orders_df.iterrows():
            print(f"\nAnalyzing order {idx + 1}/{len(orders_df)}...", end='\r')

            order_data = row.to_dict()
            image_path = order_data.pop(image_column) if image_column and image_column in order_data else None

            result = self.analyze_order(order_data, image_path)
            results_list.append(result)

        print("\n")

        # Convert to DataFrame
        results_df = pd.DataFrame(results_list)

        # Print summary
        print("\n" + "="*80)
        print("BATCH ANALYSIS SUMMARY")
        print("="*80)

        if "overall_assessment" in results_df.columns:
            print("\nRisk Assessment Distribution:")
            print(results_df['overall_assessment'].value_counts())

        if "flags" in results_df.columns:
            all_flags = []
            for flags in results_df["flags"]:
                all_flags.extend(flags)

            if all_flags:
                flag_counts = pd.Series(all_flags).value_counts()
                print("\nFlags Summary:")
                print(flag_counts)

        # Save if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            results_df.to_json(output_path, orient="records", indent=2)
            print(f"\n Results saved to {output_path}")

        return results_df
    
    def generate_report(self,
                        analysis_result: Dict,
                        save_path: str = None)-> str:
        """
        Generate human-readable report from analysis
        
        Args:
            analysis_result: Result from analyze_order
            save_path: Path to save report
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("="*80)
        report.append("E-COMMERCE QUALITY CONTROL ANALYSIS REPORT")
        report.append("="*80)

        report.append(f"\nOrder ID: {analysis_result['order_id']}")
        report.append(f"Analysis Time: {analysis_result['timestamp']}")
        report.append(f"\nOverall Risk Score: {analysis_result['overall_risk_score']:.1%}")
        report.append(f"Assessment: {analysis_result['overall_assessment']}")

        # Flags
        if analysis_result['flags']:
            report.append(f"\n  ACTIVE FLAGS ({len(analysis_result['flags'])}):")
            for flag in analysis_result['flags']:
                report.append(f"  • {flag}")
        
        # Predictions
        report.append("\n" + "-"*80)
        report.append("DETAILED PREDICTIONS")
        report.append("-"*80)
        
        # Quality
        if 'quality' in analysis_result['predictions']:
            q = analysis_result['predictions']['quality']
            if 'error' not in q:
                report.append(f"\n1. Quality Prediction:")
                report.append(f"   Risk Level: {q['risk_level']}")
                report.append(f"   Risk Score: {q['risk_score']:.1%}")
                report.append(f"   Prediction: {q['prediction']}")

        # Delivery
        if 'delivery' in analysis_result['predictions']:
            d = analysis_result['predictions']['delivery']
            if 'error' not in d:
                report.append(f"\n2. Delivery Time Prediction:")
                report.append(f"   Estimated Days: {d['predicted_days']}")
                report.append(f"   Category: {d['category']}")
                report.append(f"   Confidence Interval: {d['confidence_interval']['lower']}-{d['confidence_interval']['upper']} days")
        
        # Image Classification
        if 'image_classification' in analysis_result['predictions']:
            ic = analysis_result['predictions']['image_classification']
            if 'error' not in ic:
                report.append(f"\n3. Image Classification:")
                report.append(f"   Predicted Category: {ic['predicted_class']}")
                report.append(f"   Confidence: {ic['confidence']:.1%}")

        # Object Detection
        if 'object_detection' in analysis_result['predictions']:
            od = analysis_result['predictions']['object_detection']
            if 'error' not in od:
                report.append(f"\n4. Object Detection:")
                report.append(f"   Objects Detected: {od['num_objects']}")
                for obj in od['detected_objects'][:5]:  # Show first 5
                    report.append(f"   • {obj['class']} ({obj['confidence']:.1%})")
        
        # Recommendations
        if analysis_result['recommendations']:
            report.append("\n" + "-"*80)
            report.append("RECOMMENDATIONS")
            report.append("-"*80)
            for i, rec in enumerate(analysis_result['recommendations'], 1):
                report.append(f"\n{i}. {rec}")
        
        report.append("\n" + "="*80)
        
        report_str = "\n".join(report)

        # Save if path provided
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report_str)
            print(f"\n Report saved to {save_path}")
        
        return report_str

def analyze_single_order(order_data: Dict, product_image: str = None)-> Dict:
    """
    Quick function to analyze a single order
    
    Args:
        order_data: Order features
        product_image: Path to product image
        
    Returns:
        Analysis results
    """

    system = EcommerceQualityControlSystem()
    result = system.analyze_order(order_data, product_image)

    # Print report
    report = system.generate_report(result)
    print(report)

    return result

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    
    print("="*80)
    print("E-COMMERCE QUALITY CONTROL SYSTEM - DEMO")
    print("="*80)

    # Initialize system
    system = EcommerceQualityControlSystem()

    # Example order
    sample_order = {
        'order_id': 'ORDER_12345',
        'total_price': 150.00,
        'total_freight': 25.00,
        'num_items': 2,
        'total_weight_kg': 3.5,
        'primary_category': 'electronics',
        'customer_state': 'SP',
        'seller_state': 'RJ',
        'is_same_state': 0,
        'promised_delivery_days': 10,
        'primary_payment_type': 'credit_card',
        'max_installments': 3,
        'purchase_month': 6,
        'is_weekend': 0,
        'is_holiday_season': 0,
    }
    
    # Analyze order
    result = system.analyze_order(sample_order)

    # Generate and display report
    report = system.generate_report(result, save_path='results/sample_order_report.txt')
    print(report)
        




        


