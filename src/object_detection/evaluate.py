""" Evaluation and visualization for object detection """

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import yaml
import json
from typing import List, Dict, Tuple
import subprocess
import sys

def resolve_root_dir(project_name: str) -> Path:
    """
    Resolve root directory for local vs Colab + Google Drive
    """
    gdrive_root = Path("/content/drive/MyDrive")

    if gdrive_root.exists():
        # Running on Colab with Google Drive mounted
        return gdrive_root / project_name
    else:
        # Running locally
        return Path(__file__).resolve().parent # ecommerce_ml_system root



class ObjectDetectionEvaluator:
    """ Evaluate object detection models """
    def __init__(self,
                 weights_path: str,
                 data_yaml: str,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45):
        """
        Initialize evaluator
        
        Args:
            weights_path: Path to trained weights
            data_yaml: Path to dataset YAML
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
        """
        self.root_dir = resolve_root_dir("ecommerce_ml_system")
        self.weights_path = Path(weights_path)
        self.data_yaml = Path(data_yaml)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Load class names
        with open(self.data_yaml, "r") as f:
            data_config = yaml.safe_load(f)

        self.class_names = data_config["names"]
        self.num_classes = len(self.class_names)

        # YOLOv5 directory
        self.yolov5_dir = Path("models/object_detection/yolov5")

        print(f"   Evaluator initialized")
        print(f"   Weights: {self.weights_path}")
        print(f"   Classes: {self.num_classes}")
        print(f"   Confidence threshold: {self.conf_threshold}")
        print(f"   IoU threshold: {self.iou_threshold}")

    def evaluate_on_test_set(self,
                             img_size: int = 640,
                             batch_size: int = 32,
                             device: str = "0",
                             save_dir: str = None):
        """
        Evaluate model on test set
        
        Args:
            img_size: Image size
            batch_size: Batch size
            device: Device to use
            save_dir: Directory to save results
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "="*80)
        print("EVALUATING ON TEST SET")
        print("="*80)

        if save_dir is None:
            save_dir = self.weights_path.parent.parent / "evaluation"

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Check if CUDA is available
        if device != "0" and not torch.cuda.is_available():
            print("  CUDA not available, using CPU")
            device = "cpu"

        # Build command for val.py
        val_script = self.yolov5_dir / "val.py"

        cmd = [
            sys.executable, str(val_script),
            '--data', str(self.data_yaml),
            '--weights', str(self.weights_path),
            '--batch-size', str(batch_size),
            '--imgsz', str(img_size),
            '--device', str(device),
            '--conf-thres', str(self.conf_threshold),
            '--iou-thres', str(self.iou_threshold),
            '--task', 'test',  # Evaluate on test set
            '--save-txt',
            '--save-conf',
            '--project', str(save_dir),
            '--name', 'test_results',
            '--exist-ok'
        ]

        print(f"\nRunning validation...")

        try:
            # Capture output
            result = subprocess.run(
                cmd,
                check=True,
                cwd = self.yolov5_dir,
                capture_output=True,
                text=True
            )
            print("\n Evaluation complete!")

            # Parse output for metrics
            output = result.stdout
            metrics = self._parse_val_output(output)

            # Print metrics
            print("\nTest Set Results:")
            print(f"  Precision: {metrics.get('precision', 0):.4f}")
            print(f"  Recall: {metrics.get('recall', 0):.4f}")
            print(f"  mAP@0.5: {metrics.get('map50', 0):.4f}")
            print(f"  mAP@0.5:0.95: {metrics.get('map', 0):.4f}")

            # Save metrics
            metrics_path = save_dir / "test_results" / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            print(f"\n Results saved to: {save_dir / 'test_results'}")
            return metrics
        
        except subprocess.CalledProcessError as e:
            print(f"\n Evaluation failed: {e}")
            print(f"Error output: {e.stderr}")
            return {}
    
    def _parse_val_output(self, output: str) -> Dict:
        """ Parse validation output for metrics """
        metrics = {}

        lines = output.split("\n")
        for line in lines:
            # Look for metrics in output
            if "all" in line.lower() and "precision" in output.lower():
                # Parse metrics line
                parts = line.split()
                try:
                    # Typical format: "all   100   1000   0.xxx   0.xxx   0.xxx   0.xxx"
                    if len(parts) >= 6:
                        metrics["precision"] = float(parts[-4])
                        metrics["recall"] = float(parts[-3])
                        metrics["map50"] = float(parts[-2])
                        metrics["map"] = float(parts[-1])
                except (ValueError, IndexError):
                    pass

        return metrics
    
    def plot_confusion_matrix(self, results_dir: str = None, save_path: str = None):
        """
        Plot confusion matrix from validation results
        
        Args:
            results_dir: Directory with validation results
            save_path: Path to save plot
        """
        if results_dir is None:
            results_dir = self.weights_path.parent.parent / 'evaluation' / 'test_results'
        
        confusion_matrix_path = Path(results_dir) / 'confusion_matrix.png'
        
        if confusion_matrix_path.exists():
            # Display existing confusion matrix
            img = plt.imread(confusion_matrix_path)
            
            plt.figure(figsize=(12, 10))
            plt.imshow(img)
            plt.axis('off')
            plt.title('Confusion Matrix', fontsize=14)
            plt.tight_layout()
            
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f" Confusion matrix saved to {save_path}")
            
            plt.show()
            plt.close()
        else:
            print(f"  Confusion matrix not found at {confusion_matrix_path}")
    
    def plot_pr_curve(self, results_dir: str = None, save_path: str = None):
        """
        Plot precision-recall curve
        
        Args:
            results_dir: Directory with validation results
            save_path: Path to save plot
        """
        if results_dir is None:
            results_dir = self.weights_path.parent.parent / 'evaluation' / 'test_results'
        
        pr_curve_path = Path(results_dir) / 'PR_curve.png'
        
        if pr_curve_path.exists():
            img = plt.imread(pr_curve_path)
            
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title('Precision-Recall Curve', fontsize=14)
            plt.tight_layout()
            
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f" PR curve saved to {save_path}")
            
            plt.show()
            plt.close()
        else:
            print(f"  PR curve not found at {pr_curve_path}")

    def analyze_per_class_performance(self, results_dir: str = None):
        """
        Analyze per-class performance metrics
        
        Args:
            results_dir: Directory with validation results
        """
        print("\n" + "="*80)
        print("PER-CLASS PERFORMANCE ANALYSIS")
        print("="*80)

        if results_dir is None:
            results_dir = self.weights_path.parent.parent / "evaluation" / "test_results"

        # YOLOv5 saves per-class metrics in results.txt or similar
        # This is a simplified version - actual parsing would depend on YOLOv5 version

        print("\nPer-class metrics are available in the results directory:")
        print(f"  {results_dir}")
        print("\nKey files:")
        print("  - results.csv: Training/validation metrics over epochs")
        print("  - confusion_matrix.png: Confusion matrix visualization")
        print("  - PR_curve.png: Precision-Recall curve")
        print("  - F1_curve.png: F1 score curve")

    def generate_full_report(self, output_dir: str = "results/object_detection"):
        """ 
        Generate comprehensive evaluation report
        
        Args:
            output_dir: Directory to save all outputs
        """
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE EVALUATION REPORT")
        print("="*80)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. Evaluate on test set
        metrics = self.evaluate_on_test_set()

        # 2. Plot confusion matrix
        self.plot_confusion_matrix(save_path=output_path / "confusion_matrix.png")

        # 3. Plot PR curve
        self.plot_pr_curve(save_path=output_path / "pr_curve.png")

        # 4. Analyze per-class performance
        self.analyze_per_class_performance()

        print(f"\n Full evaluation report generated in {output_dir}/")

        return metrics
    
def evaluate_detector(weights_path: str, data_yaml: str):
    """
    Convenience function to evaluate detector
    
    Args:
        weights_path: Path to model weights
        data_yaml: Path to dataset YAML
        
    Returns:
        Evaluation metrics
    """
    evaluator = ObjectDetectionEvaluator(weights_path, data_yaml)
    metrics = evaluator.generate_full_report()

    return metrics

if __name__ == "__main__":
    # Example evaluation
    weights_path = "models/object_detection/runs/object_detection/train/weights/best.pt"
    data_yaml = "data/object_detection/dataset.yaml"

    if Path(weights_path).exists() and Path(data_yaml).exists():
        evaluate_detector(weights_path, data_yaml)

    else:
        print("Weights or data YAML not found. Train a model first.")