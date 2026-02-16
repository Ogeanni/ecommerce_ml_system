"""
Train classification models for e-commerce quality control
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE

from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve
)



import xgboost as xgb
from imblearn.over_sampling import SMOTE
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')



from config import get_task_dirs

dir_name = get_task_dirs("tabular_classification")

MODEL_DIR = dir_name["saved_models"]
CHECKPOINT_DIR = dir_name["checkpoints"]
RESULTS_DIR = dir_name["results"]
LOGS_DIR = dir_name["logs"]

class QualityControlModelTrainer:
    """ Train and evaluate classification models """

    def __init__(self, splits):
        """
        Initialize trainer with data splits
        
        Args:
            splits: Dictionary with X_train, X_val, X_test, y_train, y_val, y_test
        """

        self.splits = splits
        self.models = {}
        self.results = {}

    def train_baseline_model(self):
        """ Train simple logistic regression baseline """

        print("\n" + "="*80)
        print("TRAINING BASELINE MODEL: Logistic Regression")
        print("="*80)

        # Train without class weights
        lr_basic = LogisticRegression(max_iter=1000, random_state=42)
        lr_basic.fit(self.splits["X_train"], self.splits["y_train"])

        # Evaluate
        print("\n[Without Class Weights]")
        results_basic = self.evaluate_model(lr_basic, "lr_basic")

        # Train with class weights
        lr_balanced = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        lr_balanced.fit(self.splits["X_train"], self.splits["y_train"])

        # Evaluate
        print("\n[With Balanced Class Weights]")
        results_balanced = self.evaluate_model(lr_balanced, "lr_balanced", optimize_threshold=True, cost_fp=5, cost_fn=100)

        self.models["lr_basic"] = lr_basic
        self.models["lr_balanced"] = lr_balanced
        self.results["lr_basic"] = results_basic
        self.results["lr_balanced"] = results_balanced

        return lr_balanced
    

    def train_with_smote(self):
        """ Train model with SMOTE oversampling """
        print("\n" + "="*80)
        print("TRAINING WITH SMOTE (Synthetic Minority Oversampling)")
        print("="*80)

        # Apply SMOTE to training data only
        print("\nApplying SMOTE to training data...")

        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(
            self.splits["X_train"], self.splits["y_train"])
        
        print(f"   Original training size: {len(self.splits['X_train']):,}")
        print(f"   Resampled training size: {len(X_train_resampled):,}")
        print(f"   Original positive class: {self.splits['y_train'].sum():,} ({self.splits['y_train'].mean()*100:.2f}%)")
        print(f"   Resampled positive class: {y_train_resampled.sum():,} ({y_train_resampled.mean()*100:.2f}%)")

        # Train logistic regression on resampled data
        lr_smote = LogisticRegression(max_iter=1000, random_state=42)
        lr_smote.fit(X_train_resampled, y_train_resampled)

        # Evaluate (on original validation set, NOT resampled)
        results_smote = self.evaluate_model(lr_smote, "lr_smote", optimize_threshold=True, cost_fp=5, cost_fn=100)

        self.models["lr_smote"] = lr_smote
        self.results["lr_smote"] = results_smote

        return lr_smote
    

    def train_random_forest(self):
        """ Train Random Forest classifier """
        print("\n" + "="*80)
        print("TRAINING RANDOM FOREST")
        print("="*80)

        # Calculate class weights
        n_samples = len(self.splits["y_train"])
        n_positive = self.splits["y_train"].sum()
        n_negative = n_samples - n_positive
        weight_positive = n_samples / (2*n_positive)
        weight_negative = n_samples / (2*n_negative)
        
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=25,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight={0:weight_negative, 1:weight_positive},
            random_state=42,
            n_jobs=-1
        )

        print("\nTraining Random Forest...")
        rf.fit(self.splits["X_train"], self.splits["y_train"])

        # Evaluate
        results_rf = self.evaluate_model(rf, "random_forest", optimize_threshold=True, cost_fp=5, cost_fn=100)

        self.models["random_forest"] = rf
        self.results["random_forest"] = results_rf

        return rf

    def train_gradient_boosting(self):
        """ Train Gradient Boosting classifier """
        print("\n" + "="*80)
        print("TRAINING GRADIENT BOOSTING")
        print("="*80)

        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        print("\nTraining Gradient Boosting...")
        gb.fit(self.splits["X_train"], self.splits["y_train"])

        # Evaluate 
        results_gb = self.evaluate_model(gb, "gradient_boosting", optimize_threshold=True, cost_fp=5, cost_fn=100)
        
        self.models["graident_boosting"] = gb
        self.results["gradient_boosting"] = results_gb

        return gb


    
    def evaluate_model(self, model, model_name, optimize_threshold=False, cost_fp=5, cost_fn=100):
        """
        Evaluate model on validation set
        
        Args:
            model: Trained model
            model_name: Name for logging
            
        Returns:
            Dictionary with evaluation metrics
        """

        # Predictions
        
        y_pred_proba = model.predict_proba(self.splits["X_val"])[:, 1]

        # Default Threshold
        threshold = 0.35

        if optimize_threshold:
            def compute_cost(y_true, y_pred, cost_fp, cost_fn):
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                return fp * cost_fp + fn * cost_fn
            
            thresholds = np.linspace(0.35, 0.99, 65)
            costs = []
            for t in thresholds:
                y_pred_temp = (y_pred_proba >= t).astype(int)
                cost = compute_cost(self.splits["y_val"], y_pred_temp, cost_fp, cost_fn)
                costs.append(cost)
            best_idx = np.argmin(costs)
            threshold = thresholds[best_idx]
            print(f"\nOptimal threshold for {model_name}: {threshold:.3f} (min cost: ${min(costs):,.0f})")
        
        # Apply threshold
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Calculate metrics
        results = {
            "accuracy": accuracy_score(self.splits["y_val"], y_pred),
            "precision": precision_score(self.splits["y_val"], y_pred),
            "recall": recall_score(self.splits["y_val"], y_pred),
            "f1": f1_score(self.splits["y_val"], y_pred),
            "roc_auc": roc_auc_score(self.splits["y_val"], y_pred_proba),
            "threshold": threshold
        }

        # Print results
        print(f"\nValidation Set Performance ({model_name}):")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1-Score:  {results['f1']:.4f}")
        print(f"  ROC-AUC:   {results['roc_auc']:.4f}")
        print(f"  Threshold: {results['threshold']:.4f}")

        # Confusion matrix
        cm = confusion_matrix(self.splits["y_val"], y_pred)
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
        print(f"  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")

        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(self.splits["y_val"], y_pred, 
                                   target_names=["Good Quality", "Low Quality"]))
        
        results["y_pred"] = y_pred
        results["y_pred_proba"] = y_pred_proba
        results["confusion_matrix"] = cm

        return results
    
    def compare_models(self):
        """ Compare all trained models """

        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)

        comparison = []
        for model_name, results in self.results.items():
            comparison.append({
                "model": model_name,
                "accuracy": results["accuracy"],
                "precision": results["precision"],
                "recall": results["recall"],
                "f1": results["f1"],
                "roc_auc": results["roc_auc"]
            })

        comp_df = pd.DataFrame(comparison).sort_values('f1', ascending=False)
        print("\n", comp_df.to_string(index=False))

        # Save comparison
        Path("results/tabular_classification").mkdir(parents=True, exist_ok=True)
        comp_df.to_csv("results/tabular_classification/model_comparison.csv", index=False)

        return comp_df
    

    def plot_roc_curves(self):
        """ Plot ROC curves for all models """

        plt.figure(figsize=(10,8))

        for model_name, results in self.results.items():
            fpr, tpr, _ = roc_curve(self.splits["y_val"], results["y_pred_proba"])
            plt.plot(fpr, tpr, label=f"{model_name} (AUC={results['roc_auc']:.3f})")

        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves - Model Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        Path("results/tabular_classification/plots").mkdir(parents=True, exist_ok=True)
        plt.savefig("results/tabular_classification/plots/roc_curves.png", dpi=100)
        plt.close()

        print("\n ROC curves saved to results/tabular_classification/plots/roc_curves.png")


    def save_best_model(self):
        """ Save the best performing model """
        # Find best model by F1-score
        best_model_name = max(self.results, key=lambda x: self.results[x]["f1"])
        best_model = self.models[best_model_name]
        best_f1 = self.results[best_model_name]["f1"]

        print(f"\n Best model: {best_model_name} (F1={best_f1:.4f})")

        # Save model
        joblib.dump(best_model, f"{MODEL_DIR}/best_model_{best_model_name}.plk")

        # Save model metadata
        metadata = {
            "model_name": best_model_name,
            "metrics": {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in self.results[best_model_name].items()
                if k not in ["y_pred", "y_pred_proba", "confusion_matrix"]},
            "feature_count": len(self.splits["X_train"].columns)
        }

        with open(MODEL_DIR/"best_model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return best_model, best_model_name
    

def train_all_models(splits):
    """
    Train all models and return trainer
    
    Args:
        splits: Dictionary with train/val/test splits
        
    Returns:
        Trained model trainer object
    """

    trainer = QualityControlModelTrainer(splits)

    # Train all models
    trainer.train_baseline_model()
    trainer.train_with_smote()
    trainer.train_random_forest()
    trainer.train_gradient_boosting()

    # Compare and visualize
    trainer.compare_models()
    trainer.plot_roc_curves()

    # Save best model
    trainer.save_best_model()

    return trainer


if __name__ == "__main__":
    print("Load data splits first using prepare_for_modeling.py")


