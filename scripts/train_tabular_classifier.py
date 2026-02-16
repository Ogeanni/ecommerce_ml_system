"""
Complete model training pipeline
1. Prepare data
2. Train models
3. Evaluate and compare
4. Save best model
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR

from config import get_task_dirs

dirs = get_task_dirs("tabular_classification")

MODEL_DIR = dirs["saved_models"]
CHECKPOINT_DIR = dirs["checkpoints"]
RESULTS_DIR = dirs["results"]
LOGS_DIR = dirs["logs"]

from src.tabular_classification.data_preparation import prepare_modeling_data
from src.tabular_classification.train import train_all_models

print("="*80)
print("E-COMMERCE QUALITY CONTROL - MODEL TRAINING PIPELINE")
print("="*80)

# Step 1: Prepare data
print("\n[STEP 1] Preparing data for modeling...")
data_path = PROCESSED_DATA_DIR / "featured_orders.csv"
splits, preparator = prepare_modeling_data(data_path)


# Step 2: Train all models
print("\n[STEP 2] Training models...")
trainer = train_all_models(splits)
trainer.compare_models

# Step 3: Final evaluation on test set
print("\n" + "="*80)
print("FINAL EVALUATION ON TEST SET")
print("="*80)

best_model_name = max(trainer.results, key=lambda x: trainer.results[x]["f1"])
best_model = trainer.models[best_model_name]

# Test set evaluation
from sklearn.metrics import classification_report, confusion_matrix

# Optimal threshold (from validation)
best_threshold = 0.35

# Test set probabilities
y_test_proba = best_model.predict_proba(splits["X_test"])[:, 1]

# Apply threshold
y_test_pred = (y_test_proba >= best_threshold).astype(int)

print(f"\nBest Model: {best_model_name}")
print("\nTest Set Performance:")
print(classification_report(splits["y_test"], y_test_pred, 
                           target_names=["Good Quality", "Low Quality"]))

cm = confusion_matrix(splits["y_test"], y_test_pred)
print(f"\nConfusion Matrix:")
print(f"  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
print(f"  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")

print("\n" + "="*80)
print("TRAINING PIPELINE COMPLETE!")
print("="*80)
print(f"\nBest model saved to: models/saved_models/best_model_{best_model_name}.pkl")
print(f"Results saved to: results/")
print(f"Plots saved to: results/plots/")

