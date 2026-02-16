"""Complete end-to-end training script for regression model"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


from config import get_task_dirs

dirs = get_task_dirs("regression")

MODEL_DIR = dirs["saved_models"]
CHECKPOINT_DIR = dirs["checkpoints"]
RESULTS_DIR = dirs["results"]
LOGS_DIR = dirs["logs"]



from src.regression.data_preparation import RegressionDataPreparator
from src.regression.train import train_all_regression_models
from src.regression.evaluate import RegressionEvaluator
from src.regression.predict import DeliveryTimePredictor

print("="*80)
print("REGRESSION MODEL - COMPLETE TRAINING PIPELINE")
print("="*80)
print("\nTask: Predict Delivery Time (in days)")

# ==================== CONFIGURATION ====================
CONFIG = {
    # Data
    "source_data": "data/processed/featured_orders.csv",

    # Splits
    "test_size": 0.15,
    "val_size": 0.15,
    "random_state": 42,

    # Model hyperparameters
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 15
    },
    "gradient_boosting": {
        "n_estimators": 100,
        "learning_rate": 0.1
    },
    'xgboost': {
        "n_iter": 50
    },
    
    # Regularization
    "ridge_alpha": 1.0,
    "lasso_alpha": 0.1,
}

print("\nConfiguration:")
for key, value in CONFIG.items():
    if isinstance(value, dict):
        print(f"  {key}:")
        for k, v in value.items():
            print(f"    {k}: {v}")
    else:
        print(f"  {key}: {value}")

# ==================== STEP 1: PREPARE DATA ====================
print("\n" + "="*80)
print("STEP 1: DATA PREPARATION")
print("="*80)

if Path(CONFIG["source_data"]).exists():
    source_data_path = Path(CONFIG["source_data"])
elif Path('/content/drive/MyDrive/ecommerce_ml_system').exists():
    source_data_path = Path('/content/drive/MyDrive/ecommerce_ml_system')
else:
    print(f"\n Source data not found")
    print("Please run the tabular classification pipeline first to generate this data")
    print("Run: python scripts/train_tabular_classifier.py")
    sys.exit(1)

preparator = RegressionDataPreparator(str(source_data_path))
splits = preparator.prepare_complete_dataset()

print(f"\n Data preparation complete")
print(f"   Features: {splits['X_train'].shape[1]}")
print(f"   Train samples: {len(splits['X_train']):,}")
print(f"   Val samples: {len(splits['X_val']):,}")
print(f"   Test samples: {len(splits['X_test']):,}")

# ==================== STEP 2: TRAIN MODELS ====================
print("\n" + "="*80)
print("STEP 2: TRAINING MODELS")
print("="*80)

trainer = train_all_regression_models(splits)
trainer.compare_models


# # ==================== STEP 3: EVALUATE ON TEST SET ====================
# print("\n" + "="*80)
# print("STEP 3: FINAL EVALUATION ON TEST SET")
# print("="*80)

# # Get best model
# best_model_name = min(trainer.results, key=lambda x: trainer.results[x]['mae'])
# best_model = trainer.models[best_model_name]
# print(f"\nBest Model: {best_model_name}")

# # Create evaluator
# evaluator = RegressionEvaluator(best_model, splits["X_test"], splits["y_test"])

# # Generate comprehensive report
# test_metrics = evaluator.generate_full_report(output_dir='results/regression')

# # ==================== STEP 4: COMPARE ALL MODELS ON TEST SET ====================
# print("\n" + "="*80)
# print("STEP 4: MODEL COMPARISON ON TEST SET")
# print("="*80)

# test_comparison = []
# for model_name, model in trainer.models.items():
#     # Predict
#     y_pred = model.predict(splits["X_test"])
#     y_true = splits["y_test"]


# # Calculate metrics
# from sklearn.metrics import (
#     mean_absolute_error, mean_squared_error, r2_score,
#     mean_absolute_percentage_error
# )
# import numpy as np

# mae = mean_absolute_error(y_true, y_pred)
# rmse = np.sqrt(mean_squared_error(y_true, y_pred))
# r2 = r2_score(y_true, y_pred)
# mape = mean_absolute_percentage_error(y_true, y_pred) * 100

# # Within X days accuracy
# residuals = np.abs(y_true - y_pred)
# within_2_days = (residuals <= 2).mean() * 100

# test_comparison.append({
#     'Model': model_name,
#     'MAE': mae,
#     'RMSE': rmse,
#     'R²': r2,
#     'MAPE (%)': mape,
#     'Within ±2 days (%)': within_2_days
# })
# comparison_df = pd.DataFrame(test_comparison).sort_values('MAE')
# print("\nTest Set Performance - All Models:")
# print(comparison_df.to_string(index=False))

# # Save comparison
# comparison_path = Path('results/regression/test_model_comparison.csv')
# comparison_path.parent.mkdir(parents=True, exist_ok=True)
# comparison_df.to_csv(comparison_path, index=False)
# print(f"\n Comparison saved to {comparison_path}")




# # ==================== STEP 5: TEST PREDICTIONS ====================
# print("\n" + "="*80)
# print("STEP 4: TESTING PREDICTIONS")
# print("="*80)

# # Load predictor
# predictor = DeliveryTimePredictor()

# # Test on sample orders
# print("\nTesting on sample test orders...")

# test_sample = splits["X_test"].head(10)
# actual_sample = splits["y_test"].head(10)

# # print(f"\n{'Actual':<10} {'Predicted':<12} {'Error':<10} {'Category':<30}")
# print("-" * 70)
# for i in range(len(test_sample)):
#     order = test_sample.iloc[i:i+1]
#     actual = actual_sample.iloc[i]

# result = predictor.predict_single_order(order.to_dict("records")[0])
# predicted = result["predicted_days"]
# error = actual - predicted
# category = result["delivery_category"]

# print(f"{actual:<10.1f} {predicted:<12} {error:<10.1f} {category:<30}")

# # ==================== STEP 6: SUMMARY ====================
# print("\n" + "="*80)
# print("TRAINING PIPELINE COMPLETE!")
# print("="*80)
# print(f"\n Model Performance Summary:")
# print(f"   Best Model: {best_model_name}")
# print(f"   Test MAE: {test_metrics['mae']:.2f} days")
# print(f"   Test RMSE: {test_metrics['rmse']:.2f} days")
# print(f"   Test R²: {test_metrics['r2']:.4f}")
# print(f"   Predictions within ±2 days: {test_metrics['within_2_days']:.1f}%")
# print(f"\n Files Saved:")
# print(f"   Model: models/regression/saved_models/")
# print(f"   Preprocessing: models/regression/preprocessing/")
# print(f"   Results: results/regression/")
# print(f"   Data: data/regression/processed/")
# print(f"\n Usage Example:")



