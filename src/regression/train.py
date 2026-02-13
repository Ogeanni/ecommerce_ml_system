""" Training pipeline for regression models """

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
import xgboost as xgb
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

class RegressionModelTrainer:
    """ Train regression models for delivery time prediction """
    def __init__(self, splits: Dict,):
        """
        Initialize trainer
        
        Args:
            splits: Dictionary with X_train, y_train, X_val, y_val, X_test, y_test
        """

        self.splits = splits
        self.models = {}
        self.results = {}

        self.model_dir = Path("models/regression/saved_models")
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train_linear_regression(self):
        """Train simple linear regression"""
        print("\n" + "="*80)
        print("TRAINING LINEAR REGRESSION")
        print("="*80)

        model = LinearRegression()
        model.fit(self.splits["X_train"], self.splits["y_train"])

        results = self.evaluate_model(model, "linear regression")

        self.models["linear_regression"] = model
        self.results["linear_regression"] = results

        return model
    
    def train_ridge_regression(self, alpha=1.0):
        """Train Ridge regression (L2 regularization)"""
        print("\n" + "="*80)
        print("TRAINING RIDGE REGRESSION")
        print("="*80)

        model = Ridge(alpha=alpha)
        model.fit(self.splits["X_train"], self.splits["y_train"])

        results = self.evaluate_model(model, "ridge_regression")

        self.models["ridge_regression"] = model
        self.results["ridge_regression"] = results

        return model
    
    def train_lasso_regression(self, alpha=1.0):
        """Train Lasso regression (L1 regularization)"""
        print("\n" + "="*80)
        print("TRAINING LASSO REGRESSION")
        print("="*80)

        model = Lasso(alpha=alpha, max_iter=10000)
        model.fit(self.splits["X_train"], self.splits["y_train"])

        results = self.evaluate_model(model, "lasso_regression")

        # Print feature selection info
        n_features_used = np.sum(model.coef_ != 0)
        print(f"\nFeature Selection:")
        print(f"  Total features: {len(model.coef_)}")
        print(f"  Features used: {n_features_used}")
        print(f"  Features dropped: {len(model.coef_) - n_features_used}")

        self.models["lasso_regression"] = model
        self.results["lasso_regression"] = results

        return model
    
    def train_random_forest(self, n_estimators=200, max_depth=None):
        """Train Random Forest regressor"""
        print("\n" + "="*80)
        print("TRAINING RANDOM FOREST")
        print("="*80)

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        print(f"\nTraining with {n_estimators} trees...")
        model.fit(self.splits["X_train"], self.splits["y_train"])

        results = self.evaluate_model(model, "random_forest")

        # Plot feature importance
        self.plot_feature_importance(model, "random_forest")

        self.models["random_forest"] = model
        self.results["random_forest"] = results

        return model
    
    def train_gradient_boosting(self, n_estimators=100, learning_rate=0.1):
        """Train Gradient Boosting regressor"""
        print("\n" + "="*80)
        print("TRAINING GRADIENT BOOSTING")
        print("="*80)

        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        print(f"\nTraining with {n_estimators} estimators...")
        model.fit(self.splits["X_train"], self.splits["y_train"])

        results = self.evaluate_model(model, "gradient_boosting")

        self.models["gradient_boosting"] = model
        self.results["gradient_boosting"] = results

        return model
    
    def train_xgboost(self, n_iter=50):
        """Train XGBoost regressor"""
        print("\n" + "="*80)
        print("TRAINING XGBOOST")
        print("="*80)
        
        xgbr = xgb.XGBRegressor(
            objective = "reg:squarederror",
            random_state=42,
            n_jobs=-1
        )

        parameters_dist = {
            "n_estimators": [100, 200, 400, 500],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 6, 7, 10],
            "min_child_weight": [1, 3, 5, 7],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "gamma": [0, 0.1, 0.5, 1.0]
        }

        search = RandomizedSearchCV(
            estimator=xgbr,
            param_distributions=parameters_dist,
            n_iter=n_iter,
            cv=3,
            scoring="neg_mean_absolute_error",
            verbose=1,
            random_state=42
        )

        print(f"\nTraining XGBoost...")
        print(f"\nSearching Randomized CV...")
        search.fit(
            self.splits["X_train"],
            self.splits["y_train"])
        
        best_model = search.best_estimator_
        print(f"\nBest XGBoost params: {search.best_params_}")
        
        results = self.evaluate_model(best_model, "xgboost")

        # Plot feature importance
        self.plot_feature_importance(best_model, "xgboost")

        self.models["xgboost"] = best_model
        self.results["xgboost"] = results

        return best_model
    
    def evaluate_model(self, model, model_name: str) -> Dict:
        """
        Evaluate model on validation set
        
        Args:
            model: Trained model
            model_name: Name for logging
            
        Returns:
            Dictionary with metrics
        """

        # Predictions
        y_pred = model.predict(self.splits["X_val"])
        y_true = self.splits["y_val"]

        # Calculate metrics
        results = {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_absolute_error(y_true, y_pred)),
            "r2": r2_score(y_true, y_pred),
            "mape": mean_absolute_percentage_error(y_true, y_pred) * 100
        }
        # Print results
        print(f"\nValidation Set Performance ({model_name}):")
        print(f"  MAE (Mean Absolute Error): {results['mae']:.2f} days")
        print(f"  RMSE (Root Mean Squared Error): {results['rmse']:.2f} days")
        print(f"  R² Score: {results['r2']:.4f}")
        print(f"  MAPE (Mean Absolute % Error): {results['mape']:.2f}%")

        # Store predictions
        results["y_pred"] = y_pred
        results["y_true"] = y_true

        return results

    def plot_feature_importance(self, model, model_name: str, top_n=20):
        """Plot feature importance for tree-based models"""
        if not hasattr(model, "feature_importances_"):
            return

        importances = model.feature_importances_
        feature_names = self.splits["X_train"].columns

        # Create dataframe
        feat_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)

        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(feat_imp_df)), feat_imp_df['importance'])
        plt.yticks(range(len(feat_imp_df)), feat_imp_df['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        # Save
        save_path = Path(f'results/regression/feature_importance_{model_name}.png')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n Feature importance saved to {save_path}")

    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)

        comparison = []
        for model_name, results in self.results.items():
            comparison.append({
                "Model": model_name,
                "MAE": results["mae"],
                "RMSE": results["rmse"],
                "R2": results["r2"],
                "MAPE (%)": results["mape"]
            })
        comp_df = pd.DataFrame(comparison).sort_values('MAE')
        print("\n", comp_df.to_string(index=False))

        # Save comparison
        save_path = Path("results/regression/model_comparison.csv")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        comp_df.to_csv(save_path, index=False)
        
        return comp_df
    
    def plot_predictions_vs_actual(self, model_name: str = None, save_path: str = None):
        """Plot predicted vs actual values"""
        if model_name is None:
            # Use best model
            model_name = min(self.results, key=lambda x: self.results[x]['mae'])
        
        results = self.results[model_name]
        y_true = results['y_true']
        y_pred = results['y_pred']
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
        axes[0].plot([y_true.min(), y_true.max()], 
                    [y_true.min(), y_true.max()], 
                    'r--', lw=2, label='Perfect prediction')
        axes[0].set_xlabel('Actual Delivery Days')
        axes[0].set_ylabel('Predicted Delivery Days')
        axes[0].set_title(f'Predictions vs Actual - {model_name}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Delivery Days')
        axes[1].set_ylabel('Residuals (Actual - Predicted)')
        axes[1].set_title(f'Residual Plot - {model_name}')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f'results/regression/predictions_vs_actual_{model_name}.png'
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n Prediction plot saved to {save_path}")


    def save_best_model(self):
        """Save the best performing model"""
        # Find best model by MAE
        best_model_name = min(self.results, key=lambda x: self.results[x]["mae"])
        best_model = self.models[best_model_name]
        best_mae = self.results[best_model_name]["mae"]

        print(f"\n Best model: {best_model_name} (MAE={best_mae:.2f} days)")

        # Save model
        model_path = self.model_dir/f"best_model_{best_model_name}.pkl"
        joblib.dump(best_model, model_path)

        # Save metadata
        metadata = {
            "model_name": best_model_name,
            "metrics": {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in self.results[best_model_name].items()
                if k not in ["y_pred", "y_true"]
            },
            "feature_count": len(self.splits["X_train"].columns) 
        }

        metadata_path = self.model_dir/"best_model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"   Saved to: {model_path}")

        return best_model, best_model_name
    
def train_all_regression_models(splits: Dict):
    """
    Train all regression models
    Args:
        splits: Dictionary with train/val/test splits
    """

    trainer = RegressionModelTrainer(splits)

    # Train all models
    trainer.train_linear_regression()
    trainer.train_ridge_regression()
    trainer.train_lasso_regression()
    trainer.train_random_forest(n_estimators=200, max_depth=5)
    trainer.train_gradient_boosting(n_estimators=100)
    trainer.train_xgboost(n_iter=200)

    # Compare models
    trainer.compare_models()

    # Plot predictions for best model
    trainer.plot_predictions_vs_actual()

    # Save best model
    trainer.save_best_model()

    return trainer

if __name__ == "__main__":
    print("Load prepared data first using data_preparation.py")
   


    