""" Evaluation and visualization for regression models """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
from pathlib import Path
from typing import Dict
import joblib

from config import get_task_dirs

dir_name = get_task_dirs("tabular_classification")

MODEL_DIR = dir_name["saved_models"]
CHECKPOINT_DIR = dir_name["checkpoints"]
PREPROCESSING = dir_name["preprocessing"]
RESULTS_DIR = dir_name["results"]
LOGS_DIR = dir_name["logs"]


class RegressionEvaluator:
    """Evaluate regression models"""
    def __init__(self, model, X_test, y_test, feature_names=None):
        """
        Initialize evaluator
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            feature_names: List of feature names
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names if feature_names else X_test.columns.tolist()
        self.predictions = None

    def evaluate(self) -> Dict:
        """
        Evaluate model on test set
        
        Returns:
            Dictionary with metrics
        """
        print("\n" + "="*80)
        print("EVALUATING ON TEST SET")
        print("="*80)

        # Make predictions
        self.predictions = self.model.predict(self.X_test)

        # Calculate metrics
        metrics = {
            "mae": mean_absolute_error(self.y_test, self.predictions),
            "rmse": np.sqrt(mean_absolute_error(self.y_test, self.predictions)),
            "r2": r2_score(self.y_test, self.predictions),
            "mape": mean_absolute_percentage_error(self.y_test, self.predictions) * 100
        }

        print("\nTest Set Performance:")
        print(f"  MAE (Mean Absolute Error): {metrics['mae']:.2f} days")
        print(f"  RMSE (Root Mean Squared Error): {metrics['rmse']:.2f} days")
        print(f"  R² Score: {metrics['r2']:.4f}")
        print(f"  MAPE (Mean Absolute % Error): {metrics['mape']:.2f}%")

        # Additional analysis
        residuals = self.y_test - self.predictions
        metrics['mean_residual'] = residuals.mean()
        metrics['std_residual'] = residuals.std()

        print(f"\nResidual Statistics:")
        print(f"  Mean residual: {metrics['mean_residual']:.2f} days")
        print(f"  Std of residuals: {metrics['std_residual']:.2f} days")

        # Prediction accuracy within thresholds
        within_1_day = (np.abs(residuals) <= 1).mean() * 100
        within_2_days = (np.abs(residuals) <= 2).mean() * 100
        within_3_days = (np.abs(residuals) <= 3).mean() * 100
        
        print(f"\nPrediction Accuracy:")
        print(f"  Within 1 day: {within_1_day:.2f}%")
        print(f"  Within 2 days: {within_2_days:.2f}%")
        print(f"  Within 3 days: {within_3_days:.2f}%")
        
        metrics['within_1_day'] = within_1_day
        metrics['within_2_days'] = within_2_days
        metrics['within_3_days'] = within_3_days
        
        return metrics
    
    def plot_predictions_vs_actual(self, save_path=None):
        """Plot predicted vs actual delivery days"""
        if self.predictions is None:
            self.predictions = self.model.predict(self.X_test)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        axes[0].scatter(self.y_test, self.predictions, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
        
        # Perfect prediction line
        min_val = min(self.y_test.min(), self.predictions.min())
        max_val = max(self.y_test.max(), self.predictions.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        
        axes[0].set_xlabel('Actual Delivery Days', fontsize=12)
        axes[0].set_ylabel('Predicted Delivery Days', fontsize=12)
        axes[0].set_title('Predictions vs Actual', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Add R² to plot
        r2 = r2_score(self.y_test, self.predictions)
        axes[0].text(0.05, 0.95, f'R² = {r2:.4f}', 
                    transform=axes[0].transAxes, fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Residuals plot
        residuals = self.y_test - self.predictions
        axes[1].scatter(self.predictions, residuals, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Delivery Days', fontsize=12)
        axes[1].set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
        axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n Prediction plot saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_residual_distribution(self, save_path=None):
        """Plot distribution of residuals"""
        if self.predictions is None:
            self.predictions = self.model.predict(self.X_test)
        
        residuals = self.y_test - self.predictions
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Residuals (days)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        axes[0].text(0.05, 0.95, 
                    f'Mean: {residuals.mean():.2f}\nStd: {residuals.std():.2f}',
                    transform=axes[0].transAxes, fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n Residual distribution plot saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_error_by_actual_value(self, save_path=None):
        """Plot error distribution by actual delivery time"""
        if self.predictions is None:
            self.predictions = self.model.predict(self.X_test)
        
        residuals = self.y_test - self.predictions
        
        # Create bins for actual values
        bins = pd.cut(self.y_test, bins=10)
        
        # Calculate error stats per bin
        error_stats = pd.DataFrame({
            'actual': self.y_test,
            'residual': residuals,
            'bin': bins
        }).groupby('bin')['residual'].agg(['mean', 'std', 'count'])
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_labels = [f'{int(interval.left)}-{int(interval.right)}' for interval in error_stats.index]
        x_pos = np.arange(len(x_labels))
        
        ax.bar(x_pos, error_stats['mean'], yerr=error_stats['std'], 
               capsize=5, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_xlabel('Actual Delivery Days (bins)', fontsize=12)
        ax.set_ylabel('Mean Residual (days)', fontsize=12)
        ax.set_title('Prediction Error by Actual Delivery Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n Error by value plot saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_error_distribution_by_range(self, save_path=None):
        """Plot percentage of predictions within different error ranges"""
        if self.predictions is None:
            self.predictions = self.model.predict(self.X_test)
        
        absolute_errors = np.abs(self.y_test - self.predictions)
        
        # Define error ranges
        ranges = [1, 2, 3, 5, 7, 10]
        percentages = []
        
        for range_val in ranges:
            pct = (absolute_errors <= range_val).mean() * 100
            percentages.append(pct)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(range(len(ranges)), percentages, edgecolor='black', alpha=0.7)
        ax.set_xticks(range(len(ranges)))
        ax.set_xticklabels([f'±{r} days' for r in ranges])
        ax.set_ylabel('Percentage of Predictions (%)', fontsize=12)
        ax.set_xlabel('Error Range', fontsize=12)
        ax.set_title('Prediction Accuracy by Error Range', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels on bars
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n Error range plot saved to {save_path}")
        
        plt.show()
        plt.close()

    def analyze_worst_predictions(self, top_n=10):
        """Analyze worst predictions"""
        if self.predictions is None:
            self.predictions = self.model.predict(self.X_test)
        
        print("\n" + "="*80)
        print("WORST PREDICTIONS ANALYSIS")
        print("="*80)
        
        # Calculate absolute errors
        absolute_errors = np.abs(self.y_test - self.predictions)
        
        # Get worst predictions
        worst_indices = np.argsort(absolute_errors)[-top_n:][::-1]
        
        print(f"\nTop {top_n} Worst Predictions:")
        print(f"{'Actual':<10} {'Predicted':<12} {'Error':<10} {'% Error':<10}")
        print("-" * 50)
        
        for idx in worst_indices:
            actual = self.y_test.iloc[idx]
            predicted = self.predictions[idx]
            error = actual - predicted
            pct_error = (abs(error) / actual) * 100
            
            print(f"{actual:<10.2f} {predicted:<12.2f} {error:<10.2f} {pct_error:<10.2f}%")


    def generate_full_report(self, output_dir=RESULTS_DIR):
        """Generate comprehensive evaluation report"""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE EVALUATION REPORT")
        print("="*80)
        
        output_path = output_dir
        
        # 1. Evaluate
        metrics = self.evaluate()
        
        # 2. Plot predictions vs actual
        self.plot_predictions_vs_actual(
            save_path=output_path / 'predictions_vs_actual.png'
        )
        
        # 3. Plot residual distribution
        self.plot_residual_distribution(
            save_path=output_path / 'residual_distribution.png'
        )
        
        # 4. Plot error by actual value
        self.plot_error_by_actual_value(
            save_path=output_path / 'error_by_actual_value.png'
        )
        
        # 5. Plot error distribution by range
        self.plot_error_distribution_by_range(
            save_path=output_path / 'error_distribution_by_range.png'
        )
        
        # 6. Analyze worst predictions
        self.analyze_worst_predictions(top_n=10)
        
        # Save metrics
        import json
        metrics_serializable = {k: float(v) for k, v in metrics.items()}
        with open(output_path / 'test_metrics.json', 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        print(f"\n Full evaluation report generated in {output_dir}/")
        
        return metrics
    
def evaluate_regression_model(model_path: str, X_test, y_test):
    """
    Convenience function to evaluate regression model
    
    Args:
        model_path: Path to saved model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Evaluation metrics
    """
    model = joblib.load(model_path)
    evaluator = RegressionEvaluator(model, X_test, y_test)
    metrics = evaluator.generate_full_report()
    
    return metrics


if __name__ == "__main__":
    print("Load trained model and test data first")