"""
Evaluation and visualization for image classification
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)
from pathlib import Path
import tensorflow as tf

from config import MODELS_DIR, LOGS_DIR



class ImageClassificationEvaluator:
    """Evaluate image classification models"""
    
    def __init__(self, model, data, class_names):
        """
        Initialize evaluator
        
        Args:
            model: Trained Keras model
            data: Dictionary with test data
            class_names: List of class names
        """
        self.model = model
        self.data = data
        self.class_names = class_names
        self.y_pred = None
        self.y_pred_proba = None
        self.y_true = None

    def evaluate(self, dataset='test', verbose=1):
        """
        Evaluate model on dataset
        
        Args:
            dataset: Which dataset to evaluate ('train', 'val', or 'test')
            verbose: Verbosity level
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "="*80)
        print(f"EVALUATING ON {dataset.upper()} SET")
        print("="*80)
        
        X = self.data[f'X_{dataset}']
        y_true = self.data[f'y_{dataset}']

        # ---- EVALUATION ----
        if isinstance(X, tf.data.Dataset):
            results = self.model.evaluate(X, verbose=verbose)

            # Extract true labels from dataset
            y_true = np.concatenate([y.numpy() for _, y in X])
            y_pred_proba = self.model.predict(X, verbose=verbose)
        else:
            results = self.model.evaluate(X, y_true, verbose=verbose)
            y_pred_proba = self.model.predict(X, verbose=verbose)

        y_pred = np.argmax(y_pred_proba, axis=1)

        # ---- METRICS (dynamic, includes top-k) ----
        metrics = {
        name: float(value)
        for name, value in zip(self.model.metrics_names, results)}

        print(f"\n{dataset.upper()} SET RESULTS:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")

        # ---- STORE FOR REPORTING ----
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba

        return metrics
    
    def print_classification_report(self, dataset='test'):
        """
        Print detailed classification report
        
        Args:
            dataset: Which dataset to evaluate
        """
        print("\n" + "="*80)
        print("CLASSIFICATION REPORT")
        print("="*80)
        

        
        if self.y_pred is None:
            self.evaluate(dataset=dataset, verbose=0)
        
        # Print report
        report = classification_report(
            self.y_true, 
            self.y_pred, 
            target_names=self.class_names,
            digits=4
        )
        print(f"\n{report}")
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_true, self.y_pred, average=None
        )
        
        print("\nPER-CLASS METRICS:")
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 80)
        
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:<20} {precision[i]:<12.4f} {recall[i]:<12.4f} "
                  f"{f1[i]:<12.4f} {support[i]:<10}")
    
    def plot_confusion_matrix(self, dataset='test', save_path=None, normalize=False):
        """
        Plot confusion matrix
        
        Args:
            dataset: Which dataset to evaluate
            save_path: Path to save plot
            normalize: Whether to normalize confusion matrix
        """
        print("\n" + "="*80)
        print("CONFUSION MATRIX")
        print("="*80)
        
       
        
        if self.y_pred is None:
            self.evaluate(dataset=dataset, verbose=0)
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt=fmt, 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n Confusion matrix saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_prediction_samples(self, dataset='test', num_samples=25, 
                               correct_only=False, incorrect_only=False,
                               save_path=None):
        """
        Plot sample predictions
        
        Args:
            dataset: Which dataset to use
            num_samples: Number of samples to plot
            correct_only: Show only correct predictions
            incorrect_only: Show only incorrect predictions
            save_path: Path to save plot
        """
        X = self.data[f'X_{dataset}']
        y_true = self.y_true
    
        
        if self.y_pred is None:
            self.evaluate(dataset=dataset, verbose=0)
        
        # Filter samples
        if correct_only:
            indices = np.where(self.y_pred == y_true)[0]
            title_suffix = "(Correct Predictions)"
        elif incorrect_only:
            indices = np.where(self.y_pred != y_true)[0]
            title_suffix = "(Incorrect Predictions)"
        else:
            indices = np.arange(len(X))
            title_suffix = ""
        
        # Select random samples
        if len(indices) < num_samples:
            num_samples = len(indices)
        
        selected_indices = np.random.choice(indices, num_samples, replace=False)
        
        # Create grid
        grid_size = int(np.sqrt(num_samples))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        axes = axes.flatten()
        
        for i, idx in enumerate(selected_indices):
            img = X[idx]
            true_label = y_true[idx]
            pred_label = self.y_pred[idx]
            confidence = self.y_pred_proba[idx][pred_label]
            
            # Display image
            if img.shape[-1] == 1:
                axes[i].imshow(img.squeeze(), cmap='gray')
            else:
                axes[i].imshow(img)
            
            # Color: green for correct, red for incorrect
            color = 'green' if true_label == pred_label else 'red'
            
            # Title with true and predicted labels
            title = f"True: {self.class_names[true_label]}\n"
            title += f"Pred: {self.class_names[pred_label]}\n"
            title += f"Conf: {confidence:.2%}"
            
            axes[i].set_title(title, fontsize=8, color=color)
            axes[i].axis('off')
        
        plt.suptitle(f'Prediction Samples {title_suffix}', fontsize=14, y=0.995)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n Prediction samples saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_top_k_accuracy(self, dataset='test', k_values=[1, 3, 5], save_path=None):
        """
        Plot top-k accuracy for different k values
        
        Args:
            dataset: Which dataset to use
            k_values: List of k values to evaluate
            save_path: Path to save plot
        """
       
        
        if self.y_pred_proba is None:
            self.evaluate(dataset=dataset, verbose=0)
        
        # Calculate top-k accuracy for each k
        accuracies = []
        for k in k_values:
            top_k_preds = np.argsort(self.y_pred_proba, axis=1)[:, -k:]
            correct = np.any(top_k_preds == self.y_true.reshape(-1, 1), axis=1)
            accuracy = correct.mean()
            accuracies.append(accuracy)
            print(f"Top-{k} Accuracy: {accuracy:.4f}")
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(k_values)), accuracies, color='skyblue', edgecolor='navy')
        plt.xticks(range(len(k_values)), [f'Top-{k}' for k in k_values])
        plt.ylabel('Accuracy')
        plt.title('Top-K Accuracy')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 0.02, f'{acc:.2%}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n Top-k accuracy plot saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def analyze_misclassifications(self, dataset='test', save_path=None):
        """
        Analyze most common misclassifications
        
        Args:
            dataset: Which dataset to use
            save_path: Path to save plot
        """
        print("\n" + "="*80)
        print("MISCLASSIFICATION ANALYSIS")
        print("="*80)
        
        
        
        if self.y_pred is None:
            self.evaluate(dataset=dataset, verbose=0)
        
        # Find misclassifications
        misclassified = self.y_true != self.y_pred
        
        # Count misclassification pairs
        misclass_pairs = {}
        for true_label, pred_label in zip(self.y_true[misclassified], self.y_pred[misclassified]):
            pair = (self.class_names[true_label], self.class_names[pred_label])
            misclass_pairs[pair] = misclass_pairs.get(pair, 0) + 1
        
        # Sort by frequency
        sorted_pairs = sorted(misclass_pairs.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTotal misclassifications: {misclassified.sum()} / {len(self.y_true)} ({misclassified.mean():.2%})")
        print(f"\nTop 10 Misclassification Pairs:")
        print(f"{'True Label':<20} {'Predicted As':<20} {'Count':<10}")
        print("-" * 60)
        
        for (true_label, pred_label), count in sorted_pairs[:10]:
            print(f"{true_label:<20} {pred_label:<20} {count:<10}")
        
        # Visualize top misclassifications
        if len(sorted_pairs) > 0:
            top_n = min(10, len(sorted_pairs))
            pairs = [f"{t} → {p}" for (t, p), _ in sorted_pairs[:top_n]]
            counts = [count for _, count in sorted_pairs[:top_n]]
            
            plt.figure(figsize=(12, 6))
            plt.barh(range(len(pairs)), counts, color='salmon', edgecolor='darkred')
            plt.yticks(range(len(pairs)), pairs)
            plt.xlabel('Number of Misclassifications')
            plt.title('Top 10 Most Common Misclassifications')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"\n Misclassification analysis saved to {save_path}")
            
            plt.show()
            plt.close()
    
    def generate_full_report(self, dataset='test', output_dir='results/image_classification'):
        """
        Generate comprehensive evaluation report with all visualizations
        
        Args:
            dataset: Which dataset to evaluate
            output_dir: Directory to save all outputs
        """
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE EVALUATION REPORT")
        print("="*80)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Evaluate
        metrics = self.evaluate(dataset=dataset)
        
        # 2. Classification report
        self.print_classification_report(dataset=dataset)
        
        # 3. Confusion matrix
        self.plot_confusion_matrix(
            dataset=dataset,
            save_path=output_path / 'confusion_matrix.png'
        )
        
        # 4. Normalized confusion matrix
        self.plot_confusion_matrix(
            dataset=dataset,
            save_path=output_path / 'confusion_matrix_normalized.png',
            normalize=True
        )
        
        # 5. Correct predictions
        self.plot_prediction_samples(
            dataset=dataset,
            num_samples=25,
            correct_only=True,
            save_path=output_path / 'correct_predictions.png'
        )
        
        # 6. Incorrect predictions
        self.plot_prediction_samples(
            dataset=dataset,
            num_samples=25,
            incorrect_only=True,
            save_path=output_path / 'incorrect_predictions.png'
        )
        
        # 7. Top-k accuracy
        self.plot_top_k_accuracy(
            dataset=dataset,
            save_path=output_path / 'top_k_accuracy.png'
        )
        
        # 8. Misclassification analysis
        self.analyze_misclassifications(
            dataset=dataset,
            save_path=output_path / 'misclassification_analysis.png'
        )
        
        print(f"\n Full evaluation report generated in {output_dir}/")