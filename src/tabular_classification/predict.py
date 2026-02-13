"""Model inference pipeline for predicting quality on new orders"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class QualityPredictor:
    """Make predictions on new orders"""
    def __init__(self,
                 model_path = "models/tabular_classification/saved_models",
                 preprocessing_path = "models/tabular_classification/preprocessing"):
        """
        Load trained model and preprocessing objects
        
        Args:
            model_path: Path to saved model directory
            preprocessing_path: Path to preprocessing objects
        """
        print("Loading model and preprocessing objects...")

        preprocessing_dir = Path(preprocessing_path)

        # Load preprocessing objects
        self.scalers = joblib.load(f"{preprocessing_dir}/scalers.pkl")
        print(f"    Loaded scalers")

        try:
            self.encoders = joblib.load(f"{preprocessing_dir}/encoders.pkl")
            print(f"    Loaded encoders")
        except:
            self.encoders = {}

        # Load feature names
        with open(f"{preprocessing_dir}/feature_names.json", "r") as f:
            self.feature_names = json.load(f)
        print(f"    Loaded {len(self.feature_names)} feature names")

        # Load model metadata to find best model
        if model_path:
            # Try load this best model
            model_dir = Path("models/tabular_classification/saved_models")
            with open(model_dir/"best_model_metadata.json", "r") as f:
                metadata = json.load(f)

            model_name = metadata["model_name"]
            model_path = model_dir/f"best_model_{model_name}.pkl"

        # Load model
        print(f"\nLoading model from {model_path}...")
        self.model = joblib.load(model_path)
        print(f"    Loaded model: {model_name}")
        
        print(" Model ready for predictions!\n")
        

    def preprocess_new_order(self, order_data):
        """
        Preprocess a single new order or batch of orders
        
        Args:
            order_data: DataFrame with raw order features (after feature engineering)
            
        Returns:
            Preprocessed features ready for prediction
        """

        df = order_data.copy()

        # Handle Missing Values
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().any() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

        categorical_cols = df.select_dtypes(include=["obejct", "catgeory"]).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna("unknown")

        # Encode categorical variables
        for col in categorical_cols:
            n_unique = df[col].nunique()
            if n_unique <= 10:
                # One-hot encode
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
            else:
                # Label encode
                if col in self.encoders:
                    le = self.encoders[col]
                    # Handle unseen categories
                    df[col] = df[col].apply(lambda x: x if x in le.classes_ else "unknown")
                    df[col] = le.transform(df[col].astype(str))
                else:
                    # If encoder not found, convert to numeric
                    df[col] = pd.factorize(df[col])[0]

        # Scale numerical features
        cols_to_scale = self.scalers["columns"]
        scaler = self.scalers["standard"]

        # Only scale columns that exist
        cols_to_scale_existing = [col for col in cols_to_scale if col in df.columns]
        if cols_to_scale_existing:
            df[cols_to_scale_existing] = scaler.transform(df[cols_to_scale_existing])

        # Ensure all required features are present
        # Add missing features as zeros
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0

        # Select only the features used in training (in correct order)
        df = df[self.feature_names]

        return df
    
    def predict(self, order_data, return_proba = True):
        """
        Predict quality for new orders
        
        Args:
            order_data: DataFrame with order features
            return_proba: Whether to return probabilities
            
        Returns:
            Dictionary with predictions
        """
        # Preprocess
        X = self.preprocess_new_order(order_data)

        # Predict
        predictions = self.model.predict(X)

        results = {
            "predictions": predictions,
            "quality_labels": ["Good Quality" if p == 0 else "Low Quality" for p in predictions]
        }

        if return_proba:
            probabilities = self.model.predict_proba(X)
            results["probablities"] = probabilities
            results["risk_scores"] = probabilities[:, 1]            # Probability of low quality

        return results
    
    def predict_single_order(self, order_features, threshold=0.35):
        """
        Predict quality for a single order with detailed output
        
        Args:
            order_features: Dictionary or DataFrame row with order features
            threshold: Decision threshold for classification
            
        Returns:
            Dictionary with detailed prediction
        """
        # Convert to DataFrame if dictionary
        if isinstance(order_features, dict):
            df = pd.DataFrame([order_features])
        else:
            df = pd.DataFrame([order_features])

        # Get predictions
        results = self.predict(df, return_proba=True)

        risk_score = results["risk_scores"][0]
        prediction = 1 if risk_score >= threshold else 0

        # Determine risk level
        if risk_score < 0.3:
            risk_level = "LOW"
        elif risk_score < 0.6:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return {
            "prediction": prediction,
            "quality_label": ["Low Quality" if prediction == 1 else "Good Quality"],
            "risk_score": float(risk_score),
            "risk_level": risk_level,
            "confidence": float(max(results["probabilities"][0])),
            "should_flag": risk_score >= threshold
        }

    def batch_predict(self, orders_df, output_path = None, threshold=0.35):
        """
        Predict quality for batch of orders and save results
        
        Args:
            orders_df: DataFrame with multiple orders
            output_path: Path to save predictions (optional)
            
        Returns:
            DataFrame with predictions added
        """
        print(f"Making predictions for {len(orders_df)} orders...")
        
        # Get predictions
        results = self.predict(orders_df, return_proba=True)

        # Add predictions to original dataframe
        output_df = orders_df.copy()
        output_df["predicted_quality"] = results["predictions"]
        output_df["quality_label"] = results["quality_labels"]
        output_df["risk_score"] = results["risk_scores"]

        # Add risk level
        output_df["risk_level"] = pd.cut(
            output_df["risk_score"],
            bins = [0, 0.3, 0.6, 1.0],
            labels = ["LOW", "MEDIUM", "HIGH"]
        )

        # Save if path provided
        if output_path:
            output_df.to_csv(output_path, index=False)
            print(f" Predictions saved to: {output_path}")

        # Print summary
        print("\nPrediction Summary:")
        print(f"  Total orders: {len(output_df)}")
        print(f"  Predicted low quality: {(output_df['predicted_quality'] == 1).sum()} ({(output_df['predicted_quality'] == 1).mean()*100:.1f}%)")
        print(f"  Predicted good quality: {(output_df['predicted_quality'] == 0).sum()} ({(output_df['predicted_quality'] == 0).mean()*100:.1f}%)")
        print(f"\nRisk Level Distribution:")
        print(output_df['risk_level'].value_counts())
        
        return output_df
    

def load_predictor():
    """Convenience function to load predictor"""
    return QualityPredictor()

if __name__ == "main":
    print("="*80)
    print("E-COMMERCE QUALITY CONTROL - PREDICTION EXAMPLE")
    print("="*80 + "\n")
    
    # Load predictor
    predictor = QualityPredictor()

    # Example 1: Single order prediction
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Order Prediction")
    print("="*80)
    
    # Sample order (get this from your system)
    sample_order = {
        'total_price': 150.00,
        'total_freight': 25.00,
        'num_items': 2,
        'total_weight_kg': 3.5,
        'primary_category': 'electronics',
        'customer_state': 'SP',
        'seller_state': 'SP',
        'is_same_state': 1,
        'promised_delivery_days': 10,
        'primary_payment_type': 'credit_card',
        'max_installments': 3,
        # ... other features would be included
    }

    # Make prediction
    result = predictor.predict_single_order(sample_order)

    print(f"\nOrder Prediction Results:")
    print(f"  Quality Prediction: {result['quality_label']}")
    print(f"  Risk Score: {result['risk_score']:.2%}")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Flag for Review: {'YES' if result['should_flag'] else 'NO'}")

    # Example 2: Batch prediction
    print("\n" + "="*80)
    print("EXAMPLE 2: Batch Prediction on Test Set")
    print("="*80)
    
    # Load some sample data (use your featured dataset)
    try:
        test_data = pd.read_csv("data/processed/final_dataset.csv").head(50)

        # Make batch predictions
        predictions = predictor.batch_predict(test_data, output_path="results/sample_predictions.csv")
        print("\nSample predictions:")
        print(predictions[['order_id', 'quality_label', 'risk_score', 'risk_level']].head(10))
    except FileNotFoundError:
        print("Test data not found. Run feature engineering first.")




