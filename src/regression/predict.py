"""Inference pipeline for regression model"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from typing import Union, List, Dict

class DeliveryTimePredictor:
    """Predict delivery time for new orders"""

    def __init__(self,
                 model_path: str = "models/regression/saved_models",
                 preprocessing_path: str = "models/regression/preprocessing"):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model
            preprocessing_path: Path to preprocessing objects
        """
        # Load preprocessing objects
        preprocessing_dir = Path(preprocessing_path)

        print("Loading preprocessing objects...")

        # Load Scaler
        self.scalers = joblib.load(preprocessing_dir/"scalers.pkl")
        print(" Loaded scalers")

        # Load Encoders
        try:
            self.encoders = joblib.load(preprocessing_dir/"encoders.pkl")
            print(" Loaded encoders")
        except:
            self.encoders = {}

        # Load feature names
        with open(preprocessing_dir/"feature_names.json", "r") as f:
            self.feature_names = json.load(f)
        print(f" Loaded {len(self.feature_names)} feature names")

        # Load model
        if model_path is None:
            # Try to load best model
            model_dir = Path("models/regression/saved_models")
            with open(model_dir/"best_model_metadata.json", "r") as f:
                metadata = json.load(f)

            model_name = metadata["model_name"]
            model_path = model_dir/f"best_model_{model_name}.pkl"

        print(f"\nLoading model from {model_path}...")
        self.model = joblib.load(model_path)
        print(" Model loaded successfully\n")

    def preprocess_features(self, order_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess features for prediction
        
        Args:
            order_data: DataFrame with order features
            
        Returns:
            Preprocessed features
        """
        df = order_data.copy()

        # Handle missing values (same as training)
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna("unknown")

        # Encode categorical features (same as training)
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
                    df[col] = pd.factorize(df[col])[0]

        # Scale Features
        cols_to_scale = self.scalers["columns"]
        scaler = self.scalers["standard"]

        cols_to_scale_existing = [col for col in cols_to_scale if col in df.columns]
        if cols_to_scale_existing:
            df[cols_to_scale_existing] = scaler.transform(df[cols_to_scale_existing])

        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0

        # Select features in correct order
        df = df[self.feature_names]

        return df
    
    def predict(self, order_data: Union[pd.DataFrame, Dict]) -> np.ndarray:
        """
        Predict delivery time for orders
        
        Args:
            order_data: DataFrame or dictionary with order features
            
        Returns:
            Array of predicted delivery days
        """
        # Convert dict to DataFrame if needed
        if isinstance(order_data, Dict):
            order_data = pd.DataFrame([order_data])

        # Preprocess
        X = self.preprocess_features(order_data)

        # Predict
        predictions = self.model.predict(X)

        return predictions
    
    def predict_single_order(self, order_feature: Dict) -> Dict:
        """
        Predict delivery time for a single order with detailed output
        
        Args:
            order_features: Dictionary with order features
            
        Returns:
            Dictionary with prediction details
        """
        # Get prediction
        prediction = self.predict(order_feature)[0]

        # Round to nearest day
        predicted_days = int(np.round(prediction))

        # Create confidence interval (simplified)
        # In production, use model's prediction intervals if available
        lower_bound = max(1, predicted_days - 2)
        upper_bound = predicted_days + 2

        return {
            "predicted_days": predicted_days,
            "predicted_days_exact": float(prediction),
            "confidence_interval": {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            },
            "delivery_category": self._categorize_delivery_time(predicted_days)
        }
    
    def _categorize_delivery_time(self, days: int) -> str:
        """Categorize delivery time"""
        if days <= 7:
            return 'Express (1 week or less)'
        elif days <= 14:
            return 'Fast (1-2 weeks)'
        elif days <= 21:
            return 'Standard (2-3 weeks)'
        elif days <= 30:
            return 'Slow (3-4 weeks)'
        else:
            return 'Very Slow (over 1 month)'
        
    def batch_predict(self, orders_df: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
        """
        Predict delivery times for multiple orders
        
        Args:
            orders_df: DataFrame with multiple orders
            output_path: Path to save predictions
            
        Returns:
            DataFrame with predictions added
        """
        print(f"Predicting delivery times for {len(orders_df)} orders...")
        
        # Get predictions
        predictions = self.predict(orders_df)

        # Add to dataframe
        output_df = orders_df.copy()
        output_df["predicted_delivery_days"] = predictions
        output_df["predicted_delivery_days_rounded"] = np.round(predictions).astype(int)
        output_df["delivery_category"] = output_df["predicted_delivery_days_rounded"].apply(self._categorize_delivery_time)

        # Save if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            output_df.to_csv(output_path, index=False)
            print(f" Predictions saved to {output_path}")

        # Print summary
        print("\nPrediction Summary:")
        print(f"  Mean predicted delivery: {predictions.mean():.2f} days")
        print(f"  Median predicted delivery: {np.median(predictions):.2f} days")
        print(f"  Min predicted delivery: {predictions.min():.2f} days")
        print(f"  Max predicted delivery: {predictions.max():.2f} days")
        
        print("\nDelivery Category Distribution:")
        print(output_df['delivery_category'].value_counts())
        
        return output_df
    
    def predict_with_features_explanation(self, order_features: Dict, top_n: int = 10):
        """
        Predict with feature importance explanation
        
        Args:
            order_features: Order features
            top_n: Number of top features to show
            
        Returns:
            Prediction with feature importance
        """
        # Get prediction
        result = self.predict_single_order(order_features)

        # Get feature importance if available
        if hasattr(self.model, "feature_importance_"):
            importances = self.model.feature_importances_

            # Create feature importance dataframe
            feat_imp = pd.DataFrame({
                "feature": self.feature_names,
                "importance": importances
            }).sort_values("importance", ascending=False).head(top_n)

            result["top_features"] = feat_imp.to_dict("records")

        return result
    
def predict_delivery_time(order_features: Dict, model_path: str = None) -> Dict:
    """
    Convenience function to predict delivery time
    
    Args:
        order_features: Dictionary with order features
        model_path: Path to model (optional)
        
    Returns:
        Prediction results
    """
    predictor = DeliveryTimePredictor(model_path)
    result = predictor.predict_single_order(order_features)

    return result

# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    print("="*80)
    print("DELIVERY TIME PREDICTION - EXAMPLE")
    print("="*80)
    
    # Load predictor
    try:
        predictor = DeliveryTimePredictor()

        # Example order
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
            'purchase_month': 6,
            'is_weekend': 0,
            'is_holiday_season': 0,
        }

        # Predict
        print("\nSample Order:")
        for key, value in sample_order.items():
            print(f"  {key}: {value}")

        result = predictor.predict_single_order(sample_order)

        print("\n" + "="*80)
        print("PREDICTION RESULTS")
        print("="*80)
        print(f"\nPredicted Delivery Time: {result['predicted_days']} days")
        print(f"Exact Prediction: {result['predicted_days_exact']:.2f} days")
        print(f"Confidence Interval: {result['confidence_interval']['lower']}-{result['confidence_interval']['upper']} days")
        print(f"Category: {result['delivery_category']}")

        # Test batch prediction
        print("\n" + "="*80)
        print("BATCH PREDICTION TEST")
        print("="*80)
        
        # Load test data if available
        test_data_path = Path("data/regression/processed/test.csv")
        if test_data_path.exists():
            test_df = pd.read_csv(test_data_path)

            # Remove target if present
            if "actual_delivery_days" in test_df.columns:
                actual_days = test_df["actual_delivery_days"].copy()
                test_df = test_df.drop(columns=["actual_delivery_days"])
            else:
                actual_days = None

        # Predict on first 10 orders
        sample_df = test_df.head(10)
        predictions_df = predictor.batch_predict(sample_df)

        print("\nSample Predictions:")
        if actual_days is not None:
            predictions_df["actual_days"] = actual_days.head(10).values
            print(predictions_df[['predicted_delivery_days_rounded', 'actual_days', 'delivery_category']].to_string(index=False))
        else:
            print(predictions_df[['predicted_delivery_days_rounded', 'delivery_category']].head(10).to_string(index=False))

    except FileNotFoundError as e:
        print(f"\n Error: {e}")
        print("Train a model first using scripts/train_regression_model.py")

    






