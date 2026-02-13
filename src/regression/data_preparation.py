""" Data preparation for regression (delivery time prediction)"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
import joblib
import json
from typing import Dict, Tuple


class RegressionDataPreparator:
    """ Prepare data for delivery time regression """
    def __init__(self, source_data_path: str = "data/processed/featured_orders.csv"):
        """
         Initialize data preparator
        
        Args:
            source_data_path: Path to processed e-commerce data
        """
        self.source_data_path = Path(source_data_path)
        self.output_dir = Path("data/regression")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.scalers = {}
        self.encoders = {}

    def load_and_prepare_data(self) -> Dict:
        """ 
        Load and prepare regression dataset
        
        Returns:
            Prepared DataFrame 
        """
        print("\n" + "="*80)
        print("LOADING AND PREPARING REGRESSION DATA")
        print("="*80)

        # Load source data
        print(f"\nLoading data from: {self.source_data_path}")
        df = pd.read_csv(self.source_data_path, parse_dates = [
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date"
        ])
        print(f"Loaded {len(df):,} orders")

        # Create target variable (actual delivery days)
        df["actual_delivery_days"] = (df["order_delivered_customer_date"] - 
                                      df["order_purchase_timestamp"]).dt.days     

        # Filter valid delivery times
        df = df[(df["actual_delivery_days"] > 0) & (df["actual_delivery_days"] < 90)].copy()
        print(f"After filtering: {len(df):,} orders")

        # Analyze target distribution
        self._analyze_target_distribution(df)

        return df
    
    def _analyze_target_distribution(self, df: pd.DataFrame):
        """ Analyze target variable distribution """

        print("\n" + "="*80)
        print("TARGET VARIABLE ANALYSIS")
        print("="*80)

        target = df["actual_delivery_days"]

        print(f"\nActual Delivery Days Statistics:")
        print(f"  Count: {len(target):,}")
        print(f"  Mean: {target.mean():.2f} days")
        print(f"  Median: {target.median():.2f} days")
        print(f"  Std: {target.std():.2f} days")
        print(f"  Min: {target.min():.2f} days")
        print(f"  Max: {target.max():.2f} days")
        print(f"  25th percentile: {target.quantile(0.25):.2f} days")
        print(f"  75th percentile: {target.quantile(0.75):.2f} days")

    def select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Select features for regression
        
        Args:
            df: DataFrame with all data
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """

        print("\n" + "="*80)
        print("FEATURE SELECTION")
        print("="*80)

        # Target variable
        target = df["actual_delivery_days"]

        # Features to exclude (identifiers, target-related, dates)
        exclude_features = [
            "order_id", "customer_id", "primary_seller_id",
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
            "actual_delivery_days",
            "delivery_delay_days",                                      # Derived from target
            "is_late", "is_very_late", "is_extremely_late",             # Derived from target
            "delivery_speed",                                           # Derived from target
            "is_early",                                                 # Derived from target
            "delivery_efficiency",                                      # Derived from target
            "is_low_quality",
            "review_score",
            "promised_realistic", "overpromised",
            "price_delay_interaction", "value_per_delivery_day",
            "holiday_delivery_risk", "value_risk", "promised_delivery_days"                                         
        ]
        # Select features
        feature_cols = [col for col in df.columns if col not in exclude_features]
        features = df[feature_cols].copy()

        print(f"\nSelected {len(feature_cols)} features:")
        print(f"  Numerical features: {features.select_dtypes(include=[np.number]).shape[1]}")
        print(f"  Categorical features: {features.select_dtypes(include=['object', 'category']).shape[1]}")

        return features, target
    
    def handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """ Handle missing values in features """
        print("\n" + "="*80)
        print("HANDLING MISSING VALUES")
        print("="*80)

        missing_before = X.isnull().sum().sum()
        print(f"\nMissing values before: {missing_before}")

        # Numerical columns: fill with median
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if X[col].isnull().sum() > 0:
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
                print(f"  Filled {col} with median: {median_val:.2f}")

        # Categorical columns: fill with mode or 'unknown'
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            if X[col].isnull().sum() > 0:
                X[col] = X[col].fillna("unknown")
                print(f"  Filled {col} with 'unknown'")

        missing_after = X.isnull().sum().sum()
        print(f"\nMissing values after: {missing_after}")

        return X
    

    def encode_categorical_features(self, X: pd.DataFrame, fit: bool = None) -> pd.DataFrame:
        """ Encode categorical variables """
        print("\n" + "="*80)
        print("ENCODING CATEGORICAL FEATURES")
        print("="*80)

        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        print(f"\nFound {len(categorical_cols)} categorical columns")

        for col in categorical_cols:
            n_unique = X[col].nunique()

            if n_unique <= 10:
                # One-hot encode low cardinality
                print(f"  One-hot encoding: {col} ({n_unique} categories)")

                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X, dummies], axis=1)
                X = X.drop(columns=[col])
            else:
                # Label encode high cardinality
                print(f"  Label encoding: {col} ({n_unique} categories)")

                if fit:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    self.encoders[col] = le
                else:
                    if col in self.encoders:
                        le = self.encoders[col]
                        # Handle unseen categories
                        X[col] = X[col].apply(lambda x: x if x in le.classes_ else "unknown")
                        X[col] = le.transform(X[col].astype(str))
                    else:
                        X[col] = pd.factorize(X[col])[0]
        print(f"\nFinal feature count: {X.shape[1]}")

        return X
    
    def create_train_val_test_split(self,
                                    X: pd.DataFrame,
                                    y: pd.Series,
                                    test_size: float = 0.15,
                                    val_size: float = 0.15,
                                    random_state: int = 42) -> Dict:
        """
        Create train/val/test splits
        
        Args:
            X: Features
            y: Target
            test_size: Test set proportion
            val_size: Validation set proportion
            random_state: Random seed
            
        Returns:
            Dictionary with splits
        """

        print("\n" + "="*80)
        print("CREATING TRAIN/VAL/TEST SPLITS")
        print("="*80)

         # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=random_state)

        splits = {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test 
        }

        # Print split info
        print(f"\nSplit sizes:")
        print(f"  Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Val: {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")

        print(f"\nTarget statistics per split:")
        for split_name in ["train", "val", "test"]:
            y_split = splits[f"y_{split_name}"]

            print(f"  {split_name.capitalize()}:")
            print(f"    Mean: {y_split.mean():.2f} days")
            print(f"    Median: {y_split.median():.2f} days")
            print(f"    Std: {y_split.std():.2f} days")

        return splits
    
    def scale_features(self,
                       X_train: pd.DataFrame,
                       X_val: pd.DataFrame,
                       X_test: pd.DataFrame) -> tuple:
        """
        Scale numerical features
        
        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            
        Returns:
            Tuple of scaled dataframes
        """
        print("\n" + "="*80)
        print("SCALING FEATURES")
        print("="*80)

        # Identify numerical columns
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

        # Don't scale binary features
        binary_cols = []
        for col in numerical_cols:
            if X_train[col].nunique() == 2 and set(X_train[col].unique()).issubset({0, 1}):
                binary_cols.append(col)

        cols_to_scale = [col for col in numerical_cols if col not in binary_cols]
        print(f"\nScaling {len(cols_to_scale)} numerical features")
        print(f"Keeping {len(binary_cols)} binary features unscaled")

        for df in [X_train, X_val, X_test]:
            df[cols_to_scale] = df[cols_to_scale].replace([np.inf, -np.inf], np.nan)
            df[cols_to_scale] = df[cols_to_scale].fillna(df[cols_to_scale].median())


        # Fit scaler on training data
        scaler = StandardScaler()
        X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
        X_val[cols_to_scale] = scaler.transform(X_val[cols_to_scale])
        X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

        self.scalers["standard"] = scaler
        self.scalers["columns"] = cols_to_scale

        return X_train, X_val, X_test
    
    def save_preprocessing_objects(self, feature_names: list):
        """ Save preprocessing objects for later use """

        save_dir = Path("models/regression/preprocessing")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save Scalers
        if self.scalers:
            joblib.dump(self.scalers, save_dir/"scalers.pkl")

        # Save Encoders
        if self.encoders:
            joblib.dump(self.encoders, save_dir/"encoders.pkl")

        # Save feature names
        with open(save_dir/"feature_names.json", "w") as f:
            json.dump(feature_names, f)

        print(f"\n Saved preprocessing objects to {save_dir}/")

    def prepare_complete_dataset(self) -> Dict:
        """
        Complete data preparation pipeline
        
        Returns:
            Dictionary with prepared train/val/test splits
        """

        # Load data
        df = self.load_and_prepare_data()

        # Select features
        X, y = self.select_features(df)

        # Handle missing values
        X = self.handle_missing_values(X)

        # Encode categorical features
        X = self.encode_categorical_features(X, fit=True)

        # Create splits
        splits = self.create_train_val_test_split(X, y)

        # Scale features
        X_train, X_val, X_test = self.scale_features(splits["X_train"], splits["X_val"], splits["X_test"])

        # Update splits with scaled data
        splits["X_train"] = X_train
        splits["X_val"] = X_val
        splits["X_test"] = X_test

        # Save preprocessing objects
        self.save_preprocessing_objects(X_train.columns.tolist())

        # Save processed data
        self._save_splits(splits)

        return splits
    
    def _save_splits(self, splits: Dict):
        """ Save train/val/test splits """
        save_dir = self.output_dir/"processed"
        save_dir.mkdir(parents=True, exist_ok=True)

        for split_name in ["train", "val", "test"]:
            X = splits[f"X_{split_name}"]
            y = splits[f"y_{split_name}"]

            # Combine X and y
            df = X.copy()
            df["actual_delivery_days"] = y.values

            # Save
            filepath = save_dir/f"{split_name}.csv"
            df.to_csv(filepath, index=False)
            print(f"  Saved {split_name} set: {filepath}")
        print(f"\n All splits saved to {save_dir}/")

    
def prepare_regression_data(source_path: str = "data/processed/final_dataset.csv"):
    """
    Convenience function to prepare regression data
    
    Args:
        source_path: Path to source dataset
        
    Returns:
        Dictionary with train/val/test splits
    """

    preparator = RegressionDataPreparator(source_path)
    splits = preparator.prepare_complete_dataset()

    return splits

if __name__ == "__main__":
    # Test data preparation
    splits = prepare_regression_data()

    print("\n" + "="*80)
    print("DATA PREPARATION COMPLETE")
    print("="*80)
    print(f"\nFinal dataset shapes:")
    print(f"  Features: {splits['X_train'].shape[1]}")
    print(f"  X_train: {splits['X_train'].shape}")
    print(f"  X_val: {splits['X_val'].shape}")
    print(f"  X_test: {splits['X_test'].shape}")
    

