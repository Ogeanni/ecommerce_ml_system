"""
Prepare featured data for model training
- Handle missing values
- Encode categorical variables
- Scale numerical features
- Split into train/val/test sets
"""

import pandas as pd
import numpy as np

import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

class ModelDataPreparator:

    def __init__(self, config_path = "config/modeling_config.yaml"):
        
        self.scalers = {}
        self.encoders = {}
        self.feature_names = None

    
    def prepare_data(self, df: pd.DataFrame, test_size=0.15, val_size=0.15, random_state=42):
        """
        Complete data preparation pipeline
        
        Args:
            df: Featured dataset
            test_size: Proportion for test set
            val_size: Proportion for validation set (from remaining after test)
            random_state: Random seed
            
        Returns:
            Dictionary with train, val, test splits
        """

        print("="*80)
        print("PREPARING DATA FOR MODELING")
        print("="*80)

        # Step 1: Separate features and target
        X, y = self.separate_features_target(df)

        # Step 2: Handle missing values
        X = self.handle_missing_values(X)

        # Step 3: Encode categorical variables
        X = self.encode_categorical_features(X)

        # Step 4: Time-based split (important for time-series data)
        X_train, X_temp, y_train, y_temp = self.time_based_split(df, X, y, test_size=test_size)

        # Split temp into validation and test
        val_ratio = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5,
                                                        random_state=random_state, stratify=y_temp)
        
        # Step 5: Scale numerical features (fit on train only!)
        X_train_scaled = self.scale_features(X_train, fit=True)
        X_test_scaled = self.scale_features(X_test, fit=False)
        X_val_scaled = self.scale_features(X_val, fit=False)

        # Store feature names
        self.feature_names = X_train_scaled.columns.tolist()

        # Create splits dictionary
        splits = {
            "X_train": X_train_scaled,
            "X_val": X_val_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test
        }

        # Print summary
        self.print_split_summary(splits)

        return splits
    

    def separate_features_target(self, df: pd.DataFrame):
        """ Separate features and target, drop unnecessary columns """

        # Columns to drop (not features)
        drop_columns = [
            "order_id",                          # Identifier
            "customer_id",                       # Identifier
            "primary_seller_id",                 # Identifier
            "order_purchase_timestamp",          # Raw timestamp
            "order_delivered_customer_date",     # Raw timestamp (used for target)
            "order_estimated_delivery_date",     # Raw timestamp
            "is_low_quality",                    # Target variable
            "review_score",                      # Used to create target
            "delivery_delay_days"                # Used to create target
        ]

        # Drop columns that exist in dataframe
        drop_columns = [col for col in drop_columns if col in df.columns]

        # Separate target
        y = df["is_low_quality"].copy()

        # Separate features
        X = df.drop(columns=drop_columns)

        print(f"   Features: {X.shape[1]} columns")
        print(f"   Target: {y.name}")
        print(f"   Samples: {len(X):,}")

        return X, y
    

    def handle_missing_values(self, X: pd.DataFrame):
        """ Handle missing values in features """
        print("\n[2/5] Handling missing values...")

        missing_before = X.isnull().sum().sum()

        # Numerical columns: fill with median
        numerical_cols = X.select_dtypes(include=[np.number]).columns #select only numeric columns (int, float, etc.) and .columns - get their names as a list/index
        for col in numerical_cols:
            if X[col].isnull().sum() > 0:
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val) # Replace all NaN in that column with the median value.
                print(f"   Filled {col} with median: {median_val:.2f}")

        # Categorical columns: fill with 'unknown'
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            if X[col].isnull().sum() > 0:
                X[col] = X[col].fillna("unknown")
                print(f"   Filled {col} with 'unknown'")

        missing_after = X.isnull().sum().sum()

        print(f"   Missing values before: {missing_before}")
        print(f"   Missing values after: {missing_after}")

        return X
    

    def encode_categorical_features(self, X: pd.DataFrame):
        """ Encode Categorical Features """

        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist() # Give me all columns that are categorical in nature — either stored as strings (object) or as pandas categorical type (category)
        print(f"   Found {len(categorical_cols)} categorical columns")

        for col in categorical_cols:
            # Check number of unique values
            n_unique = X[col].nunique()

            if n_unique <= 10:
                # One-hot encode (for low cardinality)
                print(f"   One-hot encoding: {col} ({n_unique} categories)")
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X, dummies], axis=1)
                X = X.drop(columns=col)

            else:
                # Label encode (for high cardinality)
                print(f"   Label encoding: {col} ({n_unique} categories)")
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.encoders[col] = le

        print(f"   Final feature count: {X.shape[1]}")
        return X
    
    
    def time_based_split(self, df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, test_size=0.15):
        """
        Split data based on time (important for e-commerce data)
        
        Args:
            df: Original dataframe with timestamps
            X: Features
            y: Target
            test_size: Proportion for test set
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print("\n[4/5] Creating time-based train/test split...")

        # Sort by purchase timestamp
        if "order_purchase_timestamp" in df:
            df_sorted = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
            X_sorted = X.loc[df_sorted.index].reset_index(drop=True)
            y_sorted = y.loc[df_sorted.index].reset_index(drop=True) # Make sure features X and target y are in the same chronological order as the orders.

            # Split at time cutoff
            split_idx = int(len(df_sorted) * (1 - test_size))

            X_train = X_sorted.iloc[:split_idx]
            X_temp = X_sorted.iloc[split_idx:]
            y_train = y_sorted.iloc[:split_idx]
            y_temp = y_sorted.iloc[split_idx:]

            print(f"   Time-based split at index: {split_idx}")

        else:
            # Fallback to random split if no timestamp
            print("   Warning: No timestamp found, using random split")
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

        return X_train, X_temp, y_train, y_temp

    
    def scale_features(self, X: pd.DataFrame, fit=False):
        """
        Scale numerical features
        
        Args:
            X: Features dataframe
            fit: Whether to fit scaler (True for train, False for val/test)
            
        Returns:
            Scaled features
        """

        print("\n[5/5] Scaling features (fitting on training data)...")
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        # Don't scale binary features (0/1)
        binary_cols = []
        for col in numerical_cols:
            if X[col].nunique() == 2 and set(X[col].unique()).issubset({0,1}):
                binary_cols.append(col)

        cols_to_scale = [col for col in numerical_cols if col not in binary_cols]

        # Replace inf/-inf → NaN
        X[cols_to_scale] = X[cols_to_scale].replace([np.inf, -np.inf], np.nan)
        # Impute NaNs
        X[cols_to_scale] = X[cols_to_scale].fillna(X[cols_to_scale].median())

        if fit:
            print(f"   Scaling {len(cols_to_scale)} numerical features")
            print(f"   Keeping {len(binary_cols)} binary features unscaled")
  
            # Fit scaler
            scaler = StandardScaler()
            X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
            self.scalers["standard"] = scaler
            self.scalers["columns"] = cols_to_scale

        else:
            # Transform using fitted scaler
            scaler = self.scalers['standard']
            cols_to_scale = self.scalers['columns']
            X[cols_to_scale] = scaler.transform(X[cols_to_scale])
        
        return X
    

    def print_split_summary(self, splits):
        """ Print summary of data splits """
        print("\n" + "="*80)
        print("DATA SPLIT SUMMARY")
        print("="*80)
        
        print(f"\nTrain set: {len(splits['X_train']):,} samples")
        print(f"  Low quality: {splits['y_train'].sum():,} ({splits['y_train'].mean()*100:.2f}%)")
        print(f"  Good quality: {(~splits['y_train'].astype(bool)).sum():,} ({(1-splits['y_train'].mean())*100:.2f}%)")
        
        print(f"\nValidation set: {len(splits['X_val']):,} samples")
        print(f"  Low quality: {splits['y_val'].sum():,} ({splits['y_val'].mean()*100:.2f}%)")
        print(f"  Good quality: {(~splits['y_val'].astype(bool)).sum():,} ({(1-splits['y_val'].mean())*100:.2f}%)")
        
        print(f"\nTest set: {len(splits['X_test']):,} samples")
        print(f"  Low quality: {splits['y_test'].sum():,} ({splits['y_test'].mean()*100:.2f}%)")
        print(f"  Good quality: {(~splits['y_test'].astype(bool)).sum():,} ({(1-splits['y_test'].mean())*100:.2f}%)")
        
        print(f"\nFeatures: {len(self.feature_names)}")


    def save_preprocessing_objects(self, save_dir="models/preprocessing"):
        """ Save scalers and encoders for later use """

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Save Scalers
        if self.scalers:
            joblib.dump(self.scalers, f"{save_dir}/scalers.plk")

        # Save Encoders
        if self.encoders:
            joblib.dump(self.encoders, f"{save_dir}/encoders.plk")

        # Save Feature Names
        if self.feature_names:
            with open(f"{save_dir}/feature_names.json", "w") as f:
                json.dump(self.feature_names, f)

        print(f"\n Saved preprocessing objects to {save_dir}/")


def prepare_modeling_data(df_path="data/processed/featured_orders.csv"):
    """
        Convenience function to prepare data for modeling
    
    Args:
        df_path: Path to featured dataset
        
    Returns:
        Dictionary with train/val/test splits
        """

        # Load data
    df = pd.read_csv(df_path, parse_dates=[
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date"
        ])

    # Prepare data
    preparator = ModelDataPreparator()
    splits = preparator.prepare_data(df)

    # Save preprocessing objects
    preparator.save_preprocessing_objects()

    return splits, preparator
    

if __name__ == "__main__":
    splits, preparator = prepare_modeling_data()
