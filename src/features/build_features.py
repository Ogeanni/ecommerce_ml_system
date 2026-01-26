import pandas as pd
import numpy as np
import os
from pathlib import Path

class FeatureEngineering:
    """ Create features from merged Olist dataset """

    def __init__(self, df:pd.DataFrame):
        """
        Initialize with merged dataframe
        Args:
            df: merged orders dataframe
        """

        self.df = df.copy()
        self.feature_names = []


    def create_all_features(self) -> pd.DataFrame:
        """
        Create all features in sequence
        
        Returns:
            DataFrame with all engineered features
        """
        
        # Create features in order
        self.df = self.create_temporal_features(self.df)
        self.df = self.create_price_features(self.df)
        self.df = self.create_delivery_features(self.df)
        self.df = self.create_product_features(self.df)
        self.df = self.create_customer_features(self.df)
        self.df = self.create_seller_features(self.df)
        self.df = self.create_payment_features(self.df)
        self.df = self.create_interaction_features(self.df)
        
        print(f"\n Feature engineering complete!")
        print(f"Total features created: {len(self.feature_names)}")
        print(f"Final dataset shape: {self.df.shape}")
        
        return self.df


    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Create time base features from orders purchase timestamp"""

        # Extract components from purchase timestamp
        df["purchase_year"] = df["order_purchase_timestamp"].dt.year
        df["purchase_month"] = df["order_purchase_timestamp"].dt.month
        df["purchase_day"] = df["order_purchase_timestamp"].dt.day
        df["purchase_day_of_week"] = df["order_purchase_timestamp"].dt.dayofweek
        df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
        df["purchase_quarter"] = df["order_purchase_timestamp"].dt.quarter

        df["is_weekend"] = (df["purchase_day_of_week"] >=5).astype(int)
        df["is_business_hours"] = ((df["purchase_hour"] >=9) & (df["purchase_hour"] <=18)).astype(int)
        df["is_holiday_season"] = df["purchase_month"].isin([11,12]).astype(int)
        df["is_month_end"] = (df["purchase_day"] >= 25).astype(int)


        # Brazilian seasons (Southern Hemisphere)
        def get_season(month):
            if month in [12, 1, 2]:
                return 'summer'
            elif month in [3, 4, 5]:
                return 'autumn'
            elif month in [6, 7, 8]:
                return 'winter'
            else:
                return 'spring'
            

        df["season"] = df["purchase_month"].apply(get_season)

        # Calculate expected delivery days (promise)
        df["promised_delivery_days"] = (df["order_estimated_delivery_date"] - 
                                        df["order_delivered_customer_date"]).dt.days
        
        features = ["purchase_year", "purchase_month", "purchase_day",
                    "purchase_day_of_week", "purchase_hour", "purchase_quarter",
                    "is_weekend", "is_business_hours", "is_holiday_season", "is_month_end",
                    "season", "promised_delivery_days"]
        
        self.feature_names.extend(features)

        print(f"   Created {len(features)} temporal features")

        return df
        
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Create price and cost related features from order_items table"""

        # Total order cost
        df["total_order_cost"] = df["total_price"] + df["total_freight"]

        # Price per item
        df["price_per_item"] = df["total_price"] / df["num_items"]

        # Freight efficiency
        df["freight_to_price_ratio"] = df["total_freight"] / (df["total_price"] +1) # +1 to avoid division by zero
        df["freight_per_item"] = df["total_freight"] / df["num_items"]

        # Price spread (range of item prices in order)
        df["price_range"] = df["max_item_price"] - df["min_item_price"]

        # Price categories based on quartiles (calculate on training data in real scenario)
        df["order_value_category"] = pd.cut(
            df["total_price"],
            bins=[0, 50, 100, 200, np.inf],
            labels=["low", "medium", "high", "premium"]
        )

        # Binary indicators
        df["is_small_order"] = (df["total_price"] < 50).astype(int)
        df["is_large_order"] = (df["total_price"] > 500).astype(int)
        df["is_high_freight"] = (df["freight_to_price_ratio"] > 0.2).astype(int)
        df["is_bulk_purchase"] = (df["num_items"] >= 5).astype(int)

        # Standardized price (z-score)
        df["price_zscore"] = (df["total_price"] - df["total_price"].mean()) / df["total_price"].std()

        features = ["total_order_cost", "price_per_item", "freight_to_price_ratio",
                    "freight_per_item", "price_range", "order_value_category", "is_small_order",
                    "is_large_order", "is_high_freight", "is_bulk_purchase", "price_zscore"]
        
        self.feature_names.extend(features)

        print(f"   Created {len(features)} price features")
        return df
    

    def create_delivery_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Create delivery performance features"""

        # Actual delivery time
        df["actual_delivery_days"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days

        # Delivery delay already created in target creation (delivery_delay_days)
        # But let's create additional variants

        # Late delivery indicators
        df["is_late"] = (df["delivery_delay_days"] > 0).astype(int)
        df["is_very_late"] = (df["delivery_delay_days"] > 5).astype(int)
        df["is_extremely_late"] = (df["delivery_delay_days"] > 10).astype(int)
        df["is_early"] = (df["actual_delivery_days"] < 0).astype(int)

        # Delivery speed categories
        def categorize_delivery_speed(days):
            if days <= 7:
                return 'express'
            elif days <= 15:
                return 'standard'
            elif days <= 30:
                return 'slow'
            else:
                return 'very_slow'
            
        df["delivery_speed"] = df["actual_delivery_days"].apply(categorize_delivery_speed)

        # Promise realism
        df["promised_realistic"] = (df["promised_delivery_days"] <=30).astype(int)
        df["overpromised"] = ((df["promised_delivery_days"] < 7) & (df["actual_delivery_days"] >10)).astype(int)

        # Delivery efficiency ratio
        df["delivery_efficiency"] = df["actual_delivery_days"] / (df["promised_delivery_days"]+1)

        features = ["actual_delivery_days", "is_late", "is_very_late", "is_extremely_late",
                    "is_early", "delivery_speed", "promised_realistic", "overpromised",
                    "delivery_efficiency"]
        
        self.feature_names.extend(features)

        print(f"   Created {len(features)} delivery features")
        return df
    
    

    def create_product_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Create product related features """

        # Weight features
        df["total_weight_kg"] = df["total_weight_g"] / 1000
        df["avg_weight_kg"] = df["avg_weight_g"] / 1000
        df["max_weight_kg"] = df["max_weight_g"] / 1000

        # Weight categories
        df["weight_category"] = pd.cut(
            df["total_weight_kg"],
            bins=[0, 1, 5, 10, np.inf],
            labels=["light", "medium", "heavy", "very_heavy"]
        )

        df["is_heavy_order"] = (df["total_weight_kg"] >10).astype(int)

        # Volume features (approximate)
        df["avg_volume_cm3"] = (df["avg_length_cm"] * df["avg_height_cm"] * df["avg_width_cm"]
        ).fillna(0)

        df["avg_volume_liters"] = df["avg_volume_cm3"] / 1000
        df["is_bulky"] = (df["avg_volume_liters"] > 50).astype(int)

        # Photo quality indicator
        df["has_multiple_photos"] = (df["avg_photos_qty"] >= 3).astype(int)
        
        # Product diversity
        df["is_multi_product"] = (df["num_unique_products"] > 1).astype(int)
        df["is_multi_category"] = (df["num_unique_products"] > 2).astype(int)  # Proxy for categories
        
        # Handle missing categories
        df["primary_category"] = df["primary_category"].fillna("unknown")

        # Category-based risk indicators (based on common e-commerce patterns)
        fragile_categories = [
            'housewares', 'computers_accessories', 'electronics', 
            'telephony', 'tablets_printing_image', 'small_appliances'
        ]
        
        fashion_categories = [
            'fashion_male_clothing', 'fashion_female_clothing', 
            'fashion_shoes', 'fashion_underwear_beach', 'watches_gifts'
        ]
        
        df["is_fragile_category"] = df["primary_category"].isin(fragile_categories).astype(int)
        df["is_fashion_category"] = df["primary_category"].isin(fashion_categories).astype(int)

        features = [
            "total_weight_kg", "avg_weight_kg", "max_weight_kg", "weight_category",
            "is_heavy_order", "avg_volume_cm3", "avg_volume_liters", "is_bulky",
            "has_multiple_photos", "is_multi_product", "is_multi_category",
            "is_fragile_category", "is_fashion_category"]

        self.feature_names.extend(features)

        print(f"   Created {len(features)} product features")
        return df
    


    def create_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Create customer geography features """

        # Major cities in Brazil
        major_cities = [
            'sao paulo', 'rio de janeiro', 'brasilia', 'salvador', 
            'fortaleza', 'belo horizonte', 'manaus', 'curitiba',
            'recife', 'porto alegre'
        ]

        df["customer_city_lower"] = df["customer_city"].str.lower()
        df["is_major_city"] = df["customer_city_lower"].isin(major_cities).astype(int)

        # Brazilian regions (grouping states)
        region_mapping = {
            # Southeast (most developed)
            'SP': 'southeast', 'RJ': 'southeast', 'MG': 'southeast', 'ES': 'southeast',
            # South
            'PR': 'south', 'SC': 'south', 'RS': 'south',
            # Northeast
            'BA': 'northeast', 'CE': 'northeast', 'PE': 'northeast', 'MA': 'northeast',
            'RN': 'northeast', 'PB': 'northeast', 'AL': 'northeast', 'SE': 'northeast', 'PI': 'northeast',
            # North
            'AM': 'north', 'PA': 'north', 'RO': 'north', 'AC': 'north', 
            'RR': 'north', 'AP': 'north', 'TO': 'north',
            # Center-West
            'GO': 'center_west', 'MT': 'center_west', 'MS': 'center_west', 'DF': 'center_west'
        }

        df["customer_region"] = df["customer_state"].map(region_mapping).fillna("unknown")

        # State indicators for most common states
        df["is_sao_paulo"] = (df["customer_state"] == "SP").astype(int)
        df["is_rio_janeiro"] = (df["customer_state"] == "RJ").astype(int)

        # Drop temporary column
        df = df.drop(columns=["customer_city_lower"])

        features = ["is_major_city", "customer_region", "is_sao_paulo", "is_rio_janeiro"]
        self.feature_names.extend(features)
        
        print(f"   Created {len(features)} customer features")
        return df
    

    def create_seller_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Create seller geography and matching features """

        # Seller region (same mapping as customer)
        region_mapping = {
            'SP': 'southeast', 'RJ': 'southeast', 'MG': 'southeast', 'ES': 'southeast',
            'PR': 'south', 'SC': 'south', 'RS': 'south',
            'BA': 'northeast', 'CE': 'northeast', 'PE': 'northeast', 'MA': 'northeast',
            'RN': 'northeast', 'PB': 'northeast', 'AL': 'northeast', 'SE': 'northeast', 'PI': 'northeast',
            'AM': 'north', 'PA': 'north', 'RO': 'north', 'AC': 'north', 
            'RR': 'north', 'AP': 'north', 'TO': 'north',
            'GO': 'center_west', 'MT': 'center_west', 'MS': 'center_west', 'DF': 'center_west'
        }

        df["seller_region"] = df["seller_state"].map(region_mapping).fillna("unknown")

        # Geographic matching
        df["is_same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)
        df["is_same_region"] = (df["customer_region"] == df["seller_region"]).astype(int)
        df["is_same_city"] = (df["customer_city"] == df["seller_city"]).astype(int)
        
        # Multi-seller complexity
        df["is_multi_seller"] = (df["num_unique_sellers"] > 1).astype(int)
        
        # Seller in major hub
        df["seller_in_sao_paulo"] = (df["seller_state"] == "SP").astype(int)

        features = ["seller_region", "is_same_state", "is_same_region", "is_same_city",
            "is_multi_seller", "seller_in_sao_paulo"]
        
        self.feature_names.extend(features)
        
        print(f"   Created {len(features)} seller features")
        return df
    


    def create_payment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Create payment related features """

         # Payment method indicators
        df["pays_credit_card"] = (df["primary_payment_type"] == "credit_card").astype(int)
        df["pays_boleto"] = (df["primary_payment_type"] == "boleto").astype(int)
        df["pays_voucher"] = (df["primary_payment_type"] == "voucher").astype(int)
        df["pays_debit"] = (df["primary_payment_type"] == "debit").astype(int)

        # Installment features
        df["uses_installments"] = (df["max_installments"] > 1).astype(int)

        # Installment categories
        def categorize_installments(n):
            if n == 1:
                return 'cash'
            elif n <= 6:
                return 'short_term'
            else:
                return 'long_term'
            
        df["installment_catgeory"] = df["max_installments"].apply(categorize_installments)

        # Payment value per installment
        df["installment_amount"] = df["total_payment_value"] / df["max_installments"]

        # Payment complexity
        df["is_split_payment"] = (df["num_payment_methods"] >1).astype(int)

        # Payment value matches order cost (within tolerance)
        df["payment_matches_order"] = (np.abs(df["total_payment_value"] - df["total_order_cost"]) < 5
        ).astype(int)

        # Delayed payment risk (boleto takes days to clear)
        df["has_delayed_payment_method"] = df["pays_boleto"].copy()

        features = ["pays_credit_card", "pays_boleto", "pays_voucher", "pays_debit",
                    "uses_installments", "installment_catgeory", "installment_amount",
                    "is_split_payment", "payment_matches_order", "has_delayed_payment_method"]
        
        self.feature_names.extend(features)

        print(f"   Created {len(features)} payment features")
        return df
    


    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Create interaction features between different aspects """

        # Price × Delivery interactions
        df["price_delay_interaction"] = df["total_price"] * df['delivery_delay_days'].clip(lower=0)
        df["value_per_delivery_day"] = df["total_price"] / (df['actual_delivery_days'] + 1)

        # Weight × Distance interactions
        df["weight_distance_risk"] = df["total_weight_kg"] * (1 - df["is_same_state"])
        df["freight_weight_efficiency"] = df["total_freight"] / (df["total_weight_kg"]+0.1)

        # Order complexity score
        df["order_complexity"] = (
            df["num_items"] *
            (df["num_unique_products"] + 1) *
            (df["num_unique_sellers"] + 1)
        )

        # Value risk score
        df["value_risk"] = (
            df["total_price"] * 
            (1 + df["is_late"]) * 
            (1 + df["is_multi_seller"])
        )

        # Handling risk for fragile items
        df["handling_risk_score"] = (
            df["is_fragile_category"] * 
            df["total_weight_kg"] * 
            (1 - df["is_same_region"])
        )

        # Distance premium (high freight for low value)
        df["distance_premium"] = (df["freight_to_price_ratio"] * (1 - df["is_same_state"]))
        
        # Holiday × delivery interaction
        df["holiday_delivery_risk"] = (df["is_holiday_season"] * df["actual_delivery_days"])
        
        # Bulk × weight interaction
        df["bulk_weight_score"] = df["num_items"] * df["avg_weight_kg"]

        features = ["price_delay_interaction", "value_per_delivery_day", "weight_distance_risk",
                    "freight_weight_efficiency", "order_complexity", "value_risk",
                    "handling_risk_score", "distance_premium", "holiday_delivery_risk",
                    "bulk_weight_score"]
        
        self.feature_names.extend(features)
        
        print(f"   Created {len(features)} interaction features")
        return df
    

    def get_feature_list(self) -> list:
        """ Return list of all created feature names """
        self.feature_names


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to engineer all features
    
    Args:
        df: Merged orders dataframe
        
    Returns:
        DataFrame with all features
    """

    engineer = FeatureEngineering(df)
    df_featured = engineer.create_all_features()

    return df_featured


if __name__ == "__main__":

    print("="*80)
    print("FEATURE ENGINEERING FOR E-COMMERCE CONTROL QUALITY")
    print("="*80)

    # Load merged data
    print("\nLoading merged dataset...")

    df = pd.read_csv("data/processed/merged_data.csv", parse_dates=[
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ])

    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Engineer features
    df_featured = engineer_features(df)

    # Save featured dataset
    print("\nSaving featured dataset...")
    df_featured.to_csv('data/processed/featured_orders.csv', index=False)

    print(f" Saved to: data/processed/featured_orders.csv")
    print(f"Final shape: {df_featured.shape[0]:,} rows × {df_featured.shape[1]} columns")


    # Display feature summary
    print("\n" + "="*80)
    print("FEATURE SUMMARY")
    print("="*80)
    
    print(f"\nTotal features created: {len(FeatureEngineering(df).feature_names)}")
    
    # Show sample of data
    print("\nSample of featured data:")
    print(df_featured[['total_price', 'num_items', 'is_low_quality', 
                       'delivery_delay_days', 'customer_region', 
                       'is_same_state']].head(10))
    
    # Show target distribution
    print("\nTarget Distribution:")
    print(df_featured['is_low_quality'].value_counts())
    print(f"\nLow Quality: {df_featured['is_low_quality'].mean()*100:.2f}%")
