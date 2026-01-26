import pandas as pd
import numpy as np
from pathlib import Path

def load_files(folder_path):
    folder_path = Path(folder_path)
    dataframes = {}

    for file in folder_path.glob("*.csv"):
        dataframes[file.stem] = pd.read_csv(file)

    return dataframes



def create_merged_orders_dataset(dfs):

    # Load Data
    orders = dfs["olist_orders_dataset"]
    order_date_cols = ["order_purchase_timestamp", "order_approved_at",
                       "order_delivered_carrier_date", "order_delivered_customer_date",
                       "order_estimated_delivery_date"]
    orders[order_date_cols] = orders[order_date_cols].apply(pd.to_datetime)

    reviews = dfs["olist_order_reviews_dataset"]
    orders_items = dfs["olist_order_items_dataset"]
    payments = dfs["olist_order_payments_dataset"]
    customers = dfs["olist_customers_dataset"]
    products = dfs["olist_products_dataset"]
    sellers = dfs["olist_sellers_dataset"]
    geolocation = dfs["olist_geolocation_dataset"]
    category_translation = dfs["product_category_name_translation"]

    print(f"Loaded {len(orders):,} orders")

    # ==================== FILTER ORDERS TABLE ====================

    # Keep only delivered orders from 2017 onwards with delivery dates
    orders_clean = orders[
        (orders["order_status"] == "delivered") &
        (orders["order_purchase_timestamp"] >= "2017-01-01") &
        (orders["order_delivered_customer_date"].notna())
    ][["order_id", "customer_id", "order_purchase_timestamp",
       "order_delivered_customer_date", "order_estimated_delivery_date"]]
    
    print(f"Filtered to {len(orders_clean):,} orders")

    # ==================== JOIN REVIEWS TABLE ====================

    reviews_clean = reviews[["order_id", "review_score"]].drop_duplicates("order_id")

    df = orders_clean.merge(reviews_clean,
                            on="order_id",
                            how="left").dropna(subset=["review_score"])
    
    print(f"After reviews: {len(df):,} orders")

    # ==================== AGGREGATE ORDER ITEMS ====================

    order_items_agg = orders_items.groupby("order_id").agg({
        "price": ["sum", "mean", "min", "max", "std"],
        "freight_value": ["sum", "mean"],
        "order_item_id": "count",
        "product_id": "nunique",
        "seller_id": ["nunique", "first"]
    }).reset_index()
    
    order_items_agg.columns = [
        "order_id", "total_price", "avg_item_price", "min_item_price",
        "max_item_price", "price_std", "total_freight", "avg_freight",
        "num_items", "num_unique_products", "num_unique_sellers", "primary_seller_id"
    ]

    order_items_agg["price_std"] = order_items_agg["price_std"].fillna(0)

    df = df.merge(order_items_agg, on="order_id", how="left")

    print(f"After order items: {len(df):,} orders")

    # ==================== ADD PRODUCTS TABLE ====================

    products_translated = products.merge(category_translation, on="product_category_name", how="left")
    product_clean = products_translated[[
        "product_id", "product_category_name_english", "product_weight_g", "product_length_cm",
        "product_height_cm", "product_width_cm", "product_photos_qty"
]].rename(columns={'product_category_name_english': 'product_category'})
    
    order_items_with_products = orders_items.merge(product_clean, on="product_id", how="left")

    product_agg = order_items_with_products.groupby("order_id").agg({
        "product_weight_g": ["sum", "mean", "max"],
        "product_length_cm": "mean",
        "product_height_cm": "mean",
        "product_width_cm": "mean",
        "product_photos_qty": "mean",
        "product_category": lambda x: x.mode()[0] if len(x.mode()) > 0 else "unknown"
    }).reset_index()

    product_agg.columns = ["order_id", "total_weight_g", "avg_weight_g", "max_weight_g",
                            "avg_length_cm", "avg_height_cm", "avg_width_cm",
                            "avg_photos_qty", "primary_category"]
    
    df = df.merge(product_agg, on="order_id", how="left")

    print(f"After products: {len(df):,} orders")

    # ==================== ADD CUSTOMERS TABLE ====================

    customers_clean = customers[["customer_id", "customer_zip_code_prefix",
                                 "customer_city", "customer_state"]]
    
    df = df.merge(customers_clean, on="customer_id", how="left")

    print(f"After customers: {len(df):,} orders")

    # ==================== ADD SELLERS TABLE ====================

    sellers_clean = sellers[["seller_id", "seller_zip_code_prefix", 
                         "seller_city", "seller_state"]]
    
    df = df.merge(sellers_clean, left_on="primary_seller_id", right_on="seller_id",
                  how="left").drop(columns=["seller_id"])
    
    print(f"After sellers: {len(df):,} orders")

    # ==================== ADD PAYMENTS TABLE ====================

    payments_agg = payments.groupby("order_id").agg({
        "payment_type": lambda x: x.mode()[0] if len(x.mode()) > 0 else "unknown",
        "payment_installments": "max",
        "payment_value": "sum",
        "payment_sequential": "count"
    }).reset_index()

    payments_agg.columns = ["order_id", "primary_payment_type", "max_installments",
                            "total_payment_value", "num_payment_methods"]
    
    df = df.merge(payments_agg, on="order_id", how="left")

    print(f"After payments: {len(df):,} orders")


    # ==================== CREATE TARGET VARIABLE ====================

    df["delivery_delay_days"] = (df["order_delivered_customer_date"] 
                                 - df["order_estimated_delivery_date"]).dt.days
    
    df["is_low_quality"] = ((df['review_score'] <= 2) |  (df['delivery_delay_days'] > 7)).astype(int)

    print(df['is_low_quality'].value_counts())
    print(f"\nLow Quality: {df['is_low_quality'].sum() / len(df) * 100:.2f}%")

    # ==================== SAVE MERGED DATA ====================

    df.to_csv("data/processed/merged_data.csv" ,index=False)

    print(f"\n Complete! Final dataset: {df.shape[0]:,} rows * {df.shape[1]} columns")
    print(f"Saved to: data/processed/merged_orders.csv")


if __name__ == "__main__":

    dataframes = load_files("/Users/user/Documents/Projects/ml_projects/ecommerce_ml_system/data/raw/olist")
    create_merged_orders_dataset(dataframes)