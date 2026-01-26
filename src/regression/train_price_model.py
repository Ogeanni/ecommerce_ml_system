import pandas as pd
import numpy as np
import math, copy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score

from scipy.sparse import csr_matrix, hstack

import joblib
import os


def load_data(file_path, sample_size=10000):

    df = pd.read_csv(file_path, encoding="latin1")

    df =  df.sample(n=min(sample_size, len(df)), random_state=42)

    # Retain rows where the price of an item is greater than zero
    df = df.loc[df["price"] > 0]

    df[["main_category", "sub_category", "sub_sub_category"]] = df["category_name"].str.split("/", expand=True, n=2)
    df["name_length"] = df.item_description.str.len()
    df["desc_length"] = df.item_description.str.len()

    df.brand_name.fillna("Unknown", inplace=True)
    df.main_category.fillna("Unknown", inplace=True)
    df.sub_category.fillna("Unknown", inplace=True)
    df.sub_sub_category.fillna("Unknown", inplace=True)

    return df

def feature_engineering(train_df, y_train):

    tmp = train_df.copy()
    tmp["y"] = y_train.values   
    # Numerical features
    X_train_num = train_df[["item_condition_id", "shipping", "name_length", "desc_length"]]

    global_mean = y_train.mean()

    brand_counts = train_df["brand_name"].value_counts()
    X_train_brand_freq = train_df["brand_name"].map(brand_counts)

    brand_avg_price = tmp.groupby("brand_name")["y"].mean()
    X_train_brand_target = train_df["brand_name"].map(brand_avg_price).fillna(global_mean)

    # Category Encoding
    encoder_cat = OneHotEncoder(sparse_output=True, handle_unknown="ignore") 
    categorical_features = ["main_category", "sub_category", "sub_sub_category"]

    # Fit and transform
    X_train_encoded_cat = encoder_cat.fit_transform(train_df[categorical_features])
  

    train_df["name"] = train_df["item_description"].str.lower().str.strip()
    train_df["item_description"] = train_df["item_description"].str.lower().str.strip()
    train_df["item_description"] =  train_df["item_description"].replace("No description yet", "")

     # TF-IDF text features
    tfidf_name = TfidfVectorizer(max_features=5000,ngram_range=(1, 2),stop_words="english")
    tfidf_desc = TfidfVectorizer(max_features=5000,ngram_range=(1, 2),stop_words="english")

    X_train_name = tfidf_name.fit_transform(train_df["name"])
    X_train_desc = tfidf_desc.fit_transform(train_df["item_description"])

     # Store fitted encoders/vectorizers for later use
    fitted_objects = {
        'brand_counts': brand_counts,
        'brand_avg_price': brand_avg_price,
        'encoder_cat': encoder_cat,
        'tfidf_name': tfidf_name,
        'tfidf_desc': tfidf_desc
    }

    # inspect the first few TF-IDF features
    #name_feature_names = tfidf_name.get_feature_names_out()
    #print(name_feature_names[:10])
    
    
     # Combine everything
    X_train_final = hstack([
                            csr_matrix(X_train_num.values),
                            csr_matrix(X_train_brand_freq.values[:, None]),
                            csr_matrix(X_train_brand_target.values[:, None]),
                            X_train_encoded_cat, 
                            X_train_name,
                            X_train_desc ])

    return X_train_final, fitted_objects



def transform_test_data(test_df, fitted_objects):
    X_test_num = test_df[["item_condition_id", "shipping", "name_length", "desc_length"]]

    X_test_brand_freq = test_df["brand_name"].map(fitted_objects["brand_counts"]).fillna(0)
    X_test_brand_target = test_df["brand_name"].map(fitted_objects["brand_avg_price"]).fillna(fitted_objects["brand_avg_price"].mean())

    # Category Transform
    categorical_features = ["main_category", "sub_category", "sub_sub_category"]
    X_test_cat = fitted_objects["encoder_cat"].transform(test_df[categorical_features])

    # TF-IDF text features
    X_test_name = fitted_objects["tfidf_name"].transform(test_df["name"])
    X_test_desc = fitted_objects["tfidf_desc"].transform(test_df["item_description"])

    # Combine all features

    X_test_final = hstack([
                           csr_matrix(X_test_num.values),
                           csr_matrix(X_test_brand_freq.values[:,None]),
                           csr_matrix(X_test_brand_target.values[:,None]),
                           X_test_cat,
                           X_test_name,
                           X_test_desc])
    
    return X_test_final


def train_model(X_train, X_test, y_train, y_test):

    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()

    #linear_model = LinearRegression()
    #ridge_model = Ridge(alpha=1.0)

    models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Ridge': Ridge(alpha=1.0)
}
    
    for name, model in models.items():
        model.fit(X_train_dense, y_train)
        score = model.score(X_test_dense, y_test)
        print(f"{name} R²: {score:.4f}")

    #ridge_model.fit(X_train_dense, y_train)

    print("Evaluating model.....")
    train_pred = model.predict(X_train_dense)
    test_pred = model.predict(X_test_dense)

    train_pred_price = np.expm1(train_pred)
    test_pred_price = np.expm1(test_pred)
    
    # RMSE tells you “how wrong my guesses/predictions are, on average”
    #R² — “How smart is my guess/prediction?
    # If I just guessed/predicted the average number of candies for every bag, how much worse am I than my smart guesses?”
    #RMSE measures average error (difference between predicted and actual).
    train_rmse = np.expm1(np.sqrt(mean_squared_error(y_train, train_pred)))
    test_rmse = np.expm1(np.sqrt(mean_squared_error(y_test, test_pred)))
    test_r2 = r2_score(y_test, test_pred)

    np.expm1(0.6721)
    
    print(f"\nTrain RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test R2: {test_r2:.4f}")

    rmse_diff = abs(train_rmse - test_rmse)
    gap_pct = rmse_diff / test_rmse * 100
    print(f"Train-Test RMSE gap: {gap_pct:.1f}%")


    #return ridge_model, test_pred
    return model, score


def save_model(model):

    model_path = 'models/saved_models/price_prediction_model.pkl'
    joblib.dump(model, model_path)
    print(f"\n✓ Model saved to {model_path}")


if __name__ == "__main__":

    # Load data
    df = load_data("/Users/user/Documents/Projects/ml_projects/ecommerce_ml_system/data/raw/train2.csv", sample_size=10000)
    #print(df.columns)
   
    y_log = np.log1p(df["price"])
    X = df[["train_id", "name", "item_condition_id", "category_name", "brand_name",
            "shipping", "item_description", "main_category",
            "sub_category", "sub_sub_category", "name_length", "desc_length"]]
    
    
    X_train_df, X_test_df, y_train, y_test = train_test_split(X, y_log, test_size=0.3, random_state=42)

    # Fit train features
    X_train_final, fitted_objects = feature_engineering(X_train_df, y_train)
    X_test_final = transform_test_data(X_test_df, fitted_objects)

    

    # Train
    #model, predictions = train_model(X_train_final, X_test_final, y_train, y_test)
    model, score = train_model(X_train_final, X_test_final, y_train, y_test)

    # Save Model
    save_model(model)