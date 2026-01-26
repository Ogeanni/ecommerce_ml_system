import os
import zipfile

def setup_mercari_data():

    zip_path = "data/raw/mercari-price-suggestion-challenge.zip"

    if os.path.exists(zip_path):
        print("Found Zip File.....Extracting")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("data/raw/mercari/")
            print("Mercari Data Extracted Successfully")

    else:
        print("Zip not found")



def load_olist_data():
    zip_path = "/Users/user/Documents/Projects/ml_projects/ecommerce_ml_system/data/raw/olist_dataset.zip"

    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall("data/raw/olist/")
            print("Extracted Successfully")

    else:
        print("Zip not found")



if __name__ == "__main__":
    load_olist_data()


