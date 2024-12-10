import pandas as pd

def load_csv_data():
    try:
        return {
            "tags": pd.read_csv("./database/tags.csv").to_dict(orient="records"),
            "categories": pd.read_csv("./database/category.csv").to_dict(orient="records"),
            "colors": pd.read_csv("./database/colours.csv").to_dict(orient="records"),
            "brands": pd.read_csv("./database/brands.csv").to_dict(orient="records"),
            "products": pd.read_csv("./database/products.csv").to_dict(orient="records"),
        }
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return {}