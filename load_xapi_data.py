# load_xapi_data.py
"""
Script to download and load the xAPI-Edu-Data dataset from Kaggle
and perform a quick exploration.
"""

import kagglehub
import pandas as pd

def load_xapi_data():
    # Download latest version of dataset
    path = kagglehub.dataset_download("aljarah/xAPI-Edu-Data")
    print("Dataset downloaded to:", path)

    # Load CSV file
    df = pd.read_csv(path + "/xAPI-Edu-Data.csv")

    # Quick exploration
    print("\n--- Head of dataset ---")
    print(df.head())

    print("\n--- Info ---")
    print(df.info())

    print("\n--- Target distribution ---")
    print(df['Class'].value_counts())

    return df

if __name__ == "__main__":
    df = load_xapi_data()