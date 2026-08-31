import pandas as pd
import os

def load_and_inspect_data(file_path):
    """Loads the HR dataset and outputs its structural footprint."""

    # 1. Verify the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find the dataset at: {file_path}. Please check the path.")

    # 2. Load the data into a Pandas DataFrame
    df = pd.read_csv(file_path)

    # 3. Print the system state and data architecture
    print("=== R.E.T.A.I.N. Data Ingestion Protocol ===")
    print(f"Dataset Successfully Loaded.")
    print(f"Total Employee Records: {df.shape[0]}")
    print(f"Tracked Behavioral/Demographic Features: {df.shape[1]}\n")

    print("=== Feature Data Types & Missing Values ===")
    print(df.info())

    return df


if __name__ == "__main__":
    # Point the script to your raw data folder
    # Make sure this matches the exact name of your downloaded file!
    DATA_PATH = "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"

    employee_data = load_and_inspect_data(DATA_PATH)