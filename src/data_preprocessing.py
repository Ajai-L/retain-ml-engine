import os
import joblib
import pandas as pd
from typing import Tuple, Dict
from sklearn.preprocessing import LabelEncoder

# Defined constants
RAW_DATA_PATH = "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
PROCESSED_DATA_PATH = "data/processed/hr_data_encoded.csv"
ENCODERS_PATH = "models/label_encoders.pkl"

COLUMNS_TO_DROP = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]


def preprocess_data(
    input_path: str = RAW_DATA_PATH,
    output_path: str = PROCESSED_DATA_PATH,
    encoders_save_path: str = ENCODERS_PATH
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Preprocesses raw HR employee data for model training:
    - Drops zero-variance & identifier columns
    - Encodes target variable 'Attrition' to binary integers (Yes=1, No=0)
    - Vectorizes categorical string columns using LabelEncoder
    - Saves fitted encoders and exports preprocessed dataset CSV
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    df = pd.read_csv(input_path)
    print(f"[Preprocessing] Loaded raw dataset from '{input_path}' with shape {df.shape}.")

    # 1. Cleanse Noise: Drop zero-variance and identifier columns
    existing_drop_cols = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=existing_drop_cols)
    print(f"[Preprocessing] Dropped columns: {existing_drop_cols}. New shape: {df.shape}.")

    # 2. Encode Target: Map Attrition (Yes -> 1, No -> 0)
    if "Attrition" in df.columns:
        if df["Attrition"].dtype == object or isinstance(df["Attrition"].iloc[0], str):
            attrition_map = {"Yes": 1, "No": 0}
            df["Attrition"] = df["Attrition"].map(attrition_map)
            print("[Preprocessing] Encoded target column 'Attrition' (Yes=1, No=0).")

    # 3. Vectorize Features: LabelEncoder for remaining categorical string columns
    encoders: Dict[str, LabelEncoder] = {}
    categorical_cols = df.select_dtypes(include=["object", "category", "string", str]).columns.tolist()

    print(f"[Preprocessing] Categorical feature columns to encode: {categorical_cols}")

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # 4. Save fitted LabelEncoders for API inference pipeline
    os.makedirs(os.path.dirname(encoders_save_path), exist_ok=True)
    joblib.dump(encoders, encoders_save_path)
    print(f"[Preprocessing] Saved fitted LabelEncoders to '{encoders_save_path}'.")

    # 5. Output State: Export processed dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Preprocessing] Exported preprocessed dataset to '{output_path}'.")

    return df, encoders


if __name__ == "__main__":
    processed_df, _ = preprocess_data()
    print("=== Preprocessing Complete ===")
    print(processed_df.head())
