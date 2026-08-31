import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

PROCESSED_DATA_PATH = "data/processed/hr_data_encoded.csv"
ENCODERS_PATH = "models/label_encoders.pkl"
MODEL_OUTPUT_PATH = "models/retain_rf_model.pkl"


def train_predictive_engine(
    data_path: str = PROCESSED_DATA_PATH,
    encoders_path: str = ENCODERS_PATH,
    model_output_path: str = MODEL_OUTPUT_PATH,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """
    Trains a Random Forest classifier optimized for Recall to identify employee flight risk:
    - Splits encoded data into 80/20 train/test sets
    - Trains Random Forest with balanced class weights
    - Evaluates Precision, Recall, F1-Score, ROC-AUC, and Feature Importances
    - Serializes trained model bundle to models/retain_rf_model.pkl
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at '{data_path}'. Please run preprocessing first.")

    df = pd.read_csv(data_path)
    print(f"[Training] Loaded processed dataset from '{data_path}' with shape {df.shape}.")

    if "Attrition" not in df.columns:
        raise ValueError("Target column 'Attrition' not found in dataset.")

    # 1. Data Splitting: Separate features and target
    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[Training] Data split: {X_train.shape[0]} train rows, {X_test.shape[0]} test rows (80/20 ratio).")

    # 2. Model Instantiation: Hyperparameters optimized for Recall (flight risk sensitivity)
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        class_weight="balanced",  # Optimizes sensitivity/recall for flight risks
        random_state=random_state,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    print("[Training] Random Forest Classifier trained successfully with class_weight='balanced'.")

    # 3. Evaluation Metrics
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 50)
    print("=== R.E.T.A.I.N. Model Evaluation Report ===")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=["Stayed (0)", "Attrited (1)"]))

    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Extract Feature Importances
    importances = rf_model.feature_importances_
    feature_importances_dict = dict(
        sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    )

    print("\nTop 10 Feature Importances:")
    for feature, importance in list(feature_importances_dict.items())[:10]:
        print(f"  - {feature:25s}: {importance:.4f}")

    # Load LabelEncoders if available
    encoders = {}
    if os.path.exists(encoders_path):
        encoders = joblib.load(encoders_path)
        print(f"[Training] Loaded label encoders from '{encoders_path}'.")

    # 4. Serialization: Model bundle containing classifier, feature metadata, encoders, importances
    model_bundle: Dict[str, Any] = {
        "model": rf_model,
        "feature_names": feature_names,
        "feature_importances": feature_importances_dict,
        "encoders": encoders
    }

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model_bundle, model_output_path)
    print(f"[Training] Model bundle successfully serialized to '{model_output_path}'.")

    return rf_model, feature_importances_dict


if __name__ == "__main__":
    train_predictive_engine()
