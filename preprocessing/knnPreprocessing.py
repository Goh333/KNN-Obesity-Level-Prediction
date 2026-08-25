import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
DATA_PATH = "csv/ObesityDataSet_raw_and_data_sinthetic.csv"
PKL_DIR = "pkl"

os.makedirs(PKL_DIR, exist_ok=True)

def load_data():
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=544)
        X = dataset.data.features
        y = dataset.data.targets.squeeze()
        return X, y
    except Exception as e:
        print(f"ucimlrepo fetch failed ({e}); falling back to local CSV.")
        df = pd.read_csv(DATA_PATH)
        y = df["NObeyesdad"]
        X = df.drop(columns=["NObeyesdad"])
        return X, y

def preprocess_data():
    start_time = time.time()

    print("============================================================")
    print("DATASET LOADING")
    print("============================================================")

    X, y = load_data()
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")

    print("\n============================================================")
    print("FEATURE ENCODING AND SEPARATION")
    print("============================================================")

    # Identify column types
    binary_cols = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
    ordinal_cols = ["CAEC", "CALC"]
    nominal_cols = ["MTRANS"]
    numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

    # Keep only columns that actually exist
    binary_cols = [c for c in binary_cols if c in X.columns]
    ordinal_cols = [c for c in ordinal_cols if c in X.columns]
    nominal_cols = [c for c in nominal_cols if c in X.columns]
    numeric_cols = [c for c in numeric_cols if c in X.columns]

    # Explicit category orders for OrdinalEncoder
    binary_categories = [["Female", "Male"], ["no", "yes"], ["no", "yes"], ["no", "yes"], ["no", "yes"]]
    binary_categories = binary_categories[: len(binary_cols)]
    ordinal_categories = [["no", "Sometimes", "Frequently", "Always"]] * len(ordinal_cols)

    # Encode target labels in their natural CLINICAL order
    class_order = [
        "Insufficient_Weight", "Normal_Weight", "Overweight_Level_I",
        "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III",
    ]

    target_encoder = LabelEncoder()
    target_encoder.classes_ = np.array(class_order)
    y_encoded = target_encoder.transform(y)
    print(f"\nClasses (in ordinal order): {list(target_encoder.classes_)}")

    # ColumnTransformer with new OrdinalEncoder logic
    preprocessor = ColumnTransformer(
        transformers=[
            ("bin", OrdinalEncoder(categories=binary_categories), binary_cols),
            ("ord", OrdinalEncoder(categories=ordinal_categories), ordinal_cols),
            ("nom", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), nominal_cols),
            ("num", StandardScaler(), numeric_cols),
        ]
    )

    print("\n============================================================")
    print("DATA SPLITTING")
    print("============================================================")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=RANDOM_STATE, stratify=y_encoded
    )
    
    print(f"Training set samples: {len(X_train)}")
    print(f"Testing set samples:  {len(X_test)}")

    print("\n============================================================")
    print("FEATURE SCALING AND TRANSFORMATION")
    print("============================================================")
    
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    export_path = os.path.join(PKL_DIR, 'processed_data.pkl')
    joblib.dump({
        'X_train_scaled': X_train_scaled, 
        'X_test_scaled': X_test_scaled,
        'y_train': y_train, 
        'y_test': y_test
    }, export_path)

    print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
    print(f"Data saved to: {export_path}")

if __name__ == "__main__":
    preprocess_data()