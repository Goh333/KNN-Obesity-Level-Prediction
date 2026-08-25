import os
import time
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Define global paths for loading data and saving outputs
DATA_PATH = "csv/ObesityDataSet_raw_and_data_sinthetic.csv"
PKL_DIR = "pkl"

# Ensure the output directory exists
os.makedirs(PKL_DIR, exist_ok=True)

# Main orchestrator function for preprocessing
def preprocess_data():
    start_time = time.time()

    print("============================================================")
    print("DATASET LOADING")
    print("============================================================")

    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    print("\n============================================================")
    print("FEATURE ENCODING AND SEPARATION")
    print("============================================================")

    X = df.drop(columns=["NObeyesdad"])
    y = df["NObeyesdad"]

    # Identify column types to match the group's exact methodology
    binary_cols = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
    nominal_cols = ["CAEC", "CALC", "MTRANS"]
    numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

    categorical_cols = binary_cols + nominal_cols

    # Encode the target labels from text to integers
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    print(f"Target classes encoded: {list(target_encoder.classes_)}")

    # Create the preprocessing blueprint
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical_cols),
            ("num", StandardScaler(), numeric_cols) 
        ]
    )

    print("\n============================================================")
    print("DATA SPLITTING")
    print("============================================================")
    
    # Split data into 80% training and 20% testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    
    print(f"Training set samples: {len(X_train)}")
    print(f"Testing set samples:  {len(X_test)}")

    print("\n============================================================")
    print("FEATURE SCALING AND TRANSFORMATION")
    print("============================================================")
    
    # Apply the preprocessor only on training data to prevent data leakage
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    export_path = os.path.join(PKL_DIR, 'processed_data.pkl')
    
    # Save the processed data for the tuning script to use
    joblib.dump({
        'X_train_scaled': X_train_scaled, 
        'X_test_scaled': X_test_scaled,
        'y_train': y_train, 
        'y_test': y_test
    }, export_path)

    print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
    print(f"Data saved to: {export_path}")

# Entry point of the script
if __name__ == "__main__":
    preprocess_data()