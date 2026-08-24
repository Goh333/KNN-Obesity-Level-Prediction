import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

# Define global paths for loading data and saving outputs
DATA_PATH = "data/ObesityDataSet_raw_and_data_sinthetic.csv"
SAVED_MODEL_PATH = "saved_model"

# Ensure the output directory exists before we try to save files there
os.makedirs(SAVED_MODEL_PATH, exist_ok=True)

# Reads the dataset from the CSV file and prints its initial shape.
def load_data():
    df = pd.read_csv(DATA_PATH)
    print("=" * 60 + "\nDATASET LOADING\n" + "=" * 60)
    print(f"Dataset shape: {df.shape}")
    return df

# Displays basic health metrics of the dataset, such as missing values and duplicate rows.
def explore_data(df):
    print("\n" + "=" * 60 + "\nDATA EXPLORATION\n" + "=" * 60)
    print(f"Missing values:\n{df.isnull().sum()}\n")
    print(f"Duplicate rows: {df.duplicated().sum()}")

# Removes exact duplicate rows from the dataset to prevent data leakage during training.
def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print("\n" + "=" * 60 + "\nDUPLICATE REMOVAL\n" + "=" * 60)
    print(f"Duplicates removed: {before - len(df)}")
    return df

# Separates features (X) and target (y), encodes the target, and sets up scaling for KNN.
def prepare_data(df):
    X = df.drop("NObeyesdad", axis=1)
    
    # Encode the target labels from text to integers
    target_encoder = LabelEncoder()
    y = pd.Series(target_encoder.fit_transform(df["NObeyesdad"]), name="NObeyesdad")

    # Automatically identify text columns and number columns
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()
    num_features = X.select_dtypes(exclude=["object"]).columns.tolist()

    # Create the preprocessing blueprint (StandardScaler is strictly required for KNN)
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat_features),
            ("num", StandardScaler(), num_features) 
        ]
    )
    return X, y, preprocessor, target_encoder

# Splits the data into 80% training and 20% testing sets while preserving class proportions.
def split_data(X, y):
    return train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Main orchestrator function: runs all steps, scales data, and saves to a .pkl file.
def preprocess_data():
    df = load_data()
    explore_data(df)
    df = remove_duplicates(df)
    
    X, y, preprocessor, target_encoder = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Apply the preprocessor (Fit ONLY on training data to prevent data leakage)
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # Bundle all the processed objects together and save them
    export_path = os.path.join(SAVED_MODEL_PATH, 'processed_data.pkl')
    with open(export_path, 'wb') as f:
        pickle.dump({
            'X_train_scaled': X_train_scaled, 
            'X_test_scaled': X_test_scaled,
            'y_train': y_train, 
            'y_test': y_test,
            'target_encoder': target_encoder
        }, f)
        
    print(f"\nSuccess! Cleaned and scaled data saved to '{export_path}'")

if __name__ == "__main__":
    preprocess_data()