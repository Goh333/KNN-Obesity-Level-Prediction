import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
import time
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

# --------------------------------------------------------------------------
# BEST HYPERPARAMETERS (Obtained from tuning results)
# --------------------------------------------------------------------------
BEST_PARAMS = {
    "n_neighbors": 5,
    "weights": "distance",
    "metric": "manhattan",
    "p": 1
}

# --------------------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------------------
def load_data():
    data_path = "csv/ObesityDataSet_raw_and_data_sinthetic.csv" 
    df = pd.read_csv(data_path)
    y = df["NObeyesdad"]
    X = df.drop(columns=["NObeyesdad"])
    return X, y

# Main orchestrator function for final model training and evaluation
def main():
    print("============================================================")
    print("FINAL KNN MODEL EVALUATION")
    print("============================================================")

    # 1. LOAD DATA
    X, y = load_data()

    # 2. STRICT PREPROCESSING
    binary_cols = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
    nominal_cols = ["CAEC", "CALC", "MTRANS"]          
    numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    categorical_cols = binary_cols + nominal_cols

    # Encode target labels (7 obesity classes -> integers)
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    # ColumnTransformer: one-hot encode categoricals, standardscaler for numeric
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ]
    )

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
    )

    # 3. FIT KNN PIPELINE WITH BEST HYPERPARAMETERS
    best_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                KNeighborsClassifier(
                    n_jobs=-1,
                    **BEST_PARAMS
                ),
            ),
        ]
    )

    print("Fitting final KNN Pipeline model...")
    best_model.fit(X_train, y_train)

    # Record Training Time
    train_start = time.time()
    best_model.fit(X_train, y_train)
    training_time = time.time() - train_start

    # 4. EVALUATION
    run_start = time.time()
    final_preds = best_model.predict(X_test)
    final_probs = best_model.predict_proba(X_test)
    run_time = time.time() - run_start
    
    # Binarize labels for calculating multiclass ROC-AUC
    y_test_bin = label_binarize(y_test, classes=np.unique(y_train))

    print(f"Model Training Time:  {training_time:.5f} seconds")
    print(f"Model Inference Time: {run_time:.5f} seconds")

    print("\n============================================================")
    print("PERFORMANCE METRICS")
    print("============================================================")
    
    # Calculate and print the weighted average metrics for the report table
    print(f"Accuracy:  {accuracy_score(y_test, final_preds) * 100:.2f}%")
    print(f"Precision: {precision_score(y_test, final_preds, average='weighted') * 100:.2f}%")
    print(f"Recall:    {recall_score(y_test, final_preds, average='weighted') * 100:.2f}%")
    print(f"F1-Score:  {f1_score(y_test, final_preds, average='weighted') * 100:.2f}%")
    print(f"ROC-AUC:   {roc_auc_score(y_test_bin, final_probs, multi_class='ovr', average='weighted') * 100:.2f}%")

    print("\nClassification report:\n")
    print(classification_report(y_test, final_preds, target_names=target_encoder.classes_, digits=4))

    # Ensure output directories exist
    os.makedirs("results/graphs", exist_ok=True)
    os.makedirs("pkl", exist_ok=True)

    # Generate Confusion Matrix
    cm = confusion_matrix(y_test, final_preds)
    fig, ax = plt.subplots(figsize=(9, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_encoder.classes_)
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix — KNN (Best Params)")
    plt.tight_layout()
    plt.savefig("results/graphs/knn_confusion_matrix.png", dpi=150)
    plt.close()
    
    print("\n============================================================")
    print("EXPORTING FILES")
    print("============================================================")
    print("Saved results/graphs/knn_confusion_matrix.png")

    # 5. SAVE THE FINAL MODEL FOR GUI
    joblib.dump(best_model, "pkl/knn_model.pkl")
    joblib.dump(target_encoder, "pkl/knn_target_encoder.pkl")
    print("Saved trained pipeline to pkl/knn_model.pkl")
    print("Saved target label encoder to pkl/knn_target_encoder.pkl")

# Entry point of the script
if __name__ == "__main__":
    main()