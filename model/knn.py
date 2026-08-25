import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler, label_binarize
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

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# --------------------------------------------------------------------------
# BEST HYPERPARAMETERS (found previously via tuning)
# --------------------------------------------------------------------------
BEST_PARAMS = {
    "n_neighbors": 5,
    "weights": "distance",
    "metric": "manhattan",
    "p": 1,
}

# --------------------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------------------
def load_data():
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=544)
        X = dataset.data.features
        y = dataset.data.targets.squeeze()
        return X, y
    except Exception as e:
        print(f"ucimlrepo fetch failed ({e}); falling back to local CSV.")
        df = pd.read_csv("csv/ObesityDataSet_raw_and_data_sinthetic.csv")
        y = df["NObeyesdad"]
        X = df.drop(columns=["NObeyesdad"])
        return X, y

def main():
    print("============================================================")
    print("FINAL KNN MODEL EVALUATION")
    print("============================================================")

    X, y = load_data()

    # --------------------------------------------------------------------------
    # 2. PREPROCESSING
    # --------------------------------------------------------------------------
    binary_cols = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
    ordinal_cols = ["CAEC", "CALC"]                    
    nominal_cols = ["MTRANS"]                          
    numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

    binary_cols = [c for c in binary_cols if c in X.columns]
    ordinal_cols = [c for c in ordinal_cols if c in X.columns]
    nominal_cols = [c for c in nominal_cols if c in X.columns]
    numeric_cols = [c for c in numeric_cols if c in X.columns]

    binary_categories = [["Female", "Male"], ["no", "yes"], ["no", "yes"], ["no", "yes"], ["no", "yes"]]
    binary_categories = binary_categories[: len(binary_cols)]
    ordinal_categories = [["no", "Sometimes", "Frequently", "Always"]] * len(ordinal_cols)

    class_order = [
        "Insufficient_Weight", "Normal_Weight", "Overweight_Level_I",
        "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III",
    ]

    target_encoder = LabelEncoder()
    target_encoder.classes_ = np.array(class_order)
    y_encoded = target_encoder.transform(y)

    preprocessor = ColumnTransformer(
        transformers=[
            ("bin", OrdinalEncoder(categories=binary_categories), binary_cols),
            ("ord", OrdinalEncoder(categories=ordinal_categories), ordinal_cols),
            ("nom", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), nominal_cols),
            ("num", StandardScaler(), numeric_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
    )

    # --------------------------------------------------------------------------
    # 3. FIT KNN WITH BEST-KNOWN HYPERPARAMETERS
    # --------------------------------------------------------------------------
    best_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", KNeighborsClassifier(n_jobs=-1, **BEST_PARAMS)),
        ]
    )

    print("Fitting KNN with best-known hyperparameters...")
    train_start = time.time()
    best_model.fit(X_train, y_train)
    training_time = time.time() - train_start

    # --------------------------------------------------------------------------
    # 4. EVALUATION ON TEST SET
    # --------------------------------------------------------------------------
    run_start = time.time()
    final_preds = best_model.predict(X_test)
    final_probs = best_model.predict_proba(X_test)
    run_time = time.time() - run_start

    print("\n============================================================")
    print("PERFORMANCE METRICS")
    print("============================================================")

    final_acc = accuracy_score(y_test, final_preds)
    print(f"Test accuracy: {final_acc:.4f}")
    print(f"Model training time:  {training_time:.5f} seconds")
    print(f"Model inference time: {run_time:.5f} seconds")

    print("\nClassification report:\n")
    print(classification_report(y_test, final_preds, target_names=target_encoder.classes_, digits=4))

    # ROC-AUC calculations
    y_test_bin = label_binarize(y_test, classes=np.unique(y_train))
    macro_roc_auc = roc_auc_score(y_test_bin, final_probs, multi_class="ovr", average="macro")
    weighted_roc_auc = roc_auc_score(y_test_bin, final_probs, multi_class="ovr", average="weighted")
    per_class_roc_auc = roc_auc_score(y_test_bin, final_probs, multi_class="ovr", average=None)

    print(f"\nMacro-average ROC-AUC (OvR): {macro_roc_auc:.4f}")
    print(f"Weighted-average ROC-AUC (OvR): {weighted_roc_auc:.4f}")

    roc_auc_df = pd.DataFrame({
        "Class": target_encoder.classes_,
        "ROC-AUC": per_class_roc_auc,
    }).sort_values("ROC-AUC", ascending=False)

    print("\nPer-class ROC-AUC (OvR):")
    print(roc_auc_df.to_string(index=False))

    precision = precision_score(y_test, final_preds, average="weighted")
    recall = recall_score(y_test, final_preds, average="weighted")
    f1 = f1_score(y_test, final_preds, average="weighted")

    print("\nSummary metrics:")
    print(f"Accuracy:  {final_acc * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {weighted_roc_auc * 100:.2f}%")

    # --------------------------------------------------------------------------
    # 5. GENERATE GRAPHS AND SAVE MODEL
    # --------------------------------------------------------------------------
    os.makedirs("results/graphs", exist_ok=True)
    os.makedirs("pkl", exist_ok=True)
    
    print("\n============================================================")
    print("GENERATING VISUALIZATIONS & EXPORTING MODEL")
    print("============================================================")

    # Confusion matrix
    cm = confusion_matrix(y_test, final_preds)
    fig, ax = plt.subplots(figsize=(9, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_encoder.classes_)
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix — KNN (Best Params)")
    plt.tight_layout()
    plt.savefig("results/graphs/knn_confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved results/graphs/knn_confusion_matrix.png")

    # Learning Curve
    print("Generating Learning Curve...")
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=best_model,
        X=X_train,
        y=y_train,
        cv=5,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, marker='o', label="Training Accuracy")
    plt.plot(train_sizes, val_mean, marker='s', label="Validation Accuracy")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
    plt.xlabel("Training Samples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve - KNN")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/graphs/knn_learning_curve.png", dpi=150)
    plt.close()
    print("Saved results/graphs/knn_learning_curve.png")

    # Export Model
    joblib.dump(best_model, "pkl/knn_model.pkl")
    joblib.dump(target_encoder, "pkl/knn_target_encoder.pkl")
    print("\nSaved trained pipeline to pkl/knn_model.pkl")
    print("Saved target label encoder to pkl/knn_target_encoder.pkl")

if __name__ == "__main__":
    main()