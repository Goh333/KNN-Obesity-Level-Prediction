import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, classification_report, confusion_matrix)
from sklearn.preprocessing import label_binarize

# Builds final model, tests on 20% unseen data, calculates rubric metrics, and draws confusion matrix.
def main():
    print("=" * 60 + "\nMODEL EVALUATION & EXPORT (KNN)\n" + "=" * 60)

    # 1. Load the preprocessed data
    with open('saved_model/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
        
    X_train, X_test = data['X_train_scaled'], data['X_test_scaled']
    y_train, y_test = data['y_train'], data['y_test']
    class_names = data['target_encoder'].classes_

    # 2. Train the model using the optimal parameters (K=3, Manhattan distance, distance weights)
    knn = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='manhattan', p=1)
    knn.fit(X_train, y_train)

    # 3. Generate predictions on the unseen test data
    y_pred = knn.predict(X_test)
    y_proba = knn.predict_proba(X_test)
    
    # Binarize labels for calculating multiclass ROC-AUC
    y_test_bin = label_binarize(y_test, classes=np.unique(y_train))

    # 4. Calculate and print the weighted average metrics for the report table
    print(f"Accuracy:  {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted') * 100:.2f}%")
    print(f"Recall:    {recall_score(y_test, y_pred, average='weighted') * 100:.2f}%")
    print(f"F1-Score:  {f1_score(y_test, y_pred, average='weighted') * 100:.2f}%")
    print(f"ROC-AUC:   {roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='weighted') * 100:.2f}%")

    # 5. Print a detailed breakdown per class (helps explain model limitations)
    print("\nPer-Class Classification Report:\n" + "-" * 50)
    print(classification_report(y_test, y_pred, target_names=class_names))

    # 6. Generate and save the Confusion Matrix heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix - KNN")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("results/graphs/KNN_Confusion_Matrix.png")
    
    # 7. Export the trained model for future use
    with open('saved_model/knn_model.pkl', 'wb') as f:
        pickle.dump(knn, f)
    print("\nConfusion matrix and model successfully saved!")

if __name__ == "__main__":
    main()