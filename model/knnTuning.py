import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Ensure the graphs directory exists
os.makedirs('results/graphs', exist_ok=True)

# Main orchestrator function for hyperparameter tuning
def main():
    print("============================================================")
    print("KNN HYPERPARAMETER TUNING")
    print("============================================================")

    # Load the preprocessed data
    try:
        data = joblib.load('pkl/processed_data.pkl')
        X_train, y_train = data['X_train_scaled'], data['y_train']
    except FileNotFoundError:
        print("Error: 'pkl/processed_data.pkl' not found. Run knnPreprocessing.py first.")
        return

    # Define the exact parameter grid from the report
    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
        "p": [1, 2]
    }
    
    print("Initializing GridSearchCV (5-fold CV)...")
    
    # Initialize and run GridSearchCV
    start_time = time.time()
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)
    tuning_time = time.time() - start_time

    print("\n============================================================")
    print("TUNING RESULTS")
    print("============================================================")
    
    print("Best Parameters:")
    for key, value in grid.best_params_.items():
        print(f"  - {key}: {value}")
        
    print(f"\nBest CV Accuracy: {grid.best_score_ * 100:.2f}%")
    print(f"Time Taken:       {tuning_time:.2f} seconds")
    
    print("\n============================================================")
    print("GENERATING GRAPHS")
    print("============================================================")
    
    # Generate and save the performance line charts for each parameter
    cv_results = pd.DataFrame(grid.cv_results_)
    
    for param in ["n_neighbors", "weights", "metric", "p"]:
        grouped = cv_results.groupby(f"param_{param}")["mean_test_score"].mean().sort_index()
        
        plt.figure(figsize=(8, 5))
        plt.plot([str(v) for v in grouped.index], grouped.values * 100, marker="o", linewidth=2, color="#1f77b4")
        plt.title(f"Mean CV Accuracy vs {param.capitalize()}")
        plt.ylabel("Mean CV Accuracy (%)")
        plt.xlabel(param.capitalize())
        plt.grid(alpha=0.3)
        
        save_path = f"results/graphs/knn_{param}_tuning.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Tuning graphs saved to 'results/graphs/'")

# Entry point of the script
if __name__ == "__main__":
    main()