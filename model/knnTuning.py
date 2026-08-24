import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import warnings
warnings.filterwarnings('ignore') # Hides unnecessary terminal warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Ensure the graphs folder exists
os.makedirs('results/graphs', exist_ok=True)

# Loads processed data, runs GridSearchCV for optimal hyperparameters, and plots accuracy results.
def main():
    print("=" * 60 + "\nPARAMETER TUNING (KNN)\n" + "=" * 60)

    # 1. Load the preprocessed data
    with open('saved_model/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    X_train, y_train = data['X_train_scaled'], data['y_train']

    # 2. Define the exact parameter grid from your report
    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
        "p": [1, 2]
    }

    # 3. Initialize and run GridSearchCV (5-fold cross validation)
    print("Running GridSearchCV...")
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    # Output the winning parameters to the console
    print(f"\nBest Parameters: {grid.best_params_}")
    print(f"Best CV Accuracy: {grid.best_score_ * 100:.2f}%")

    # 4. Generate and save the performance line charts for each parameter
    cv_results = pd.DataFrame(grid.cv_results_)
    
    for param in ["n_neighbors", "weights", "metric", "p"]:
        # Calculate the mean test score for each parameter value
        grouped = cv_results.groupby(f"param_{param}")["mean_test_score"].mean().sort_index()
        
        # Plot the data
        plt.figure(figsize=(8, 5))
        plt.plot([str(v) for v in grouped.index], grouped.values * 100, marker="o", linewidth=2)
        plt.title(f"Mean CV Accuracy vs {param}")
        plt.ylabel("Mean CV Accuracy (%)")
        plt.grid(alpha=0.3)
        
        # Save the graph
        plt.savefig(f"results/graphs/knn_{param}.png")
        plt.close()
    
    print("Tuning graphs successfully saved to 'results/graphs/'")

if __name__ == "__main__":
    main()