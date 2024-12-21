import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from surprise import Dataset, Reader, SVD, SVDpp, KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline
from surprise.model_selection import cross_validate, train_test_split

# path to dataset file
file_path = "ml-latest-small/ratings.csv"

reader = Reader(line_format="user item rating timestamp", sep=",", skip_lines=1)

data = Dataset.load_from_file(file_path, reader=reader)

# algorithms = {
#     "SVD": SVD(),
#     # "SVDpp": SVDpp(),
#     "KNNBasic": KNNBasic(),
#     "KNNWithMeans": KNNWithMeans(),
#     "KNNWithZScore": KNNWithZScore(),
#     "KNNBaseline": KNNBaseline(),
# }

sim_options = {
    "name": "cosine",  # Use cosine similarity (can be changed to "pearson" or others)
    "user_based": True,  # Set to False for item-based collaborative filtering
}

algorithms = {
    "KNNBasic": KNNBasic(),
    "KNNWithMeans": KNNWithMeans(),
    "KNNWithZScore": KNNWithZScore(),
    "KNNBaseline": KNNBaseline(),
}

# Create a dictionary to store results
all_results = {}

# Test each algorithm and save results
for algo_name, algo in algorithms.items():
    print(f"Testing algorithm: {algo_name}")
    # Perform cross-validation
    result = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=10, verbose=True)

    # Prepare data for saving
    result_dict = {
        "Fold": [f"Fold {i + 1}" for i in range(len(result['test_rmse']))] + ["Mean", "Std"],
        "RMSE": list(result["test_rmse"]) + [np.mean(result["test_rmse"]), np.std(result["test_rmse"])],
        "MAE": list(result["test_mae"]) + [np.mean(result["test_mae"]), np.std(result["test_mae"])],
        "Fit time": list(result["fit_time"]) + [np.mean(result["fit_time"]), np.std(result["fit_time"])],
        "Test time": list(result["test_time"]) + [np.mean(result["test_time"]), np.std(result["test_time"])],
    }

    # Convert to DataFrame
    result_df = pd.DataFrame(result_dict).set_index("Fold").T
    all_results[algo_name] = result_df

# # Save all results to an Excel file with multiple sheets
# output_path = "./result/all_results.xlsx"
# with pd.ExcelWriter(output_path) as writer:
#     for algo_name, result_df in all_results.items():
#         result_df.to_excel(writer, sheet_name=algo_name)
# print(f"Results saved to {output_path}")



# Prepare data for visualization
rmse_data = {}
mae_data = {}

for algo_name, result_df in all_results.items():
    rmse_data[algo_name] = result_df.iloc[0, :-2]  # Exclude Mean and Std
    mae_data[algo_name] = result_df.iloc[1, :-2]   # Exclude Mean and Std

# Convert data to DataFrame for plotting
rmse_df = pd.DataFrame(rmse_data)
mae_df = pd.DataFrame(mae_data)

# Plot RMSE results
plt.figure(figsize=(12, 6))
rmse_df.plot(kind="line", marker="o", ax=plt.gca())
plt.title("RMSE across folds for each algorithm", fontsize=14)
plt.xlabel("Folds", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
plt.legend(title="Algorithm", fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot MAE results
plt.figure(figsize=(12, 6))
mae_df.plot(kind="line", marker="o", ax=plt.gca())
plt.title("MAE across folds for each algorithm", fontsize=14)
plt.xlabel("Folds", fontsize=12)
plt.ylabel("MAE", fontsize=12)
plt.legend(title="Algorithm", fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()

