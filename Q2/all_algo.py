import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from surprise import Dataset, Reader, SVD, SVDpp, KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline
from surprise.model_selection import cross_validate, train_test_split

# path to dataset file
file_path = "Q2/ml-latest-small/ratings.csv"

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
    "SVD": SVD,
    "KNNBasic": KNNBasic,
    # "KNNWithMeans": KNNWithMeans,
    # "KNNWithZScore": KNNWithZScore,
    # "KNNBaseline": KNNBaseline,
}

# # Create a dictionary to store results
# all_results = {}

# # Test each algorithm and save results
# for algo_name, algo in algorithms.items():
#     print(f"Testing algorithm: {algo_name}")
#     # Perform cross-validation
#     result = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=10, verbose=True)

#     # Prepare data for saving
#     result_dict = {
#         "Fold": [f"Fold {i + 1}" for i in range(len(result['test_rmse']))] + ["Mean", "Std"],
#         "RMSE": list(result["test_rmse"]) + [np.mean(result["test_rmse"]), np.std(result["test_rmse"])],
#         "MAE": list(result["test_mae"]) + [np.mean(result["test_mae"]), np.std(result["test_mae"])],
#         "Fit time": list(result["fit_time"]) + [np.mean(result["fit_time"]), np.std(result["fit_time"])],
#         "Test time": list(result["test_time"]) + [np.mean(result["test_time"]), np.std(result["test_time"])],
#     }

#     # Convert to DataFrame
#     result_df = pd.DataFrame(result_dict).set_index("Fold").T
#     all_results[algo_name] = result_df

def algorithms_cross_validate(algorithms, data, sim_option):
    all_results = {}
    for algo_name, algo in algorithms.items():
        # print(type(algo))
        # exit()
        if algo_name == "SVD":
            algo_instance = algo()
        else:
            algo_instance = algo(sim_options=sim_option)

        print(f"测试算法: {algo_name}")
        # Perform cross-validation
        result = cross_validate(algo_instance, data, measures=["RMSE", "MAE"], cv=10, verbose=True)

        # Prepare data for saving
        result_dict = {
            "Fold": [f"Fold {i + 1}" for i in range(len(result['test_rmse']))] + ["Mean", "Std"],
            # "RMSE": list(result["test_rmse"]) + [np.mean(result["test_rmse"]), np.std(result["test_rmse"])],
            # "MAE": list(result["test_mae"]) + [np.mean(result["test_mae"]), np.std(result["test_mae"])],
            # "Fit time": list(result["fit_time"]) + [np.mean(result["fit_time"]), np.std(result["fit_time"])],
            # "Test time": list(result["test_time"]) + [np.mean(result["test_time"]), np.std(result["test_time"])],

            "RMSE": np.append(result["test_rmse"], [np.mean(result["test_rmse"]), np.std(result["test_rmse"])]),
            "MAE": np.append(result["test_mae"], [np.mean(result["test_mae"]), np.std(result["test_mae"])]),
            "Fit time": np.append(result["fit_time"], [np.mean(result["fit_time"]), np.std(result["fit_time"])]),
            "Test time": list(np.append(result["test_time"], [np.mean(result["test_time"]), np.std(result["test_time"])])),
        }

    
        all_results[algo_name] = result_dict
    # print(all_results)
    return all_results




def plot_rmse_mae(all_results, sim_option):
    # exit()
    option1 = sim_option["name"]
    import os
    import time
    root_path = "Q2/figures"
    time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    file_path = os.path.join(root_path, f"{option1}_{time_str}")
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # Prepare data for visualization
    rmse_data = {}
    mae_data = {}

    for algo_name, result_dict in all_results.items():

        rmse_data[algo_name] = result_dict["RMSE"][:-2] # 排除Std和Mean
        mae_data[algo_name] = result_dict["MAE"][:-2]

        # 将算法总结果存入文件
        result_df = pd.DataFrame(result_dict).set_index("Fold").T
        # result_df = pd.DataFrame()
        # result_df.loc["RMSE"] = result_dict["RMSE"]
        # result_df.loc["MAE"] = result_dict["MAE"]
        # result_df.loc["Fit time"] = result_dict["Fit time"]
        # result_df.loc["Test time"] = result_dict["Test time"]


  
        result_df.to_excel(os.path.join(file_path, f"{algo_name}.xlsx"))


    # Convert data to DataFrame for plotting
    rmse_df = pd.DataFrame(rmse_data)
    mae_df = pd.DataFrame(mae_data)

    # Plot RMSE results
    plt.figure(figsize=(12, 6))
    rmse_df.plot(kind="line", marker="o", ax=plt.gca())
    plt.title(f"RMSE 各算法交叉验证结果({option1})", fontsize=14)
    plt.xlabel("Folds", fontsize=12)
    plt.ylabel("RMSE", fontsize=12)
    plt.legend(title="Algorithm", fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # 保存到png文件
    plt.savefig(os.path.join(file_path, f"RMSE_{option1}.png"))
    # 保存到SVG文件
    plt.savefig(os.path.join(file_path, f"RMSE_{option1}.svg"))

    # Plot MAE results
    plt.figure(figsize=(12, 6))
    mae_df.plot(kind="line", marker="o", ax=plt.gca())
    plt.title(f"MAE 各算法交叉验证结果({option1})", fontsize=14)

    plt.xlabel("Folds", fontsize=12)
    plt.ylabel("MAE", fontsize=12)
    plt.legend(title="Algorithm", fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # 保存到png文件
    plt.savefig(os.path.join(file_path, f"MAE_{option1}.png"))
    # 保存到SVG文件
    plt.savefig(os.path.join(file_path, f"MAE_{option1}.svg"))

if __name__ == "__main__":
    all_results = algorithms_cross_validate(algorithms, data, sim_options)
    plot_rmse_mae(all_results, sim_options)