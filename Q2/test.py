import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, SVDpp, KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline
from surprise.model_selection import cross_validate, train_test_split

# path to dataset file
file_path = "ml-latest-small/ratings.csv"

# As we're loading a custom dataset, we need to define a reader. In the
# movielens-100k dataset, each line has the following format:
# 'user item rating timestamp', separated by '\t' characters.
reader = Reader(line_format="user item rating timestamp", sep=",", skip_lines=1)

data = Dataset.load_from_file(file_path, reader=reader)
# train_set, test_set = train_test_split(data, test_size=0.25)

algo1 = SVD()
algo2 = SVDpp()
algo3 = KNNBasic()
algo4 = KNNWithMeans()
algo5 = KNNWithZScore()
algo6 = KNNBaseline()

# We can now use this dataset as we please, e.g. calling cross_validate
result1 = cross_validate(algo1, data, measures=['RMSE', 'MAE'], cv=10, verbose=True)

# 创建横向表格
result1_dict = {
    "Fold": [f"Fold {i+1}" for i in range(len(result1['test_rmse']))] + ["Mean", "Std"],
    "RMSE": list(result1['test_rmse']) + [np.mean(result1['test_rmse']), np.std(result1['test_rmse'])],
    "MAE": list(result1['test_mae']) + [np.mean(result1['test_mae']), np.std(result1['test_mae'])],
    "Fit time": list(result1['fit_time']) + [np.mean(result1['fit_time']), np.std(result1['fit_time'])],
    "Test time": list(result1['test_time']) + [np.mean(result1['test_time']), np.std(result1['test_time'])],
}

# 转置表格
result1s_df = pd.DataFrame(result1_dict).set_index("Fold").T
result1 = pd.DataFrame(result1s_df)
result1.to_excel("./result/SVD.xlsx")
print("结果已保存为SVD.xlsx")

