# 数据挖掘作业

## Q1 层次聚类与GMM聚类实现

[层次聚类](Q1/HierarchicalClustering.py)与[GMM聚类](Q1/GMM.py)的实现，读取数据方法实现[read_data](Q1/read.py)。

关键参数：
HerarchicalClustering:
    - n_clusters：int 聚类数
    - linkage：str 聚类方式，可选'single', 'complete', 'average'

GMM:
    - n_clusters：int 聚类数
    - init：str 参数初始化方式，可选'random', 'kmeans'

使用方法

```python
import HierarchicalClustering
# 或 import GMM

from read import read_data

data = read_data('Q1/8gau.txt')

# 调用聚类方法
model = HierarchicalClustering(n_clusters=8, linkage='single')
model.fit(data)

labels = model.labels
```

## Q2 协同过滤实验对比

[算法对比](Q2/all_algo.py)基于SVD、KNNBasic、KNNWithMeans、KNNWithZScore四种协同过滤算法，使用cosine、MSD、Pearson三种相似度度量方式，user_based参数为True或False，通过10折交叉验证对MovieLens数据集进行评估。
实验方法：执行[算法对比](Q2/all_algo.py)中的main函数，输出RMSE、MAE结果并绘制折线图。存储至[图像输出](Q2/figures/)文件夹。
```python
# 构造sim_options
sim_options = {
    'name': 'cosine',
    'user_based': False
}

# 调用算法对比方法
from all_algo import algorithms_cross_validate, plot_rmse_mae
plot_rmse_mae(algorithms_cross_validate(algorithms, data, sim_option), sim_option)
```


