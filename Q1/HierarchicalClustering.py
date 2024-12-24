# 基于numpy实现层次聚类算法
import matplotlib.pyplot as plt
from typing import Literal
import tqdm
log_path = "logs"


import numpy as np

class HierarchicalClustering:
    def __init__(self, n_clusters=2, linkage: Literal['single', 'complete', 'average']='single', 
                 verbose=False):
        self.verbose = verbose
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X):
        n_samples = X.shape[0]
        self.labels_ = np.arange(n_samples)
        self.n_samples = n_samples
        # 输入数据的维度
        self.n_features = X.shape[1]
        # 构造距离矩阵
        print("初始化距离")
        self.distances = np.zeros((n_samples, n_samples))
        # for i in range(n_samples):
        #     for j in range(i+1, n_samples):
        #         self.distances[i, j] = np.sqrt(np.sum((X[i] - X[j]) ** 2))
        #         self.distances[j, i] = self.distances[i, j]

        # 使用numpy的广播机制，计算所有点之间的距离
        self.distances = np.sqrt(np.sum((X[:, np.newaxis] - X) ** 2, axis=2))
        
        # self.distances = np.triu(self.distances)
        self.distances[self.distances == 0] = np.inf
        self.clusters = [[i] for i in range(n_samples)]
        self.n_clusters_ = n_samples
        print("开始聚类")
        # while self.n_clusters_ > self.n_clusters:

        #     min_distance = np.min(self.distances)
        #     # 获取最小距离的下标i,j
        #     min_index = np.argmin(self.distances)

        #     i, j = min_index // self.n_clusters_, min_index % self.n_clusters_
        #     if self.verbose:
        #         print(self.clusters)
        #         print(self.distances)
        #         print(min_index)
        #         print(i, j)
        #         print("------------")
        #     # i,j合并为新的类，其中若i,j合并后的新的类距离其他类的坐标即为i,j类分别的距离按照linkage方式计算
        #     self.distances[i, :] = self.linkage_func(i, j)
        #     # 自环设置为无穷大
        #     self.distances[i, i] = np.inf
        #     self.distances[:, i] = self.distances[i, :]
        #     self.distances = np.delete(self.distances, j, axis=0)
        #     self.distances = np.delete(self.distances, j, axis=1)
        #     self.clusters[i] += self.clusters[j]
        #     self.clusters.pop(j)
        #     self.n_clusters_ -= 1
        # self.labels_ = np.zeros(n_samples)
        # for i, cluster in enumerate(self.clusters):
        #     for j in cluster:
        #         self.labels_[j] = i

        # 每次循环合并一个类，使用tqdm显示进度条
        for _ in tqdm.tqdm(range(n_samples - self.n_clusters)):
            min_distance = np.min(self.distances)
            min_index = np.argmin(self.distances)
            i, j = min_index // self.n_clusters_, min_index % self.n_clusters_
            self.distances[i, :] = self.linkage_func(i, j)
            self.distances[i, i] = np.inf
            self.distances[:, i] = self.distances[i, :]
            self.distances = np.delete(self.distances, j, axis=0)
            self.distances = np.delete(self.distances, j, axis=1)
            self.clusters[i] += self.clusters[j]
            self.clusters.pop(j)
            self.n_clusters_ -= 1

        self.labels_ = np.zeros(n_samples)
        for i, cluster in enumerate(self.clusters):
            for j in cluster:
                self.labels_[j] = i


    def linkage_func(self, i, j):
        if self.linkage == 'single':
            return np.minimum(self.distances[i, :], self.distances[j, :])
        elif self.linkage == 'complete':
            return np.maximum(self.distances[i, :], self.distances[j, :])
        elif self.linkage == 'average':
            return (self.distances[i, :] * len(self.clusters[i]) + self.distances[j, :] * len(self.clusters[j])) / (len(self.clusters[i]) + len(self.clusters[j]))
        
    @property
    def labels(self):
        return self.labels_
        
if __name__ == '__main__':
    logs = "Q1/logs"
    from read import read_data
    import os
    import time   
    data = read_data("Q1/8gau.txt")
    print(data.shape)
    # # 测试用，取前200个点
    # data = data[:200]
    linkages = ["single", "complete", "average"]
    n_clusters_m = 18
    n_clusters_start = 2

    for linkage in linkages:
        for n_clusters in range(n_clusters_start, n_clusters_m):
            timestamp = str(time.strftime("%Y%m%d%H%M%S", time.localtime()))
            test_name = timestamp + "_HC_" + str(n_clusters) + "_" + linkage
            print(test_name)
            hc = HierarchicalClustering(n_clusters=n_clusters, verbose=False,linkage=linkage)
            hc.fit(data)
            print(hc.labels_)
            title = "HC " + str(n_clusters) + " " + linkage
            plt.title(title)
            plt.scatter(data[:, 0], data[:, 1], c=hc.labels_)
            # 在各个类别的中心点上标注类别
            for i in range(n_clusters):
                plt.text(np.mean(data[hc.labels_ == i, 0]), np.mean(data[hc.labels_ == i, 1]), str(i), fontsize=12, color="red")

            plt.savefig(os.path.join(logs, test_name + ".png"))
            # 以svg格式保存图片，方便放大查看
            plt.savefig(os.path.join(logs, test_name + ".svg"))
            plt.clf()
            # plt.show()


    # n_clusters = 14
    # linkage = "single"
    # test_name = str(time()) + "_HC_" + str(n_clusters) + "_" + linkage
    # hc = HierarchicalClustering(n_clusters=14, verbose=False,linkage="single")
    # hc.fit(data)
    # print(hc.labels_)
    # plt.scatter(data[:, 0], data[:, 1], c=hc.labels_)
    # plt.savefig(os.path.join(logs, test_name + ".png"))
    # plt.clf()
    # # plt.show()


    