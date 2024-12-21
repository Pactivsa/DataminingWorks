# 基于numpy实现层次聚类算法
import matplotlib.pyplot as plt
import os
log_path = "logs"

# 设置matplotlib为无头模式
os.environ['MPLCONFIGDIR'] = log_path
plt.switch_backend('agg')
import math


import numpy as np

class HierarchicalClustering:
    def __init__(self, n_clusters=2, linkage='single'):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X):
        n_samples = X.shape[0]
        self.labels_ = np.arange(n_samples)
        self.n_samples = n_samples
        self.n_features = X.shape[1]
        self.distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                self.distances[i, j] = np.sqrt(np.sum((X[i] - X[j]) ** 2))
                self.distances[j, i] = self.distances[i, j]
        self.distances = np.triu(self.distances)
        self.distances[self.distances == 0] = np.inf
        self.clusters = [[i] for i in range(n_samples)]
        self.n_clusters_ = n_samples
        while self.n_clusters_ > self.n_clusters:
            min_distance = np.min(self.distances)
            min_index = np.argmin(self.distances)
            i, j = min_index // n_samples, min_index % n_samples
            self.distances[i, :] = self.linkage_func(i, j)
            self.distances[:, i] = self.distances[i, :]
            self.distances = np.delete(self.distances, j, axis=0)
            self.distances = np.delete(self.distances, j, axis=1)
            self.clusters[i] += self.clusters[j]
            self.clusters.pop(j)
            self.n_clusters_ -= 1
            for k in range(self.n_samples):
                if k == i:
                    continue
                self.distances[i, k] = np.sqrt(np.sum((X[i] - X[k]) ** 2))
                self.distances[k, i] = self.distances[i, k]
        self.labels_ = np.zeros(n_samples)
        for i, cluster in enumerate(self.clusters):
            for j in cluster:
                self.labels_[j] = i

    def linkage_func(self, i, j):
        if self.linkage == 'single':
            return np.minimum(self.distances[i, :], self.distances[j, :])
        elif self.linkage == 'complete':
            return np.maximum(self.distances[i, :], self.distances[j, :])