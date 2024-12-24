# 基于python实现高斯混合模型Gaussion Mixture Model

import numpy as np
from typing import Literal
import tqdm
class KMeans:
    def __init__(self, n_clusters=2, max_iter=100, tol=1e-3):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n_samples = X.shape[0]
        self.n_samples = n_samples
        self.n_features = X.shape[1]
        self.labels_ = np.zeros(n_samples)
        # 随机初始化质心
        self.centers = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        for _ in range(self.max_iter):
            for i in range(n_samples):
                self.labels_[i] = np.argmin(np.sum((X[i] - self.centers) ** 2, axis=1))
            new_centers = np.array([np.mean(X[self.labels_ == i], axis=0) for i in range(self.n_clusters)])
            if np.sum((new_centers - self.centers) ** 2) < self.tol:
                break
            self.centers = new_centers

class GMM:
    def __init__(self, n_clusters=2, max_iter=100, tol=1e-3,init_params:Literal['kmeans', 'random']='kmeans'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init_params = init_params

    def fit(self, X,task_name=""):
        n_samples = X.shape[0]
        self.n_samples = n_samples
        self.n_features = X.shape[1]
        self.labels_ = np.zeros(n_samples)

        # self.means = np.random.rand(self.n_clusters, self.n_features)
        if self.init_params == 'random':
            # self.weights = np.ones(self.n_clusters) / self.n_clusters
            # self.means = np.random.rand(self.n_clusters, self.n_features)
            # self.covariances = np.array([np.eye(self.n_features) for _ in range(self.n_clusters)])
            # 随机初始化类
            print("随机初始化")
            self.labels_ = np.random.randint(0, self.n_clusters, n_samples)

        elif self.init_params == 'kmeans':
            print("kmeans初始化")
            kmeans = KMeans(n_clusters=self.n_clusters)
            kmeans.fit(X)
            self.labels_ = kmeans.labels_
            print("kmeans初始化完成")
        plt.scatter(X[:, 0], X[:, 1], c=self.labels_)
        # plt.savefig("Q1/logs/"+task_name+"_init.png")
        plt.savefig("Q1/logs/"+task_name+"_init.svg")
        plt.clf()

        self.weights = np.array([np.sum(self.labels_ == i) for i in range(self.n_clusters)]) / self.n_samples
        self.means = np.array([np.mean(X[self.labels_ == i], axis=0) for i in range(self.n_clusters)])
        self.covariances = np.array([np.cov(X[self.labels_ == i].T) for i in range(self.n_clusters)])

        # for _ in range(self.max_iter):
        #     self.expectation(X)
        #     self.maximization(X)
        print("开始迭代")
        for _ in tqdm.tqdm(range(self.max_iter)):
            self.expectation(X)
            self.maximization(X)
            # if np.sum((new_means - self.means) ** 2) < self.tol:
            #     break
            # self.means = new_means
    

    def gaussian(self, x, mean, covariance):
        try:
            result = np.exp(-0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(covariance)), x - mean)) / np.sqrt(np.linalg.det(covariance) * (2 * np.pi) ** self.n_features)
        except Exception as e:
            print(covariance)
            raise e
        return result        
        # return np.exp(-0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(covariance)), x - mean)) / np.sqrt(np.linalg.det(covariance) * (2 * np.pi) ** self.n_features)

    def expectation(self, X):
        self.responsibilities = np.zeros((self.n_samples, self.n_clusters))
        for i in range(self.n_samples):
            for j in range(self.n_clusters):
                self.responsibilities[i, j] = self.weights[j] * self.gaussian(X[i], self.means[j], self.covariances[j])
            self.responsibilities[i] /= np.sum(self.responsibilities[i])
            self.labels_[i] = np.argmax(self.responsibilities[i])

    def maximization(self, X):
        for i in range(self.n_clusters):
            weight = np.sum(self.responsibilities[:, i])
            self.weights[i] = weight / self.n_samples
            self.means[i] = np.sum(X * self.responsibilities[:, i].reshape(-1, 1), axis=0) / weight
            self.covariances[i] = np.dot((X - self.means[i]).T, (X - self.means[i]) * self.responsibilities[:, i].reshape(-1, 1)) / weight
            
    @property
    def labels(self):
        return self.labels_
if __name__ == '__main__':
    from read import read_data
    from matplotlib import pyplot as plt
    import time
    import os
    logs = "Q1/logs"
    data = read_data("Q1/8gau.txt")
    n_clusters = 18
    init_params = ['kmeans', 'random']
    
    for n_clusters in range(2, n_clusters):
        for init_param in init_params:
            timestamp = str(time.strftime("%Y%m%d%H%M%S", time.localtime()))
            test_name = timestamp + "_GMM_" + str(n_clusters) + "_" + init_param
            print(test_name)
            gmm = GMM(n_clusters=n_clusters, max_iter=200,init_params=init_param)
            gmm.fit(data,task_name=test_name)
            title = "GMM " + str(n_clusters) + " " + init_param
            plt.title(title)
            plt.scatter(data[:, 0], data[:, 1], c=gmm.labels_)
            # 在中心点上标注数字
            for i in range(n_clusters):
                plt.text(gmm.means[i][0], gmm.means[i][1], str(i), fontsize=12, color="red")

            plt.savefig(os.path.join(logs, test_name + ".png"))
            plt.savefig(os.path.join(logs, test_name + ".svg"))
            plt.clf()
            # plt.show()
        # plt.show()
    # data = data[:100]
    # gmm = GMM(n_clusters=5, max_iter=100,init_params='random')
    # gmm.fit(data)
    # plt.scatter(data[:, 0], data[:, 1], c=gmm.labels_)
    # plt.show()
    # km = KMeans(n_clusters=5)
    # km.fit(data)
    # plt.scatter(data[:, 0], data[:, 1], c=km.labels_)
    # plt.show()