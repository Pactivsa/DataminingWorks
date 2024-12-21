# 基于python实现高斯混合模型Gaussion Mixture Model

import numpy as np

class GMM:
    def __init__(self, n_clusters=2, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        n_samples = X.shape[0]
        self.n_samples = n_samples
        self.n_features = X.shape[1]
        self.labels_ = np.zeros(n_samples)
        self.weights = np.ones(self.n_clusters) / self.n_clusters
        self.means = np.random.rand(self.n_clusters, self.n_features)
        self.covariances = np.array([np.eye(self.n_features) for _ in range(self.n_clusters)])
        for _ in range(self.max_iter):
            self.expectation(X)
            self.maximization(X)

    def expectation(self, X):
        self.responsibilities = np.zeros((self.n_samples, self.n_clusters))
        for i in range(self.n_samples):
            for j in range(self.n_clusters):
                self.responsibilities[i, j] = self.weights[j] * self.gaussian(X[i], self.means[j], self.covariances[j])
            self.responsibilities[i] /= np.sum(self.responsibilities[i])
            self.labels_[i] = np.argmax(self.responsibilities[i])

    def maximization(self, X):
        for j in range(self.n_clusters):
            rj = np.sum(self.responsibilities[:, j])
            self.weights[j] = rj / self.n_samples
            self.means[j] = np.sum(X * self.responsibilities[:, j].reshape(-1, 1), axis=0) / rj
            self.covariances[j] = np.dot((X - self.means[j]).T, (X - self.means[j]) * self.responsibilities[:, j].reshape(-1, 1)) / rj

    def gaussian(self, x, mean, covariance):
        return np.exp(-0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(covariance)), x - mean)) / np.sqrt(np.linalg.det(covariance) * (2 * np.pi) ** self.n_features)


if __name__ == '__main__':
    from read import read_data
    X, y = read_data()

    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    print(gmm.labels_)
