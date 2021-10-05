import time
# import math
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors


class KNearestNeighbors:

    def __init__(self, k=1, distance='euclidean'):
        self.k = k

        self.best_k = 1

        if distance == 'euclidean':
            self.dist_fn = euclidean_distance

        self.data = None
        self.labels = None

        self.fit_time = 0.
        self.pred_time = 0.

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, value):
        self._k = value

    def _query(self, sample, k):

        indices_distances = []

        for i in range(self.data.shape[0]):
            dist = self.dist_fn(self.data[i], sample)
            indices_distances.append((dist, i))

        # Sort list based on ascending distance value
        k_neighbors = sorted(indices_distances)[:k]

        # Convert tuple to np.array of indices and distances for k neighbors
        indices_distances = np.array(k_neighbors).T

        return indices_distances[1].astype(int), indices_distances[0]

    def fit(self, X_train, X_val, y_train, y_val):

        self.data, self.labels = X_train, y_train

        print(f'Learning best k...')

        max_acc = 0
        best_k = 1
        tic = time.time()

        for k in range(1, self.k + 1, 2):

            preds = np.zeros_like(y_val)

            nn = NearestNeighbors(n_neighbors=k, algorithm='brute', n_jobs=1).fit(X_train)
            distances, indices = nn.kneighbors(X_val, n_neighbors=k)

            for i in range(X_val.shape[0]):
                # indices, distances = self._query(X_val[i], k)
                preds[i] = stats.mode(y_train[indices[i]])[0][0]

            acc = accuracy_score(y_val, preds)

            if acc > max_acc:
                max_acc = acc
                best_k = k

        self.best_k = best_k
        self.fit_time = time.time() - tic

        print(f'Best k = {best_k}')

    def predict(self, X_test):

        preds = np.zeros(X_test.shape[0])
        tic = time.time()

        for i in range(X_test.shape[0]):
            indices, distances = self._query(X_test[i], self.best_k)
            preds[i] = stats.mode(self.labels[indices])[0][0]

        self.pred_time = time.time() - tic

        return preds.astype(int)


def euclidean_distance(x, y):

    dist = np.sum((x - y) ** 2)
    return np.sqrt(dist)
