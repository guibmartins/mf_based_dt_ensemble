import numpy as np
import time as t
from numpy.linalg import solve
from sklearn.metrics import mean_squared_error as mse
from sklearn.decomposition import NMF, TruncatedSVD


def get_samples(R):
    samples = [
        (i, j, R[i, j])
        for i in range(R.shape[0])
        for j in range(R.shape[1])
        if R[i, j] > 0
    ]

    return samples


class MF:

    def __init__(self, factors, with_bias=False, loss='mse'):
        self.k = factors
        self.loss_function = self._mse if loss == 'mse' else self._mae
        self.with_bias = with_bias
        self.fit_time = 0
        self.pred_time = 0

    def fit(self, R, lrate=0.1, reg=.01, iterations=50, custom_seed=None, verbose=False):
        # Reset the random seed generator
        np.random.seed(custom_seed)

        # Get matrix R dimensions (n x m)
        n, m = R.shape

        # Random initialize low-level matrices P and Q
        # considering a normal uniform distribution
        self.P = np.random.rand(n, self.k)
        self.Q = np.random.rand(m, self.k)
        # self.P = np.random.normal(scale=1./self.k, size=(self.n, self.k))
        # self.Q = np.random.normal(scale=1./self.k, size=(self.m, self.k))

        # Initialize bias
        self.bias_global = np.mean(R[R > 0]) if self.with_bias else 0
        self.bias_row = np.zeros(n)
        self.bias_col = np.zeros(m)

        # Making R matrix into samples <row, column, rating>
        self.samples = get_samples(R)

        # Get time at the beginning of training
        start = t.time()

        training_log = []
        tmp = 1e5
        for i in range(iterations):
            # Shuffle training samples
            np.random.shuffle(self.samples)

            # Optimize latent factors through stochastic gradient descent (SGD)
            self.sgd(alpha=lrate, beta=reg)

            # Compute the general loss using some appropriate error metric
            loss = self.loss_function()

            # Get training information for historical purposes
            training_log.append([i, loss])

            if (i + 1) % 10 == 0 and verbose:
                print(f'Iteration: {i + 1} | error: {loss:.4f}')

            # If error is optimally low
            # if abs(loss - tmp) < 5e-2:
            #     break
            # tmp = loss

        # Get time at the end of training
        self.fit_time = t.time() - start
        print(f'Fitting time: {self.fit_time:.4f} sec.')

        return training_log

    def sgd(self, alpha, beta):

        for i, j, r in self.samples:
            # Calculate error (real rating - predicted rating)
            loss = r - self.get_pred_rating(i, j)

            # Make a copy of current row i of P to update Q
            P_i = self.P[i, :]

            # Update latent matrices P and Q
            self.P[i, :] += alpha * (loss * self.Q[j, :] - beta * self.P[i, :])
            self.Q[j, :] += alpha * (loss * P_i - beta * self.Q[j, :])

            # Update biases
            if self.with_bias:
                self.bias_row[i] += alpha * (loss - beta * self.bias_row[i])
                self.bias_col[j] += alpha * (loss - beta * self.bias_col[j])

    def _mae(self):
        error = 0
        for i, j, r in self.samples:
            error += np.abs(r - self.get_pred_rating(i, j))

        return error

    def _mse(self):
        error = 0
        for i, j, r in self.samples:
            error += (r - self.get_pred_rating(i, j)) ** 2

        # return np.sqrt(error)
        return error / float(len(self.samples))

    def get_pred_rating(self, i, j):
        return self.bias_global + self.bias_row[i] + self.bias_col[j] + np.dot(self.P[i, :], self.Q.T[:, j])

    def predict_all(self, verbose=False):
        # Get time at the beginning of testing
        start = t.time()

        bias = self.bias_global + self.bias_row[:, np.newaxis] + self.bias_col[np.newaxis, :]
        R_hat = bias + np.dot(self.P, self.Q.T)

        self.pred_time = t.time() - start

        if verbose:
            print(f'Testing time: {self.pred_time:.4f} sec.')

        return R_hat

    def predict(self, X_test, verbose=False):

        start = t.time()
        preds = [self.get_pred_rating(i, j) for i, j, _ in X_test]
        self.pred_time = t.time() - start

        if verbose:
            print(f'Testing time: {self.pred_time:.4f} sec.')

        return preds


class ALS_MF:

    def __init__(self, factors, reg):
        self.k = factors
        self.fit_time = 0.
        self.pred_time = 0.
        self.samples = None
        self.P = None
        self.Q = None
        self._lambda = reg

    def fit(self, X_train, epochs=50, verbose=False):

        # Get matrix R dimensions (n x m)
        n, m = X_train.shape

        # Random initialize low-level matrices P and Q
        # considering a normal uniform distribution
        self.P = np.random.randn(n, self.k)
        self.Q = np.random.randn(m, self.k)

        # Making X_train (matrix) into sample format: <row_i, column_j, rating_r>
        self.samples = get_samples(X_train)

        # Get time at the beginning of training
        start = t.time()

        train_log = []

        for i in range(epochs):

            np.random.seed(None)

            self.P = self._als2(X_train, self.P, self.Q)
            self.Q = self._als2(X_train.T, self.Q, self.P)

            # Compute the general loss using some appropriate error metric
            loss = self._mse(X_train)

            # Get training information for historical purposes
            train_log.append(loss)

            if (i + 1) % 10 == 0 and verbose:
                print(f'Iteration: {i + 1} | error: {loss:.4f}')

            # Get time at the end of training
        self.fit_time = t.time() - start
        print(f'Fitting time: {self.fit_time:.4f} sec.')

        return train_log

    def predict(self, X_test, verbose=False):

        start = t.time()
        preds = [self.get_pred_rating(i, j) for i, j, _ in X_test]
        self.pred_time = t.time() - start

        if verbose:
            print(f'Testing time: {self.pred_time:.4f} sec.')

        return preds

    def predict_all(self, verbose=False):
        # Get time at the beginning of testing
        start = t.time()
        R_hat = np.dot(self.P, self.Q.T)
        self.pred_time = t.time() - start

        if verbose:
            print(f'Testing time: {self.pred_time:.4f} sec.')

        return R_hat

    def get_pred_rating(self, i, j):
        return np.dot(self.P[i, :], self.Q.T[:, j])

    def _als(self, X, solve_vec, fixed_vec):

        A = np.dot(fixed_vec.T, fixed_vec) + np.eye(self.k) * self._lambda

        for i in range(solve_vec.shape[0]):
            solve_vec[i, :] = solve(A, np.dot(X[i, :], fixed_vec))

        return solve_vec

    def _als2(self, X, solve_vec, fixed_vec):

        A = np.dot(fixed_vec.T, fixed_vec) + (np.eye(self.k) * self._lambda)
        b = np.dot(X, fixed_vec)
        solve_vec = np.dot(b, np.linalg.inv(A))

        return solve_vec

    def _mse(self, X_train):
        pred = self.predict_all()
        mask = np.nonzero(X_train)
        e = mse(X_train[mask].flatten(), pred[mask].flatten())

        # for i, j, r in self.samples:
        #    error += (r - self.get_pred_rating(i, j)) ** 2
        # return np.sqrt(error)

        return e


class ProbabilisticMF:

    def __init__(self, n_factors, lambda_p=1, lambda_q=1, n_labels=1):
        self.k = n_factors
        self._lambda_p = lambda_p
        self._lambda_q = lambda_q
        self.P = None
        self.Q = None
        self._min = 1.
        self._max = 1.
        self._n_labels = n_labels
        self.fit_time = 0.
        self.pred_time = 0.
        self.samples = None

    def _init_latent_factors(self, X):

        # Get train data matrix dimensions
        n, m = X.shape

        # Perform mean value imputation
        # nan_mask = np.isnan(X)
        # self.data[nan_mask] = self.data[~nan_mask].mean()

        #zero_mask = (X == 0)
        #X[zero_mask] = np.mean(X[~zero_mask])
        #self._lambda_p = 1 / np.mean(np.var(X, axis=1))
        #self._lambda_q = 1 / np.mean(np.var(X, axis=0))

        # Low precision reflects uncertainty; prevents overfitting.
        # Set to the mean variance across users and items.
        # self.alpha_u = 1 / self.data.var(axis=1).mean()
        # self.alpha_v = 1 / self.data.var(axis=0).mean()

        self.P = np.zeros((self.k, n), dtype=np.float64)
        self.Q = np.random.normal(0., 1./self._lambda_q, (self.k, m))

    def _update_latent_factors(self, X):

        n, m = X.shape
        Id = np.identity(self.k)

        for i in range(n):
            Q_j = self.Q[:, X[i, :] > 0]
            R_ij = X[i, X[i, :] > 0]
            A_inv = np.linalg.inv(np.dot(Q_j, Q_j.T) + self._lambda_p * Id)
            self.P[:, i] = np.dot(A_inv, np.dot(R_ij, Q_j.T))

        for j in range(m):
            P_i = self.P[:, X[:, j] > 0]
            R_ij = X[X[:, j] > 0, j]
            A_inv = np.linalg.inv(np.dot(P_i, P_i.T) + self._lambda_q * Id)
            self.Q[:, j] = np.dot(A_inv, np.dot(R_ij, P_i.T))

    def _loss(self, X):

        X_hat = np.dot(self.P.T, self.Q)
        X_PQ = (X[X > 0] - X_hat[X > 0])
        loss = np.sum(np.dot(X_PQ, X_PQ.T)) +\
            self._lambda_p * np.sum(np.dot(self.P, self.P.T)) +\
            self._lambda_q * np.sum(np.dot(self.Q, self.Q.T))

        return -0.5 * loss

    def _update_scale(self):
        X_hat = np.dot(self.P.T, self.Q)
        self._min = np.min(X_hat)
        self._max = np.max(X_hat)

    def _mse(self, squared=True):
        error = 0
        for i, j, r in self.samples:
            error += (r - self.get_pred_rating(i, j)) ** 2

        if squared:
            return error / float(len(self.samples))

        return np.sqrt(error / float(len(self.samples)))

    def get_pred_rating(self, i, j):
        x_hat = np.dot(self.P[:, i], self.Q.T[j, :])
        return 0 if self._max == self._min else (((x_hat - self._min) / (self._max - self._min)) * self._n_labels)

    def fit(self, X_train, epochs=100, verbose=False):
        train_log = []
        rmse_train = []

        self.samples = get_samples(X_train)

        start = t.time()
        self._init_latent_factors(X_train)
        self._update_scale()
        rmse_train.append(self._mse())

        for it in range(epochs):

            # Calculate the log a-posteriori
            self._update_latent_factors(X_train)
            loss = self._loss(X_train)
            train_log.append(loss)

            if (it + 1) % 10 == 0:
                self._update_scale()
                rmse_train.append(self._mse())

                if verbose:
                    print(f'Loss (log p a-posteriori) at epoch {it + 1}: {loss}')

        self._update_scale()
        self.fit_time = t.time() - start

        if verbose:
            print(f'Training time: {self.fit_time:.4f} sec.')

        return train_log, rmse_train

    def predict(self, X_test, verbose=False):

        start = t.time()
        preds = [self.get_pred_rating(i, j) for i, j, _ in X_test]
        self.pred_time = t.time() - start

        if verbose:
            print(f'Testing time: {self.pred_time:.4f} sec.')

        return preds

    def predict_all(self, verbose=False):
        # Get time at the beginning of testing
        start = t.time()

        # Dot product that calculates X_hat based on latent matrices P and Q
        X_hat = np.dot(self.P.T, self.Q)

        self.pred_time = t.time() - start

        if verbose:
            print(f'Testing time: {self.pred_time:.4f} sec.')

        return X_hat


class NMF:

    def __init__(self, n_factors, reg_p=0.1, reg_q=0.1, custom_seed=None):
        self.k = n_factors
        self._lambda_p = reg_p
        self._lambda_p = reg_q
        self._seed = custom_seed
        self.samples = None
        self.fit_time = 0.
        self.pred_time = 0.

    def _init_latent_factors(self, data_dims):
        n, m = data_dims
        self.P = np.random.rand(n, self.k)
        self.Q = np.random.rand(m, self.k)

    def _sgd(self, l_rate, data_dims):
        n, m = data_dims
        p_num = np.zeros(shape=(n, self.k))
        p_denom = np.zeros_like(p_num)
        q_num = np.zeros(shape=(m, self.k))
        q_denom = np.zeros_like(q_num)

        for i, j, r in self.samples:
            r_hat = 0
            for k in range(self.k):
                r_hat += self.Q[j, k] * self.P[i, k]

            r_hat = np.dot(self.Q[j, :], self.P.T[:, i])
            # est = global_mean + bu[u] + bi[i] + r_hat
            e = r - r_hat

    def _mse(self):
        pass

    def fit(self, X_train, l_rate=0.1, epochs=50, verbose=False):

        # Reset the random seed generator
        np.random.seed(self._seed)

        # Get training samples in tuple format
        self.samples = get_samples(X_train)

        # Initialize Low-rank matrices P and q
        self._init_latent_factors(X_train.shape)

        # Get time at the beginning of training
        start = t.time()

        train_loss = []
        for it in range(epochs):
            # Shuffle training samples
            np.random.shuffle(self.samples)

            # Optimize latent factors through stochastic gradient descent (SGD)
            self._sgd(l_rate, X_train.shape)

        self.fit_time = t.time() - start

        if verbose:
            print(f'Fit time: {self.fit_time:.4f} sec.')

    def predict(self):
        pass
