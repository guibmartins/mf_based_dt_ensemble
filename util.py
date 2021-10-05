import datetime as d
import pickle

#import opfython.math.general as g
# import sklearn.preprocessing
# from scipy.stats import wilcoxon, f_oneway, levene, ttest_ind, ttest_rel
import numpy as np
import pandas as pd
import imblearn
from surprise import NMF, SVD


def over_sampling(X, Y):
    over_sampler = imblearn.over_sampling.RandomOverSampler(sampling_strategy='minority')
    X_os, Y_os = over_sampler.fit_resample(X, Y)
    return X_os, Y_os


def under_sampling(X, Y):
    under_sampler = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='majority')
    X_us, Y_us = under_sampler.fit_resample(X, Y)
    return X_us, Y_us


def get_samples(R):
    samples = [
        (i + 1, j, R[i, j])
        for i in range(R.shape[0])
        for j in range(R.shape[1])
        if R[i, j] > 0
    ]

    return samples


def get_samples_from_df(df):
    samples = [
        (i, j, df.loc[i][j])
        for i in df.index
        for j in df.columns
    ]

    return samples


def set_mf_instance(alg, k, epochs, lr, reg):

    if alg == 'svd':
        return SVD(n_factors=k, n_epochs=epochs, lr_all=lr, reg_all=reg, biased=True)
    elif alg == 'pmf':
        return SVD(n_factors=k, n_epochs=epochs, lr_all=lr, reg_all=reg, biased=False)
    else:
        return NMF(n_factors=k, n_epochs=epochs, reg_pu=reg, reg_qi=reg)


def pickle_save(obj, file_name):

    print(f'Saving model to file: {file_name} ...')

    # Opening a destination file
    with open(file_name, 'wb') as dest_file:
        # Dumping model to file
        pickle.dump(obj, dest_file)

    print('Model saved.')


def pickle_load(file_name):

    print(f'Loading model from file: {file_name} ...')

    # Trying to open the file
    obj = pickle.load(open(file_name, 'rb'))
    print('Model loaded.')
    return obj


def load_data(load_func, adj_labels=True):
    X, Y = load_func(return_X_y=True)

    # Adjust label identifiers
    if adj_labels:
        Y += 1

    return X, Y.astype(int)


def load_cmc(data_path, header=None, adj_labels=False, sampling=None):

    if header is None:
        header = list(range(1, 9 + 1))

    df = pd.read_csv(data_path, sep=',', index_col=False, names=header, engine='python')
    df.dropna(how='all', inplace=True)

    X = df[df.columns[:-1]].values.astype(np.float32)
    Y = df[df.columns[-1]].values.squeeze().astype(int)

    if adj_labels:
        Y += 1

    if sampling == 'over':
        X, Y = over_sampling(X, Y)
    elif sampling == 'under':
        X, Y = under_sampling(X, Y)

    return X, Y


def load_blood(data_path, header=None, adj_labels=True, sampling=None):

    if header is None:
        header = list(range(1, 5 + 1))

    df = pd.read_csv(data_path, sep=',', index_col=False, names=header, engine='python')
    df.dropna(how='all', inplace=True)

    X = df[df.columns[:-1]].values.astype(np.float32)
    Y = df[df.columns[-1]].values.squeeze().astype(int)

    if adj_labels:
        Y += 1

    if sampling == 'over':
        X, Y = over_sampling(X, Y)
    elif sampling == 'under':
        X, Y = under_sampling(X, Y)

    return X, Y


def sparsity_based_split(M, s=.5):
    # Turn matrix into tuple <i, j, r> format
    samples = [
        [i, j, M[i, j]]
        for i in range(M.shape[0])
        for j in range(M.shape[1])
        if M[i, j] > 0
    ]

    # Shuffle first axis indexes
    np.random.shuffle(samples)

    # Define a cut
    cut = int((1 - s) * len(samples))

    return np.asarray(samples[:cut]), np.asarray(samples[cut:])


def pivot(data, shape):
    data = np.array(data, dtype=int)
    M = np.zeros(shape=shape)

    for i in data:
        M[i[0], i[1]] = i[2]

    return M


def mae(y_true, y_pred):
    e = np.sum(np.abs(y_true - y_pred)) / float(len(y_true))
    return e


def mse(y_true, y_pred, rooted=False):
    e = np.sum((y_true - y_pred) ** 2) / float(len(y_true))

    if rooted:
        return np.sqrt(e)

    return e


def get_datetime():
    current_date = d.datetime.now()
    return current_date.strftime("%b-%d-%Y_%H-%M-%S-%f")
