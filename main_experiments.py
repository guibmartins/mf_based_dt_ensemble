# OS handling modules
import sys
import time
import time as t

# project modules
import util
import knn

# data science modules
import numpy as np
import pandas as pd

# import opfython.math.general as g
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# machine learning modules
from opfython.models.supervised import SupervisedOPF
from sklearn.model_selection import KFold, train_test_split # StratifiedKFold
from sklearn.preprocessing import StandardScaler
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import GridSearchCV as _GridSearchCV

# recommender systems modules
from surprise import NMF, SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split as mf_train_test_split
from surprise.model_selection import RandomizedSearchCV

# https://scikit-learn.org/stable/datasets/index.html#toy-datasets
from sklearn.datasets import load_breast_cancer, load_digits, load_iris

# graphical plotting
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set_style(style="darkgrid")


# Columns names of dataframe, which will be saved as excel file (k-folds)
col_names = [
    'acc_baseline', 'acc_mf',  # accuracy columns
    'bacc_baseline', 'bacc_mf',  # balanced accuracy columns
    'baseline_pred_time_total', 'baseline_pred_time_mean',  # baseline's prediction time
    'mf_fit_time', 'mf_pred_time', 'mf_time_total',   # MF training + prediction time
    'mf_hpt_time'
]

# Dictionary of datasets
ds_dict = {
    'cancer': load_breast_cancer,
    'digits': load_digits,
    'iris': load_iris,
    'cmc': './data/cmc.csv',
    'blood': './data/blood.csv'
}

baseline_dict = {'opf': SupervisedOPF, 'knn': knn.KNearestNeighbors}
mf_dict = {'nmf': NMF, 'pmf': SVD, 'svd': SVD}


def main(base_alg, mf_alg, ds_key, iterations=10, n_models=1, k_factors=1, k_folds=5, seed=None):

    # Load chosen dataset
    if ds_key == 'cmc':
        X, y = util.load_cmc(ds_dict.get(ds_key))
    elif ds_key == 'blood':
        X, y = util.load_blood(ds_dict.get(ds_key), adj_labels=True, sampling='over')
    else:
        X, y = util.load_data(ds_dict.get(ds_key), adj_labels=True)

    # Set different sparsity degrees
    sp_levels = [x / 100 for x in range(10, 90, 10)]

    # Custom seed (experimental purposes)
    np.random.seed(seed)

    # Define the custom seed for each iteration
    custom_seeds = np.random.randint(10e3, size=iterations)

    for i in range(iterations):

        # Perform a stratified k-fold cross validation
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=custom_seeds[i])
        kf_idx = list(range(1, kf.get_n_splits(X, y) + 1))

        sp_list = [pd.DataFrame(index=kf_idx, columns=col_names) for _ in range(len(sp_levels))]
        run_time = {'baseline': np.zeros((n_models, 2)), 'mf': np.zeros(2)}
        fold_idx = 1
        y_preds = {}

        # Run a k-fold cross validation
        for idx_hold, idx_test in kf.split(X, y):

            # Standardize features
            std_scaler = StandardScaler().fit(X[idx_hold])
            X_hold = std_scaler.transform(X[idx_hold])
            X_test = std_scaler.transform(X[idx_test])

            # Set test labels
            Y_test = y[idx_test]

            # Reset running time
            # idx_0 = training time; idx_1 = test time
            run_time.update({'baseline': np.zeros((n_models, 2))})

            # Train/classify data for each model
            for j in range(n_models):

                # Split holdout set into train/dev subsets
                X_train, X_val, Y_train, Y_val = train_test_split(
                   X_hold, y[idx_hold], test_size=0.2, shuffle=True, random_state=custom_seeds[i]) #, stratify=y[idx_hold])

                # Instantiate the chosen classifier
                base_model = baseline_dict.get(base_alg)()

                # Training step
                run_time['baseline'][j, 0] = t.time()

                # Exhaustive search - finding the best parameters
                if base_alg == 'knn':
                    # Set the number of neighbors (k)
                    base_model.k = int(X_val.shape[0])

                    # Fit the knn model
                    base_model.fit(X_train, X_val, Y_train, Y_val)
                else:
                    # Learning the model
                    base_model.fit(X_train, Y_train)

                run_time['baseline'][j, 0] = t.time() - run_time.get('baseline')[j, 0]

                # Testing step
                run_time['baseline'][j, 1] = t.time()
                y_pred = base_model.predict(X_test)
                run_time['baseline'][j, 1] = t.time() - run_time.get('baseline')[j, 1]

                print(f"Training: {run_time['baseline'][j, 0]} seconds | ", end='')
                print(f"Classification: {run_time['baseline'][j, 1]} seconds.")

                # Add entry to pred class dictionary
                y_preds.update({str(j + 1): y_pred})

                # Save current model to file
                util.pickle_save(base_model, f'_{mf_alg}_{base_alg}_model_{j+1}.sav')

            # Dataframe of predicted labels for each model
            df_M = pd.DataFrame(data=y_preds).T
            df_M.columns = idx_test

            # Array of true classes (evaluation purposes)
            # y_true = np.concatenate([Y_test for _ in range(n_models)], axis=0).flatten().astype(int)

            # Array of baseline predicted classes (evaluation purposes)
            # y_base_pred = df_M.values.flatten().astype(int)

            # STEP 2: run a matrix factorization technique over the new data
            # Building the tuple structure (learner i, sample j, prediction r) from matrix of predictions
            D = pd.DataFrame(data=util.get_samples_from_df(df_M), columns=['base', 'sample', 'base_pred'])

            reader = Reader(rating_scale=(Y_test.min(), Y_test.max()))
            data = Dataset.load_from_df(D, reader)

            # Hyper-parameters optimization: randomized search k-fold cv
            param_grid = {'n_epochs': [100]}

            if mf_alg == 'svd':
                param_grid.update(
                    {'lr_all': [0.01, 0.05, 0.1, 0.2],
                     'reg_all': [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2],
                     'biased': [True]})
            elif mf_alg == 'pmf':
                param_grid.update(
                    {'lr_all': [0.01, 0.05, 0.1, 0.2],
                     'reg_all': [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2],
                     'biased': [False]})
            else:
                param_grid.update(
                    {'reg_pu': [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2],
                     'reg_qi': [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2]})

            print('Performing randomized search...', end='')
            gs = RandomizedSearchCV(
                mf_dict.get(mf_alg),
                param_grid, n_iter=5,
                measures=['mae'],
                cv=3,
                n_jobs=-1,
                random_state=custom_seeds[i])

            mf_hpt_time = time.time()
            gs.fit(data)
            mf_hpt_time = time.time() - mf_hpt_time

            alg = gs.best_estimator['mae']
            print('done')

            # For each sparsity degree (i.e., test set size)
            for s_idx, sparsity in enumerate(sp_levels):

                # Reset running time
                # idx_0 = training time; idx_1 = test time
                run_time.update({'mf': np.zeros(3)})

                print(f'Sparsity degree: {sparsity}...')

                # Split `df_M` into training and test sets
                R_train, R_test = mf_train_test_split(
                    data, test_size=sparsity, random_state=custom_seeds[i], shuffle=True)

                # Initial time (training)
                run_time['mf'][0] = t.time()

                # Fit the MF model on training data
                alg.fit(R_train)

                # End time (training)
                run_time['mf'][0] = t.time() - run_time.get('mf')[0]

                # Initial time (testing)
                run_time['mf'][1] = t.time()

                # Predict on test data
                preds = alg.test(R_test)

                # End time (testing)
                run_time['mf'][1] = t.time() - run_time.get('mf')[1]

                run_time['mf'][2] = t.time()
                preds = alg.fit(R_train).test(R_test)
                run_time['mf'][2] = t.time() - run_time.get('mf')[2]

                # Evaluation: accuracy score (ACC) and balanced accuracy score (BACC)
                # Array of true labels
                y_true = y[[p.iid for p in preds]].astype(int)

                # Array of MF predicted labels
                y_base_pred = np.array([int(p.r_ui) for p in preds])
                y_mf_pred = np.array([p.est for p in preds])

                # Evaluate MF algorithm
                acc_baseline = accuracy_score(y_true, y_base_pred)
                acc_mf = accuracy_score(y_true, np.rint(y_mf_pred).astype(int))
                bacc_baseline = balanced_accuracy_score(y_true, y_base_pred)
                bacc_mf = balanced_accuracy_score(y_true, np.rint(y_mf_pred).astype(int))

                # Measure running time for the baseline technique
                # Predict on % of sparsity of the samples (the unseen ones, i.e., MF test set)
                _X = pd.DataFrame(data=R_test, columns=['base_id', 'sample_id', 'base_pred'])

                # Baseline's prediction time with respect only to unobserved labels (R_test)
                baseline_pred_time = t.time()

                # Filtering predicted samples for a given learner (model)
                for j in range(n_models):
                    idx_query = _X[_X['base_id'] == str(j + 1)]['sample_id']
                    tmp_model = util.pickle_load(f'_{mf_alg}_{base_alg}_model_{j + 1}.sav')
                    if len(idx_query) > 0:
                        tmp_X_test = std_scaler.transform(X[idx_query])
                        _ = tmp_model.predict(tmp_X_test)

                baseline_pred_time = t.time() - baseline_pred_time

                sp_list[s_idx].loc[fold_idx] = [
                    acc_baseline, acc_mf,
                    bacc_baseline, bacc_mf,
                    run_time.get('baseline')[:, 1].sum() * sparsity,
                    baseline_pred_time,
                    run_time.get('mf')[0],
                    run_time.get('mf')[1],
                    run_time.get('mf')[2],
                    mf_hpt_time
                ]

            fold_idx += 1
            y_preds.clear()

        # Save results (all k-fold for all iterations) to excel file
        out_dir = f'./out/SIBGRAPI_2022/k_{k_factors}/{base_alg}/{mf_alg}/{ds_key}'
        out_file = f'{base_alg}_{mf_alg}_{ds_key}_n{n_models}_k{k_factors}_results_{util.get_datetime()}.xlsx'

        with pd.ExcelWriter('/'.join([out_dir, out_file])) as writer:
            for df, sp in zip(sp_list, sp_levels):
                df.to_excel(writer, sheet_name=f'S_{sp * 100}')


if __name__ == '__main__':

    if len(sys.argv) == 1:
        ValueError('Need to define all required parameters before continue.')

    # Get parameters
    base_algo = str(sys.argv[1])    # which baseline algorithm to be used (OPF or Naive Bayes)
    mf_algo = str(sys.argv[2])      # which MF technique to be used
    dataset = str(sys.argv[3])      # dataset key string
    it = int(sys.argv[4])           # number of iterations (stats)
    n = int(sys.argv[5])            # number of classifier models to be generated
    k = int(sys.argv[6])            # number of latent factors to be used with MF
    fd = int(sys.argv[7])           # Number of folds to cross validate
    custom_seed = int(sys.argv[8])  # Custom seed generator

    main(base_algo, mf_algo, dataset, iterations=it, n_models=n, k_factors=k, k_folds=fd, seed=custom_seed)
