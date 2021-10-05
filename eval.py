import datetime as d
from scipy.stats import wilcoxon, f_oneway, levene, ttest_ind, ttest_rel
import numpy as np


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
    return current_date.strftime("%b-%d-%Y_%H-%M-%S")


# Calculate the Wilcoxon signed-rank test
def wilcoxon_test(data_1, data_2, alpha=0.05, zm='wilcox', alt='two-sided', verbose=False):

    # confidence interval
    ci = 100 * (1 - alpha)

    data_1 = np.array(data_1)
    data_2 = np.array(data_2)

    if verbose:
        print(f'G1 (Mean, Std): {data_1.mean()}, {data_1.std()}', end='\n\n')
        print(f'G2 (Mean, Std): {data_2.mean()}, {data_2.std()}', end='\n\n')

    stat, p = wilcoxon(data_1, data_2, zero_method=zm, alternative=alt)

    if verbose:
        print(f'Statistics: {stat:.2f}')
        print(f'p-value: {p:.6f}', end='\n\n')

    stats_log = []
    stats_log.extend([
        data_1.mean(),
        data_1.std(),
        data_2.mean(),
        data_2.std(),
        stat, p])

    # Fail to reject H0 (samples have the same distribution)
    msg_h = f'Reject H0 (samples come from different distributions) with {ci}% of confidence.'
    if p > alpha:
        msg_h = f'Fail to reject H0 (samples have the same distribution) with {ci}% of confidence.'

    if verbose:
        print(msg_h)

    stats_log.extend([msg_h])
    return stats_log


# Perform one-way ANOVA correlation test
def oneway_anova_test(data_1, data_2, alpha=0.05, verbose=False):
    stats_log = []

    # confidence interval
    ci = 100 * (1 - alpha)

    data_1 = np.array(data_1)
    data_2 = np.array(data_2)

    if verbose:
        print('Obs.: G1 is the control group.')
        print(f'G1 (Mean, Std): {data_1.mean()}, {data_1.std()}')
        print(f'G2 (Mean, Std): {data_2.mean()}, {data_2.std()}', end='\n\n')

    stat, p = f_oneway(data_1, data_2)

    if verbose:
        print(f'statistic: {stat:.2f}\np-value: {p:.6f}', end='\n\n')

    msg_h = f'Reject H0 (samples come from different distributions) with {ci}% of confidence.'
    if p > alpha:
        msg_h = f'Fail to reject H0 (samples have the same distribution) with {ci}% of confidence.'

    if verbose:
        print(msg_h)

    stats_log.extend([
        data_1.mean(), data_1.std(),
        data_2.mean(), data_2.std(),
        stat, p,
        msg_h
    ])

    return stats_log


# Calculate the T-test for the means of two independent samples of scores.
def t_independent_test(data_1, data_2, alpha=0.05, alt='two-sided', verbose=False):
    stats_log = []

    # confidence interval
    ci = 100 * (1 - alpha)

    data_1 = np.array(data_1)
    data_2 = np.array(data_2)

    if verbose:
        print('Obs.: G1 is the control group.')
        print(f'G1 (Mean, Std): {data_1.mean()}, {data_1.std()}')
        print(f'G2 (Mean, Std): {data_2.mean()}, {data_2.std()}', end='\n\n')

    stat, p = ttest_ind(data_1, data_2, alternative=alt)

    if verbose:
        print(f'statistic: {stat:.2f}\np-value: {p:.6f}', end='\n\n')

    msg_h = f'Reject H0 (samples come from different distributions) with {ci}% of confidence.'
    if p > alpha:
        msg_h = f'Fail to reject H0 (samples have the same distribution) with {ci}% of confidence.'

    if verbose:
        print(msg_h)

    stats_log.extend([
        data_1.mean(), data_1.std(),
        data_2.mean(), data_2.std(),
        stat, p,
        msg_h
    ])

    return stats_log


# Calculate the T-test for the means of two independent samples of scores.
def t_dependent_test(data_1, data_2, alpha=0.05, alt='two-sided', verbose=False):
    stats_log = []

    # confidence interval
    ci = 100 * (1 - alpha)

    data_1 = np.array(data_1)
    data_2 = np.array(data_2)

    if verbose:
        print('Obs.: G1 is the control group.')
        print(f'G1 (Mean, Std): {data_1.mean()}, {data_1.std()}')
        print(f'G2 (Mean, Std): {data_2.mean()}, {data_2.std()}', end='\n\n')

    stat, p = ttest_rel(data_1, data_2, alternative=alt)

    if verbose:
        print(f'statistic: {stat:.2f}\np-value: {p:.6f}', end='\n\n')

    msg_h = f'Reject H0 (samples come from different distributions) with {ci}% of confidence.'
    if p > alpha:
        msg_h = f'Fail to reject H0 (samples have the same distribution) with {ci}% of confidence.'

    if verbose:
        print(msg_h)

    stats_log.extend([
        data_1.mean(), data_1.std(),
        data_2.mean(), data_2.std(),
        stat, p,
        msg_h
    ])

    return stats_log


# Perform Levene test for equal variances.
def levene_test(data_1, data_2, alpha=0.05, verbose=False):
    stats_log = []

    # confidence interval
    ci = 100 * (1 - alpha)

    data_1 = np.array(data_1)
    data_2 = np.array(data_2)

    if verbose:
        print('Obs.: G1 is the control group.')
        print(f'G1 (Mean, Std): {data_1.mean()}, {data_1.std()}')
        print(f'G2 (Mean, Std): {data_2.mean()}, {data_2.std()}', end='\n\n')

    stat, p = levene(data_1, data_2)

    if verbose:
        print(f'statistic: {stat:.2f}\np-value: {p:.6f}', end='\n\n')

    msg_h = f'Reject H0 (samples come from different distributions) with {ci}% of confidence.'
    if p > alpha:
        msg_h = f'Fail to reject H0 (samples have the same distribution) with {ci}% of confidence.'

    if verbose:
        print(msg_h)

    stats_log.extend([
        data_1.mean(), data_1.std(),
        data_2.mean(), data_2.std(),
        stat, p,
        msg_h
    ])

    return stats_log
