import os

import numpy as np
import pandas as pd

from src import config as cfg


def sx(results_join, precision_requested):
    ''' get sensitivity and threshold at x precision'''
    results_join['delta'] = results_join['precision'] - precision_requested

    over = results_join[results_join['delta'] > 0].sort_values(by='delta', ascending=True, inplace=False)
    under = results_join[results_join['delta'] < 0].sort_values(by='delta', ascending=False, inplace=False)

    if len(over) == 0:
        threshold = under.iloc[0]['ins_buzz']
    elif len(under) == 0:
        threshold = over.iloc[0]['ins_buzz']
    else:
        threshold = over.iloc[0]['ins_buzz'] + (under.iloc[0]['ins_buzz'] - over.iloc[0]['ins_buzz']) / 2


    sensitivity = np.mean(results_join[results_join['correct']]['ins_buzz'] > threshold)

    return {'threshold': threshold.__round__(2), 'precision': precision_requested.__round__(2), 'sensitivity': sensitivity.__round__(2)}


def calculate_threshold(modelname, precision_requested, tolerance=0.01):
    dir_tests = os.path.join(cfg.DIR_MODELS, modelname, cfg.SUBDIR_TESTS)

    try:
        metrics = pd.read_csv(os.path.join(dir_tests, cfg.FNAME_METRICS))
    except FileNotFoundError:
        raise FileNotFoundError(f'metrics not available for model "{modelname}"; run test_model({modelname}) and proceed') from None

    # filter for where precision is +- tolerance/2 of requested
    metrics['delta'] = abs(metrics['precision'] - precision_requested)
    metrics = metrics[metrics['delta'] <= tolerance/2]

    return metrics['threshold'].mean()
