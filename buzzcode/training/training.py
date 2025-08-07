import glob
import os
import re

import numpy as np
import pandas as pd

from buzzcode import config as cfg

def clean_name(name_in, prefix, extension):
    name_out = re.sub(prefix, '', name_in)
    name_out = re.sub(extension, '', name_out)

    return name_out


def build_classes(translation):
    classes = translation['to'].unique().tolist()
    classes = [c for c in classes if not c in  ['ignore', np.nan, '', 'exclude']]
    classes = sorted(classes)

    return classes


def build_weights(data_train, classes):
    weights = pd.DataFrame()
    weights['target'] = range(len(classes))
    weights['class'] = classes

    def count_samples(target_in):
        return sum(1 for s in data_train if s['targets'][target_in])

    weights['frames'] = [count_samples(t) for t in weights['target']]
    frames_total = weights['frames'].sum()

    n_classes_present = sum(weights['frames'] > 0)

    def weighter(frames_class):
        if frames_class == 0:
            return 1

        weight = frames_total / (frames_class * n_classes_present)

        return weight

    weights['weight'] = [weighter(f) for f in weights['frames']]

    return weights


def labels_from_path(path_in):
    base = os.path.basename(path_in)
    base = os.path.splitext(base)[0]
    labels = re.split(pattern='\\+', string=base)

    return labels

def enumerate_paths(setname, sample_subpath, labels_raw=None, exclusive=False, fold=None):
    dir_set = os.path.join(cfg.TRAIN_DIR_SET, setname)
    dir_samples = os.path.join(dir_set, sample_subpath)

    paths = glob.glob(os.path.join(dir_samples, '**', '*.pickle'), recursive=True)

    # drop paths outside of fold
    def filter_fold(paths_original):
        folds = pd.read_csv(os.path.join(dir_set, 'folds.csv'))
        idents_in_fold = folds['ident'][folds['fold']==fold].to_list()

        paths_new = []
        for path in paths_original:
            ident = re.sub(dir_samples, '', path)
            ident = re.sub('^/', '', ident)
            ident = os.path.dirname(ident)
            if ident in idents_in_fold:
                paths_new.append(path)

        return paths_new

    if fold is not None:
        paths = filter_fold(paths)

    # drop paths outside of classes
    if labels_raw is not None:
        paths = [p for p in paths if any([c in labels_from_path(p) for c in labels_raw])]
        if exclusive:
            paths = [p for p in paths if len(labels_from_path(p))==1]

    return paths


def can_write_model(modelname):
    dir_model = os.path.join(cfg.DIR_MODELS, modelname)

    if not os.path.exists(dir_model):
        return True

    if modelname == 'test':
        return True

    if not os.listdir(dir_model):
        return True

    return False

