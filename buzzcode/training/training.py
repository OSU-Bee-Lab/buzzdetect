import glob
import os
import random
import re

import numpy as np
import pandas as pd

import buzzcode.config as cfg
from buzzcode.utils import read_pickle_exhaustive


def path_to_labels(path_in):
    # TODO: this should probably go in some utils file for training
    base = os.path.basename(path_in)
    base = os.path.splitext(base)[0]
    labels = re.split(pattern='\\+', string=base)

    return labels


def get_fold_paths(paths_in, folds, foldname):
    idents_fold = folds[folds['fold'] == foldname]['ident']

    def path_in_fold(path_in):
        return any([re.search(i, path_in) for i in idents_fold])

    paths_fold = [p for p in paths_in if path_in_fold(p)]

    return paths_fold


def load_path_samples(paths_in):
    samples_all = []
    for path in paths_in:
        embeddings = read_pickle_exhaustive(path)
        labels = path_to_labels(path)

        samples_path = [{'embeddings': e, 'labels_raw': labels} for e in embeddings]
        samples_all.extend(samples_path)

    return samples_all


def load_fold_samples(foldname, setname, augment=False):
    dir_set = os.path.join(cfg.DIR_TRAIN_SET, setname)

    paths_data = glob.glob(os.path.join(dir_set, 'samples_embeddings', '**', '*.pickle'), recursive=True)

    folds = pd.read_csv(os.path.join(dir_set, 'folds.csv'))

    paths_fold = get_fold_paths(
        paths_in=paths_data,
        folds=folds,
        foldname=foldname
    )

    if foldname == 'train' and augment:
        paths_fold += glob.glob(os.path.join(dir_set, 'augment_*', '**', '*.pickle'), recursive=True)

    samples = load_path_samples(paths_fold)

    return samples


def labels_to_targets(labels, classes):
    y_blank = [0 for _ in classes]
    y_true = y_blank.copy()
    for label in labels:
        y_true[classes.index(label)] = 1

    return np.array(y_true, dtype=np.int64)


def add_fold_targets(s, classes):
    s_up = s.copy()
    s_up.update({'targets': labels_to_targets(s['labels_translate'], classes)})
    return s_up


def build_fold_dataset(foldname, setname, translation_dict, classes, augment, shuffle=True):
    samples = load_fold_samples(foldname, setname, augment=augment)

    # so far, datasets fit in memory easily; will have to look into random reads if we run oom
    if shuffle:
        random.seed(42)
        random.shuffle(samples)

    samples = [add_labels_translate(s, translation_dict) for s in samples]
    samples = [s for s in samples if s['labels_translate']]  # drop empty labels

    samples = [add_fold_targets(s, classes) for s in samples]

    return samples


def clean_name(name_in, prefix, extension):
    name_out = re.sub(prefix, '', name_in)
    name_out = re.sub(extension, '', name_out)

    return name_out


def translate_labels(labels_raw, translation_dict):
    """
    Translates a list of raw labels using a translation dictionary as built by build_translation_dict.
    Leaves labels unchanged if no key is found (not in translate.csv).
    Drops labels where the translation value is NaN (empty "to" value in translate.csv).

    Args:
        labels_raw (list): The raw labels to translate.
        translation_dict (dict): A dictionary mapping raw labels to their translations.

    Returns:
        list: Translated labels with NaN values removed.
    """
    labels_translated = [
        translation_dict.get(l, l)  # Translate if found, else leave unchanged
        for l in labels_raw
    ]
    # Filter out NaN values
    labels_translated = [label for label in labels_translated if label is not np.nan]
    return labels_translated


def add_labels_translate(s, translation_dict):
    s_up = s.copy()
    s_up.update({'labels_translate': translate_labels(s['labels_raw'], translation_dict)})
    return s_up


def build_translation_dict(translation):
    """
    where key is "from" and value is "to"

    """
    if "from" not in translation.columns or "to" not in translation.columns:
        raise ValueError("DataFrame must contain 'from' and 'to' columns.")

    translation_dict = translation.set_index("from")["to"].to_dict()
    return translation_dict
