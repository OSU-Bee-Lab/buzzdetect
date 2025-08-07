import os
import random

import pandas
import pandas as pd

from buzzcode import config as cfg, config
from buzzcode.training.training import build_classes, labels_from_path, enumerate_paths
from buzzcode.utils import read_pickle_exhaustive


def build_translation_dict(translation):
    return {translation['from']: translation['to'] for _, translation in translation.iterrows()}

def translate_labels(labels_raw: list, translation_dict: dict):
    """
    Translates a list of raw labels according to a translation dict, as built by build_translation_dict()..

    When a raw label has a "to" value of "ignore"  or no value, no target will be generated for that label.
    If a sample is only labeled with "ignore" or blank labels, it will be excluded from the dataset.

    When a raw label has a "to" value of "exclude", any sample matching that label will be dropped
    from the dataset, no matter what other labels are present for the sample.

    Args:
        labels_raw (list): The raw labels to translate.
        translation (DataFrame): A translation data frame with columns "from" and "to" containing the labels to translate from and to, respectively..

    Returns:
        list: Translated labels with NaN values removed.
    """
    labels_translated = [
        translation_dict.get(l, 'ignore')  # Translate if found, else leave unchanged
        for l in labels_raw
    ]

    return labels_translated


def labels_to_targets(labels_translate, classes):
    return [c in labels_translate for c in classes]


def build_fold_dataset(setname, fold, translation, augmenttype, labels_keep_raw=None, exclusive=False):
    translation_dict = build_translation_dict(translation)
    classes = build_classes(translation)

    paths_samples = enumerate_paths(
        setname=setname,
        sample_subpath=os.path.join(cfg.TRAIN_DIRNAME_EMBEDDINGSAMPLES, augmenttype),
        fold=fold,
        labels_raw=labels_keep_raw,
        exclusive=exclusive
    )

    samples = []
    for p in paths_samples:
        labels_raw = labels_from_path(p)
        labels_translate = translate_labels(labels_raw, translation_dict)

        # drop samples where any "to" label is set to exclude
        if 'exclude' in labels_translate:
            continue

        targets = labels_to_targets(labels_translate, classes)

        # drop samples where every label is set to ignore
        if sum(targets) == 0:
            continue

        # else, we're good to go!
        embeddings = read_pickle_exhaustive(p)

        samples_path = [{'embeddings': e, 'targets': targets} for e in embeddings]
        samples.extend(samples_path)

    # TODO: shuffle takes a long time; can I just shuffle indices? Does that work with tf dataset?
    random.seed(42)
    random.shuffle(samples)

    return samples


def load_augment_noise(setname, translation, name_noise):
    augment_df = pd.read_csv(os.path.join(cfg.TRAIN_DIR_SET, setname, 'augment_noise_' + name_noise + '.csv'))
    data_augment = []

    props = augment_df['prop'].unique()
    for prop in props:
        classes_keep = augment_df[augment_df['prop'] == prop]['class'].unique()
        augmenttype = 'augment_noise_prop' + str(prop)
        data_augment += build_fold_dataset(setname=setname, fold='train', translation=translation, augmenttype=augmenttype, labels_keep_raw=classes_keep, exclusive=False)

    return data_augment


def load_augment_volume(setname, translation, name_volume):
    augment_df = pd.read_csv(os.path.join(cfg.TRAIN_DIR_SET, setname, 'augment_volume_' + name_volume + '.csv'))
    data_augment = []

    props = augment_df['prop'].unique()
    for prop in props:
        classes_keep = augment_df[augment_df['prop'] == prop]['class'].unique()
        augmenttype = 'augment_volume_prop' + str(prop)
        data_augment += build_fold_dataset(setname=setname, fold='train', translation=translation,
                                           augmenttype=augmenttype, labels_keep_raw=classes_keep, exclusive=False)

    return data_augment