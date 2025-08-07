import json
import multiprocessing
import os
import pickle
from itertools import product, cycle

import numpy as np
import pandas as pd

import buzzcode.config as cfg
from buzzcode.training.training import enumerate_paths
from buzzcode.utils import read_pickle_exhaustive


def combine(frames_source, frames_augment, prop_combine, limit='square'):
    """
        Combines source frames with augmentation frames while avoiding temporal correlations.
        Uses two different strategies depending on the size of frames_augment:
        1. When len(frames_augment) < limit: Makes every combination of source and augment frames
        2. Else: Makes random combinations of frames_source and frames_augment (where each combination is unique)

        Args:
            frames_source: List of source frames to be augmented
            frames_augment: List of frames to use for augmentation
            proportion: Proportion of the output frame that will be composed of the source frame
            limit: Either 'square' or an integer limiting augmentations per source frame
        """
    def combine_frames(frame_source, frame_augment):
        return frame_source * prop_combine + frame_augment * (1 - prop_combine)

    if limit == 'square':
        limit = len(frames_source)

    n_frames_augment = len(frames_augment)

    if n_frames_augment <= limit:
        # if we aren't over our limit on augment frames, we'll end up using them all. Just return the product.
        frames_out = []
        for source, augment in product(frames_source, frames_augment):
            frames_out.append(combine_frames(source, augment))
    else:
        # if we are over the limit on augment frames, we need to make sure we don't end up with weird correlations,
        # e.g., one frame of source is augmented entirely with augment frames from the same event (one minute of rainstorm)
        shuffled_augment = frames_augment.copy()
        np.random.shuffle(shuffled_augment)
        augment_cycler = cycle(shuffled_augment)

        frames_out = []
        for frame_source in frames_source:
            for _ in range(limit):
                try:
                    frame_augment = next(augment_cycler)
                    frames_out.append(combine_frames(frame_source, frame_augment))
                except StopIteration:
                    break  # exit loop when cycler is exhausted

    return frames_out


def augment_combine_classes(setname, class_source, class_augment, limit, prop_combine):
    print(f'AUGMENT: combining {class_source} with {class_augment}')
    paths_source = enumerate_paths(
        setname=setname,
        sample_subpath=os.path.join(cfg.TRAIN_DIRNAME_AUDIOSAMPLES, cfg.TRAIN_DIRNAME_RAW),
        fold='train',
        exclusive=True,
        labels_raw=[class_source]
    )

    frames_source = []
    for p in paths_source:
        frames_source.extend(read_pickle_exhaustive(p))

    if not frames_source:
        print(f'AUGMENT: no source frames found for {class_source}')
        return

    paths_augment = enumerate_paths(
        setname=setname,
        sample_subpath=os.path.join(cfg.TRAIN_DIRNAME_AUDIOSAMPLES, cfg.TRAIN_DIRNAME_RAW),
        fold='train',
        exclusive=True,
        labels_raw=[class_augment]
    )

    frames_augment = []
    for p in paths_augment:
        frames_augment.extend(read_pickle_exhaustive(p))

    if not frames_augment:
        print(f'AUGMENT: no augment frames found for {class_augment}')
        return

    path_out = os.path.join(cfg.TRAIN_DIR_SET, setname, cfg.TRAIN_DIRNAME_AUDIOSAMPLES, cfg.TRAIN_DIRNAME_AUGMENT_COMBINE, class_source + '+' + class_augment + '.pickle')
    frames_out = combine(frames_source, frames_augment, prop_combine=prop_combine, limit=limit)

    with open(path_out, 'wb') as file:
        for f in frames_out:
            pickle.dump(f, file)


# TODO: detect already combined files, skip
    # drop nan
def augment_combine_set(setname, prop_combine, cpus=4, limit=4):
    print("DEBUG: entered set")
    dir_set = os.path.join(cfg.TRAIN_DIR_SET, setname)
    dir_combine = os.path.join(dir_set, cfg.TRAIN_DIRNAME_AUDIOSAMPLES, cfg.TRAIN_DIRNAME_AUGMENT_COMBINE)
    os.makedirs(dir_combine)
    with open(os.path.join(dir_combine, 'config_combine.txt'), 'x') as file:
        file.write(json.dumps({'limit': limit, 'prop': prop_combine}))

    print("DEBUG: reading augment")
    df_augment = pd.read_csv(os.path.join(dir_set, 'augment_combine.csv')).dropna()

    print("DEBUG: starting pool")
    with multiprocessing.Pool(cpus) as pool:
        arglist = [(setname, class_source, class_augment, limit, prop_combine) for class_source, class_augment in zip(df_augment['source'], df_augment['augment'])]
        pool.starmap_async(augment_combine_classes, arglist)
        pool.close()
        pool.join()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    augment_combine_set(
        setname='test',
        cpus=4,
        limit=1,
        prop_combine=0.5,
    )
