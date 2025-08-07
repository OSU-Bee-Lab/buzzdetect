import multiprocessing
import os
import pickle
import re

import numpy as np

import buzzcode.config as cfg
from buzzcode.training.training import enumerate_paths
from buzzcode.utils import read_pickle_exhaustive


def path_augment(path_audiosamples, prop, dirname_augment):
    return re.sub(
        os.path.join(cfg.TRAIN_DIRNAME_AUDIOSAMPLES, cfg.TRAIN_DIRNAME_RAW),
        os.path.join(cfg.TRAIN_DIRNAME_AUDIOSAMPLES, dirname_augment + f'_prop{prop}'),
        path_audiosamples
    )

def noise(frames, prop_noise):
    frames_noisy = [f + prop_noise * np.random.uniform(-1, 1, len(f)) for f in frames]
    return frames_noisy

def volume(frames, prop_volume):
    frames_volume = [f * prop_volume for f in frames]
    return frames_volume

def augment_file(path_audiosamples, props_noise: list, props_volume: list, overwrite=False):
    if props_noise is None:
        props_noise = []

    if props_volume is None:
        props_volume = []

    if props_noise.__class__ is not list:
        props_noise = [props_noise]

    if not overwrite:
        props_noise = [p for p in props_noise if not os.path.exists(path_augment(path_audiosamples, p, cfg.TRAIN_DIRNAME_AUGMENT_NOISE))]

    if props_volume.__class__ is not list:
        props_volume = [props_volume]

    if not overwrite:
        props_volume = [p for p in props_volume if not os.path.exists(path_augment(path_audiosamples, p, cfg.TRAIN_DIRNAME_AUGMENT_VOLUME))]

    if not props_volume and not props_noise:
        print(f"AUGMENT: all augmentations complete for {path_audiosamples}")
        return False

    print(f"AUGMENT: augmenting {path_audiosamples}")
    frames_raw = read_pickle_exhaustive(path_audiosamples)

    for prop_noise in props_noise:
        path_out = path_augment(path_audiosamples, prop_noise, cfg.TRAIN_DIRNAME_AUGMENT_NOISE)
        frames_noisy = noise(frames_raw, prop_noise)
        os.makedirs(os.path.dirname(path_out), exist_ok=True)

        with open(path_out, 'wb') as file:  # overwrites
            for f in frames_noisy:
                pickle.dump(f, file)

    for prop_volume in props_volume:
        path_out = path_augment(path_audiosamples, prop_volume, cfg.TRAIN_DIRNAME_AUGMENT_VOLUME)
        frames_noisy = volume(frames_raw, prop_volume)
        os.makedirs(os.path.dirname(path_out), exist_ok=True)

        with open(path_out, 'wb') as file:  # overwrites
            for f in frames_noisy:
                pickle.dump(f, file)

    return True

def augment_set(setname, cpus, props_noise:list, props_volume:list, overwrite=False):
    dir_set = os.path.join(cfg.TRAIN_DIR_SET, setname)
    if not os.path.exists(dir_set):
        raise FileNotFoundError(f'Set directory {dir_set} does not exist')

    paths_samples = enumerate_paths(
        setname=setname,
        sample_subpath=os.path.join(cfg.TRAIN_DIRNAME_AUDIOSAMPLES, cfg.TRAIN_DIRNAME_RAW),
        fold='train'
    )

    if not paths_samples:
        raise FileNotFoundError(f'No matching audio samples found')

    with multiprocessing.Pool(cpus) as pool:
        args = zip(paths_samples, [props_noise] * len(paths_samples), [props_volume]*len(paths_samples), [overwrite] * len(paths_samples))
        pool.starmap_async(augment_file, args)
        pool.close()
        pool.join()

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    augment_set(
        setname='standard',
        cpus=6,
        props_noise=[0.05, 0.075, 0.2],
        props_volume=[2.5, 0.75]
    )
