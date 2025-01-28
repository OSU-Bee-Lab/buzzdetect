import glob
import json
import multiprocessing
import os
import pickle
import re
from itertools import product, cycle, islice

import numpy as np
import pandas as pd
import tensorflow as tf

import buzzcode.config as cfg
from buzzcode.utils import setthreads, read_pickle_exhaustive

setthreads(1)


def combiner(frames_source, frames_augment, limit='square'):
    """
        Combines source frames with augmentation frames while avoiding temporal correlations.
        Uses two different strategies depending on the size of frames_augment:
        1. When len(frames_augment) < limit: Makes every combination of source and augment frames
        2. Else: Makes random combinations of frames_source and frames_augment (where each combination is unique)

        Args:
            frames_source: List of source frames to be augmented
            frames_augment: List of frames to use for augmentation
            limit: Either 'square' or an integer limiting augmentations per source frame
        """

    def combo_full():
        # if we aren't over our limit on augment frames, we'll end up using them all. Just yield the product.
        for source, augment in product(frames_source, frames_augment):
            yield (source + augment) / 2

    def combo_limited():
        # if we are over the limit on augment frames, we need to make sure we don't end up with weird correlations,
        # e.g., one frame of source is augmented entirely with augment frames from the same event (one minute of rainstorm)
        shuffled_augment = frames_augment.copy()
        np.random.shuffle(shuffled_augment)
        augment_cycler = cycle(shuffled_augment)

        for frame_source in frames_source:
            for _ in range(limit):
                try:
                    frame_augment = next(augment_cycler)
                    yield (frame_source + frame_augment) / 2
                except StopIteration:
                    break  # exit loop when cycler is exhausted

    if limit == 'square':
        limit = len(frames_source)

    n_frames_augment = len(frames_augment)

    if n_frames_augment <= limit:
        return combo_full()
    else:
        return combo_limited()


def combine_set(setname, cpus=2, limit=10, batchsize=1500):
    # TODO:
    # TODO: make memory management to replace batch size!
    # add add augmentation with white noise (should apply to negative cases, too)
    # change volume
    # change frequency response?
    dir_set = os.path.join(cfg.dir_sets, setname)
    path_folds = os.path.join(dir_set, 'folds.csv')
    if not os.path.exists(path_folds):
        raise FileNotFoundError(
            f"The required file '{path_folds}' is missing. Please add it before running the script.")

    with open(os.path.join(dir_set, 'config_augment.txt'), 'x') as file:
        file.write(json.dumps({'limit_combinations': limit}))

    dir_combine = os.path.join(dir_set, 'augment_combine')
    os.makedirs(dir_combine)

    with open(os.path.join(dir_set, 'config_set.txt'), 'r') as f:
        config_set = json.load(f)

    folds = pd.read_csv(path_folds)

    # read in training audio
    idents_train = folds[folds['fold'] == 'train']['ident']

    def path_in_train(path_in):
        return any([re.search(i, path_in) for i in idents_train])

    def read_class_audio(classname):
        pattern = os.path.join(dir_set, 'samples_audio', '**', classname + '.pickle')
        paths_in = glob.glob(pattern, recursive=True)
        paths_in = [p for p in paths_in if path_in_train(p)]

        frames = []
        for path in paths_in:
            frames_path = read_pickle_exhaustive(path)
            frames.extend(frames_path)

        return frames

    augmentation = pd.read_csv(os.path.join(dir_set, 'augmentation_combine.csv'))
    classes_to_augment = set(list(augmentation['source']) + list(augmentation['augment']))
    frames_by_class = {k: read_class_audio(k) for k in classes_to_augment}
    frames_by_class = {k: frames_by_class[k] for k in frames_by_class.keys() if frames_by_class[k]}  # drop empty

    # make queue with source and augment names for which we have frames
    q_combinations = multiprocessing.Queue()
    for s, a in zip(augmentation['source'], augmentation['augment']):
        if s not in frames_by_class.keys():
            continue
        if a not in frames_by_class.keys():
            continue

        q_combinations.put((s, a))
    for _ in range(cpus):
        q_combinations.put('terminate')

    def construct_combined(class_source, class_augment, embedder):
        """ build the dataset files for overlapped audio """
        path_out = os.path.join(dir_combine, class_source + '+' + class_augment + '.pickle')

        frames_source = frames_by_class[class_source]
        frames_augment = frames_by_class[class_augment]

        frame_generator = combiner(frames_source, frames_augment, limit=limit)

        with open(path_out, 'wb') as file:
            while True:
                batch = list(islice(frame_generator, batchsize))
                if not batch:  # stop when no more batches
                    break

                batch_flat = np.concatenate(batch)
                embeddings = embedder(batch_flat)
                if len(embeddings) != len(batch):
                    raise ValueError('something has gone horribly wrong; number of embeddings out != frames in')

                embeddings = tf.squeeze(embeddings)  # shape=(1, 1024) -> (1024,)

                for e in embeddings:
                    pickle.dump(e, file)

    def worker_combiner(worker_id):
        # makes more sense to do this with starmap, but pickling + tensorflow is a nightmare
        from buzzcode.embedders import load_embedder_model, load_embedder_config
        config_embedder = load_embedder_config(config_set['embedder'])
        embedder = load_embedder_model(config_set['embedder'], framehop_s=config_embedder['framelength'])

        c = q_combinations.get()
        while c != 'terminate':
            print(f'AUGMENTATION: worker {worker_id} combining {c[0]} with {c[1]}')
            construct_combined(c[0], c[1], embedder)
            c = q_combinations.get()

    proc_combiners = []
    for c in range(cpus):
        proc_combiners.append(multiprocessing.Process(target=worker_combiner, args=[c]))
        proc_combiners[-1].start()

    for proc_combiner in proc_combiners:
        proc_combiner.join()


if __name__ == '__main__':
    combine_set(
        setname='test',
        cpus=4,
        limit=10,
        batchsize=1000
    )
