import glob
import multiprocessing
import os
import pickle
from itertools import islice
import json
import numpy as np

import re

import buzzcode.config as cfg
from buzzcode.embedding.embedders import load_embedder_config, load_embedder_model
from buzzcode.utils import read_pickle_generator


def embed_samples(path_audiosamples, embedder, batch_size=2000):
    def batched(generator, size):
        while True:
            batch = list(islice(generator, size))
            if not batch:
                break
            yield batch

    gen = read_pickle_generator(path_audiosamples)
    all_embeddings = []
    total_samples = 0

    for batch in batched(gen, batch_size):
        audio_flattened = np.concatenate(batch)
        embeddings = embedder(audio_flattened)
        if len(embeddings) != len(batch):
            raise ValueError(
                f'EMBEDDING: in path {path_audiosamples} something has gone horribly wrong\n'
                f'samples out ({len(embeddings)}) != samples in ({len(batch)})'
            )
        total_samples += len(batch)
        all_embeddings.extend(embeddings)

    return all_embeddings

def embed_file(path_audiosamples, embeddername, dir_set, overwrite):
    path_out = re.sub(
        os.path.join(dir_set, cfg.TRAIN_DIRNAME_AUDIOSAMPLES),
        os.path.join(dir_set, cfg.TRAIN_DIRNAME_EMBEDDINGSAMPLES),
        path_audiosamples,
    )

    if os.path.exists(path_out) and not overwrite:
        print(f'EMBEDDING: embedding for {path_audiosamples} already exists; skipping embedding')

    print(f"EMBEDDING: building embeddings for {path_audiosamples}")
    config_embedder = load_embedder_config(embeddername)
    embedder = load_embedder_model(embeddername, framehop_s=config_embedder['framelength'] )

    dir_out = os.path.dirname(path_out)
    os.makedirs(dir_out, exist_ok=True)

    samples_embeddings = embed_samples(path_audiosamples, embedder)
    with open(path_out, 'wb') as file:  # overwrites
        for e in samples_embeddings:
            pickle.dump(e, file)


def embed_set(setname, cpus, overwrite=False):
    dir_set = os.path.join(cfg.TRAIN_DIR_SET, setname)
    if not os.path.exists(dir_set):
        raise FileNotFoundError(f'Set directory {dir_set} does not exist')

    with open(os.path.join(dir_set, 'config_embedder.txt'), 'r') as file:
        config_embedder = json.load(file)

    embeddername=config_embedder['embeddername']

    paths_audiosamples = glob.glob(os.path.join(dir_set, 'samples_audio', '**', '*.pickle'), recursive=True)

    if not paths_audiosamples:
        raise FileNotFoundError(f'Audio samples not yet built for set "{setname}"')

    if cpus > 1:
        with multiprocessing.Pool(cpus) as pool:
            arglist = [(path, embeddername, dir_set, overwrite) for path in paths_audiosamples]
            pool.starmap_async(embed_file, arglist)
            pool.close()
            pool.join()

    # if you have a ton of small files, this will be much faster than multiprocessing,
    # since each worker has to load a new embedder model
    # TODO: look back into producer/consumer setup where model is loaded once;
    #  this previously caused tensorflow to leak memory due to repeated graph construction
    elif cpus == 1:
        config_embedder = load_embedder_config(embeddername)
        embedder = load_embedder_model(embeddername, framehop_s=config_embedder['framelength'])

        for path_audiosamples in paths_audiosamples:
            path_out = re.sub(
                os.path.join(dir_set, cfg.TRAIN_DIRNAME_AUDIOSAMPLES),
                os.path.join(dir_set, cfg.TRAIN_DIRNAME_EMBEDDINGSAMPLES),
                path_audiosamples,
            )

            if os.path.exists(path_out) and not overwrite:
                print(f'EMBEDDING: embedding for {path_audiosamples} already exists; skipping embedding')
                continue

            print(f"EMBEDDING: building embeddings for {path_audiosamples}")

            dir_out = os.path.dirname(path_out)
            os.makedirs(dir_out, exist_ok=True)

            samples_embeddings = embed_samples(path_audiosamples, embedder)
            with open(path_out, 'wb') as file:  # overwrites
                for e in samples_embeddings:
                    pickle.dump(e, file)

    else:
        raise ValueError(f'cpus must be > 0, not {cpus}')

    print(f'EMBEDDING: complete for set "{setname}" :D')
    return None

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    embed_set(
        setname='standard',
        cpus=1,
        overwrite=False
    )

