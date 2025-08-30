import glob
import json
import os
import pickle
import re
from itertools import islice

import numpy as np

import buzzcode.config as cfg
from buzzcode.embedding.load_embedder import load_embedder
from buzzcode.utils import read_pickle_generator, setthreads

setthreads(1)


def embed_file(path_audiosamples, path_embeddings, embedder):
    def embed_batch(batch_size=2000):
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
            embeddings = embedder.embed(audio_flattened)
            if len(embeddings) != len(batch):
                raise ValueError(
                    f'EMBEDDING: in path {path_audiosamples} something has gone horribly wrong\n'
                    f'samples out ({len(embeddings)}) != samples in ({len(batch)})'
                )
            total_samples += len(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    print(f"EMBEDDING: building embeddings for {path_audiosamples}")

    dir_out = os.path.dirname(path_embeddings)
    os.makedirs(dir_out, exist_ok=True)

    samples_embeddings = embed_batch()
    with open(path_embeddings, 'wb') as file:  # overwrites
        for e in samples_embeddings:
            pickle.dump(e, file)


def embed_set(setname, overwrite=False):
    dir_set = os.path.join(cfg.TRAIN_DIR_SET, setname)
    dir_samples_audio = os.path.join(dir_set, cfg.TRAIN_DIRNAME_AUDIOSAMPLES)
    dir_samples_embeddings = os.path.join(dir_set, cfg.TRAIN_DIRNAME_EMBEDDINGSAMPLES)

    with open(os.path.join(dir_set, 'config_set.json'), 'r') as file:
        config_set = json.load(file)

    if not os.path.exists(dir_set):
        raise FileNotFoundError(f'Set directory {dir_set} does not exist')

    paths_audiosamples = glob.glob(os.path.join(dir_samples_audio, '**', '*.pickle'), recursive=True)
    if not paths_audiosamples:
        raise FileNotFoundError(f'Audio samples not yet built for set "{setname}"')

    assignments = []
    for p in paths_audiosamples:
        path_embeddings = re.sub(re.escape(dir_samples_audio), re.escape(dir_samples_embeddings), p)
        exists = os.path.exists(path_embeddings)

        assignments.append({'path_audiosamples': p, 'path_embeddings': path_embeddings, 'exists': exists})

    if not overwrite:
        assignments = [a for a in assignments if not a['exists']]

    if not assignments:
        print(f'EMBEDDING: all embeddings created for set {setname}')
        return

    embedder = load_embedder(embeddername=config_set['embeddername'], framehop_prop=1, load_model=True)
    for a in assignments:
        embed_file(a['path_audiosamples'], a['path_embeddings'], embedder)

    print(f'EMBEDDING: complete for set "{setname}" :D')
    return None

if __name__ == '__main__':
    embed_set(
        setname='standard',
        overwrite=False
    )

