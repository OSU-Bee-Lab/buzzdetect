import glob
import json
import multiprocessing
import os
import pickle
import re

import numpy as np

from buzzcode.config import DIR_TRAIN_SET
from buzzcode.utils import setthreads, read_pickle_exhaustive

setthreads(1)


def embed_samples(path_samples, embedder):
    samples_audio = read_pickle_exhaustive(path_samples)

    audio_flattened = np.concatenate(samples_audio)
    samples_embeddings = embedder(audio_flattened)
    if len(samples_embeddings) != len(samples_audio):
        raise ValueError('something has gone horribly wrong; samples out != samples in')

    return samples_embeddings


embedding_to_audio_prop = (449 + 212) / (6399 + 3021)


def embeddings_done(path_audio, dir_samples_audio, dir_samples_embeddings, tolerance=0.95):
    path_embeddings = re.sub(dir_samples_audio, dir_samples_embeddings, path_audio)

    if not os.path.exists(path_embeddings):
        return False

    size_audio = os.path.getsize(path_audio)
    size_embeddings = os.path.getsize(path_embeddings)
    size_embeddings_expected = size_audio * embedding_to_audio_prop

    if size_embeddings < (size_embeddings_expected * tolerance):
        return False

    return True


def embed_set(setname, overwrite=False, cpus=2):
    dir_set = os.path.join(DIR_TRAIN_SET, setname)
    dir_samples_audio = os.path.join(dir_set, 'samples_audio')
    dir_samples_embeddings = os.path.join(dir_set, 'samples_embeddings')
    if not os.path.exists(dir_samples_audio):
        raise FileNotFoundError('samples_audio directory does not exist for this set')

    with open(os.path.join(dir_set, 'config_set.txt')) as f:
        config_set = json.load(f)

    embeddername = config_set['embedder']

    paths_audio = glob.glob(os.path.join(dir_samples_audio, '**', '*.pickle'), recursive=True)
    if not overwrite:
        paths_audio = [p for p in paths_audio if not embeddings_done(p, dir_samples_audio, dir_samples_embeddings)]

    q_paths = multiprocessing.Queue()  # better to have persistent workers to avoid re-loading embedders hundreds of times
    for p in paths_audio:
        q_paths.put(p)

    for c in range(cpus):
        q_paths.put('terminate')

    def worker_embedder(worker_id):
        path_in = q_paths.get()
        if path_in == 'terminate':  # early return if cpus>idents
            return

        from buzzcode.embedders import load_embedder_model, load_embedder_config
        config_embedder = load_embedder_config(embeddername)
        embedder = load_embedder_model(embeddername, framehop_s=config_embedder[
            'framelength'])  # set framehop to 1 frame; see embed_samples()

        while path_in != 'terminate':
            path_out = re.sub(dir_samples_audio, dir_samples_embeddings, path_in)
            print(f"EMBEDDING: worker {worker_id} creating embeddings for {re.sub(dir_samples_audio, '', path_in)}")
            dir_out = re.sub(dir_samples_audio, dir_samples_embeddings, path_in)
            dir_out = os.path.dirname(dir_out)
            os.makedirs(dir_out, exist_ok=True)

            samples_embeddings = embed_samples(path_in, embedder)
            with open(path_out, 'wb') as file:  # overwrites
                for e in samples_embeddings:
                    pickle.dump(e, file)

            path_in = q_paths.get()

    proc_embedders = []
    for c in range(cpus):
        proc_embedders.append(
            multiprocessing.Process(target=worker_embedder, name=f"embedder_proc{c}", args=[c]))
        proc_embedders[-1].start()
        pass
