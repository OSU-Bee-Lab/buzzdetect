from datetime import datetime
import numpy as np
import argparse
import pickle
import warnings
import os
import re




class Timer:
    def __init__(self):
        self.time_start = datetime.now()
        # self.state = 'running'  # not pursuing it right now, but I could add exceptions for illegal start/stop

    def stop(self):
        self.time_end = datetime.now()

    def restart(self):
        self.time_start = datetime.now()

    def get_current(self):
        return datetime.now() - self.time_start

    def get_total(self, decimals=2):
        self.time_total = self.time_end - self.time_start
        self.total_formatted = self.time_total.total_seconds().__round__(decimals)

        return self.total_formatted


def clip_name(filepath, clippath):
    clip = re.sub(clippath, "", filepath)
    return clip


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def search_dir(dir_in, extensions):
    if not (extensions.__class__ is list and extensions[0].__class__ is str):
        raise ValueError("input extensions should be list of strings")

    # convert extensions into regex, if they aren't already
    for i in range(len(extensions)):
        e = extensions[i]
        if e[len(e) - 1] != "$":
            e = e+"$"

        e = e.lower()
        extensions[i] = e

    paths = []
    for root, dirs, files in os.walk(dir_in):
        for file in files:
            if True in [bool(re.search(e, file.lower())) for e in extensions]:
                paths.append(os.path.join(root, file))

    return paths


def save_pickle(path_pickle, obj):
    os.makedirs(os.path.dirname(path_pickle), exist_ok=True)

    with open(path_pickle, 'wb') as file_pickle:
        pickle.dump(obj, file_pickle)


def load_pickle(path_pickle):
    with open(path_pickle, 'rb') as file_pickle:
        obj = pickle.load(file_pickle)

    return obj


def setthreads(threads_desired=1):
    if threads_desired < 1:
        warnings.warn(f'cannot set threadcount below 1; continuing without modification')

    import tensorflow as tf
    threads_current = tf.config.threading.get_inter_op_parallelism_threads()

    if threads_current == threads_desired:
        return

    elif threads_current == 0:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

    else:
        warnings.warn(f'tensorflow threads already initialized to {threads_current}; cannot set')


def make_chunklist(duration, chunklength, chunk_overlap=0, chunk_min=0):
    if chunk_overlap > chunklength:
        raise ValueError('chunk overlap must be less than chunk length')

    if chunklength > duration:
        return [(0, duration)]

    chunklist = []
    chunk_start = 0
    chunk_stop = chunklength
    while chunk_stop < duration:
        chunklist.append((chunk_start, chunk_stop))

        chunk_start = chunk_stop - chunk_overlap
        chunk_stop = chunk_start + chunklength
        if (duration - chunk_stop) <= chunk_min:
            chunklist.append((chunk_start, duration))
            break

    return chunklist
