import os
import pickle
import re
from datetime import datetime


class Timer:
    def __init__(self):
        self.time_start = datetime.now()
        self.time_end = datetime.now()

    def stop(self):
        self.time_end = datetime.now()

    def restart(self):
        self.time_start = datetime.now()

    def get_current(self):
        return datetime.now() - self.time_start

    def get_total(self, decimals=2):
        time_total = self.time_end - self.time_start
        total_formatted = time_total.total_seconds().__round__(decimals)

        return total_formatted


def search_dir(dir_in, extensions=None):
    if extensions is not None and not (extensions.__class__ is list and extensions[0].__class__ is str):
        raise ValueError("input extensions should be None, or list of strings")

    paths = []
    for root, dirs, files in os.walk(dir_in):
        for file in files:
            paths.append(os.path.join(root, file))

    if extensions is None:
        return paths

    # convert extensions into regex, if they aren't already
    for i, extension in enumerate(extensions):
        if extension[-1] != "$":
            extension = extension + "$"

        extension = extension.lower()
        extensions[i] = extension

    paths = [p for p in paths if True in [bool(re.search(e, p.lower())) for e in extensions]]
    return paths


def read_pickle_exhaustive(path_pickle):
    elements = []
    with open(path_pickle, 'rb') as f:
        while True:
            try:
                element = pickle.load(f)
                elements.append(element)
            except EOFError:
                break
    return elements


def read_pickle_generator(path_pickle):
    with open(path_pickle, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def shortpath(path, dir_root):
    return re.sub(dir_root, "", path)

def build_ident(path, root_dir, tag=None):
    ident = re.sub(root_dir, '', path)
    ident = os.path.splitext(ident)[0]

    if tag is not None:
        ident = re.sub(re.escape(tag), '', ident)

    ident = re.sub('^/', '', ident)

    return ident
