import os
import re
import argparse

# given a list of paths, create the directories necessary to hold the files
def unique_dirs(paths, make=False):
    path_dirs = []
    for path in paths:
        path_dir = os.path.dirname(path)
        path_dirs.append(path_dir)

    dirs_unique = list(set(path_dirs))

    if make == True:
        for path_dir in dirs_unique:
            os.makedirs(path_dir, exist_ok=True)

    return dirs_unique


def size_to_runtime(size_GB, kbps=256):
    runtime = (
            size_GB *  # gigabytes
            8 *  # convert to gigabits
            (10 ** 6) *  # convert to kilobits
            (1 / kbps)
    )  # convert to seconds

    return runtime


def runtime_to_size(runtime, kbps=256):
    size_GB = (
            runtime *  # seconds
            kbps *  # kilobits
            (1 / 8) *  # kilobytes
            (10 ** -6)  # gigabytes
    )

    return size_GB


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


def search_dir(dir_in, extension):
    paths = []
    for root, dirs, files in os.walk(dir_in):
        for file in files:
            if file.endswith(extension):
                paths.append(os.path.join(root, file))

    return paths