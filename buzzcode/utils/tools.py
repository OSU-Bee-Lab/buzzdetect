import os
import re
import argparse
from datetime import datetime


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
