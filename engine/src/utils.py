import os
import re
from datetime import datetime

def get_ext(path):
    return os.path.splitext(path)[1].lower().lstrip('.')

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
    """Walk dir_in and yield matching paths one at a time (rather than
    collecting a full list first), so a caller processing thousands of files
    across a large/slow-to-stat tree can act on each match as it's found
    instead of blocking until the entire tree has been walked."""
    if extensions is not None and not (extensions.__class__ is list and extensions[0].__class__ is str):
        raise ValueError("input extensions should be None, or list of strings")

    patterns = None
    if extensions is not None:
        patterns = []
        for extension in extensions:
            if extension[-1] != "$":
                extension = extension + "$"
            patterns.append(re.compile(extension.lower()))

    for root, dirs, files in os.walk(dir_in):
        for file in files:
            path = os.path.join(root, file)
            if patterns is not None and not any(p.search(path.lower()) for p in patterns):
                continue
            yield path


def build_ident(path, root_dir, tag=None):
    ident = re.sub(root_dir, '', path)
    ident = os.path.splitext(ident)[0]

    if tag is not None:
        ident = re.sub(re.escape(tag), '', ident)

    ident = re.sub('^/', '', ident)

    return ident
