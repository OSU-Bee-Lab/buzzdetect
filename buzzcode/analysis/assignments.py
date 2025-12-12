import logging
import re


class AssignLog:
    def __init__(self, message: str, level_str: str):
        self.message = message
        self.level_str = level_str
        self.level_int = loglevels[level_str]


class AssignStream:
    def __init__(self, path_audio, duration_audio, chunklist, dir_audio, terminate=False):
        self.path_audio = path_audio
        self.duration_audio = duration_audio
        self.chunklist = chunklist
        self.terminate = terminate

        if dir_audio is None:
            self.shortpath = None
        else:
            self.shortpath = self.path_audio.replace(dir_audio, '')
            self.shortpath = re.sub('^/', '', self.shortpath)



class AssignAnalyze:
    def __init__(self, path_audio, shortpath, chunk, samples):
        self.path_audio = path_audio
        self.shortpath = shortpath
        self.chunk = chunk
        self.samples = samples


class AssignWrite:
    def __init__(self, path_audio, chunk, results):
        self.path_audio = path_audio
        self.chunk = chunk
        self.results = results


loglevels = {
    'NOTSET': logging.NOTSET,
    'DEBUG': logging.DEBUG,
    'PROGRESS': logging.INFO-5,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}
