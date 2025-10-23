import logging
import re


class AssignLog:
    def __init__(self, msg: str, level_str: str):
        self.item = msg
        self.level_str = level_str
        self.level_int = log_levels[level_str]


class AssignStream:
    def __init__(self, path_audio, duration_audio, chunklist, terminate=False, dir_audio=None):
        self.path_audio = path_audio
        self.duration_audio = duration_audio
        self.chunklist = chunklist
        self.terminate = terminate
        self.shortpath = None

        if dir_audio is not None:
            self.shortpath = path_audio.replace(dir_audio, '')
            self.shortpath = re.sub('^/', '', self.shortpath)


class AssignAnalyze:
    def __init__(self, path_audio, chunk, samples):
        self.path_audio = path_audio
        self.chunk = chunk
        self.samples = samples


class AssignWrite:
    def __init__(self, path_audio, chunk, results):
        self.path_audio = path_audio
        self.chunk = chunk
        self.results = results


level_progress = {'level': logging.INFO-5, 'levelName': 'PROGRESS'}
logging.addLevelName(level=level_progress['level'], levelName=level_progress['levelName'])
log_levels = {
    'NOTSET': logging.NOTSET,
    level_progress['levelName']: level_progress['level'],
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}
