import logging
import re


class AssignLog:
    def __init__(self, msg: str, level_str: str, terminate_gui: bool = False):
        self.item = msg
        self.level_str = level_str
        self.level_int = loglevels[level_str]
        self.terminate_gui = terminate_gui


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


level_progress = {'level': logging.INFO-5, 'levelName': 'PROGRESS'}
logging.addLevelName(level=level_progress['level'], levelName=level_progress['levelName'])
loglevels = {
    'NOTSET': logging.NOTSET,
    level_progress['levelName']: level_progress['level'],
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}
