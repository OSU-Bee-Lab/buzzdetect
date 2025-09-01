import logging


class AssignLog:
    def __init__(self, msg: str, level):
        self.item = msg

        if level.__class__ is str:
            self.level_int = log_levels[level]
        elif level.__class__ is int:
            self.level_int = level
        else:
            raise ValueError(f"level must be str or int, not {level.__class__}")


class AssignStream:
    def __init__(self, path_audio, duration_audio, chunklist, terminate=False, dir_audio=None):
        self.path_audio = path_audio
        self.duration_audio = duration_audio
        self.chunklist = chunklist
        self.terminate = terminate
        self.shortpath = None if dir_audio is None else path_audio.replace(dir_audio, '')


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


log_levels = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL}
