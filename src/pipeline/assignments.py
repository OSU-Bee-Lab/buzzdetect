import os
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

import src.config as cfg
from src.pipeline.loglevels import loglevels
from src.utils import build_ident


@dataclass
class AssignFile:
    path_audio: str
    dir_audio: str
    dir_results: str
    duration_audio: float | None = None
    chunklist: list[tuple[float, float]] | None = None

    def __post_init__(self):
        self.ident = build_ident(self.path_audio, self.dir_audio, tag=None)

        self.path_results_base = self.dir_results + '/' + self.ident
        self.path_results_partial = self.path_results_base + cfg.SUFFIX_RESULT_PARTIAL
        self.path_results_complete = self.path_results_base + cfg.SUFFIX_RESULT_COMPLETE

        self.extension_audio = os.path.splitext(self.path_audio)[1]

        self.shortpath_audio = self.ident + '.' + self.extension_audio
        self.shortpath_results_complete = self.ident + cfg.SUFFIX_RESULT_COMPLETE

@dataclass
class AssignChunk:
    file : AssignFile
    chunk: tuple[float, float] | None = None
    samples: np.ndarray = None
    results: tf.Tensor = None


@dataclass
class AssignLog:
    message: str
    level_str: str
    terminate: bool = False

    def __post_init__(self):
        self.level_int = loglevels[self.level_str]
