from queue import Full

import tensorflow as tf
import librosa
import numpy as np

from src.pipeline.assignments import AssignChunk
from src.stream.audio import mark_eof
import os

import pandas as pd
import soundfile as sf

from src import config as cfg
from src.stream.results_coverage import melt_coverage, get_gaps, smooth_gaps, gaps_to_chunklist
from src.pipeline.assignments import AssignFile, AssignLog
from src.pipeline.coordination import Coordinator
from src.stream.audio import get_duration
from src.inference.models import BaseModel

class WorkerStreamer:
    def __init__(self,
                 id_streamer,
                 model: BaseModel,
                 chunklength: float,
                 coordinator: Coordinator, ):

        self.model = model
        self.id_streamer = id_streamer
        self.coordinator = coordinator

        self.chunklength = chunklength
        self.framelength_s = self.model.embedder.framelength_s
        self.resample_rate = self.model.embedder.samplerate

    def __call__(self):
        self.run()

    def log(self, msg, level_str):
        self.coordinator.q_log.put(AssignLog(message=f'streamer {self.id_streamer}: {msg}', level_str=level_str))

    def handle_bad_read(self, a_file: AssignFile):
        # we've found that many of our .mp3 files give an incorrect .frames count, or else headers are broken
        # this appears to be because our recorders ran out of battery while recording
        # SoundFile does not handle this gracefully, so we catch it here.

        final_frame = a_file.track.tell()
        mark_eof(path_audio=a_file.path_audio, final_frame=final_frame)

        final_second = final_frame/a_file.track.samplerate

        msg = f"Unreadable audio at {round(final_second, 1)}s out of {round(a_file.duration_audio, 1)}s for {a_file.shortpath_audio}."
        if 1 - (final_second / a_file.duration_audio) > cfg.BAD_READ_ALLOWANCE:
            # if we get a bad read in the middle of a file, this deserves a warning.
            level = 'WARNING'
            msg += '\nAborting early due to corrupt audio data.'
        else:
            # but bad reads at the ends of files are almost guaranteed when the batteries run out
            level = 'DEBUG'
            msg += '\nBad audio is near file end, results should be mostly unaffected.'

        self.log(msg, level)

    def _chunk_file(self, a_file: AssignFile):
        self.log(f'Building assignments for {a_file.shortpath_audio}', 'INFO')

        if os.path.exists(a_file.path_results_complete):
            self.log(f'Skipping {a_file.shortpath_audio}; already analyzed', 'DEBUG')
            a_file.chunklist = []
            return

        if os.path.getsize(a_file.path_audio) < cfg.FILE_SIZE_MINIMUM:
            self.log(f'Skipping {a_file.shortpath_audio}; below minimum analyzeable size', 'DEBUG')
            a_file.chunklist = []
            return

        a_file.track = sf.SoundFile(a_file.path_audio)
        a_file.duration_audio = get_duration(a_file, q_log=self.coordinator.q_log)

        # if the file hasn't been started, chunk the whole file
        if not os.path.exists(a_file.path_results_partial):
            gaps = [(0, a_file.duration_audio)]
        else:
            # otherwise, read the results and calculate chunks
            df = pd.read_csv(a_file.path_results_partial)
            coverage = melt_coverage(df, self.framelength_s)
            gaps = get_gaps(
                range_in=(0, a_file.duration_audio),
                coverage_in=coverage
            )

            gaps = smooth_gaps(
                gaps,
                range_in=(0, a_file.duration_audio),
                framelength=self.framelength_s,
                gap_tolerance=self.framelength_s / 4
            )

            # if we find no gaps, this file was actually finished and never cleaned!
            # as of 2026-05-18, this shouldn't be possible because cleanup is incorporated into the writer.
            # before, cleanup happened at the end of all analyses. I'll leave this, just in case.
            # output to the finished file
            if not gaps:
                self.log(f'Discovered non-cleaned file at {a_file.shortpath_audio}; cleaning results', 'DEBUG')
                df.sort_values("start", inplace=True)
                df.to_csv(a_file.path_results_complete, index=False)
                os.remove(a_file.path_results_partial)
                a_file.chunklist = []
                return

        a_file.chunklist = gaps_to_chunklist(gaps, self.chunklength)
        return

    def queue_chunk(self, a_file, chunk: tuple[float, float]):
        sample_from = int(chunk[0] * a_file.track.samplerate)
        sample_to = int(chunk[1] * a_file.track.samplerate)
        read_size = sample_to - sample_from

        a_file.track.seek(sample_from)
        samples = a_file.track.read(read_size, dtype=np.float32)
        if a_file.track.channels > 1:
            samples = np.mean(samples, axis=1)

        n_samples = len(samples)

        if n_samples < read_size:
            self.handle_bad_read(a_file)
            chunk = (chunk[0], round(chunk[0] + (n_samples/a_file.track.samplerate), 1))
            continue_file = False
        else:
            continue_file = True

        samples = librosa.resample(y=samples, orig_sr=a_file.track.samplerate, target_sr=self.resample_rate)
        samples = tf.convert_to_tensor(samples, dtype=tf.float32)

        a_chunk = AssignChunk(file=a_file, chunk=chunk, samples=samples, last_chunk=not continue_file)
        self.coordinator.put_analyze(a_chunk)

        return continue_file

    def stream_to_queue(self, a_file: AssignFile):
        self._chunk_file(a_file)

        for chunk in a_file.chunklist:
            # reading and resampling can be very slow, so opportunistically bail mid-file
            # rather than committing to the next chunk's work
            if self.coordinator.event_exitanalysis.is_set():
                return
            continue_file = self.queue_chunk(a_file=a_file, chunk=chunk)
            if not continue_file:
                break

    def run(self):
        self.log('launching', 'INFO')
        while True:
            a_file = self.coordinator.get_stream()
            if a_file == 'exit':
                break

            self.log(f"buffering {a_file.shortpath_audio}", 'INFO')
            self.stream_to_queue(a_file)

        self.log("terminating", 'INFO')
