from queue import Full

import librosa
import numpy as np
import soundfile as sf

from src import config as cfg
from src.pipeline.assignments import AssignFile, AssignChunk, AssignLog
from src.pipeline.coordination import Coordinator
from src.stream.audio import mark_eof


class WorkerStreamer:
    def __init__(self,
                 id_streamer,
                 resample_rate: float,
                 coordinator: Coordinator, ):

        self.id_streamer = id_streamer
        self.resample_rate = resample_rate
        self.coordinator = coordinator

    def __call__(self):
        self.run()

    def log(self, msg, level_str):
        self.coordinator.q_log.put(AssignLog(message=f'streamer {self.id_streamer}: {msg}', level_str=level_str))

    def handle_bad_read(self, track: sf.SoundFile, a_file: AssignFile):
        # we've found that many of our .mp3 files give an incorrect .frames count, or else headers are broken
        # this appears to be because our recorders ran out of battery while recording
        # SoundFile does not handle this gracefully, so we catch it here.

        final_frame = track.tell()
        mark_eof(path_audio=a_file.path_audio, final_frame=final_frame)

        final_second = final_frame/track.samplerate

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


    def stream_to_queue(self, a_file: AssignFile):
        track = sf.SoundFile(a_file.path_audio)
        samplerate_native = track.samplerate


        def queue_chunk(chunk, track, samplerate_native):
            sample_from = int(chunk[0] * samplerate_native)
            sample_to = int(chunk[1] * samplerate_native)
            read_size = sample_to - sample_from

            track.seek(sample_from)
            samples = track.read(read_size, dtype=np.float32)
            if track.channels > 1:
                samples = np.mean(samples, axis=1)

            n_samples = len(samples)

            if n_samples < read_size:
                self.handle_bad_read(track, a_file)
                chunk = (chunk[0], round(chunk[0] + (n_samples/track.samplerate), 1))
                abort_stream = True
            else:
                abort_stream = False

            samples = librosa.resample(y=samples, orig_sr=track.samplerate, target_sr=self.resample_rate)

            a_chunk = AssignChunk(file=a_file, chunk=chunk, samples=samples)
            while not self.coordinator.event_exitanalysis.is_set():
                try:
                    self.coordinator.q_analyze.put(a_chunk, timeout=3)
                    break
                except Full:
                    continue
            return abort_stream

        for chunk in a_file.chunklist:
            abort_stream = queue_chunk(chunk, track, samplerate_native)
            if abort_stream:
                return True # Note! This is whether to continue to *next file*, the abort for this file is handled by return
                # need to refactor to make clearer...a lot of these methods could be independent functions

            if self.coordinator.event_exitanalysis.is_set():
                self.log("exit event set, terminating", 'DEBUG')
                return False

        return True

    def run(self):
        self.log('launching', 'INFO')
        a_stream = self.coordinator.q_stream.get()
        while a_stream is not None:
            self.log(f"buffering {a_stream.shortpath_audio}", 'INFO')
            keep_streaming = self.stream_to_queue(a_stream)
            if not keep_streaming:
                break

            a_stream = self.coordinator.q_stream.get()

        self.log("terminating", 'INFO')
