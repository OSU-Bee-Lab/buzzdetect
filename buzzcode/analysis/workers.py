import logging
import os
import re
import warnings
from _queue import Empty
from queue import Queue

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf

from buzzcode.analysis.assignments import AssignLog, AssignStream, AssignAnalyze, AssignWrite
from buzzcode.analysis.formatting import format_activations, format_detections
from buzzcode.audio import handle_badread
from buzzcode.config import SUFFIX_RESULT_PARTIAL
from buzzcode.models.load_model import load_model
from buzzcode.utils import Timer


# source code modified from Sergey Pleshakov's code here: https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
# might also consider loguru
class PrintFormatter(logging.Formatter):
    gray = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "[%(levelname)s] %(message)s"

    FORMATS = {
        logging.DEBUG: gray + format + reset,
        logging.INFO: gray + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

class WorkerLogger:
    def __init__(self, path_log, q_log: Queue[AssignLog], event_closelogger, verbosity):
        self.path_log = path_log
        self.q_log = q_log
        self.event_closelogger = event_closelogger
        self.verbosity = verbosity

        self.log = logging.getLogger()
        self.log.setLevel(0)

        self.handle_stream = logging.StreamHandler()
        self.handle_stream.setLevel(verbosity)
        self.handle_stream.setFormatter(PrintFormatter())
        self.log.addHandler(self.handle_stream)

        self.format_file = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        self.handle_file = logging.FileHandler(path_log)
        self.handle_file.setLevel(logging.DEBUG)  # always write everything to file
        self.handle_file.setFormatter(self.format_file)
        self.log.addHandler(self.handle_file)

    def __call__(self):
        self.run()

    def getwrite(self, time):
        a_log = self.q_log.get(timeout=time)
        logging.log(msg=a_log.item, level=a_log.level_int)

    def run(self):
        # Main loop: drain while work is still coming or writers alive
        while True:
            try:
                self.getwrite(1)
            except Empty:
                if self.event_closelogger.is_set():
                    break
                # else: another writer might still put, so continue

        # Final catch-up: keep draining until there's a real timeout
        while True:
            try:
                self.getwrite(1)
            except Empty:
                break


        logging.info("logger: exiting")


class WorkerStreamer:
    def __init__(self, id_streamer, streamer_count, q_stream: Queue[AssignStream], q_analyze: Queue[AssignAnalyze], q_log, resample_rate):
        self.id_streamer = id_streamer
        self.streamer_count = streamer_count
        self.q_stream = q_stream
        self.q_assignments = q_analyze
        self.q_log = q_log
        self.resample_rate = resample_rate

    def __call__(self):
        self.run()

    def stream_to_queue(self, a_stream: AssignStream):
        track = sf.SoundFile(a_stream.path_audio)
        samplerate_native = track.samplerate

        def queue_assignment(chunk, track, samplerate_native):
            sample_from = int(chunk[0] * samplerate_native)
            sample_to = int(chunk[1] * samplerate_native)
            read_size = sample_to - sample_from

            track.seek(sample_from)
            samples = track.read(read_size, dtype=np.float32)
            if track.channels > 1:
                samples = np.mean(samples, axis=1)

            # we've found that many of our .mp3 files give an incorrect .frames count, or else headers are broken
            # this appears to be because our recorders ran out of battery while recording
            # SoundFile does not handle this gracefully, so we catch it here.
            n_samples = len(samples)

            if n_samples < read_size:
                handle_badread(path_audio=a_stream.path_audio, track=track, duration_audio=a_stream.duration_audio, end_intended=chunk[1])

            samples = librosa.resample(y=samples, orig_sr=track.samplerate, target_sr=self.resample_rate)
            a_analyze = AssignAnalyze(path_audio=a_stream.path_audio, chunk=chunk, samples=samples)

            self.q_assignments.put(a_analyze)

        for chunk in a_stream.chunklist:
            queue_assignment(chunk, track, samplerate_native)

    def run(self):
        a_stream = self.q_stream.get()
        while not a_stream.terminate:
            self.q_log.put(AssignLog(msg=f"streamer {self.id_streamer}: buffering {a_stream.shortpath}", level='INFO'))

            self.stream_to_queue(a_stream)
            a_stream = self.q_stream.get()
        self.q_log.put(AssignLog(msg=f"streamer {self.id_streamer}: terminating", level='INFO'))

        # Decrement streamer count
        with self.streamer_count.get_lock():
            self.streamer_count.value -= 1

class WorkerWriter:
    def __init__(self, classes_out, threshold, analyzer_count, q_write: Queue[AssignWrite], classes, framehop_s, digits_time,
                 dir_audio, dir_out, q_log, event_analysisdone, digits_results):
        # TODO: what if someone starts an analysis with activations, then finishes with detections?
        if classes_out is not None and threshold is not None:
            raise ValueError("cannot specify both classes_out and threshold")

        if classes_out is None and threshold is None:
            raise ValueError("must specify either classes_out or threshold")

        self.classes_out = classes_out
        self.threshold = threshold
        self.analyzer_count = analyzer_count
        self.q_write = q_write
        self.classes = classes
        self.framehop_s = framehop_s
        self.digits_time = digits_time
        self.dir_audio = dir_audio
        self.dir_out = dir_out
        self.q_log = q_log
        self.shutdown_event = event_analysisdone
        self.digits_results = digits_results

        if classes_out is not None:
            def format_func(results, time_start):
                out = format_activations(
                    results=results,
                    classes=classes,
                    framehop_s=framehop_s,
                    time_start=time_start,
                    digits_time=digits_time,
                    classes_keep=classes_out,
                    digits_results=digits_results
                )

                return out

        else:
            def format_func(results, time_start):
                out = format_detections(
                    results,
                    threshold,
                    classes,
                    framehop_s,
                    digits_time,
                    time_start
                )

                return out

        self.format = format_func

    def __call__(self):
        self.run()

    def write_results(self, assignment: AssignWrite):
        output = self.format(
            results=assignment.results.numpy(),
            time_start=assignment.chunk[0]
        )

        base_out = os.path.splitext(assignment.path_audio)[0]
        base_out = re.sub(self.dir_audio, self.dir_out, base_out)
        path_out = base_out + SUFFIX_RESULT_PARTIAL

        os.makedirs(os.path.dirname(path_out), exist_ok=True)

        # Check if file exists to determine if we need to write headers
        file_exists = os.path.exists(path_out)

        # Append to existing file or create new one with headers
        output.to_csv(path_out, mode='a', header=not file_exists, index=False)

    def run(self):
        while True:
            try:
                self.write_results(self.q_write.get(timeout=5))
            except Empty:
                with self.analyzer_count.get_lock():
                    current_analyzer_count = self.analyzer_count.value

                if current_analyzer_count == 0:
                    break
                else:
                    continue

        self.q_log.put(AssignLog(msg="writer: terminating", level='INFO'))
        self.shutdown_event.set()



class WorkerAnalyzer:
    def __init__(self, id_analyzer, processor, modelname, framehop_prop, q_write: Queue[AssignWrite], q_analyze: Queue[
        AssignAnalyze], q_log, streamer_count, analyzer_count, dir_audio):
        self.id_analyzer = id_analyzer
        self.processor = processor
        self.modelname = modelname
        self.framehop_prop = framehop_prop
        self.q_write = q_write
        self.q_analyze = q_analyze
        self.q_log = q_log
        self.streamer_count = streamer_count
        self.analyzer_count = analyzer_count
        self.dir_audio = dir_audio

        self.timer_analysis = Timer()
        self.timer_bottleneck = Timer()

        self.model = None


    def __call__(self):
        self.run()

    def _initialize_model(self):
        # lazy load because models can't be pickled
        self.model = load_model(self.modelname, framehop_prop=self.framehop_prop, initialize=True)

    def _managememory(self):
        if self.processor == 'CPU':
            tf.config.set_visible_devices([], 'GPU')
            visible_devices = tf.config.get_visible_devices()
            for device in visible_devices:
                assert device.device_type != 'GPU'
        elif self.processor == 'GPU':
            # let memory grow when processing on GPU
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if not gpus:
                warnings.warn("GPU not found; using CPU")
                self.processor = 'CPU'
            else:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

            if len(gpus) > 1:
                warnings.warn("multi-processing on GPU is not yet explicitly supported; ")

        self.q_log.put(AssignLog(msg=f"analyzer {self.id_analyzer}: processing on {self.processor}", level='INFO'))


    def report_rate(self, chunk, path_audio):
        chunk_duration = chunk[1] - chunk[0]

        self.timer_analysis.stop()
        analysis_rate = (chunk_duration / self.timer_analysis.get_total()).__round__(1)
        path_short = re.sub(self.dir_audio, '', path_audio)

        msg = (f"analyzer {self.id_analyzer}: analyzed {path_short}, "
                 f"chunk ({float(chunk[0])}, {float(chunk[1])}) "
                 f"in {self.timer_analysis.get_total()}s (rate: {analysis_rate})")

        self.q_log.put(AssignLog(msg=msg, level='INFO'))

        self.timer_analysis.restart()

    def analyze_assignment(self, assignment: AssignAnalyze):
        results = self.model.predict(assignment.samples)

        self.q_write.put(AssignWrite(path_audio=assignment.path_audio, chunk=assignment.chunk, results=results))
        self.report_rate(assignment.chunk, assignment.path_audio)


    def run(self):
        self._initialize_model()
        def loop_waiting():
            self.timer_bottleneck.restart()
            while True:
                try:
                    assignment = self.q_analyze.get(timeout=5)
                    self.timer_bottleneck.stop()
                    msg = f"BUFFER BOTTLENECK: analyzer {self.id_analyzer} received assignment after {self.timer_bottleneck.get_total().__round__(1)}s"
                    self.q_log.put(AssignLog(msg=msg, level='DEBUG'))
                    self.analyze_assignment(assignment)
                    return 'running'
                except Empty:
                    # Check if streamers are still active using shared counter
                    with self.streamer_count.get_lock():
                        current_streamer_count = self.streamer_count.value

                    if current_streamer_count > 0:
                        continue
                    else:
                        return 'TERMINATE'

        while True:
            try:
                self.analyze_assignment(self.q_analyze.get(timeout=0))
            except Empty:
                state = loop_waiting()
                if state == 'TERMINATE':
                    break
        self.q_log.put(AssignLog(msg=f"analyzer {self.id_analyzer}: terminating", level='INFO'))

        # Decrement analyzer count
        with self.analyzer_count.get_lock():
            self.analyzer_count.value -= 1


