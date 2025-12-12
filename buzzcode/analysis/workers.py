import logging
import multiprocessing
import os
import re
import threading
from _queue import Empty
from queue import Queue

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf

import buzzcode.config as cfg
from buzzcode.analysis.assignments import AssignLog, AssignStream, AssignAnalyze, AssignWrite, loglevels
from buzzcode.analysis.formatting import format_activations, format_detections
from buzzcode.analysis.results_coverage import melt_coverage, get_gaps, smooth_gaps, gaps_to_chunklist
from buzzcode.audio import mark_eof, get_duration
from buzzcode.models.load_model import load_model
from buzzcode.utils import Timer, build_ident, shortpath, search_dir


class FilterDropProgress(logging.Filter):
    def filter(self, record):
        return record.levelno != 'PROGRESS'


class Coordinator:
    """
    Manages the coordination of analysis processes.

    Note! multiprocessed workers still must take queues/events/threads as individuals arguments (rather than receiving the one coordinator)
    because threading queues cannot be pickled.
    """
    def __init__(self,
                 analyzers_cpu: int,
                 analyzer_gpu: bool=False,
                 streamers_total: int=None,
                 depth: int=None,
                 q_gui: Queue=None,
                 event_stopanalysis: multiprocessing.Event=None,
                 force_threads: bool = False,):

        self.analyzers_cpu = analyzers_cpu
        self.analyzer_gpu = analyzer_gpu

        self.analyzers_total = analyzers_cpu + analyzer_gpu
        self.analyzers_live = multiprocessing.Value('i', self.analyzers_total)
        self.streamers_total = self._setup_streamers(self.analyzers_total) if streamers_total is None else streamers_total
        self.streamers_live = multiprocessing.Value('i', self.streamers_total)

        self.queue_depth = self._setup_depth() if depth is None else depth
        self.q_gui = q_gui

        # these can be tweaked (don't forget to change corresponding queues),
        # but we find that leaving everything threaded is, strangely, the most
        # performant approach. The time to pickle and enqueue seems to outweigh
        # the savings of parallel processing. Leaving
        self.concurrency_logger = threading.Thread
        self.concurrency_streamer = threading.Thread
        self.concurrency_writer = threading.Thread
        self.concurrency_analyzer = threading.Thread

        self.q_log = Queue()
        self.q_stream = Queue()
        self.q_analyze = Queue(maxsize=self.queue_depth)
        self.q_write = Queue()

        self.event_stopanalysis = event_stopanalysis if event_stopanalysis is not None else multiprocessing.Event()
        self.event_analysisdone = multiprocessing.Event()
        self.event_closelogger = multiprocessing.Event()

        self.force_threads = force_threads

    def _setup_queues(self):
        if self.analyzer_gpu or self.force_threads:
            concurrency = Queue
        else:
            concurrency = multiprocessing.Queue

        q_stream = concurrency()  # don't set a max size here! These are just the paths to analyze; need to build all before analysis
        q_analyze = concurrency(maxsize=self.queue_depth)

        return q_stream, q_analyze

    def _setup_streamers(self, n_analyzers):
        if self.analyzer_gpu:
            n_streamers = n_analyzers*8
        else:
            n_streamers = n_analyzers

        return n_streamers

    def _setup_depth(self):
        return self.streamers_total * 2

    def early_exit(self, message, level):
        self.q_log.put(AssignLog(message=message, level_str=level))
        self.event_stopanalysis.set()


class WorkerLogger:
    def __init__(self, 
                 path_log,
                 q_log: Queue[AssignLog],
                 event_analysisdone: multiprocessing.Event,
                 event_stopanalysis: multiprocessing.Event,
                 verbosity_print: str='PROGRESS',
                 verbosity_log: str="DEBUG",
                 log_progress: bool=False,
                 q_gui: Queue[AssignLog]=None,):
        self.path_log = path_log

        self.verbosity_print_int = loglevels[verbosity_print]
        
        self.q_log = q_log
        self.q_gui = q_gui
        self.event_analysisdone = event_analysisdone
        self.event_stopanalysis = event_stopanalysis

        self.log = logging.getLogger()
        self.log.setLevel(0)

        self.format_file = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        self.handle_file = logging.FileHandler(path_log)
        self.handle_file.setLevel(verbosity_log)

        if not log_progress:
            self.handle_file.addFilter(FilterDropProgress())

        self.handle_file.setFormatter(self.format_file)
        self.log.addHandler(self.handle_file)

    def __call__(self):
        self.run()

    def write_log(self, time):
        a_log = self.q_log.get(timeout=time)
        logging.log(msg=a_log.message, level=a_log.level_int)
        if self.q_gui is not None and a_log.level_int >= self.verbosity_print_int:
            self.q_gui.put(a_log)


    def run(self):
        # Main loop: drain while work is still coming or writers alive
        while True:
            try:
                self.write_log(1)
            except Empty:
                if self.event_analysisdone.is_set():
                    logging.debug('logger: queue is empty and event_analysisdone is set; breaking into final getwrite loop')
                    break

                if self.event_stopanalysis.is_set():
                    logging.debug('logger: queue is empty and event_stopanalysis is set; breaking into final getwrite loop')
                    break
                # else: another writer might still put, so continue

        # Just in case there was a race condition, try a final draining loop.
        # at this point, the workers are all closed, so a limit of 5 seconds should
        # be plenty to avoid early exits
        while True:
            try:
                self.write_log(5)
            except Empty:
                break

        logging.info("logger: exiting")

class FileIn:
    def __init__(self, path_audio, dir_audio, dir_out):
        self.path_audio = path_audio
        self.dir_audio = dir_audio
        self.dir_out = dir_out

        self.path_audio_short = shortpath(self.path_audio, self.dir_audio)

        self.ident = build_ident(path_audio, dir_audio)
        self.conflicting_ident = None

        self.path_out_partial = os.path.join(self.dir_out, self.ident + cfg.SUFFIX_RESULT_PARTIAL)
        self.path_out_complete = os.path.join(self.dir_out, self.ident + cfg.SUFFIX_RESULT_COMPLETE)

        self.duration_audio = None


class WorkerChecker:
    def __init__(self,
                 dir_audio: str,
                 dir_out: str,
                 framelength_s: float,
                 chunklength: float,
                 coordinator: Coordinator,
                 ):

        self.dir_audio = dir_audio
        self.dir_out = dir_out
        self.framelength_s = framelength_s
        self.chunklength = chunklength
        self.coordinator = coordinator

        self.files_in = self._build_inputs()


    def _build_inputs(self):
        paths_in = search_dir(self.dir_audio, extensions=list(sf.available_formats().keys()))
        files_in = [FileIn(p, self.dir_audio, self.dir_out) for p in paths_in]

        # check for conflicting idents; i.e., two files have identical names but different extensions
        # causes results to be written to the same file.
        idents = [f.ident for f in files_in]

        idents_conflicting = []
        for file_in in files_in:
            if idents.count(file_in.ident) > 1:
                idents_conflicting.append(file_in.ident)
                file_in.conflicting_ident = True
            else:
                file_in.conflicting_ident = False

        for ident_conflicting in set(idents_conflicting):
            paths_conflicting = [f.path_audio for f in files_in if f.ident == ident_conflicting]
            msg = (f'The following files have conflicting names and will be skipped:\n'
                   f'{', '.join(paths_conflicting)}\n'
                   f'These files must be renamed before they can be analyzed.')
            self.coordinator.q_log.put(AssignLog(msg, 'WARNING'))

        return [f for f in files_in if not f.conflicting_ident]


    def _chunk_file(self, file_in: FileIn):
        self.coordinator.q_log.put(AssignLog(message=f'Checking results for {file_in.path_audio_short}', level_str='INFO'))

        if os.path.exists(file_in.path_out_complete):
            self.coordinator.q_log.put(AssignLog(f'Skipping {file_in.path_audio_short}; fully analyzed', 'DEBUG'))
            return None


        if os.path.getsize(file_in.path_audio) < cfg.FILE_SIZE_MINIMUM:
            self.coordinator.q_log.put(AssignLog(
                message=f'Skipping {file_in.path_audio_short}; below minimum analyzeable size', level_str='INFO'))
            return None

        if file_in.duration_audio is None:
            file_in.duration_audio = get_duration(file_in.path_audio, q_log=self.coordinator.q_log)

        # if the file hasn't been started, chunk the whole file
        if not os.path.exists(file_in.path_out_partial):
            gaps = [(0, file_in.duration_audio)]
        else:
            # otherwise, read the file and calculate chunks
            df = pd.read_csv(file_in.path_out_partial)
            coverage = melt_coverage(df, self.framelength_s)
            gaps = get_gaps(
                range_in=(0, file_in.duration_audio),
                coverage_in=coverage
            )
            gaps = smooth_gaps(
                gaps,
                range_in=(0, file_in.duration_audio),
                framelength=self.framelength_s,
                gap_tolerance=self.framelength_s / 4
            )

            # if we find no gaps, this file was actually finished and never cleaned!
            # output to the finished file
            if not gaps:
                df.sort_values("start", inplace=True)
                df.to_csv(file_in.path_out_complete, index=False)
                os.remove(file_in.path_out_partial)
                return None

        return gaps_to_chunklist(gaps, self.chunklength)

    def queue_assignments(self):
        # exit if no compatible files
        if not self.files_in:
            alog = AssignLog(
                message=f"Exiting analysis: no compatible audio files found in raw directory {self.dir_audio}.\n"
                f"audio format must be one of: \n{'\n'.join(sf.available_formats().keys())}",
                level_str='WARNING'
            )

            self.coordinator.early_exit(
                message=alog.message,
                level='ERROR'
            )

            return False


        assignments_stream = []
        for file_in in self.files_in:
            c = self._chunk_file(file_in)
            if c is not None:
                assignments_stream.append(
                    AssignStream(
                        path_audio=file_in.path_audio,
                        duration_audio=file_in.duration_audio,
                        dir_audio=self.dir_audio,
                        chunklist=c,
                        terminate=False
                    )
                )

        # exit if no files with chunks
        if not assignments_stream:
            self.coordinator.q_log.put(
                AssignLog(message=f'All files in {self.dir_audio} are fully analyzed; exiting analysis',
                          level_str='WARNING')
            )

            self.coordinator.early_exit(
                message=f'All files in {self.dir_audio} are fully analyzed; exiting analysis',
                level='WARNING'
            )

            self.coordinator.event_analysisdone.set()
            return False

        # else, queue 'em up
        for a_stream in assignments_stream:
            self.coordinator.q_stream.put(a_stream)

        for _ in range(self.coordinator.streamers_total):
            self.coordinator.q_stream.put(
                AssignStream(
                    path_audio=None,
                    dir_audio=None,
                    duration_audio=None,
                    chunklist=None,
                    terminate=True
                )
            )

        return True

    def cleanup(self):
        for file_in in self.files_in:
            self._chunk_file(file_in)


class WorkerStreamer:
    def __init__(self,
                 id_streamer,
                 resample_rate,
                 streamers_live: multiprocessing.Value,
                 q_stream: Queue,
                 q_analyze: Queue,
                 q_log: Queue,
                 event_stopanalysis: multiprocessing.Event,):
        self.id_streamer = id_streamer
        self.resample_rate = resample_rate

        self.q_log = q_log
        self.q_stream = q_stream
        self.q_analyze = q_analyze
        self.event_stopanalysis = event_stopanalysis
        self.streamers_live = streamers_live

    def __call__(self):
        self.run()

    def log(self, msg, level_str):
        self.q_log.put(AssignLog(message=f'streamer {self.id_streamer}: {msg}', level_str=level_str))

    def handle_bad_read(self, track: sf.SoundFile, a_stream: AssignStream):
        # we've found that many of our .mp3 files give an incorrect .frames count, or else headers are broken
        # this appears to be because our recorders ran out of battery while recording
        # SoundFile does not handle this gracefully, so we catch it here.

        final_frame = track.tell()
        mark_eof(path_audio=a_stream.path_audio, final_frame=final_frame)

        final_second = final_frame/track.samplerate

        msg = f"Unreadable audio at {round(final_second, 1)}s out of {round(a_stream.duration_audio, 1)}s for {a_stream.shortpath}."
        if 1 - (final_second/a_stream.duration_audio) > cfg.BAD_READ_ALLOWANCE:
            # if we get a bad read in the middle of a file, this deserves a warning.
            level = 'WARNING'
            msg += '\nAborting early due to corrupt audio data.'
        else:
            # but bad reads at the ends of files are almost guaranteed when the batteries run out
            level = 'DEBUG'
            msg += '\nBad audio is near file end, results should be mostly unaffected.'

        self.log(msg, level)


    def stream_to_queue(self, a_stream: AssignStream):
        track = sf.SoundFile(a_stream.path_audio)
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
                self.handle_bad_read(track, a_stream)
                chunk = (chunk[0], round(chunk[0] + (n_samples/track.samplerate), 1))
                abort_stream = True
            else:
                abort_stream = False

            samples = librosa.resample(y=samples, orig_sr=track.samplerate, target_sr=self.resample_rate)
            a_analyze = AssignAnalyze(path_audio=a_stream.path_audio, shortpath=a_stream.shortpath, chunk=chunk, samples=samples)

            self.q_analyze.put(a_analyze)
            return abort_stream

        for chunk in a_stream.chunklist:
            abort_stream = queue_chunk(chunk, track, samplerate_native)
            if abort_stream:
                return

            if self.event_stopanalysis.is_set():
                self.log('stop analysis event is set, terminating', 'DEBUG')
                return

    def run(self):
        self.log('launching', 'INFO')
        a_stream = self.q_stream.get()
        while not a_stream.terminate and not self.event_stopanalysis.is_set():
            self.log(f"buffering {a_stream.shortpath}", 'INFO')

            self.stream_to_queue(a_stream)
            a_stream = self.q_stream.get()

        self.log("terminating", 'INFO')

        # Decrement streamer count
        with self.streamers_live.get_lock():
            self.streamers_live.value -= 1
            self.log(f"after terminating, streamer count is {self.streamers_live.value}", 'DEBUG')


class WorkerWriter:
    def __init__(self,
                 classes_out,
                 threshold,
                 classes,
                 framehop_s,
                 digits_time,
                 dir_audio,
                 dir_out,
                 digits_results,
                 q_write: Queue,
                 q_log: Queue,
                 event_stopanalysis: multiprocessing.Event,
                 event_analysisdone: multiprocessing.Event,
                 analyzers_live: multiprocessing.Value, ):

        self.classes_out = classes_out
        self.threshold = threshold
        self.classes = classes
        self.framehop_s = framehop_s
        self.digits_time = digits_time
        self.dir_audio = dir_audio
        self.dir_out = dir_out
        self.digits_results = digits_results

        self.q_log = q_log
        self.q_write = q_write
        self.event_stopanalysis = event_stopanalysis
        self.event_analysisdone = event_analysisdone
        self.analyzers_live = analyzers_live

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

    def log(self, msg, level_str):
        self.q_log.put(AssignLog(message=f'writer: {msg}', level_str=level_str))

    def write_results(self, assignment: AssignWrite):
        output = self.format(
            results=assignment.results.numpy(),
            time_start=assignment.chunk[0]
        )

        base_out = os.path.splitext(assignment.path_audio)[0]
        base_out = re.sub(self.dir_audio, self.dir_out, base_out)
        path_out = base_out + cfg.SUFFIX_RESULT_PARTIAL

        os.makedirs(os.path.dirname(path_out), exist_ok=True)

        # Check if file exists to determine if we need to write headers
        file_exists = os.path.exists(path_out)

        # Append to existing file or create new one with headers
        output.to_csv(path_out, mode='a', header=not file_exists, index=False)

    def run(self):
        self.log('launching', 'INFO')
        while True:
            if self.event_stopanalysis.is_set():
                self.log('stop analysis event is set, terminating', 'DEBUG')
                break
            try:
                self.write_results(self.q_write.get(timeout=5))
            except Empty:
                with self.analyzers_live.get_lock():
                    if self.analyzers_live.value == 0:
                        break
                    else:
                        self.log('Write queue is empty, but workers are active. Continuing.', 'DEBUG')
                        continue

        self.log(msg="terminating", level_str='INFO')
        self.event_analysisdone.set()


class WorkerAnalyzer:
    def __init__(self,
                 id_analyzer,
                 processor: str,
                 modelname: str,
                 framehop_prop: float,
                 q_log: Queue,
                 q_write: Queue,
                 q_analyze: Queue,
                 event_stopanalysis: multiprocessing.Event,
                 streamers_live: multiprocessing.Value,
                 analyzers_live: multiprocessing.Value,):

        self.id_analyzer = id_analyzer
        self.processor = processor
        self.model = load_model(modelname, framehop_prop, initialize=False)
        self.timer_analysis = Timer()
        self.timer_bottleneck = Timer()

        self.q_log = q_log
        self.q_write = q_write
        self.q_analyze = q_analyze
        self.event_stopanalysis = event_stopanalysis
        self.streamers_live = streamers_live
        self.analyzers_live = analyzers_live

    def __call__(self):
        self.run()

    def log(self, msg, level_str):
        self.q_log.put(AssignLog(message=f'analyzer {self.id_analyzer}: {msg}', level_str=level_str))

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
                self.log("GPU not found; using CPU", 'WARNING')
                self.processor = 'CPU'
            else:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

            if len(gpus) > 1:
                self.log("Use of multiple GPUs is untested", 'WARNING')

        self.log(f"processing on {self.processor}", 'INFO')


    def report_rate(self, chunk, path_short):
        chunk_duration = chunk[1] - chunk[0]

        self.timer_analysis.stop()
        analysis_rate = (chunk_duration / self.timer_analysis.get_total(5)).__round__(1)

        msg = (f"analyzed {path_short}, chunk ({float(chunk[0])}, {float(chunk[1])}) "
                 f"in {self.timer_analysis.get_total()}s (rate: {analysis_rate})")

        self.log(msg, 'PROGRESS')
        self.timer_analysis.restart()

    def analyze_assignment(self, assignment: AssignAnalyze):
        results = self.model.predict(assignment.samples)

        self.q_write.put(AssignWrite(path_audio=assignment.path_audio, chunk=assignment.chunk, results=results))
        self.report_rate(assignment.chunk, assignment.shortpath)


    def run(self):
        self.log('launching', 'INFO')
        self.model.initialize()
        def loop_waiting():
            self.timer_bottleneck.restart()
            while True:
                try:
                    assignment = self.q_analyze.get(timeout=1)
                    self.timer_bottleneck.stop()
                    msg = f"BUFFER BOTTLENECK: analyzer {self.id_analyzer} received assignment after {self.timer_bottleneck.get_total().__round__(1)}s"
                    self.log(msg, 'DEBUG')
                    self.analyze_assignment(assignment)
                    return 'running'
                except Empty:
                    if self.event_stopanalysis.is_set():
                        self.log('stop analysis event is set, terminating', 'DEBUG')
                        return 'TERMINATE'
                    # Check if streamers are still active using shared counter
                    with self.streamers_live.get_lock():
                        if self.streamers_live.value > 0:
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
        self.log("terminating", 'INFO')

        # Decrement analyzer count
        with self.analyzers_live.get_lock():
            self.analyzers_live.value -= 1
            self.log(f"after exiting, analyzer count is {self.analyzers_live.value}", 'DEBUG')

