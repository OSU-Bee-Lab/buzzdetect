import logging
import multiprocessing
import os
import re
from _queue import Empty
from queue import Queue, Full

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf

import buzzcode.config as cfg
import threading
from buzzcode.analysis.assignments import AssignLog, AssignStream, AssignAnalyze, AssignWrite, loglevels
from buzzcode.analysis.formatting import format_activations, format_detections
from buzzcode.analysis.results_coverage import melt_coverage, get_gaps, smooth_gaps, gaps_to_chunklist
from buzzcode.audio import mark_eof, get_duration
from buzzcode.models.load_model import load_model
from buzzcode.utils import Timer, build_ident, shortpath, search_dir


class FilterDropProgress(logging.Filter):
    def filter(self, record):
        return record.levelno != 'PROGRESS'

class ExitSignal:
    def __init__(self, message, level, end_reason):
        self.message = message
        self.level = level
        self.end_reason = end_reason

class Coordinator:
    """
    Manages the coordination of analysis processes.
    """
    def __init__(self,
                 analyzers_cpu: int,
                 analyzer_gpu: bool=False,
                 streamers_total: int=None,
                 depth: int=None,
                 q_gui: Queue[AssignLog]=None,
                 event_analysisdone: multiprocessing.Event=None,
                 q_earlyexit: multiprocessing.Queue=None,):

        self.analyzers_cpu = analyzers_cpu
        self.analyzer_gpu = analyzer_gpu

        self.analyzers_total = analyzers_cpu + analyzer_gpu
        self.streamers_total = self._setup_streamers(self.analyzers_total) if streamers_total is None else streamers_total

        self.queue_depth = self._setup_depth() if depth is None else depth
        self.q_gui = q_gui

        self.q_log = Queue()
        self.q_stream = Queue()
        self.q_analyze = Queue(maxsize=self.queue_depth)
        self.q_write = Queue()

        self.streamers_done = threading.Event()
        self.analyzers_done = threading.Event()
        self.writer_done = threading.Event()

        # these arguments are for running the analysis as a process under the GUI; if they aren't set,
        # just initialize as non-multiprocessing versions
        self.event_exitanalysis = event_analysisdone if event_analysisdone is not None else threading.Event()
        self.q_earlyexit = q_earlyexit if q_earlyexit is not None else Queue()

        self.end_reason = None

    def log(self, msg, level_str):
        self.q_log.put(AssignLog(message=f'coordinator: {msg}', level_str=level_str))

    def _setup_streamers(self, n_analyzers):
        if self.analyzer_gpu:
            n_streamers = n_analyzers*8
        else:
            n_streamers = n_analyzers

        return n_streamers

    def _setup_depth(self):
        return self.streamers_total * 2

    def exit_analysis(self, exit_signal: ExitSignal):
        """ Note! Does not kill the logger; this needs to be done by analyzer process after cleanup """
        self.q_log.put(AssignLog(message=exit_signal.message, level_str=exit_signal.level))
        self.end_reason = exit_signal.end_reason
        self.event_exitanalysis.set()

    # pass threads as arguments because they're defined after the coordinator is initialized
    def wait_for_exit(self,
                      threads_streamers: list[threading.Thread],
                      threads_analyzers: list[threading.Thread],
                      thread_writer: threading.Thread,
                      ):

        def watch_workers():
            for t in threads_streamers:
                t.join()
            self.log('streamers done', 'DEBUG')
            self.streamers_done.set()

            for t in threads_analyzers:
                t.join()
            self.log('analyzers done', 'DEBUG')
            self.analyzers_done.set()

            thread_writer.join()
            self.log('writer done', 'DEBUG')
            self.writer_done.set()

            self.exit_analysis(ExitSignal(message='Analysis complete', level='INFO', end_reason='completed'))

        def watch_queue():
            exit_message = self.q_earlyexit.get()
            self.exit_analysis(ExitSignal(message=exit_message, level='WARNING', end_reason='interrupted'))

        thread_worker = threading.Thread(target=watch_workers, daemon=True)
        thread_worker.start()

        thread_exit = threading.Thread(target=watch_queue, daemon=True)
        thread_exit.start()

        self.event_exitanalysis.wait()


class WorkerLogger:
    def __init__(self, 
                 path_log,
                 coordinator: Coordinator,
                 verbosity_print: str='PROGRESS',
                 verbosity_log: str="DEBUG",
                 log_progress: bool=False):

        self.path_log = path_log
        self.coordinator = coordinator
        self.verbosity_print_int = loglevels[verbosity_print]

        self.log = logging.getLogger('buzzdetect')
        self.log.setLevel('DEBUG')

        self.format_file = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        self.handle_file = logging.FileHandler(path_log)
        self.handle_file.setLevel(verbosity_log)
        if not log_progress:
            self.handle_file.addFilter(FilterDropProgress())
        self.handle_file.setFormatter(self.format_file)
        self.log.addHandler(self.handle_file)

        self.handle_console = logging.StreamHandler()
        self.handle_console.setLevel(self.verbosity_print_int)
        self.log.addHandler(self.handle_console)

    def __call__(self):
        self.run()

    def write_log(self, a_log):
        self.log.log(msg=a_log.message, level=a_log.level_int)
        if self.coordinator.q_gui is not None and a_log.level_int >= self.verbosity_print_int:
            self.coordinator.q_gui.put(a_log)

    def run(self):
        a_log = self.coordinator.q_log.get()
        while not a_log.terminate:
            self.write_log(a_log)
            a_log = self.coordinator.q_log.get()

        self.write_log(AssignLog(message='logger closing', level_str='DEBUG'))


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
            self.coordinator.exit_analysis(
                ExitSignal(
                    message=f"Exiting analysis: no compatible audio files found in raw directory {self.dir_audio}.\n"
                            f"audio format must be one of: \n{', '.join(sf.available_formats().keys())}",
                    level='WARNING',
                    end_reason='no files'
                )
            )

            return


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
            self.coordinator.exit_analysis(
                ExitSignal(
                    message=f"All files in {self.dir_audio} are fully analyzed; exiting analysis",
                    level='INFO',
                    end_reason='fully analyzed'
                )
            )
            return

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

        return

    def cleanup(self):
        for file_in in self.files_in:
            self._chunk_file(file_in)


class WorkerStreamer:
    def __init__(self,
                 id_streamer,
                 resample_rate,
                 coordinator: Coordinator,):

        self.id_streamer = id_streamer
        self.resample_rate = resample_rate
        self.coordinator = coordinator


    def __call__(self):
        self.run()

    def log(self, msg, level_str):
        self.coordinator.q_log.put(AssignLog(message=f'streamer {self.id_streamer}: {msg}', level_str=level_str))

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
            while not self.coordinator.event_exitanalysis.is_set():
                try:
                    self.coordinator.q_analyze.put(a_analyze, timeout=3)
                    break
                except Full:
                    continue
            return abort_stream

        for chunk in a_stream.chunklist:
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
        while not a_stream.terminate:
            self.log(f"buffering {a_stream.shortpath}", 'INFO')
            keep_streaming = self.stream_to_queue(a_stream)
            if not keep_streaming:
                break

            a_stream = self.coordinator.q_stream.get()

        self.log("terminating", 'INFO')



class WorkerAnalyzer:
    def __init__(self,
                 id_analyzer,
                 processor: str,
                 modelname: str,
                 framehop_prop: float,
                 coordinator: Coordinator,):

        self.id_analyzer = id_analyzer
        self.processor = processor
        self.coordinator = coordinator

        self.model = load_model(modelname, framehop_prop, initialize=False)
        self.timer_analysis = Timer()
        self.timer_bottleneck = Timer()


    def __call__(self):
        self.run()

    def log(self, msg, level_str):
        self.coordinator.q_log.put(AssignLog(message=f'analyzer {self.id_analyzer}: {msg}', level_str=level_str))

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

    def report_bottleneck(self):
        msg = f"BUFFER BOTTLENECK: analyzer {self.id_analyzer} received assignment after {self.timer_bottleneck.get_total().__round__(1)}s"
        self.log(msg, 'DEBUG')

    def analyze_assignment(self, assignment: AssignAnalyze):
        results = self.model.predict(assignment.samples)

        self.coordinator.q_write.put(AssignWrite(path_audio=assignment.path_audio, chunk=assignment.chunk, results=results))
        self.report_rate(assignment.chunk, assignment.shortpath)


    def run(self):
        self.log('launching', 'INFO')
        self.model.initialize()

        self.timer_bottleneck.restart()
        while not self.coordinator.event_exitanalysis.is_set():
            try:
                assignment = self.coordinator.q_analyze.get(timeout=1)
                self.timer_bottleneck.stop()
                if self.timer_bottleneck.get_total() > 0.01:
                    self.report_bottleneck()
                self.analyze_assignment(assignment)
                self.timer_bottleneck.restart()
            except Empty:
                # if the streamers are done, exit as usual
                if self.coordinator.streamers_done.is_set():
                    self.log('all streamers done; terminating', 'DEBUG')
                    return

                # otherwise, try polling the queue again
                pass

        self.log("exit event set, terminating", 'DEBUG')


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
                 coordinator: Coordinator,):

        self.classes_out = classes_out
        self.threshold = threshold
        self.classes = classes
        self.framehop_s = framehop_s
        self.digits_time = digits_time
        self.dir_audio = dir_audio
        self.dir_out = dir_out
        self.digits_results = digits_results
        self.coordinator = coordinator

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
        self.coordinator.q_log.put(AssignLog(message=f'writer: {msg}', level_str=level_str))

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
        while not self.coordinator.event_exitanalysis.is_set():
            try:
                # we might poll the queue just before the analysisdone event is set,
                # so use a timeout to re-check
                self.write_results(self.coordinator.q_write.get(timeout=5))
            except Empty:
                if self.coordinator.analyzers_done.is_set():
                    self.log('all analyzers done; terminating', 'DEBUG')
                    return
                pass

        self.log("exit event set, terminating", 'DEBUG')

