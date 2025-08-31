import logging
import os
import re
from _queue import Empty

import tensorflow as tf
import warnings

from buzzcode.analysis.analysis import format_activations, format_detections
from buzzcode.analysis.models import load_model
from buzzcode.audio import stream_to_queue
from buzzcode.config import SUFFIX_RESULT_PARTIAL
from buzzcode.embedding.load_embedder import load_embedder
from buzzcode.utils import Timer

log_levels = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL}

def printlog(item: str, q_log, level):
    if level.__class__ is str:
        level_int = log_levels[level]
    elif level.__class__ is int:
        level_int = level
    else:
        raise ValueError(f"level must be str or int, not {level.__class__}")

    q_log.put((item, level_int))

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

def worker_logger(path_log, q_log, shutdown_event, verbosity):
    log = logging.getLogger()
    log.setLevel(0)

    handle_stream = logging.StreamHandler()
    handle_stream.setLevel(verbosity)
    handle_stream.setFormatter(PrintFormatter())
    log.addHandler(handle_stream)

    format_file = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handle_file = logging.FileHandler(path_log)
    handle_file.setLevel(logging.DEBUG)  # always write everything to file
    handle_file.setFormatter(format_file)
    log.addHandler(handle_file)

    def getwrite(t):
        item = q_log.get(timeout=t)
        logging.log(msg=item[0], level=item[1])

    # Main loop: drain while work is still coming or writers alive
    while True:
        try:
            getwrite(1)
        except Empty:
            if shutdown_event.is_set():
                break
            # else: another writer might still put, so continue

    # Final catch-up: keep draining until there's a real timeout
    while True:
        try:
            getwrite(1)
        except Empty:
            break


    logging.info("logger: exiting")

def worker_streamer(id_streamer, streamer_count, q_control, q_assignments, q_log, resample_rate):
    c = q_control.get()
    while c != 'TERMINATE':
        printlog(f"streamer {id_streamer}: buffering {c['path_audio']}", q_log=q_log, level='INFO')
        stream_to_queue(
            path_audio=c['path_audio'],
            duration_audio=c['duration_audio'],
            chunklist=c['chunklist'],
            q_assignments=q_assignments,
            resample_rate=resample_rate
        )
        c = q_control.get()

    printlog(f"streamer {id_streamer}: terminating", q_log=q_log, level='INFO')

    # Decrement streamer count
    with streamer_count.get_lock():
        streamer_count.value -= 1


def worker_writer(classes_out, threshold, analyzer_count, q_write, classes, framehop_s, digits_time,
                  dir_audio, dir_out, q_log, shutdown_event, digits_results=2):
    # TODO: what if someone starts an analysis with activations, then finishes with detections?
    if classes_out is not None and threshold is not None:
        raise ValueError("cannot specify both classes_out and threshold")

    if classes_out is None and threshold is None:
        raise ValueError("must specify either classes_out or threshold")

    if classes_out is not None:
        def format_func(results, time_start):
            output = format_activations(
                results=results,
                classes=classes,
                framehop_s=framehop_s,
                time_start=time_start,
                digits_time=digits_time,
                classes_keep=classes_out,
                digits_results=digits_results
            )

            return output

    else:
        def format_func(results, time_start):
            output = format_detections(
                results,
                threshold,
                classes,
                framehop_s,
                digits_time,
                time_start
            )

            return output

    def write_results(path_audio, chunk, results):
        output = format_func(
            results=results.numpy(),
            time_start=chunk[0]
        )

        base_out = os.path.splitext(path_audio)[0]
        base_out = re.sub(dir_audio, dir_out, base_out)
        path_out = base_out + SUFFIX_RESULT_PARTIAL

        os.makedirs(os.path.dirname(path_out), exist_ok=True)

        # Check if file exists to determine if we need to write headers
        file_exists = os.path.exists(path_out)

        # Append to existing file or create new one with headers
        output.to_csv(path_out, mode='a', header=not file_exists, index=False)

    while True:
        try:
            w = q_write.get(timeout=5)
            write_results(
                path_audio=w['path_audio'],
                chunk=w['chunk'],
                results=w['results']
            )
        except Empty:
            with analyzer_count.get_lock():
                current_analyzer_count = analyzer_count.value

            if current_analyzer_count == 0:
                break
            else:
                continue

    printlog("writer: terminating", q_log=q_log, level='INFO')
    shutdown_event.set()


def worker_analyzer(id_analyzer, processor, modelname, embeddername, framehop_prop, q_write, q_assignments, q_log,
                    streamer_count, analyzer_count, dir_audio):
    if processor == 'CPU':
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    elif processor == 'GPU':
        # let memory grow when processing on GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

    printlog(f"analyzer {id_analyzer}: processing on {processor}", q_log=q_log, level='INFO')

    def analyze_assignment(path_audio, samples, chunk):
        embeddings = embedder.embed(samples)
        results = model(embeddings)['dense'] # hmm...
        # the results are a dict now. That's either because of Keras 3 and it's new standard behavior,
        # or it's the result of converting the Keras 2 models I'm currenlty testing to K3.
        # will the output always have the 'dense' key? I dunno! But if you see an error later,
        # check the class/shape/keys of these results.
        # TODO: change models to modular based on ABC, like embedders;
        # that will let us handle idiosyncracies

        q_write.put({
            'path_audio': path_audio,
            'chunk': chunk,
            'results': results
        })

        chunk_duration = chunk[1] - chunk[0]

        timer_analysis.stop()
        analysis_rate = (chunk_duration / timer_analysis.get_total()).__round__(1)
        path_short = re.sub(dir_audio, '', path_audio)

        printlog(f"analyzer {id_analyzer}: analyzed {path_short}, "
                 f"chunk ({float(chunk[0])}, {float(chunk[1])}) "
                 f"in {timer_analysis.get_total()}s (rate: {analysis_rate})", q_log, level='INFO')

        timer_analysis.restart()

    # ready model
    model = load_model(modelname)
    embedder = load_embedder(embeddername=embeddername, framehop_prop=framehop_prop, load_model=True)
    warnings.warn('you still have to solve embedders not loading their models, at least for k2')
    embedder.load()

    timer_analysis = Timer()
    timer_bottleneck = Timer()

    def loop_waiting():
        timer_bottleneck.restart()
        while True:
            try:
                assignment = q_assignments.get(timeout=5)
                timer_bottleneck.stop()
                printlog(
                    f"BUFFER BOTTLENECK: analyzer {id_analyzer} received assignment after {timer_bottleneck.get_total().__round__(1)}s",
                    q_log=q_log, level='DEBUG')
                analyze_assignment(
                    path_audio=assignment['path_audio'],
                    samples=assignment['samples'],
                    chunk=assignment['chunk']
                )
                return
            except Empty:
                # Check if streamers are still active using shared counter
                with streamer_count.get_lock():
                    current_streamer_count = streamer_count.value

                if current_streamer_count > 0:
                    continue
                else:
                    return 'TERMINATE'

    while True:
        try:
            assignment = q_assignments.get(timeout=0)
            analyze_assignment(
                path_audio=assignment['path_audio'],
                samples=assignment['samples'],
                chunk=assignment['chunk']
            )
        except Empty:
            state = loop_waiting()
            if state == 'TERMINATE':
                break

    printlog(f"analyzer {id_analyzer}: terminating", q_log, level='INFO')

    # Decrement analyzer count
    with analyzer_count.get_lock():
        analyzer_count.value -= 1

