import os
import re
from _queue import Empty
from datetime import datetime

import numpy as np
import tensorflow as tf

from buzzcode.analysis.analysis import format_activations, format_detections, load_model
from buzzcode.audio import stream_to_queue
from buzzcode.config import SUFFIX_RESULT_PARTIAL
from buzzcode.embedders import load_embedder_model
from buzzcode.utils import Timer


def printlog(item, q_log, verb_set, verb_item=0, do_log=True):
    time_current = datetime.now()

    if do_log:
        q_log.put(f"{time_current} - {item} \n")

    if verb_set >= verb_item:
        print(item)

    return item


def worker_logger(path_log, q_log, shutdown_event, verbosity):
    def getwrite(t):
        item = q_log.get(timeout=t)
        with open(path_log, "a") as f:
            f.write(item)

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

    # Don't use printlog here as it could cause circular dependency
    print("logger: exiting")
    with open(path_log, "a") as f:
        f.write(f"{datetime.now()} - logger: exiting \n")


def worker_streamer(id_streamer, streamer_count, q_control, q_assignments, q_log, resample_rate, verbosity):
    c = q_control.get()
    while c != 'TERMINATE':
        printlog(f"streamer {id_streamer}: buffering {c['path_audio']}", q_log=q_log, verb_set=verbosity, verb_item=1,
                 do_log=True)
        stream_to_queue(
            path_audio=c['path_audio'],
            duration_audio=c['duration_audio'],
            chunklist=c['chunklist'],
            q_assignments=q_assignments,
            resample_rate=resample_rate
        )
        c = q_control.get()

    printlog(f"streamer {id_streamer}: terminating", q_log=q_log, verb_set=verbosity, verb_item=0, do_log=True)

    # Decrement streamer count
    with streamer_count.get_lock():
        streamer_count.value -= 1


def worker_writer(classes_out, threshold, writer_count, analyzer_count, q_write, classes, framehop_s, digits_time,
                  dir_audio, dir_out, q_log, verbosity, shutdown_event, digits_results=2):
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
            results=results,
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
            # Check if analyzers are still running using shared counter
            with analyzer_count.get_lock():
                current_analyzer_count = analyzer_count.value

            if current_analyzer_count == 0:
                break
            else:
                continue

    printlog("writer: terminating", q_log=q_log, verb_set=verbosity, verb_item=2, do_log=True)

    # Decrement writer count and signal shutdown if this was the last writer
    with writer_count.get_lock():
        writer_count.value -= 1
        if writer_count.value == 0:
            shutdown_event.set()


def worker_analyzer(id_analyzer, processor, modelname, embeddername, framehop_s, q_write, q_assignments, q_log,
                    streamer_count, analyzer_count, dir_audio, verbosity):
    if processor == 'CPU':
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'

    printlog(f"analyzer {id_analyzer}: processing on {processor}", q_log=q_log, verb_set=verbosity, verb_item=2,
             do_log=True)

    def analyze_assignment(path_audio, samples, chunk):
        embeddings = embedder(samples)
        results = model(embeddings)
        results = np.array(results)
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
                 f"chunk {chunk} "
                 f"in {timer_analysis.get_total()}s (rate: {analysis_rate})", q_log, verb_set=verbosity, verb_item=1,
                 do_log=True)

        timer_analysis.restart()

    # ready model
    model = load_model(modelname)
    embedder = load_embedder_model(embeddername, framehop_s=framehop_s)

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
                    q_log=q_log, verb_set=verbosity,
                    verb_item=2,
                    do_log=True)
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

    printlog(f"analyzer {id_analyzer}: terminating", q_log, verb_set=verbosity, verb_item=0, do_log=True)

    # Decrement analyzer count
    with analyzer_count.get_lock():
        analyzer_count.value -= 1


def initialize_log(time_start, dir_out):
    log_timestamp = time_start.strftime("%Y-%m-%d_%H%M%S")
    path_log = os.path.join(dir_out, f"log {log_timestamp}.txt")
    os.makedirs(os.path.dirname(path_log), exist_ok=True)
    log = open(path_log, "x")
    log.close()

    return path_log
