import os
import re
from datetime import datetime
from queue import Empty

import numpy as np
import tensorflow as tf

from buzzcode.analysis.analysis import trim_results, load_model
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


def worker_logger(path_log, q_log, sem_writers, verbosity):
    def getwrite(t):
        item = q_log.get(timeout=t)
        with open(path_log, "a") as f:
            f.write(item)

    # Main loop: drain while work is still coming or writers alive
    while True:
        try:
            getwrite(1)
        except Empty:
            if sem_writers.get_value() == 0:
                break
            # else: another writer might still put, so continue

    # Final catch-up: keep draining until there's a real timeout
    while True:
        try:
            getwrite(1)
        except Empty:
            break

    printlog("logger: exiting", q_log=q_log, verb_set=verbosity, verb_item=1, do_log=True)


def worker_streamer(id_streamer, sem_streamers, q_control, q_assignments, q_log, resample_rate, verbosity):
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
    sem_streamers.acquire()


def worker_writer(sem_writers, sem_analyzers, q_write, classes, classes_keep, framehop_prop, framelength, digits_time,
                  dir_audio, dir_out, q_log, verbosity, digits_results=2):
    def write_results(path_audio, chunk, results):
        output = trim_results(results, classes, classes_keep=classes_keep, digits=digits_results)

        output.insert(
            column='start',
            value=range(len(output)),
            loc=0
        )

        output['start'] = output['start'] * framehop_prop * framelength
        output['start'] = output['start'] + chunk[0]

        # round to avoid float errors
        output['start'] = round(output['start'], digits_time)

        base_out = os.path.splitext(path_audio)[0]
        base_out = re.sub(dir_audio, dir_out, base_out)
        # Remove the chunk-specific suffix since we're appending to one file
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
            # if the queue is empty, are there workers?
            if sem_analyzers.get_value() == 0:
                # if not, we're done
                break
            # if so, go back to top of loop
            else:
                continue

    printlog("writer: terminating", q_log=q_log, verb_set=verbosity, verb_item=2, do_log=True)
    sem_writers.acquire()


def worker_analyzer(id_analyzer, processor, modelname, embeddername, framehop_s, q_write, q_assignments, q_log,
                    sem_streamers, sem_analyzers, dir_audio, verbosity):
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
            'chunk': assignment['chunk'],
            'results': results
        })

        chunk_duration = chunk[1] - chunk[0]

        timer_analysis.stop()
        analysis_rate = (chunk_duration / timer_analysis.get_total()).__round__(1)
        path_short = re.sub(dir_audio, '', path_audio)

        # can I just have a monitor that watches the progress and handles reporting? So it would have its own
        # verbosity, do its own rate calculations?
        printlog(f"analyzer {id_analyzer}: analyzed {path_short}, "
                 f"chunk {chunk} "
                 f"in {timer_analysis.get_total()}s (rate: {analysis_rate})", q_log, verb_set=verbosity, verb_item=1,
                 do_log=True)

        timer_analysis.restart()  # pessimistic; accounts for wait time

    # ready model
    #
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
                if sem_streamers.get_value() > 0:
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
    sem_analyzers.acquire()


def early_exit(sem_writers, sem_writer_count):
    for _ in range(sem_writer_count):
        sem_writers.acquire()


def initialize_log(time_start, dir_out):
    log_timestamp = time_start.strftime("%Y-%m-%d_%H%M%S")
    path_log = os.path.join(dir_out, f"log {log_timestamp}.txt")
    os.makedirs(os.path.dirname(path_log), exist_ok=True)
    log = open(path_log, "x")
    log.close()

    return path_log
