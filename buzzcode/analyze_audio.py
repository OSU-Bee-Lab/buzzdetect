import warnings

import numpy as np

from buzzcode.config import dir_audio_in, suffix_partial
from buzzcode.utils import search_dir, Timer, setthreads

setthreads(1)

from buzzcode.embedders import load_embedder_model, load_embedder_config
from buzzcode.analysis import load_model, translate_results, chunklist_from_base
from buzzcode.audio import stream_to_queue, get_duration
import os
import re
import multiprocessing
from queue import Empty
import json
import tensorflow as tf
import soundfile as sf
from datetime import datetime


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


def worker_writer(sem_writers, sem_analyzers, q_write, classes, framehop_prop, framelength, digits_time,
                  dir_audio, dir_out, q_log, verbosity, digits_results=2):
    def write_results(path_audio, chunk, results):
        output = translate_results(results, classes, digits=digits_results)

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
        path_out = base_out + suffix_partial

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


def analyze_batch(modelname, chunklength=2000, cpus=2, gpu=False, embeddername='yamnet', framehop_prop=1,
                  dir_audio=dir_audio_in, verbosity=1):
    # Setup
    #
    dir_model = os.path.join("models", modelname)
    if not os.path.exists(dir_model):
        warnings.warn(f'model {modelname} not found in model directory; exiting')
        return

    dir_out = os.path.join(dir_model, "output")
    timer_total = Timer()

    # processing control
    concurrent_writers = 1
    concurrent_streamers = max(int(cpus / 2), 2)
    stream_buffer_depth = max(2, int(cpus / concurrent_streamers))

    # interprocess communication
    sem_streamers = multiprocessing.Semaphore(concurrent_streamers)
    sem_analyzers = multiprocessing.Semaphore(cpus + gpu)
    sem_writers = multiprocessing.Semaphore(concurrent_writers)

    q_log = multiprocessing.Queue()
    q_control = multiprocessing.Queue()
    q_write = multiprocessing.Queue()

    # configs
    with open(os.path.join(dir_model, 'config_model.txt'), 'r') as file:
        config_model = json.load(file)
    config_embedder = load_embedder_config(config_model['embedder'])

    # for rounding off floating point error
    framelength_str = str(config_embedder['framelength'])
    framelength_str = re.sub('^.*\\.', '', framelength_str)
    framelength_digits = len(framelength_str)

    # Logging
    #
    path_log = initialize_log(
        time_start=timer_total.time_start,
        dir_out=dir_out
    )

    args_logger = {
        'path_log': path_log,
        'q_log': q_log,
        'sem_writers': sem_writers,
        'verbosity': verbosity
    }
    proc_logger = multiprocessing.Process(target=worker_logger, kwargs=args_logger)
    proc_logger.start()  # start logger early, since there can be logworthy events even if analysis doesn't happen

    # Build chunklists
    #
    paths_audio = search_dir(dir_audio, list(sf.available_formats().keys()))
    if len(paths_audio) == 0:
        m = f"no compatible audio files found in raw directory {dir_audio} \n" \
            f"audio format must be compatible with soundfile module version {sf.__version__} \n" \
            f"exiting analysis"

        printlog(m, q_log, verb_set=verbosity, verb_item=0, do_log=True)
        early_exit(sem_writers, concurrent_writers)
        return

    control = []
    for path_audio in paths_audio:
        # TODO: parallel process? This can take a long time if the interrupted analysis was very long
        if os.path.getsize(path_audio) < 5000:
            warnings.warn(f'file too small, skipping: {path_audio}')
            continue

        base_out = re.sub(dir_audio, dir_out, path_audio)
        base_out = os.path.splitext(base_out)[0]
        printlog(f"checking results for {re.sub(dir_out, '', base_out)}", q_log, verb_set=verbosity, verb_item=2,
                 do_log=True)
        duration_audio = get_duration(path_audio)

        chunklist = chunklist_from_base(
            base_out=base_out,
            duration_audio=duration_audio,
            framelength=config_embedder['framelength'],
            chunklength=chunklength
        )

        if not chunklist:
            # if this file didn't have chunks to analyze,
            # move to the next one
            continue

        control.append({
            'path_audio': path_audio,
            'chunklist': chunklist,
            'duration_audio': duration_audio
        })

    if not control:
        printlog(f"all files in {dir_audio} are fully analyzed; exiting analysis", q_log, verb_set=verbosity,
                 verb_item=0, do_log=True)
        early_exit(sem_writers, concurrent_writers)
        return

    # load control into the queue
    for c in control:
        q_control.put(c)

    # send sentinel for streamers
    for _ in range(concurrent_streamers):
        q_control.put('TERMINATE')

    q_assignments = multiprocessing.Queue(maxsize=stream_buffer_depth)

    # Launch processes
    #
    printlog(
        f"begin analysis\n"
        f"start time: {timer_total.time_start}\n"
        f"input directory: {dir_audio}\n"
        f"model: {modelname}\n"
        f"CPU count: {cpus}\n"
        f"GPU: {gpu}\n",
        q_log=q_log,
        verb_set=verbosity,
        verb_item=0,
        do_log=True
    )

    args_writer = {
        'sem_writers': sem_writers,
        'sem_analyzers': sem_analyzers,
        'q_write': q_write,
        'classes': config_model['classes'],
        'framehop_prop': framehop_prop,
        'framelength': config_embedder['framelength'],
        'digits_time': framelength_digits,
        'dir_audio': dir_audio,
        'dir_out': dir_out,
        'q_log': q_log,
        'verbosity': verbosity
    }
    proc_writer = multiprocessing.Process(target=worker_writer, name='writer_proc', kwargs=args_writer)
    proc_writer.start()

    args_streamer = {
        'sem_streamers': sem_streamers,
        'q_control': q_control,
        'q_assignments': q_assignments,
        'q_log': q_log,
        'resample_rate': config_embedder['samplerate'],
        'verbosity': verbosity
    }

    proc_streamers = []
    streamer_ids = []
    for s in range(concurrent_streamers):
        proc_streamers.append(
            multiprocessing.Process(target=worker_streamer, name=f'streamer_proc{s}', args=[s], kwargs=args_streamer))
        proc_streamers[-1].start()
        streamer_ids.append(proc_streamers[-1].pid)
        pass

    args_analyzer = {
        'modelname': modelname,
        'embeddername': embeddername,
        'framehop_s': framehop_prop * config_embedder['framelength'],
        'q_write': q_write,
        'q_assignments': q_assignments,
        'q_log': q_log,
        'sem_streamers': sem_streamers,
        'sem_analyzers': sem_analyzers,
        'dir_audio': dir_audio,
        'verbosity': verbosity
    }
    proc_analyzers = []
    for a in range(cpus):
        proc_analyzers.append(
            multiprocessing.Process(target=worker_analyzer, name=f"analysis_proc{a}", args=([a, 'CPU']),
                                    kwargs=args_analyzer))
        proc_analyzers[-1].start()
        pass

    if gpu:  # things get narsty when running GPU on a multiprocessing.Process(), so just run in main thread last
        worker_analyzer('G', 'GPU', **args_analyzer)

    # wait for analysis to finish
    proc_logger.join()

    # on terminate, clean up chunks
    for c in control:
        path_audio = c['path_audio']
        base_out = re.sub(dir_audio, dir_out, path_audio)
        base_out = os.path.splitext(base_out)[0]

        # don't need the chunks, but this will clean it up
        chunklist_from_base(
            base_out=base_out,
            duration_audio=get_duration(path_audio),
            framelength=config_embedder['framelength'],
            chunklength=chunklength
        )
        printlog(f"combining result chunks for {re.sub(dir_out, '', base_out)}", q_log, verb_set=verbosity, verb_item=1,
                 do_log=True)

    timer_total.stop()
    closing_message = f"{datetime.now()} - analysis complete; total time: {timer_total.get_total()}s"

    print(closing_message)
    file_log = open(path_log, "a")
    file_log.write(closing_message)
    file_log.close()

    return


if __name__ == "__main__":
    analyze_batch(modelname='model_general_v3', dir_audio='/home/luke/r projects/buzzdetect, repository/data/raw',
                  chunklength=500, cpus=1, verbosity=2)
