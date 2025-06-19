import warnings
import os
import re
import multiprocessing
import json
import soundfile as sf
from datetime import datetime

from buzzcode.analysis.workers import printlog, worker_logger, worker_streamer, worker_writer, worker_analyzer, \
    initialize_log

# Set the multiprocessing start method to 'spawn' for Windows compatibility
# This must be done before any other multiprocessing operations
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

from buzzcode.audio import get_duration
from buzzcode.config import DIR_AUDIO
from buzzcode.utils import search_dir, Timer, setthreads

setthreads(1)

from buzzcode.embedders import load_embedder_config
from buzzcode.analysis.coverage import chunklist_from_base


def early_exit(sem_writers, concurrent_writers):
    for _ in range(concurrent_writers):
        sem_writers.release()


def analyze_batch(modelname, classes_keep=['ins_buzz'], chunklength=2000, cpus=2, gpu=False, framehop_prop=1, dir_audio=DIR_AUDIO, verbosity=1):
    # Setup
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

    # Create multiprocessing context for spawn
    ctx = multiprocessing.get_context('spawn')

    # interprocess communication using spawn context
    q_log = ctx.Queue()
    q_control = ctx.Queue()
    q_write = ctx.Queue()

    # Use shared values instead of semaphores for counting
    streamer_count = ctx.Value('i', concurrent_streamers)
    analyzer_count = ctx.Value('i', cpus + (1 if gpu else 0))
    writer_count = ctx.Value('i', concurrent_writers)

    # configs
    with open(os.path.join(dir_model, 'config_model.txt'), 'r') as file:
        config_model = json.load(file)
    config_embedder = load_embedder_config(config_model['embedder'])

    # for rounding off floating point error
    framelength_str = str(config_embedder['framelength'])
    framelength_str = re.sub('^.*\\.', '', framelength_str)
    framelength_digits = len(framelength_str)

    # Logging
    path_log = initialize_log(
        time_start=timer_total.time_start,
        dir_out=dir_out
    )

    shutdown_event = ctx.Event()

    args_logger = {
        'path_log': path_log,
        'q_log': q_log,
        'shutdown_event': shutdown_event,
        'verbosity': verbosity
    }
    proc_logger = ctx.Process(target=worker_logger, kwargs=args_logger)
    proc_logger.start()

    # Build chunklists
    paths_audio = search_dir(dir_audio, list(sf.available_formats().keys()))
    if len(paths_audio) == 0:
        m = f"no compatible audio files found in raw directory {dir_audio} \n" \
            f"audio format must be compatible with soundfile module version {sf.__version__} \n" \
            f"exiting analysis"

        printlog(m, q_log, verb_set=verbosity, verb_item=0, do_log=True)
        shutdown_event.set()  # Signal shutdown directly
        proc_logger.join()
        return

    control = []
    for path_audio in paths_audio:
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
            continue

        control.append({
            'path_audio': path_audio,
            'chunklist': chunklist,
            'duration_audio': duration_audio
        })

    if not control:
        printlog(f"all files in {dir_audio} are fully analyzed; exiting analysis", q_log, verb_set=verbosity,
                 verb_item=0, do_log=True)
        shutdown_event.set()  # Signal shutdown directly
        proc_logger.join()
        return

    # load control into the queue
    for c in control:
        q_control.put(c)

    # send sentinel for streamers
    for _ in range(concurrent_streamers):
        q_control.put('TERMINATE')

    q_assignments = ctx.Queue(maxsize=stream_buffer_depth)

    # Launch processes
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
        'writer_count': writer_count,
        'analyzer_count': analyzer_count,
        'q_write': q_write,
        'classes': config_model['classes'],
        'classes_keep': classes_keep,
        'framehop_prop': framehop_prop,
        'framelength': config_embedder['framelength'],
        'digits_time': framelength_digits,
        'dir_audio': dir_audio,
        'dir_out': dir_out,
        'q_log': q_log,
        'verbosity': verbosity,
        'shutdown_event': shutdown_event
    }
    proc_writer = ctx.Process(target=worker_writer, name='writer_proc', kwargs=args_writer)
    proc_writer.start()

    args_streamer = {
        'streamer_count': streamer_count,
        'q_control': q_control,
        'q_assignments': q_assignments,
        'q_log': q_log,
        'resample_rate': config_embedder['samplerate'],
        'verbosity': verbosity
    }

    proc_streamers = []
    for s in range(concurrent_streamers):
        proc_streamers.append(
            ctx.Process(target=worker_streamer, name=f'streamer_proc{s}', args=[s], kwargs=args_streamer))
        proc_streamers[-1].start()

    args_analyzer = {
        'modelname': modelname,
        'embeddername': config_model['embedder'],
        'framehop_s': framehop_prop * config_embedder['framelength'],
        'q_write': q_write,
        'q_assignments': q_assignments,
        'q_log': q_log,
        'streamer_count': streamer_count,
        'analyzer_count': analyzer_count,
        'dir_audio': dir_audio,
        'verbosity': verbosity
    }
    proc_analyzers = []
    for a in range(cpus):
        proc_analyzers.append(
            ctx.Process(target=worker_analyzer, name=f"analysis_proc{a}", args=([a, 'CPU']),
                        kwargs=args_analyzer))
        proc_analyzers[-1].start()

    if gpu:
        worker_analyzer('G', 'GPU', **args_analyzer)

    # wait for analysis to finish
    proc_logger.join()

    # on terminate, clean up chunks
    for c in control:
        path_audio = c['path_audio']
        base_out = re.sub(dir_audio, dir_out, path_audio)
        base_out = os.path.splitext(base_out)[0]

        chunklist_from_base(
            base_out=base_out,
            duration_audio=get_duration(path_audio),
            framelength=config_embedder['framelength'],
            chunklength=chunklength
        )
        # Note: Can't use printlog here as logger has already exited
        print(f"combining result chunks for {re.sub(dir_out, '', base_out)}")

    timer_total.stop()
    closing_message = f"{datetime.now()} - analysis complete; total time: {timer_total.get_total()}s"

    print(closing_message)
    file_log = open(path_log, "a")
    file_log.write(closing_message)
    file_log.close()

    return


if __name__ == "__main__":
    analyze_batch(modelname='model_general_v3',
                  chunklength=960, cpus=7, verbosity=2)