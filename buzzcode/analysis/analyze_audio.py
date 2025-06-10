import warnings

from buzzcode.analysis.workers import printlog, worker_logger, worker_streamer, worker_writer, worker_analyzer, \
    early_exit, initialize_log
from buzzcode.config import DIR_AUDIO
from buzzcode.utils import search_dir, Timer, setthreads

setthreads(1)

from buzzcode.embedders import load_embedder_config
from buzzcode.analysis.coverage import chunklist_from_base
from buzzcode.audio import get_duration
import os
import re
import multiprocessing
import json
import soundfile as sf
from datetime import datetime


def analyze_batch(modelname, classes_keep='all', chunklength=2000, cpus=2, gpu=False, framehop_prop=1, dir_audio=DIR_AUDIO, verbosity=1):
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
        'classes_keep': classes_keep,
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
        'embeddername': config_model['embedder'],
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
                  chunklength=960, cpus=1, verbosity=2)  # chunklength being a multiple of framelength makes for seamless chunks
