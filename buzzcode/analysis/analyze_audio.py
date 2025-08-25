import multiprocessing
# Set the multiprocessing start method to 'spawn' for Windows compatibility
# This must be done before any other multiprocessing operations
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)


from buzzcode.utils import search_dir, Timer, setthreads

setthreads(1)

from buzzcode.embedders import load_embedder_config
from buzzcode.analysis.coverage import chunklist_from_base
from buzzcode.audio import get_duration
import buzzcode.config as cfg
import json
import os
import re
import warnings
from datetime import datetime

import soundfile as sf

from buzzcode.analysis.analysis import get_framelength_digits
from buzzcode.analysis.workers import printlog, worker_logger, worker_streamer, worker_writer, worker_analyzer
from buzzcode.training.test import pull_sx


def analyze(
        modelname: str,
        classes_out: list = None,
        precision: float = None,
        framehop_prop: float = 1,
        chunklength: float = 1000,
        cpus: int = 2,
        gpu: bool = False,
        concurrent_streamers: int = None,
        dir_audio: str = cfg.DIR_AUDIO,
        dir_out: str = None,
        verbosity: str = 'INFO'
):
    """Analyze audio files using a buzz detection model.

    Parameters
    ----------
    modelname : str
        Name of the model to use for analysis (corresponding to the directory name in the model directory)
    classes_out : list, optional
        List of strings corresponding to the names of neurons to output, by default None
        If neurons_out is specified, output values are raw neuron activations.
        Either neurons_out or precision must be specified.
    precision : float, optional
        Float of the precision value of the model to use to call buzzes
        If precision is specified, output values are binary for the buzz class.
        Calling of non-buzz events is not currently supported; if you would like
        to work with non-buzz events (e.g., rain), specify neurons_out instead.
        Either precision or neurons_out must be specified.
    framehop_prop : float, optional
        Float specifying the overlap between frames; framehop_prop=1 creates contiguous frames;
        framehop_prop=0.5 creates frames that overlap by half their length, by default 1.
    chunklength : float, optional
        Length of audio chunks in seconds, by default 1000. Try different values to tune for your machine.
    cpus : int, optional
        Number of CPU cores to use, by default 2
    gpu : bool, optional
        Whether to use GPU for processing, by default False
    concurrent_streamers : int, optional
        The number of simultaneous workers to read audio files, by default None
        If None, attempts to calculate a reasonable number of workers. If you're using GPU,
        you may need to significantly increase this number to keep the GPU fed.
    dir_audio : str, optional
        Directory containing audio files to analyze, by default DIR_AUDIO (see config.py)
    dir_out : str, optional
        Output directory for analysis results, by default None
        If None, creates 'output' subdirectory in model directory
    verbosity : str, optional
        Level of verbosity for logging (INFO, DEBUG, WARNING, ERROR), by default 'INFO')

    Returns
    -------
    None
        Results are written to output directory as files

    Notes
    -----
    This function processes audio files in parallel using multiple CPU cores
    and optionally GPU. It uses a neural network model to classify sounds
    in the audio files. Results are saved as separate files for each
    analyzed audio chunk.
    """
    timer_total = Timer()

    # Setup
    dir_model = os.path.join(cfg.DIR_MODELS, modelname)
    if not os.path.exists(dir_model):
        warnings.warn(f'model {modelname} not found in model directory; exiting')
        return

    with open(os.path.join(dir_model, 'config_model.txt'), 'r') as file:
        config_model = json.load(file)
    config_embedder = load_embedder_config(config_model['embeddername'])

    if dir_out is None:
        dir_out = os.path.join(dir_model, cfg.SUBDIR_OUTPUT)

    if precision is None:
        threshold = None
    else:
        threshold = pull_sx(modelname, precision)['threshold']

    # rounding to nearest frame allows for seamless chunks
    digits_time = get_framelength_digits(config_embedder['framelength'])
    chunklength = round(chunklength / config_embedder['framelength']) * config_embedder['framelength']
    chunklength = round(chunklength, digits_time)

    # processing control
    if concurrent_streamers is None:
        if not gpu:
            concurrent_streamers = int(cpus / 2)
        else:
            concurrent_streamers = 8
    stream_buffer_depth = 2

    # Create multiprocessing context for spawn
    ctx = multiprocessing.get_context('spawn')

    # interprocess communication using spawn context
    q_log = ctx.Queue()
    q_control = ctx.Queue()
    q_write = ctx.Queue()

    streamer_count = ctx.Value('i', concurrent_streamers)
    analyzer_count = ctx.Value('i', cpus + (1 if gpu else 0))

    shutdown_event = ctx.Event()

    # Logging
    log_timestamp = timer_total.time_start.strftime("%Y-%m-%d_%H%M%S")
    path_log = os.path.join(dir_out, f"log {log_timestamp}.log")

    args_logger = {
        'path_log': path_log,
        'q_log': q_log,
        'shutdown_event': shutdown_event,
        'verbosity': verbosity
    }
    proc_logger = ctx.Process(target=worker_logger, kwargs=args_logger)
    proc_logger.start()

    # Build chunklists
    paths_audio_raw = search_dir(dir_audio, list(sf.available_formats().keys()))
    if len(paths_audio_raw) == 0:
        m = f"no compatible audio files found in raw directory {dir_audio} \n" \
            f"audio format must be compatible with soundfile module version {sf.__version__} \n" \
            f"exiting analysis"

        printlog(m, q_log, level='ERROR')
        shutdown_event.set()  # Signal shutdown directly
        proc_logger.join()
        return

    paths_audio = []
    for path_audio in paths_audio_raw:
        path_out = re.sub(dir_audio, dir_out, path_audio)
        path_out = os.path.splitext(path_out)[0] + cfg.SUFFIX_RESULT_COMPLETE
        if os.path.exists(path_out):
            results_exist = True
            continue
        paths_audio.append(path_audio)

    if len(paths_audio) == 0:
        m = f"all files in {dir_audio} are fully analyzed; exiting analysis"
        printlog(m, q_log, level='WARNING')
        shutdown_event.set()
        proc_logger.join()
        return

    control = []
    for path_audio in paths_audio:
        if os.path.getsize(path_audio) < 5000:
            printlog(f'file too small, skipping: {path_audio}', q_log, level='WARNING')
            continue

        base_out = re.sub(dir_audio, dir_out, path_audio)
        base_out = os.path.splitext(base_out)[0]
        printlog(f"checking results for {re.sub(dir_out, '', base_out)}", q_log, level='INFO')
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
        printlog(f"all files in {dir_audio} are fully analyzed; exiting analysis", q_log, level='WARNING')
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
        level='INFO'
    )

    args_writer = {
        'analyzer_count': analyzer_count,
        'q_write': q_write,
        'classes': config_model['classes'],
        'classes_out': classes_out,
        'threshold': threshold,
        'framehop_s': config_embedder['framelength'] * framehop_prop,
        'digits_time': digits_time,
        'dir_audio': dir_audio,
        'dir_out': dir_out,
        'q_log': q_log,
        'shutdown_event': shutdown_event
    }
    proc_writer = ctx.Process(target=worker_writer, name='writer_proc', kwargs=args_writer)
    proc_writer.start()

    args_streamer = {
        'streamer_count': streamer_count,
        'q_control': q_control,
        'q_assignments': q_assignments,
        'q_log': q_log,
        'resample_rate': config_embedder['samplerate']
    }

    proc_streamers = []
    for s in range(concurrent_streamers):
        proc_streamers.append(
            ctx.Process(target=worker_streamer, name=f'streamer_proc{s}', args=[s], kwargs=args_streamer))
        proc_streamers[-1].start()

    args_analyzer = {
        'modelname': modelname,
        'embeddername': config_model['embeddername'],
        'framehop_s': framehop_prop * config_embedder['framelength'],
        'q_write': q_write,
        'q_assignments': q_assignments,
        'q_log': q_log,
        'streamer_count': streamer_count,
        'analyzer_count': analyzer_count,
        'dir_audio': dir_audio
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
    # TODO: separate logger closing from files finishing
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

    timer_total.stop()

    # TODO: after seperating logger closing from analysis finishing, move this to log
    closing_message = f"{datetime.now()} - analysis complete; total time: {timer_total.get_total()}s"
    print(closing_message)

    return


if __name__ == "__main__":
    analyze(
        modelname='lite',
        classes_out=['ins_buzz', 'ambient_rain'],
        framehop_prop=1,
        chunklength=100,
        verbosity='DEBUG'
    )
