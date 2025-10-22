import multiprocessing
import os

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# Set the multiprocessing start method to 'spawn' for Windows compatibility
# This must be done before any other multiprocessing operations
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

from buzzcode.utils import search_dir, Timer
from buzzcode.embedding.load_embedder import load_embedder
from buzzcode.analysis.coverage import chunklist_from_base
from buzzcode.audio import get_duration
from datetime import datetime
import buzzcode.config as cfg
import json
import re
import queue
import threading

import soundfile as sf

from buzzcode.analysis.workers import WorkerStreamer, WorkerWriter, WorkerAnalyzer, WorkerLogger
from buzzcode.analysis.assignments import AssignLog, AssignStream
from buzzcode.training.test import pull_sx


def run_worker(workerclass, **kwargs):
    worker = workerclass(**kwargs)
    worker()

def analyze(
        modelname: str,
        classes_out: list = None,
        precision: float = None,
        framehop_prop: float = 1,
        chunklength: float = 1000,
        cpus: int = 2,
        gpu: bool = False,
        concurrent_streamers: int = None,
        stream_buffer_depth: int = None,
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
    stream_buffer_depth : int, optional
        How many chunks should the streaming queue hold? If 1, only one streamer can enqueue at a time.
        Max RAM utilizaiton will be (concurrent streamers + stream_buffer_depth) * chunklength, since
        each streamer will hold 1 chunk while waiting to enqueue.
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
        raise FileNotFoundError(f'model {modelname} not found in model directory')

    with open(os.path.join(dir_model, 'config_model.txt'), 'r') as file:
        config_model = json.load(file)

    if dir_out is None:
        dir_out = os.path.join(dir_model, cfg.SUBDIR_OUTPUT)

    if precision is None:
        threshold = None
    else:
        threshold = pull_sx(modelname, precision)['threshold']

    embedder_noload = load_embedder(embeddername=config_model['embeddername'], framehop_prop=framehop_prop, initialize=False)
    # rounding to nearest frame allows for seamless chunks
    chunklength = round(chunklength / embedder_noload.framelength_s) * embedder_noload.framelength_s
    chunklength = round(chunklength, embedder_noload.digits_time)

    # processing control
    if concurrent_streamers is None:
        if not gpu:
            concurrent_streamers = max(cpus, 1)
        else:
            concurrent_streamers = 8

    if stream_buffer_depth is None:
        stream_buffer_depth = concurrent_streamers*2

    # Create multiprocessing context for spawn
    ctx = multiprocessing.get_context('spawn')

    # interprocess communication using spawn context
    q_log = ctx.Queue()
    q_stream = queue.Queue()
    q_analyze = queue.Queue(maxsize=stream_buffer_depth)
    q_write = ctx.Queue()

    event_analysisdone = ctx.Event()
    event_closelogger = ctx.Event()

    # Logging
    log_timestamp = timer_total.time_start.strftime("%Y-%m-%d_%H%M%S")
    path_log = os.path.join(dir_out, f"log {log_timestamp}.log")
    os.makedirs(os.path.dirname(path_log), exist_ok=True)

    proc_logger = ctx.Process(
        target=run_worker,
        name='logger_proc',
        kwargs={
            'workerclass': WorkerLogger,
            'q_log': q_log,
            'path_log': path_log,
            'event_closelogger': event_closelogger,
            'verbosity': verbosity
        }
    )
    proc_logger.start()

    if framehop_prop > 1:
        msg = (
            'Currently, analyses with framehop > 1 will produce valid results,'
            'but buzzdetect will interpret the resulting gaps as errors.'
            f'Fully analyzed files will not be converted from {cfg.SUFFIX_RESULT_PARTIAL} to {cfg.SUFFIX_RESULT_COMPLETE}.'
            f'Repeated analysis will attempt to fill gaps between frames.'
        )
        q_log.put(AssignLog(msg=msg, level_str='WARNING'))

    # Build chunklists
    paths_audio_raw = search_dir(dir_audio, list(sf.available_formats().keys()))
    if len(paths_audio_raw) == 0:
        msg = f"no compatible audio files found in raw directory {dir_audio} \n" \
            f"audio format must be compatible with soundfile module version {sf.__version__} \n" \
            f"exiting analysis"
        q_log.put(AssignLog(msg=msg, level_str='ERROR'))
        event_analysisdone.set()  # Signal shutdown directly
        proc_logger.join()
        return

    paths_audio = []
    for path_audio in paths_audio_raw:
        path_out = re.sub(dir_audio, dir_out, path_audio)
        path_out = os.path.splitext(path_out)[0] + cfg.SUFFIX_RESULT_COMPLETE
        if os.path.exists(path_out):
            continue

        paths_audio.append(path_audio)

    if not paths_audio:
        msg = f"all files in {dir_audio} are fully analyzed; exiting analysis"
        q_log.put(AssignLog(msg=msg, level_str='WARNING'))
        event_analysisdone.set()
        proc_logger.join()
        return

    a_stream_list = []
    for path_audio in paths_audio:
        if os.path.getsize(path_audio) < 5000:
            q_log.put(AssignLog(msg=f'file too small, skipping: {path_audio}', level_str='WARNING'))
            continue

        base_out = re.sub(dir_audio, dir_out, path_audio)
        base_out = os.path.splitext(base_out)[0]
        q_log.put(AssignLog(msg=f'checking results for {re.sub(dir_out, "", base_out)}', level_str='INFO'))
        duration_audio = get_duration(path_audio)

        chunklist = chunklist_from_base(
            base_out=base_out,
            duration_audio=duration_audio,
            framelength_s=embedder_noload.framelength_s,
            chunklength=chunklength
        )

        if not chunklist:
            continue


        a_stream_list.append(
            AssignStream(
                path_audio=path_audio,
                chunklist=chunklist,
                duration_audio=duration_audio,
                terminate=False
            )
        )

    if not a_stream_list:
        q_log.put(AssignLog(msg=f'all files in {dir_audio} are fully analyzed; exiting analysis', level_str='WARNING'))
        event_analysisdone.set()  # Signal shutdown directly
        proc_logger.join()
        return

    n_paths = len(set([a.path_audio for a in a_stream_list]))

    if concurrent_streamers > n_paths:
        concurrent_streamers = n_paths
    if cpus > n_paths:
        cpus = n_paths
    streamer_count = ctx.Value('i', concurrent_streamers)
    analyzer_count = ctx.Value('i', cpus + (1 if gpu else 0))

    # load control into the queue
    for a in a_stream_list:
        q_stream.put(a)

    # send sentinel for streamers
    assign_streamend = AssignStream(path_audio=None, duration_audio=None, chunklist=None, terminate=True)
    for _ in range(concurrent_streamers):
        q_stream.put(assign_streamend)

    # Launch processes
    msg = (
        f"begin analysis\n"
        f"start time: {timer_total.time_start}\n"
        f"input directory: {dir_audio}\n"
        f"model: {modelname}\n"
        f"CPU count: {cpus}\n"
        f"GPU: {gpu}\n"
    )
    q_log.put(AssignLog(msg=msg, level_str='INFO'))

    proc_writer = ctx.Process(
        target=run_worker,
        name='writer_proc',
        kwargs={
            'workerclass': WorkerWriter,
            'classes_out': classes_out,
            'threshold': threshold,
            'analyzer_count': analyzer_count,
            'q_write': q_write,
            'classes': config_model['classes'],
            'framehop_s': embedder_noload.framehop_s,
            'digits_time': embedder_noload.digits_time,
            'dir_audio': dir_audio,
            'dir_out': dir_out,
            'q_log': q_log,
            'event_analysisdone': event_analysisdone,
            'digits_results': config_model['digits_results']
        }
    )
    proc_writer.start()

    thread_streamers = []
    for s in range(concurrent_streamers):
        thread_streamer = threading.Thread(
            target=run_worker,
            name=f'streamer_proc{s}',
            kwargs={
                'workerclass': WorkerStreamer,
                'id_streamer': s,
                'streamer_count': streamer_count,
                'q_stream': q_stream,
                'q_analyze': q_analyze,
                'q_log': q_log,
                'resample_rate': embedder_noload.samplerate,
            }
        )

        thread_streamers.append(thread_streamer)
        thread_streamers[-1].start()


    thread_analyzers = []
    for a in range(cpus):
        thread_analyzer = threading.Thread(
            target=run_worker,
            name=f"analysis_proc{a}",
            kwargs={
                'workerclass': WorkerAnalyzer,
                'id_analyzer': a,
                'processor': 'CPU',
                'modelname': modelname,
                'embeddername': config_model['embeddername'],
                'framehop_prop': framehop_prop,
                'q_write': q_write,
                'q_analyze': q_analyze,
                'q_log': q_log,
                'streamer_count': streamer_count,
                'analyzer_count': analyzer_count,
                'dir_audio': dir_audio
            }
        )
        
        thread_analyzers.append(thread_analyzer)
        thread_analyzers[-1].start()
    
    if gpu:
        WorkerAnalyzer(
            id_analyzer='g',
            processor='GPU',
            modelname=modelname,
            embeddername=config_model['embeddername'],
            framehop_prop=framehop_prop,
            q_write=q_write,
            q_analyze=q_analyze,
            q_log=q_log,
            streamer_count=streamer_count,
            analyzer_count=analyzer_count,
            dir_audio=dir_audio
        )()  # call, blocking main thread in execution until analysis is finished

    # wait for analysis to finish
    # TODO: separate logger closing from files finishing
    event_analysisdone.wait()

    # on terminate, clean up chunks
    for a in a_stream_list:
        path_audio = a.path_audio
        base_out = re.sub(dir_audio, dir_out, path_audio)
        base_out = os.path.splitext(base_out)[0]

        chunklist_from_base(
            base_out=base_out,
            duration_audio=get_duration(path_audio),
            framelength_s=embedder_noload.framelength_s,
            chunklength=chunklength
        )
        # Note: Can't use printlog here as logger has already exited

    timer_total.stop()

    # TODO: after seperating logger closing from analysis finishing, move this to log
    q_log.put(AssignLog(msg=f"{datetime.now()} - analysis complete; total time: {timer_total.get_total()}s", level_str='INFO'))
    event_closelogger.set()
    proc_logger.join()

    return


if __name__ == "__main__":
    analyze(
        modelname='model_general_v3',
        classes_out='all',
        dir_audio='/Users/luke/Documents/code projects/buzzdetect_repository/data/raw/full_recordings',
        cpus=5,
        framehop_prop=1,
        chunklength=100,
        verbosity='DEBUG'
    )
