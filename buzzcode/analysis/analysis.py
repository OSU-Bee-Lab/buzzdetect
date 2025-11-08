import multiprocessing
import os
import re
import signal
import threading
from datetime import datetime
from queue import Queue

import soundfile as sf

from buzzcode import config as cfg
from buzzcode.analysis.assignments import AssignLog, AssignStream
from buzzcode.analysis.results_coverage import clean_and_chunk
from buzzcode.analysis.workers import WorkerLogger, WorkerStreamer, WorkerWriter, WorkerAnalyzer
from buzzcode.audio import get_duration
from buzzcode.models.load_model import load_model
from buzzcode.training.test import pull_sx
from buzzcode.utils import Timer, search_dir


class Coordinator:
    def __init__(self,
                 analyzers_cpu,
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
        # the savings of parallel processing.
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


def early_exit(msg: str, level: str, coordinator: Coordinator):
    coordinator.q_log.put(AssignLog(msg=msg, level_str=level))
    coordinator.event_stopanalysis.set()

def run_worker(workerclass, **kwargs):
    worker = workerclass(**kwargs)
    worker()


class GracefulKiller:
  def __init__(self, event_stopanalysis):
      signal.signal(signal.SIGINT, self.exit_gracefully)
      signal.signal(signal.SIGTERM, self.exit_gracefully)
      self.event_stopanalysis = event_stopanalysis

  def exit_gracefully(self, signum, frame):
      print(f"Received signal {signum}; exiting gracefully.")
      self.event_stopanalysis.set()


class Analyzer:
    """Audio analysis orchestrator supporting both CPU and GPU processing."""

    def __init__(
            self,
            modelname: str,
            classes_out: list = None,
            precision: float = None,
            framehop_prop: float = 1,
            chunklength: float = 200,
            dir_audio: str = cfg.DIR_AUDIO,
            dir_out: str = None,
            verbosity_print: str = 'INFO',
            verbosity_log: str = 'DEBUG',
            log_progress: bool = False,
            coordinator: Coordinator = None,
    ):
        """Initialize the analyzer with configuration parameters.

        Parameters
        ----------
        modelname : str
            Name of the model to use for analysis
        classes_out : list, optional
            List of class names to output, by default None
        precision : float, optional
            Precision value for calling detections, by default None
        framehop_prop : float, optional
            The distance between the start of two frames, as a proportion of the frame's length (1=contiguous frames, 0.5=50% overlapping frames), by default 1.
        chunklength : float, optional
            The length of analysis chunks in seconds, by default 200
        dir_audio : str, optional
            Input audio directory, by default the audio directory specified in config.py
        dir_out : str, optional
            Output directory, by default None
             If left to none, results are written to the model's "output" subdirectory
        verbosity_print : str, optional
            Level of information to be written to the console output (does not affect log files)
            By default 'INFO'
        """
        self.modelname = modelname
        self.framehop_prop = framehop_prop
        self.dir_audio = dir_audio
        self.dir_out = dir_out
        self.verbosity_print = verbosity_print
        self.verbosity_log = verbosity_log
        self.log_progress = log_progress

        self.coordinator = coordinator

        self.model = load_model(
            modelname=modelname,
            framehop_prop=framehop_prop,
            initialize=False
        )
        self.chunklength = self._setup_chunklength(chunklength)
        self.classes_out = self._setup_classes_out(classes_out)
        self.threshold = self._setup_threshold(precision)

        self.timer_total = Timer()

        if self.dir_out is None:
            self.dir_out = os.path.join(cfg.DIR_MODELS, modelname, cfg.SUBDIR_OUTPUT)

        self.a_stream_list = []

        self.proc_logger = None
        self.proc_writer = None
        self.proc_streamers = []
        self.proc_analyzers = []

    def _log_debug(self, msg):
        self.coordinator.q_log.put(AssignLog(msg=msg, level_str='DEBUG'))
        print(f'DEBUG: {msg}')

    # Setup methods
    #
    def _setup_chunklength(self, chunklength):
        # Round chunklength to nearest frame for seamless processing
        chunklength = round(
            chunklength / self.model.embedder.framelength_s
        ) * self.model.embedder.framelength_s
        chunklength = round(chunklength, self.model.embedder.digits_time)
        if chunklength < self.model.embedder.framelength_s:
            chunklength = self.model.embedder.framelength_s

        return chunklength

    def _setup_classes_out(self, classes_out):
        if classes_out == 'all':
            return self.model.config['classes']
        else:
            return classes_out

    def _setup_threshold(self, precision):
        if precision is None:
            return None
        else:
            return pull_sx(self.modelname, precision)['threshold']

    # Process launching
    #
    def _launch_logger(self):
        """Start the logging process."""
        path_log = os.path.join(
            self.dir_out,
            f"{self.timer_total.time_start.strftime('%Y-%m-%d_%H%M%S')}.log"
        )
        os.makedirs(os.path.dirname(path_log), exist_ok=True)

        self.proc_logger = self.coordinator.concurrency_logger(
            target=run_worker,
            name='logger_proc',

            # unfortunately, we can't just send the coordinator, because
            # some workers are multiprocessing.Processes and we can't
            # pickle the threading objects in coordinator
            kwargs={
                'workerclass': WorkerLogger,
                'path_log': path_log,
                'q_log': self.coordinator.q_log,
                'event_analysisdone': self.coordinator.event_analysisdone,
                'event_stopanalysis': self.coordinator.event_stopanalysis,
                'verbosity_print': self.verbosity_print,
                'verbosity_log': self.verbosity_log,
                'log_progress': self.log_progress,
                'q_gui': self.coordinator.q_gui,
            }
        )
        self.proc_logger.start()

        if self.framehop_prop > 1:
            msg = (
                'Currently, analyses with framehop > 1 will produce valid results, '
                'but buzzdetect will interpret the resulting gaps as missing data.\n'
                f'Fully analyzed files will not be converted from {cfg.SUFFIX_RESULT_PARTIAL} '
                f'to {cfg.SUFFIX_RESULT_COMPLETE}.\n'
                f'Repeated analysis will attempt to fill gaps between frames.'
            )
            self.coordinator.q_log.put(AssignLog(msg=msg, level_str='WARNING'))

    def _build_assignments(self):
        """Discover audio files and build processing assignments."""
        # Find compatible audio files
        paths_audio_raw = search_dir(self.dir_audio, list(sf.available_formats().keys()))

        if len(paths_audio_raw) == 0:
            msg = (
                f"no compatible audio files found in raw directory {self.dir_audio} \n"
                f"audio format must be compatible with soundfile module version {sf.__version__} \n"
                f"exiting analysis"
            )

            early_exit(
                msg=msg,
                level='ERROR',
                coordinator=self.coordinator,
            )
            return False

        # Filter out already-completed files
        paths_audio = []
        for path_audio in paths_audio_raw:
            path_out = re.sub(self.dir_audio, self.dir_out, path_audio)
            path_out = os.path.splitext(path_out)[0] + cfg.SUFFIX_RESULT_COMPLETE
            if not os.path.exists(path_out):
                paths_audio.append(path_audio)

        if not paths_audio:
            msg = f"all files in {self.dir_audio} are fully analyzed; exiting analysis"
            self.coordinator.q_log.put(AssignLog(msg=msg, level_str='WARNING'))
            self.coordinator.event_analysisdone.set()
            self.proc_logger.join()
            return False

        # Build assignment list
        self.a_stream_list = []
        for path_audio in paths_audio:
            if os.path.getsize(path_audio) < 5000:
                self.coordinator.q_log.put(
                    AssignLog(msg=f'Skipping miniscule file: {path_audio}', level_str='WARNING')
                )
                continue

            base_out = re.sub(self.dir_audio, self.dir_out, path_audio)
            base_out = os.path.splitext(base_out)[0]
            self.coordinator.q_log.put(
                AssignLog(msg=f'Checking results for {re.sub(self.dir_out, "", base_out)}', level_str='INFO')
            )

            duration_audio = get_duration(path_audio, q_log=self.coordinator.q_log)
            chunklist = clean_and_chunk(
                ident_out=base_out,
                duration_audio=duration_audio,
                framelength_s=self.model.embedder.framelength_s,
                chunklength=self.chunklength
            )

            if not chunklist:
                continue

            self.a_stream_list.append(
                AssignStream(
                    path_audio=path_audio,
                    chunklist=chunklist,
                    duration_audio=duration_audio,
                    terminate=False,
                    dir_audio=self.dir_audio
                )
            )

        self._log_debug(f'checking stream list')
        if not self.a_stream_list:
            self._log_debug(f'no files in stream list')
            self.coordinator.q_log.put(
                AssignLog(msg=f'All files in {self.dir_audio} are fully analyzed; exiting analysis',
                          level_str='WARNING')
            )
            self.coordinator.event_analysisdone.set()
            self.proc_logger.join()
            return False
        else:
            self._log_debug(f'found {len(self.a_stream_list)} files in stream list')

        for a in self.a_stream_list:
            self.coordinator.q_stream.put(a)

        # queue termination sentinels for streamers at end of analysis
        a_stream_terminate = AssignStream(
            path_audio=None,
            dir_audio=None,
            duration_audio=None,
            chunklist=None,
            terminate=True
        )

        for _ in range(self.coordinator.streamers_total):
            self.coordinator.q_stream.put(a_stream_terminate)

        return True

    def _log_startup(self):
        msg = (
            f'Model: {self.modelname}\n'
            f'Frame hop: {self.framehop_prop}\n'
            f'Treshold: {self.threshold}\n'
            f'Output classes: {", ".join(self.classes_out)}\n'
            f'Input directory: {self.dir_audio}\n'
            f'Output directory: {self.dir_out}\n'
            f'CPU analyzers: {self.coordinator.analyzers_cpu}\n'
            f'GPU analyzer: {self.coordinator.analyzer_gpu}\n'
            f'Chunk length: {self.chunklength}s\n'
            f'Streamers: {self.coordinator.streamers_total}\n'
            f'Queue depth: {self.coordinator.queue_depth}\n'
        )

        self.coordinator.q_log.put(AssignLog(msg=msg, level_str='INFO'))


    def _launch_streamers(self):
        """Launch streamer workers (threads or processes based on mode)."""
        for s in range(self.coordinator.streamers_total):
            streamer = self.coordinator.concurrency_streamer(
                target=run_worker,
                name=f'streamer_{s}',
                kwargs={
                    'workerclass': WorkerStreamer,
                    'id_streamer': s,
                    'resample_rate': self.model.embedder.samplerate,
                    'streamers_live': self.coordinator.streamers_live,
                    'q_stream': self.coordinator.q_stream,
                    'q_analyze': self.coordinator.q_analyze,
                    'q_log': self.coordinator.q_log,
                    'event_stopanalysis': self.coordinator.event_stopanalysis,
                }
            )
            self.proc_streamers.append(streamer)
            self.proc_streamers[-1].start()


    def _launch_writer(self):
        """Launch the writer process."""
        self.proc_writer = self.coordinator.concurrency_writer(
            target=run_worker,
            name='writer_proc',
            kwargs={
                'workerclass': WorkerWriter,
                'classes_out': self.classes_out,
                'threshold': self.threshold,
                'classes': self.model.config['classes'],
                'framehop_s': self.model.embedder.framehop_s,
                'digits_time': self.model.embedder.digits_time,
                'dir_audio': self.dir_audio,
                'dir_out': self.dir_out,
                'digits_results': self.model.config['digits_results'],
                'q_write': self.coordinator.q_write,
                'q_log': self.coordinator.q_log,
                'event_stopanalysis': self.coordinator.event_stopanalysis,
                'event_analysisdone': self.coordinator.event_analysisdone,
                'analyzers_live': self.coordinator.analyzers_live,

            }
        )
        self.proc_writer.start()

    def _launch_cpu_analyzers(self):
        for a in range(self.coordinator.analyzers_total):
            analyzer = self.coordinator.concurrency_analyzer(
                target=run_worker,
                name=f"analyzer_cpu_{a}",
                kwargs={
                    'workerclass': WorkerAnalyzer,
                    'id_analyzer': a,
                    'processor': 'CPU',
                    'modelname': self.modelname,
                    'framehop_prop': self.framehop_prop,

                    'q_log': self.coordinator.q_log,
                    'q_write': self.coordinator.q_write,
                    'q_analyze': self.coordinator.q_analyze,
                    'event_stopanalysis': self.coordinator.event_stopanalysis,
                    'streamers_live': self.coordinator.streamers_live,
                    'analyzers_live': self.coordinator.analyzers_live,
                }
            )

            self.proc_analyzers.append(analyzer)
            self.proc_analyzers[-1].start()

    def _run_gpu_analyzer(self):
        """Run GPU analyzer blocking in main thread (GPU mode only)."""
        if not self.coordinator.analyzer_gpu:
            self.coordinator.q_log.put(AssignLog(msg='GPU mode disabled; skipping GPU analyzer', level_str='WARNING'))
            return

        # GPU analyzer runs in main thread to avoid pickling issues
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
        WorkerAnalyzer(
            id_analyzer='GPU',
            processor='GPU',
            modelname=self.modelname,
            framehop_prop=self.framehop_prop,
            q_log=self.coordinator.q_log,
            q_write=self.coordinator.q_write,
            q_analyze=self.coordinator.q_analyze,
            event_stopanalysis=self.coordinator.event_stopanalysis,
            streamers_live=self.coordinator.streamers_live,
            analyzers_live=self.coordinator.analyzers_live
        )()  # Call directly, blocking execution

    # TODO: this can probably be a Cleaner class invoked at start and end of anlaysis; holds stream list in attrib
    def _cleanup(self):
        """Perform final chunklist checks and cleanup."""
        for a in self.a_stream_list:
            path_audio = a.path_audio
            base_out = re.sub(self.dir_audio, self.dir_out, path_audio)
            base_out = os.path.splitext(base_out)[0]

            clean_and_chunk(
                ident_out=base_out,
                duration_audio=get_duration(path_audio, self.coordinator.q_log),
                framelength_s=self.model.embedder.framelength_s,
                chunklength=self.chunklength
            )

        self.timer_total.stop()

        self.coordinator.q_log.put(
            AssignLog(
                msg=f"{datetime.now()} - analysis complete; total time: {self.timer_total.get_total()}s",
                level_str='INFO'
            )
        )

    def run(self):
        """Execute the complete analysis workflow."""
        self._log_debug('launching logger')
        self._launch_logger()

        self._log_debug('building assignments')
        self._build_assignments()
        if not self.a_stream_list:
            return

        self._log_debug('initializing killer')
        killer = GracefulKiller(self.coordinator.event_stopanalysis)
        # Launch all workers
        self._log_debug('launching writing')
        self._launch_writer()

        self._log_debug('launching streamers')
        self._launch_streamers()

        self._log_debug('launching analyzers')
        self._launch_cpu_analyzers()

        # GPU analyzer runs blocking in main thread if enabled
        if self.coordinator.analyzer_gpu:
            self._run_gpu_analyzer()

        # Wait for completion
        self.coordinator.event_analysisdone.wait()

        # Cleanup
        self._cleanup()
        self.coordinator.event_analysisdone.set()
        self.proc_logger.join()
