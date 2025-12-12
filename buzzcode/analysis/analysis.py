import os
import re
import signal
from datetime import datetime

from buzzcode import config as cfg
from buzzcode.analysis.assignments import AssignLog, AssignStream
from buzzcode.analysis.results_coverage import clean_and_chunk
from buzzcode.analysis.workers import WorkerLogger, WorkerStreamer, WorkerWriter, WorkerAnalyzer, WorkerChecker, \
    Coordinator
from buzzcode.audio import get_duration
from buzzcode.models.load_model import load_model
from buzzcode.training.test import pull_sx
from buzzcode.utils import Timer


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
        self.coordinator.q_log.put(AssignLog(message=msg, level_str='DEBUG'))
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
            self.coordinator.q_log.put(AssignLog(message=msg, level_str='WARNING'))

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

        self.coordinator.q_log.put(AssignLog(message=msg, level_str='INFO'))

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
            self.coordinator.q_log.put(AssignLog(message='GPU mode disabled; skipping GPU analyzer', level_str='WARNING'))
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


    def run(self):
        """Execute the complete analysis workflow."""
        self._log_debug('launching logger')
        self._launch_logger()

        self._log_debug('building assignments')

        checker = WorkerChecker(
            dir_audio=self.dir_audio,
            dir_out=self.dir_out,
            framelength_s=self.model.embedder.framelength_s,
            chunklength=self.chunklength,
            coordinator=self.coordinator
        )

        checker.queue_assignments()

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
        checker.cleanup()
        self.coordinator.event_analysisdone.set()
        self.proc_logger.join()
