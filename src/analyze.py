import multiprocessing
import os
import threading

from src import config as cfg
from src.inference.models import load_model
from src.inference.worker import WorkerInferer
from src.check.worker import WorkerChecker
from src.pipeline.coordination import Coordinator
from src.pipeline.logger import WorkerLogger
from src.pipeline.assignments import AssignLog
from src.stream.worker import WorkerStreamer
from src.utils import Timer
from src.write.thresholds import pull_sx
from src.write.worker import WorkerWriter


def run_worker(workerclass, **kwargs):
    worker = workerclass(**kwargs)
    worker()
    print(f"DEBUG, run_worker: {worker.__class__.__name__} finished.")


class Analyzer:
    """Audio analysis orchestrator."""
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

        self.thread_logger = None
        self.thread_writer = None
        self.threads_streamers = []
        self.threads_analyzers = []

    def _log_debug(self, msg):
        self.coordinator.q_log.put(AssignLog(message=msg, level_str='DEBUG'))
        print(f'DEBUG, ANALYZER: {msg}')

    def _setup_chunklength(self, chunklength):
        # Round chunklength to nearest frame for seamless processing
        chunklength = round(
            chunklength / self.model.embedder.framelength_s
        ) * self.model.embedder.framelength_s
        chunklength = round(chunklength, self.model.embedder.digits_time)
        if chunklength < self.model.embedder.framelength_s:
            chunklength = self.model.embedder.framelength_s

        return chunklength

    # Setup methods
    #
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

    def _launch_logger(self):
        """Start the logging process."""
        path_log = os.path.join(
            self.dir_out,
            f"{self.timer_total.time_start.strftime('%Y-%m-%d_%H%M%S')}.log"
        )
        os.makedirs(os.path.dirname(path_log), exist_ok=True)

        self.thread_logger = threading.Thread(
            target=run_worker,
            name='logger_proc',

            kwargs={
                'workerclass': WorkerLogger,
                'path_log': path_log,
                'verbosity_print': self.verbosity_print,
                'verbosity_log': self.verbosity_log,
                'log_progress': self.log_progress,
                'coordinator': self.coordinator,
            }
        )
        self.thread_logger.start()

        if self.framehop_prop > 1:
            msg = (
                'Currently, analyses with framehop > 1 will produce valid results, '
                'but buzzdetect will interpret the resulting gaps as missing data.\n'
                f'Fully analyzed files will not be converted from {cfg.SUFFIX_RESULT_PARTIAL} '
                f'to {cfg.SUFFIX_RESULT_COMPLETE}.\n'
                f'Repeated analysis will attempt to fill gaps between frames.'
            )
            self.coordinator.q_log.put(AssignLog(message=msg, level_str='WARNING'))

    # Process launching
    #
    def _log_startup(self):
        msg = (
            f'Model: {self.modelname}\n'
            f'Frame hop: {self.framehop_prop}\n'
            f'Threshold: {self.threshold}\n'
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
            streamer = threading.Thread(
                target=run_worker,
                name=f'streamer_{s}',
                kwargs={
                    'workerclass': WorkerStreamer,
                    'id_streamer': s,
                    'resample_rate': self.model.embedder.samplerate,
                    'coordinator': self.coordinator,
                }
            )
            self.threads_streamers.append(streamer)
            self.threads_streamers[-1].start()

    def _launch_writer(self):
        """Launch the writer process."""
        self.thread_writer = threading.Thread(
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
                'coordinator': self.coordinator

            }
        )
        self.thread_writer.start()


    def _launch_analyzers(self):
        for a in range(self.coordinator.analyzers_total):
            analyzer = threading.Thread(
                target=run_worker,
                name=f"analyzer_cpu_{a}",
                kwargs={
                    'workerclass': WorkerInferer,
                    'id_analyzer': a,
                    'processor': 'CPU',
                    'modelname': self.modelname,
                    'framehop_prop': self.framehop_prop,
                    'coordinator': self.coordinator,
                }
            )

            self.threads_analyzers.append(analyzer)

        if self.coordinator.analyzer_gpu:
            os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
            analyzer_gpu = threading.Thread(
                target=run_worker,
                name='analyzer_gpu',
                kwargs={
                    'workerclass': WorkerInferer,
                    'id_analyzer': 'GPU',
                    'processor': 'GPU',
                    'modelname': self.modelname,
                    'framehop_prop': self.framehop_prop,
                    'coordinator': self.coordinator,
                }
            )

            self.threads_analyzers.append(analyzer_gpu)

        for t in self.threads_analyzers:
            t.start()

    def run(self):
        """Execute the complete analysis workflow."""
        self._log_startup()
        self._launch_logger()


        checker = WorkerChecker(
            dir_audio=self.dir_audio,
            dir_results=self.dir_out,
            framelength_s=self.model.embedder.framelength_s,
            chunklength=self.chunklength,
            coordinator=self.coordinator
        )

        checker.queue_assignments()

        # checker can trigger an early exit based on the state of the files;
        # if this happens, don't even launch
        if self.coordinator.event_exitanalysis.is_set():
            self.coordinator.q_log.put(AssignLog(message='', level_str='INFO', terminate=True))
            return

        self._launch_writer()
        self._launch_streamers()
        self._launch_analyzers()

        self.coordinator.wait_for_exit(
            threads_streamers=self.threads_streamers,
            threads_analyzers=self.threads_analyzers,
            thread_writer=self.thread_writer
        )

        if self.coordinator.end_reason == 'completed':
            checker.cleanup()
            self.timer_total.stop()
            analysis_time = self.timer_total.get_total()

            self.coordinator.q_log.put(AssignLog(message=f'\nAll files analyzed and cleaned.\nTotal analysis time: {analysis_time.__format__(',')}s', level_str='INFO', terminate=False))

        self.coordinator.q_log.put(AssignLog(message='', level_str='INFO', terminate=True))
        self.thread_logger.join()



def analyze(
        modelname: str,
        classes_out: list = 'all',
        precision: float = None,
        framehop_prop: float = 1,
        chunklength: float = 200,
        analyzers_cpu: int = 1,
        analyzer_gpu: bool = False,
        n_streamers: int = None,
        stream_buffer_depth: int = None,
        dir_audio: str = cfg.DIR_AUDIO,
        dir_out: str = None,
        verbosity_print: str = 'PROGRESS',
        verbosity_log: str = 'DEBUG',
        log_progress: bool = False,
        q_gui: multiprocessing.Queue = None,
        event_stopanalysis: multiprocessing.Event = None,
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
        Length of audio chunks in seconds, by default 200. Try different values to tune for your machine.
    analyzers_cpu : int, optional
        Number of parallel CPU workers, by default 1
    analyzer_gpu : bool, optional
        Whether to launch a GPU worker for analysis, by default False
    n_streamers : int, optional
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
    verbosity_print : str, optional
        Level of verbosity for print statements to console (INFO, DEBUG, WARNING, ERROR), by default 'PROGRESS')
    verbosity_log : str, optional
        Level of verbosity for logging to file (INFO, DEBUG, WARNING, ERROR), by default 'DEBUG'
    log_progress : bool, optional
        Whether or not to log progress statements to file, by default False
        For long analyses with small chunks, this can result in log files megabytes in size.
    q_gui : multiprocessing.Queue, optional
        Queue for passing log messages to GUI, by default None
    event_stopanalysis : multiprocessing.Event, optional
        Event for killing analysis, by default None
        If set, allows external stopping of analysis

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

    coordinator = Coordinator(
        analyzers_cpu=analyzers_cpu,
        analyzer_gpu=analyzer_gpu,
        streamers_total=n_streamers,
        depth=stream_buffer_depth,
        q_gui=q_gui,
        event_analysisdone=event_stopanalysis
    )

    analyzer = Analyzer(
        modelname=modelname,
        classes_out=classes_out,
        precision=precision,
        framehop_prop=framehop_prop,
        chunklength=chunklength,
        dir_audio=dir_audio,
        dir_out=dir_out,
        verbosity_print=verbosity_print,
        verbosity_log=verbosity_log,
        log_progress=log_progress,
        coordinator=coordinator
    )

    analyzer.run()
