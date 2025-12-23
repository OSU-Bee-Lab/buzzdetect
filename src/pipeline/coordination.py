import multiprocessing
import threading
from queue import Queue

from src.pipeline.assignments import AssignFile, AssignChunk, AssignLog


class ExitSignal:
    def __init__(self, message, level, end_reason):
        self.message = message
        self.level = level
        self.end_reason = end_reason


class Coordinator:
    """
    Manages the coordination of analysis processes.
    """
    def __init__(self,
                 analyzers_cpu: int,
                 analyzer_gpu: bool=False,
                 streamers_total: int=None,
                 depth: int=None,
                 q_gui: Queue[AssignLog]=None,
                 event_analysisdone: multiprocessing.Event=None,
                 q_earlyexit: multiprocessing.Queue=None,):

        self.analyzers_cpu = analyzers_cpu
        self.analyzer_gpu = analyzer_gpu

        self.analyzers_total = analyzers_cpu + analyzer_gpu
        self.streamers_total = self._setup_streamers(self.analyzers_total) if streamers_total is None else streamers_total

        self.queue_depth = self._setup_depth() if depth is None else depth
        self.q_gui = q_gui

        self.q_log: Queue[AssignLog] = Queue()
        self.q_stream: Queue[AssignFile | None] = Queue()
        self.q_analyze: Queue[AssignChunk] = Queue(maxsize=self.queue_depth)
        self.q_write: Queue[AssignChunk] = Queue()

        self.streamers_done = threading.Event()
        self.analyzers_done = threading.Event()
        self.writer_done = threading.Event()

        # these arguments are for running the analysis as a process under the GUI; if they aren't set,
        # just initialize as non-multiprocessing versions
        self.event_exitanalysis = event_analysisdone if event_analysisdone is not None else threading.Event()
        self.q_earlyexit = q_earlyexit if q_earlyexit is not None else Queue()

        self.end_reason = None

    def log(self, msg, level_str):
        self.q_log.put(AssignLog(message=f'coordinator: {msg}', level_str=level_str))

    def _setup_streamers(self, n_analyzers):
        if self.analyzer_gpu:
            n_streamers = n_analyzers*8
        else:
            n_streamers = n_analyzers

        return n_streamers

    def _setup_depth(self):
        return self.streamers_total * 2

    def exit_analysis(self, exit_signal: ExitSignal):
        """ Note! Does not kill the logger; this needs to be done by analyzer process after cleanup """
        self.q_log.put(AssignLog(message=exit_signal.message, level_str=exit_signal.level))
        self.end_reason = exit_signal.end_reason
        self.event_exitanalysis.set()

    # pass threads as arguments because they're defined after the coordinator is initialized
    def wait_for_exit(self,
                      threads_streamers: list[threading.Thread],
                      threads_analyzers: list[threading.Thread],
                      thread_writer: threading.Thread,
                      ):

        def watch_workers():
            for t in threads_streamers:
                t.join()
            self.log('streamers done', 'DEBUG')
            self.streamers_done.set()

            for t in threads_analyzers:
                t.join()
            self.log('analyzers done', 'DEBUG')
            self.analyzers_done.set()

            thread_writer.join()
            self.log('writer done', 'DEBUG')
            self.writer_done.set()

            self.exit_analysis(ExitSignal(message='Analysis complete', level='INFO', end_reason='completed'))

        def watch_queue():
            exit_message = self.q_earlyexit.get()
            self.exit_analysis(ExitSignal(message=exit_message, level='WARNING', end_reason='interrupted'))

        thread_worker = threading.Thread(target=watch_workers, daemon=True)
        thread_worker.start()

        thread_exit = threading.Thread(target=watch_queue, daemon=True)
        thread_exit.start()

        self.event_exitanalysis.wait()
