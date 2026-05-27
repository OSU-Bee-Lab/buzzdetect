import multiprocessing
import threading
import warnings
from queue import Queue, Full
from dataclasses import dataclass, field

from src.pipeline.assignments import AssignFile, AssignChunk, AssignLog

# sentinel handed to a worker to tell it to stop
EXIT = 'exit'


class ExitSignal:
    def __init__(self, message, level, end_reason):
        self.message = message
        self.level = level
        self.end_reason = end_reason


@dataclass
class StreamTracker:
    chunks_streamed: list = field(default_factory=list)
    stream_in_progress: bool = True


class Coordinator:
    """
    Manages the coordination of analysis processes.

    The coordinator is the single owner of early/normal exit. Workers do not
    poll any exit flag; they call a coordinator getter and stop when it returns
    EXIT. On teardown the coordinator enqueues one EXIT sentinel per consumer
    onto each queue, so even a worker blocked in a bare queue ``.get()`` wakes
    up and receives its stop message.
    """
    def __init__(self,
                 analyzers_cpu: int,
                 analyzers_gpu: int=0,
                 streamers_total: int=None,
                 depth: int=None,
                 q_gui: Queue[AssignLog]=None,
                 event_analysisdone: multiprocessing.Event=None,
                 q_earlyexit: multiprocessing.Queue=None,):

        self.analyzers_cpu = analyzers_cpu
        self.analyzers_gpu = analyzers_gpu

        self.analyzers_total = analyzers_cpu + analyzers_gpu
        self.streamers_total = self._setup_streamers(self.analyzers_total) if streamers_total is None else streamers_total

        self.queue_depth = self._setup_depth() if depth is None else depth
        self.q_gui = q_gui

        self.assigned_chunks: dict[str, StreamTracker] = {}
        self._lock = threading.Lock()
        self._exit_lock = threading.Lock()

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

    # Worker-facing queue API
    #
    # Getters block on a bare ``.get()``. They unblock either because real work
    # arrived or because the coordinator poisoned the queue with EXIT.
    def get_stream(self):
        return self.q_stream.get()

    def put_analyze(self, a_chunk: AssignChunk):
        with self._lock:
            tracker = self.assigned_chunks.setdefault(a_chunk.file.ident, StreamTracker())
            tracker.chunks_streamed.append(a_chunk.chunk)
            if a_chunk.last_chunk:
                tracker.stream_in_progress = False

        # q_analyze is bounded, so block with a timeout. If exit is requested
        # while we're blocked on a full queue, drop the chunk silently — the
        # streamer will discover the exit on its next event check or when it
        # loops back to get_stream and pulls the sentinel.
        while True:
            if self.event_exitanalysis.is_set():
                return
            try:
                self.q_analyze.put(a_chunk, timeout=1)
                return
            except Full:
                continue

    def get_analyze(self):
        return self.q_analyze.get()

    def put_write(self, a_chunk: AssignChunk):
        self.q_write.put(a_chunk)

    def get_write(self):
        a_chunk = self.q_write.get()
        if a_chunk == EXIT:
            return EXIT

        with self._lock:
            self.assigned_chunks[a_chunk.file.ident].chunks_streamed.remove(a_chunk.chunk)
            fully_analyzed = self._ident_fully_analyzed(a_chunk.file.ident)

        return a_chunk, fully_analyzed

    def _ident_fully_analyzed(self, ident: str):
        tracker = self.assigned_chunks[ident]
        if len(tracker.chunks_streamed) > 0:
            return False
        if tracker.stream_in_progress:
            return False
        return True

    def _setup_streamers(self, n_analyzers):
        if self.analyzers_gpu > 0:
            n_streamers = n_analyzers*8
        else:
            n_streamers = n_analyzers

        return n_streamers

    def _setup_depth(self):
        return self.streamers_total * 2

    def _poison(self, queue: Queue, n: int):
        """Enqueue n EXIT sentinels so n blocked consumers each receive a stop."""
        for _ in range(n):
            queue.put(EXIT)

    def exit_analysis(self, exit_signal: ExitSignal):
        """ Note! Does not kill the logger; this needs to be done by analyzer process after cleanup """
        with self._exit_lock:
            # first caller wins: a normal-completion signal must not overwrite an
            # earlier early-exit reason (or vice versa).
            if self.end_reason is not None:
                return
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
            self._poison(self.q_analyze, self.analyzers_total)

            for t in threads_analyzers:
                t.join()
            self.log('analyzers done', 'DEBUG')
            self.analyzers_done.set()
            self._poison(self.q_write, 1)

            thread_writer.join()
            self.log('writer done', 'DEBUG')
            self.writer_done.set()

            self.exit_analysis(ExitSignal(message='Analysis complete', level='INFO', end_reason='completed'))

        def watch_queue():
            exit_message = self.q_earlyexit.get()
            self.exit_analysis(ExitSignal(message=exit_message, level='WARNING', end_reason='interrupted'))
            # wake every worker blocked in a getter
            self._poison(self.q_stream, self.streamers_total)
            self._poison(self.q_analyze, self.analyzers_total)
            self._poison(self.q_write, 1)

        thread_worker = threading.Thread(target=watch_workers, daemon=True)
        thread_worker.start()

        thread_exit = threading.Thread(target=watch_queue, daemon=True)
        thread_exit.start()

        self.event_exitanalysis.wait()
