from src.inference.models import load_model
from src.pipeline.assignments import AssignChunk, AssignLog
from src.pipeline.coordination import Coordinator
from src.pipeline.progress_json import emit_progress
from src.pipeline.benchmark import now


class WorkerInferer:
    """Takes padded chunks off the queue, runs the model, hands results on.

    Deliberately does nothing else. Padding is the streamers' job and the
    progress/log reporting is the writer's (which is why the chunk carries
    `analyzer` and `analysis_s`): the only waiting this thread should do is
    waiting for audio.
    """
    def __init__(self,
                 id_analyzer,
                 processor: str,
                 modelname: str,
                 framehop_prop: float,
                 chunklength: float,
                 coordinator: Coordinator, ):

        self.id_analyzer = id_analyzer
        self.processor = processor
        self.coordinator = coordinator

        self.model = load_model(modelname, framehop_prop, initialize=False)
        self.chunklength = chunklength
        self.t_prev = None
        self.t_predict = 0.0


    def __call__(self):
        self.run()

    def log(self, msg, level_str):
        self.coordinator.q_log.put(AssignLog(message=f'analyzer {self.id_analyzer}: {msg}', level_str=level_str))

    # Substrings that mark an inference failure as "ran out of memory" rather
    # than anything wrong with the audio or the graph. onnxruntime reports an
    # exhausted device arena as a plain Fail with the allocator's own wording,
    # so there is no exception type to catch -- the text is what there is.
    OOM_MARKERS = (
        'out of memory',
        'cudaerrormemoryallocation',
        'failed to allocate memory',
        'cublas_status_alloc_failed',
        'hipErrorOutOfMemory'.lower(),
    )

    @classmethod
    def _is_oom(cls, e):
        if isinstance(e, MemoryError):
            return True
        text = str(e).lower()
        return any(marker in text for marker in cls.OOM_MARKERS)

    def process_chunk(self, a_chunk: AssignChunk):
        try:
            t_start = now()
            a_chunk.results = self.model.predict(a_chunk.samples, n_valid=a_chunk.n_samples)
            t_done = now()
            self.t_predict = t_done - t_start
        except Exception as e:
            # Chunk length is the one setting a user can act on here, and it is
            # not obvious from an allocator's error text that it is implicated
            # at all -- the session is built for the whole chunk, so the
            # footprint scales with it. Say so, then let the failure through:
            # an analyzer that cannot infer has nothing useful left to do, and
            # run_worker winds the analysis down rather than leaving the
            # streamers filling a queue nobody drains.
            if self._is_oom(e):
                raise MemoryError(
                    f'ran out of memory inferring a {self.chunklength}s chunk on '
                    f'{self.processor}. Lower --chunklength (the inference session '
                    f'is built for a whole chunk, so its memory scales with it) or '
                    f'use fewer analyzers. Original error: {e}') from e
            raise

        # the audio is spent; don't hold a chunk's worth of RAM until the writer is done
        a_chunk.samples = None
        # all the writer needs to report the rate: who, and the wall time since
        # this analyzer finished its previous chunk (so waiting counts against it)
        a_chunk.analyzer = self.id_analyzer
        a_chunk.analysis_s = t_done - self.t_prev
        self.t_prev = t_done
        self.coordinator.put_write(a_chunk)

    def run(self):
        self.log('launching', 'INFO')
        self.log(f'processing on {self.processor}', 'INFO')
        # onnxruntime picks its own execution provider and says so itself if it
        # cannot get the one it asked for (src/inference/onnx.py). There is
        # nothing for this worker to place by hand.
        self.model.processor = self.processor
        # The session is built at one fixed input length, because CoreML cannot
        # compile a graph with an unbounded dimension. The streamers pad each
        # chunk up to it (WorkerStreamer computes the same length from the same
        # chunklength) and predict() drops the frames that padding produced.
        self.model.samples_session = self.model.session_length(self.chunklength)
        self.model.initialize()
        # The session is built, so this worker is ready for its first chunk.
        # Emitted per analyzer; a host GUI is expected to treat the stage as
        # monotonic and ignore the repeats.
        emit_progress('stage', name='analyzing', processor=self.processor)

        self.t_prev = now()
        while True:
            depth = self.coordinator.q_analyze.qsize()
            t_ask = now()
            a_chunk = self.coordinator.get_analyze()
            if a_chunk == 'exit':
                break

            t_got = now()
            self.process_chunk(a_chunk)
            t_done = now()
            self.coordinator.bench.record(
                'analyzer', self.id_analyzer,
                t_wait=t_got - t_ask,
                **self.model.timings,
                t_post=t_done - t_got - self.t_predict,
                depth=depth,
            )

        self.log("terminating", 'INFO')
