from src.inference.models import load_model
from src.pipeline.assignments import AssignChunk, AssignLog
from src.pipeline.coordination import Coordinator
from src.pipeline.progress_json import emit_progress
from src.utils import Timer


class WorkerInferer:
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
        self.timer_analysis = Timer()
        self.timer_bottleneck = Timer()


    def __call__(self):
        self.run()

    def log(self, msg, level_str):
        self.coordinator.q_log.put(AssignLog(message=f'analyzer {self.id_analyzer}: {msg}', level_str=level_str))

    def report_rate(self, a_chunk: AssignChunk):
        chunk_duration = a_chunk.chunk[1] - a_chunk.chunk[0]

        self.timer_analysis.stop()
        analysis_rate = chunk_duration / self.timer_analysis.get_total(5)

        digits_time = self.model.digits_time
        msg = (f"analyzed {a_chunk.file.shortpath_audio}, chunk ({float(a_chunk.chunk[0]):.{digits_time}f}, {float(a_chunk.chunk[1]):.{digits_time}f}) "
                 f"in {self.timer_analysis.get_total():.2f}s (rate: {analysis_rate:.1f})")

        self.log(msg, 'PROGRESS')
        emit_progress(
            'chunk_done',
            path=a_chunk.file.shortpath_audio,
            chunk_start=float(a_chunk.chunk[0]),
            chunk_end=float(a_chunk.chunk[1]),
            done=a_chunk.last_chunk,
        )
        self.timer_analysis.restart()

    def report_bottleneck(self):
        msg = f"BUFFER BOTTLENECK: analyzer {self.id_analyzer} received assignment after {self.timer_bottleneck.get_total().__round__(1)}s"
        self.log(msg, 'DEBUG')

    def process_chunk(self, a_chunk: AssignChunk):
        a_chunk.results = self.model.predict(a_chunk.samples)
        self.coordinator.put_write(a_chunk)
        self.report_rate(a_chunk)

    def run(self):
        self.log('launching', 'INFO')
        self.log(f'processing on {self.processor}', 'INFO')
        # onnxruntime picks its own execution provider and says so itself if it
        # cannot get the one it asked for (src/inference/onnx.py). There is
        # nothing for this worker to place by hand.
        self.model.processor = self.processor
        # The session is built at one fixed input length, because CoreML cannot
        # compile a graph with an unbounded dimension. predict() pads each chunk
        # up to it and drops the frames that padding produced.
        self.model.samples_session = self.model.session_length(self.chunklength)
        self.model.initialize()
        # The session is built, so this worker is ready for its first chunk.
        # Emitted per analyzer; a host GUI is expected to treat the stage as
        # monotonic and ignore the repeats.
        emit_progress('stage', name='analyzing', processor=self.processor)

        self.timer_bottleneck.restart()
        while True:
            a_chunk = self.coordinator.get_analyze()
            if a_chunk == 'exit':
                break

            self.timer_bottleneck.stop()
            if self.timer_bottleneck.get_total() > 0.01:
                self.report_bottleneck()
            self.process_chunk(a_chunk)
            self.timer_bottleneck.restart()

        self.log("terminating", 'INFO')
