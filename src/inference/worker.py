from _queue import Empty

import tensorflow as tf

from src.inference.models import load_model
from src.pipeline.assignments import AssignChunk, AssignLog
from src.pipeline.coordination import Coordinator
from src.utils import Timer


class WorkerInferer:
    def __init__(self,
                 id_analyzer,
                 processor: str,
                 modelname: str,
                 framehop_prop: float,
                 coordinator: Coordinator, ):

        self.id_analyzer = id_analyzer
        self.processor = processor
        self.coordinator = coordinator

        self.model = load_model(modelname, framehop_prop, initialize=False)
        self.timer_analysis = Timer()
        self.timer_bottleneck = Timer()


    def __call__(self):
        self.run()

    def log(self, msg, level_str):
        self.coordinator.q_log.put(AssignLog(message=f'analyzer {self.id_analyzer}: {msg}', level_str=level_str))

    def _managememory(self):
        if self.processor == 'CPU':
            tf.config.set_visible_devices([], 'GPU')
            visible_devices = tf.config.get_visible_devices()
            for device in visible_devices:
                assert device.device_type != 'GPU'
        elif self.processor == 'GPU':
            # let memory grow when processing on GPU
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if not gpus:
                self.log("GPU not found; using CPU", 'WARNING')
                self.processor = 'CPU'
            else:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

            if len(gpus) > 1:
                self.log("Use of multiple GPUs is untested", 'WARNING')

        self.log(f"processing on {self.processor}", 'INFO')


    def report_rate(self, a_chunk: AssignChunk):
        chunk_duration = a_chunk.chunk[1] - a_chunk.chunk[0]

        self.timer_analysis.stop()
        analysis_rate = (chunk_duration / self.timer_analysis.get_total(5)).__round__(1)

        msg = (f"analyzed {a_chunk.file.shortpath_audio}, chunk ({float(a_chunk.chunk[0])}, {float(a_chunk.chunk[1])}) "
                 f"in {self.timer_analysis.get_total()}s (rate: {analysis_rate})")

        self.log(msg, 'PROGRESS')
        self.timer_analysis.restart()

    def report_bottleneck(self):
        msg = f"BUFFER BOTTLENECK: analyzer {self.id_analyzer} received assignment after {self.timer_bottleneck.get_total().__round__(1)}s"
        self.log(msg, 'DEBUG')

    def process_chunk(self, a_chunk: AssignChunk):
        a_chunk.results = self.model.predict(a_chunk.samples)

        self.coordinator.q_write.put(a_chunk)
        self.report_rate(a_chunk)

    def run(self):
        self.log('launching', 'INFO')
        self.model.initialize()

        self.timer_bottleneck.restart()
        while not self.coordinator.event_exitanalysis.is_set():
            try:
                a_chunk = self.coordinator.q_analyze.get(timeout=1)
                self.timer_bottleneck.stop()
                if self.timer_bottleneck.get_total() > 0.01:
                    self.report_bottleneck()
                self.process_chunk(a_chunk)
                self.timer_bottleneck.restart()
            except Empty:
                # if the streamers are done, exit as usual
                if self.coordinator.streamers_done.is_set():
                    self.log('all streamers done; terminating', 'DEBUG')
                    return

                # otherwise, try polling the queue again
                pass

        self.log("exit event set, terminating", 'DEBUG')
