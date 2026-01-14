import os
from _queue import Empty

from src.pipeline.assignments import AssignChunk, AssignLog
from src.pipeline.coordination import Coordinator
from src.write.formatting import format_activations, format_detections


class WorkerWriter:
    def __init__(self,
                 classes_out,
                 threshold,
                 classes,
                 framehop_s,
                 digits_time,
                 dir_audio,
                 dir_out,
                 digits_results,
                 coordinator: Coordinator, ):

        self.classes_out = classes_out
        self.threshold = threshold
        self.classes = classes
        self.framehop_s = framehop_s
        self.digits_time = digits_time
        self.dir_audio = dir_audio
        self.dir_out = dir_out
        self.digits_results = digits_results
        self.coordinator = coordinator

        if self.threshold is None:
            def format_func(results, time_start):
                out = format_activations(
                    results=results,
                    classes=classes,
                    framehop_s=framehop_s,
                    time_start=time_start,
                    digits_time=digits_time,
                    classes_keep=classes_out,
                    digits_results=digits_results
                )

                return out

        else:
            def format_func(results, time_start):
                out = format_detections(
                    results,
                    threshold,
                    classes,
                    framehop_s,
                    digits_time,
                    time_start
                )

                return out

        self.format = format_func

    def __call__(self):
        self.run()

    def log(self, msg, level_str):
        self.coordinator.q_log.put(AssignLog(message=f'writer: {msg}', level_str=level_str))

    def write_results(self, a_chunk: AssignChunk):
        output = self.format(
            results=a_chunk.results.numpy(),
            time_start=a_chunk.chunk[0]
        )

        path_results_partial = a_chunk.file.path_results_partial

        os.makedirs(os.path.dirname(path_results_partial), exist_ok=True)

        # Check if file exists to determine if we need to write headers
        file_exists = os.path.exists(path_results_partial)

        # Append to existing file or create new one with headers
        output.to_csv(path_results_partial, mode='a', header=not file_exists, index=False)

    def run(self):
        self.log('launching', 'INFO')
        while not self.coordinator.event_exitanalysis.is_set():
            try:
                # we might poll the queue just before the analysisdone event is set,
                # so use a timeout to re-check
                self.write_results(self.coordinator.q_write.get(timeout=5))
            except Empty:
                if self.coordinator.analyzers_done.is_set():
                    self.log('all analyzers done; terminating', 'DEBUG')
                    return
                pass

        self.log("exit event set, terminating", 'DEBUG')
