import os

import numpy as np
import pandas as pd

from src.pipeline.assignments import AssignChunk, AssignLog
from src.pipeline.coordination import Coordinator
from src.pipeline.benchmark import now
from src.pipeline.progress_json import emit_progress
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

    def report_chunk(self, a_chunk: AssignChunk):
        # Reported from here, not the analyzer: formatting a line and writing to
        # stdout (a pipe to the GUI, which can block) is not the model's job.
        # The rate is the analyzer's -- audio seconds per wall second between
        # its finished chunks -- so it includes any time it spent waiting.
        chunk_duration = a_chunk.chunk[1] - a_chunk.chunk[0]
        d = self.digits_time
        msg = (f"analyzer {a_chunk.analyzer}: analyzed {a_chunk.file.shortpath_audio}, "
               f"chunk ({float(a_chunk.chunk[0]):.{d}f}, {float(a_chunk.chunk[1]):.{d}f})")
        if a_chunk.analysis_s:
            msg += f" in {a_chunk.analysis_s:.2f}s (rate: {chunk_duration / a_chunk.analysis_s:.1f})"

        self.coordinator.q_log.put(AssignLog(message=msg, level_str='PROGRESS'))
        emit_progress(
            'chunk_done',
            path=a_chunk.file.shortpath_audio,
            chunk_start=float(a_chunk.chunk[0]),
            chunk_end=float(a_chunk.chunk[1]),
            done=a_chunk.last_chunk,
        )

    def write_results(self, a_chunk: AssignChunk, fully_analyzed: bool):
        output = self.format(
            results=a_chunk.results,
            time_start=a_chunk.chunk[0]
        )

        path_results_partial = a_chunk.file.path_results_partial

        os.makedirs(os.path.dirname(path_results_partial), exist_ok=True)

        # Check if file exists to determine if we need to write headers
        file_exists = os.path.exists(path_results_partial)

        # Append to existing file or create new one with headers
        output.to_csv(path_results_partial, mode='a', header=not file_exists, index=False)

        if fully_analyzed:
            df = pd.read_csv(a_chunk.file.path_results_partial)
            df.sort_values("start", inplace=True)
            df.to_csv(a_chunk.file.path_results_complete, index=False)
            os.remove(a_chunk.file.path_results_partial)


    def run(self):
        self.log('launching', 'INFO')
        while True:
            depth = self.coordinator.q_write.qsize()
            t_ask = now()
            item = self.coordinator.get_write()
            if item == 'exit':
                break

            t_got = now()
            a_chunk, fully_analyzed = item
            self.write_results(a_chunk, fully_analyzed)
            t_written = now()
            self.report_chunk(a_chunk)
            self.coordinator.bench.record('writer', t_wait=t_got - t_ask, t_write=t_written - t_got,
                                          t_report=now() - t_written, depth=depth)

        self.log("terminating", 'INFO')
