import os

import pandas as pd
import soundfile as sf

from src import config as cfg
from src.check.results_coverage import melt_coverage, get_gaps, smooth_gaps, gaps_to_chunklist
from src.pipeline.assignments import AssignFile, AssignLog
from src.pipeline.coordination import Coordinator, ExitSignal
from src.stream.audio import get_duration
from src.utils import search_dir


class WorkerChecker:
    def __init__(self,
                 dir_audio: str,
                 dir_results: str,
                 framelength_s: float,
                 chunklength: float,
                 coordinator: Coordinator,
                 ):

        self.dir_audio = dir_audio
        self.dir_results = dir_results
        self.framelength_s = framelength_s
        self.chunklength = chunklength
        self.coordinator = coordinator

        self.files_in = self._build_inputs()


    def _build_inputs(self):
        files_in = []
        for p in search_dir(self.dir_audio, extensions=list(sf.available_formats().keys())):
            a_file = AssignFile(path_audio=p, dir_audio=self.dir_audio, dir_results = self.dir_results)
            files_in.append(a_file)

        # check for conflicting idents; i.e., two files have identical names but different extensions
        # causes results to be written to the same file.
        idents = [f.ident for f in files_in]

        idents_conflicting = {ident for ident in idents if idents.count(ident) > 1}

        for ident_conflicting in idents_conflicting:
            paths_conflicting = [f.shortpath_audio for f in files_in if f.ident == ident_conflicting]
            msg = (f'The following files have conflicting names and will be skipped:\n'
                   f'{', '.join(paths_conflicting)}\n'
                   f'These files must be renamed before they can be analyzed.')
            self.coordinator.q_log.put(AssignLog(msg, 'WARNING'))

        return [f for f in files_in if f.ident not in idents_conflicting]


    def _chunk_file(self, a_file: AssignFile, cleanup=False):
        if cleanup:
            self.coordinator.q_log.put(AssignLog(message=f'Cleaning up results for {a_file.shortpath_audio}', level_str='INFO'))
        else:
            self.coordinator.q_log.put(AssignLog(message=f'Building assignments for {a_file.shortpath_audio}', level_str='INFO'))

        if os.path.exists(a_file.path_results_complete):
            if not cleanup:
                self.coordinator.q_log.put(AssignLog(f'Skipping {a_file.shortpath_audio}; fully analyzed', 'DEBUG'))
            return None


        if os.path.getsize(a_file.path_audio) < cfg.FILE_SIZE_MINIMUM:
            if not cleanup:
                self.coordinator.q_log.put(AssignLog(
                    message=f'Skipping {a_file.shortpath_audio}; below minimum analyzeable size', level_str='INFO'))
            return None

        if a_file.duration_audio is None:
            a_file.duration_audio = get_duration(a_file.path_audio, q_log=self.coordinator.q_log)

        # if the file hasn't been started, chunk the whole file
        if not os.path.exists(a_file.path_results_partial) and not cleanup:
            gaps = [(0, a_file.duration_audio)]
        else:
            # otherwise, read the file and calculate chunks
            df = pd.read_csv(a_file.path_results_partial)
            coverage = melt_coverage(df, self.framelength_s)
            gaps = get_gaps(
                range_in=(0, a_file.duration_audio),
                coverage_in=coverage
            )
            gaps = smooth_gaps(
                gaps,
                range_in=(0, a_file.duration_audio),
                framelength=self.framelength_s,
                gap_tolerance=self.framelength_s / 4
            )

            # if we find no gaps, this file was actually finished and never cleaned!
            # output to the finished file
            if not gaps:
                df.sort_values("start", inplace=True)
                df.to_csv(a_file.path_results_complete, index=False)
                os.remove(a_file.path_results_partial)
                return None

        if cleanup:
            return None
        else:
            return gaps_to_chunklist(gaps, self.chunklength)

    def queue_assignments(self):
        # exit if no compatible files
        if not self.files_in:
            self.coordinator.exit_analysis(
                ExitSignal(
                    message=f"Exiting analysis: no compatible audio files found in raw directory {self.dir_audio}.\n"
                            f"audio format must be one of: \n{', '.join(sf.available_formats().keys())}",
                    level='WARNING',
                    end_reason='no files'
                )
            )

            return

        for a_file in self.files_in:
            a_file.chunklist = self._chunk_file(a_file)

        # drop finished files
        self.files_in = [f for f in self.files_in if f.chunklist is not None]

        # exit if no files with chunks
        if not self.files_in:
            self.coordinator.exit_analysis(
                ExitSignal(
                    message=f"All files in {self.dir_audio} are fully analyzed; exiting analysis",
                    level='INFO',
                    end_reason='fully analyzed'
                )
            )
            return

        # else, queue 'em up
        for a_file in self.files_in:
            self.coordinator.q_stream.put(a_file)

        # queue termination sentinels
        for _ in range(self.coordinator.streamers_total):
            self.coordinator.q_stream.put(None)
        return

    def cleanup(self):
        for a_file in self.files_in:
            self._chunk_file(a_file, cleanup=True)
