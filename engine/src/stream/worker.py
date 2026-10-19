from queue import Full

import soxr
import numpy as np

from src.pipeline.assignments import AssignChunk
import os

import pandas as pd
from src.stream.audio import build_track

from src import config as cfg
from src.stream.results_coverage import melt_coverage, get_gaps, smooth_gaps, gaps_to_chunklist
from src.pipeline.assignments import AssignFile, AssignLog
from src.pipeline.coordination import Coordinator
from src.stream.audio import get_duration
from src.inference.models import OnnxModel
from src.pipeline.progress_json import emit_progress
from src.pipeline.benchmark import now


def resampled_length(n, rate_in, rate_out):
    """How many samples soxr.resample returns for `n` samples in.

    Its one-shot call rounds half up (C's floor(x + 0.5), not Python's round,
    which rounds half to even), checked against soxr 1.1 at every rate and
    quality the engine can hand it. Integer arithmetic so a long chunk can't
    land on the wrong side of a .5.
    """
    return (2 * n * rate_out + rate_in) // (2 * rate_in)


class WorkerStreamer:
    def __init__(self,
                 id_streamer,
                 model: OnnxModel,
                 chunklength: float,
                 coordinator: Coordinator,
                 full_quality_decode: bool = False,
                 gain: float = 1.0, ):

        self.model = model
        self.id_streamer = id_streamer
        self.coordinator = coordinator

        self.chunklength = chunklength
        self.framelength_s = self.model.framelength_s
        self.resample_rate = self.model.samplerate
        # soxr QQ is ~3x cheaper than HQ and lands ~-46 dB from a VHQ reference
        # in-band, against ~-88 dB for HQ; see --full_quality_decode
        self.resample_quality = 'HQ' if full_quality_decode else 'QQ'
        self.gain = gain
        # the analyzers build their sessions at this length (see WorkerInferer.run)
        self.samples_session = self.model.session_length(chunklength)

        # Audio read past the end of the previous chunk, kept for the next one;
        # see _read_span. Native rate, downmixed. Reset per file.
        self._carry = None
        self._carry_from = None

    def __call__(self):
        self.run()

    def log(self, msg, level_str):
        self.coordinator.q_log.put(AssignLog(message=f'streamer {self.id_streamer}: {msg}', level_str=level_str))

    def handle_bad_read(self, a_file: AssignFile):
        # we've found that many of our .mp3 files give an incorrect .frames count, or else headers are broken
        # this appears to be because our recorders ran out of battery while recording
        # SoundFile does not handle this gracefully, so we catch it here.

        final_frame = a_file.track.tell()
        final_second = final_frame/a_file.track.samplerate

        msg = f"Unreadable audio at {round(final_second, 1)}s out of {round(a_file.duration_audio, 1)}s for {a_file.shortpath_audio}."
        if 1 - (final_second / a_file.duration_audio) > cfg.BAD_READ_ALLOWANCE:
            # if we get a bad read in the middle of a file, this deserves a warning.
            level = 'WARNING'
            msg += '\nAborting early due to corrupt audio data.'
        else:
            # but bad reads at the ends of files are almost guaranteed when the batteries run out
            level = 'DEBUG'
            msg += '\nBad audio is near file end, results should be mostly unaffected.'

        self.log(msg, level)

    def _chunk_file(self, a_file: AssignFile):
        if self.coordinator.event_exitanalysis.is_set():
            a_file.chunklist = []
            return

        if os.path.exists(a_file.path_results_complete):
            self.log(f'Skipping {a_file.shortpath_audio}; already analyzed', 'DEBUG')
            emit_progress('file_skip', path=a_file.shortpath_audio, reason='already_analyzed')
            a_file.chunklist = []
            return

        if os.path.getsize(a_file.path_audio) < cfg.FILE_SIZE_MINIMUM:
            self.log(f'Skipping {a_file.shortpath_audio}; below minimum analyzeable size', 'DEBUG')
            emit_progress('file_skip', path=a_file.shortpath_audio, reason='too_small')
            a_file.chunklist = []
            return

        a_file.track = build_track(a_file.path_audio)
        a_file.duration_audio = get_duration(a_file, q_log=self.coordinator.q_log)

        # if the file hasn't been started, chunk the whole file
        if not os.path.exists(a_file.path_results_partial):
            gaps = [(0, a_file.duration_audio)]
        else:
            # otherwise, read the results and calculate chunks
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
            # as of 2026-05-18, this shouldn't be possible because cleanup is incorporated into the writer.
            # before, cleanup happened at the end of all analyses. I'll leave this, just in case.
            # output to the finished file
            if not gaps:
                self.log(f'Discovered non-cleaned file at {a_file.shortpath_audio}; cleaning results', 'DEBUG')
                df.sort_values("start", inplace=True)
                df.to_csv(a_file.path_results_complete, index=False)
                os.remove(a_file.path_results_partial)
                emit_progress('file_skip', path=a_file.shortpath_audio, reason='already_analyzed')
                a_file.chunklist = []
                return

        emit_progress(
            'file_start',
            path=a_file.shortpath_audio,
            duration=a_file.duration_audio,
            work_seconds=sum(b - a for a, b in gaps),
        )
        a_file.chunklist = gaps_to_chunklist(gaps, self.chunklength)
        return

    def _read_span(self, a_file, sample_from, sample_to):
        """Native-rate mono samples [sample_from, sample_to), or fewer at EOF.

        Reads only forward. Chunks are normally contiguous, so the look-ahead
        read for one chunk is the start of the next: it is kept in `_carry`
        and the track, already positioned at its end, carries on from there.
        Seeking back to re-read it instead would be cheap on a wav and ruinous
        elsewhere -- the mts driver answers a backward seek by decoding the
        file again from its first sample, every chunk.
        """
        track = a_file.track
        parts = []
        position = sample_from
        carry = self._carry
        if carry is not None and self._carry_from <= sample_from <= self._carry_from + len(carry):
            kept = carry[sample_from - self._carry_from:sample_to - self._carry_from]
            parts.append(kept)
            position = self._carry_from + len(carry)
        else:
            # Not the chunk the carry was read for (the first one, or after a
            # gap in resumed results): an ordinary seek.
            track.seek(sample_from)

        if position < sample_to:
            new = track.read(sample_to - position, dtype=np.float32)
            if track.channels > 1:
                new = np.mean(new, axis=1)
            parts.append(new)

        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts) if len(parts) > 1 else parts[0]

    def queue_chunk(self, a_file, chunk: tuple[float, float], force_last: bool = False):
        """Read, resample and pad one chunk for the analyzers.

        The chunk is read together with enough of the audio after it to fill
        the session, not padded with silence. A frame's input runs past the
        frame -- 15 ms on a YAMNet-framed model, nearly a whole frame more on
        heavy_v1 (samples_min is 30719) -- so the last frame of a chunk padded
        with zeros is computed against silence that is not in the file, and
        moves by up to ~0.5 on heavy_v1 at every seam. That made a result
        depend on the chunk length. With the look-ahead, every frame sees the
        audio a whole-file pass would give it, and only the end of the file is
        padded. The look-ahead is input only: the frames returned are still
        those of the chunk's own audio (n_samples), so nothing is reported
        twice.
        """
        rate_in = a_file.track.samplerate
        # Rounded, not truncated: chunk times are decimal seconds, and
        # 267.84 * 44100 comes out at 11811743.999999998. int() put about one
        # chunk start in twenty a sample early at 44.1/48 kHz, which shifts
        # every frame in that chunk off the file's frame grid -- through the
        # resampler that moved heavy_v1 by up to 1.0. Contiguous chunks still
        # meet exactly, since both ends go through the same expression.
        sample_from = round(chunk[0] * rate_in)
        sample_to = round(chunk[1] * rate_in)
        read_size = sample_to - sample_from

        # Native samples that, resampled, fill the session behind the chunk.
        # One spare sample, since the two lengths are rounded separately.
        n_chunk_out = resampled_length(read_size, rate_in, self.resample_rate)
        lookahead = max(0, -(-(self.samples_session - n_chunk_out) * rate_in // self.resample_rate) + 1)

        t0 = now()
        span = self._read_span(a_file, sample_from, sample_to + lookahead)
        t1 = now()
        samples = span[:read_size]
        self._carry = span[read_size:]
        self._carry_from = sample_to
        t2 = now()

        n_samples = len(samples)

        if n_samples < read_size:
            self.handle_bad_read(a_file)
            chunk = (chunk[0], round(chunk[0] + (n_samples/a_file.track.samplerate), self.model.digits_time))
            continue_file = False
        else:
            continue_file = True

        # soxr directly rather than librosa.resample, which is a thin wrapper
        # over this same call (res_type='soxr_hq') but drags numba/llvmlite/scipy
        # in with it -- a few hundred MB of dependency for one function.
        # The chunk and its look-ahead in one call, so the resampler runs
        # straight through the seam rather than ending on the chunk's last
        # sample. n_audio is what the chunk alone would have resampled to.
        n_audio = resampled_length(n_samples, rate_in, self.resample_rate)
        samples = soxr.resample(span if continue_file else samples, rate_in, self.resample_rate,
                                quality=self.resample_quality)
        samples = samples[:self.samples_session].astype(np.float32)
        # after resampling, where there are usually fewer samples to scale.
        # Not clipped: the model takes float input with no ceiling.
        if self.gain != 1:
            samples *= np.float32(self.gain)
        t3 = now()

        # pad here rather than in the analyzer, which should only ever run the model;
        # only the end of the file still gets any
        samples = self.model.pad(samples, self.samples_session)
        t_pad = now() - t3

        last_chunk = force_last or not continue_file
        a_chunk = AssignChunk(file=a_file, chunk=chunk, samples=samples, n_samples=n_audio, last_chunk=last_chunk)
        self.coordinator.put_analyze(a_chunk)
        t4 = now()

        self.coordinator.bench.record(
            'streamer', self.id_streamer,
            t_read=t1 - t0,
            t_downmix=t2 - t1,
            t_resample=t3 - t2,
            t_pad=t_pad,
            t_put_wait=t4 - t3 - t_pad,
            audio_s=round(chunk[1] - chunk[0], 2),
        )

        return continue_file

    def stream_to_queue(self, a_file: AssignFile):
        self._carry = None
        self._carry_from = None
        try:
            t_open = now()
            self._chunk_file(a_file)
            self.coordinator.bench.record('streamer_open', self.id_streamer, t_open=now() - t_open)

            last_index = len(a_file.chunklist) - 1
            for i, chunk in enumerate(a_file.chunklist):
                # reading and resampling can be very slow, so opportunistically bail mid-file
                # rather than committing to the next chunk's work
                if self.coordinator.event_exitanalysis.is_set():
                    return
                continue_file = self.queue_chunk(a_file=a_file, chunk=chunk, force_last=i == last_index)
                if not continue_file:
                    break
        finally:
            if a_file.track is not None:
                a_file.track.close()
                a_file.track = None

    def run(self):
        self.log('launching', 'INFO')
        while True:
            if self.coordinator.event_exitanalysis.is_set():
                break
            t_idle = now()
            a_file = self.coordinator.get_stream()
            self.coordinator.bench.record('streamer_idle', self.id_streamer, t_wait_file=now() - t_idle)
            if a_file == 'exit' or self.coordinator.event_exitanalysis.is_set():
                break

            self.log(f"buffering {a_file.shortpath_audio}", 'INFO')
            self.stream_to_queue(a_file)

        self.log("terminating", 'INFO')
