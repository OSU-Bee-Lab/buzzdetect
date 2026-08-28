"""End-to-end: buzzdetect_cli.py over a small audio folder.

The engine is a subprocess with a stdout protocol, which is exactly how the
desktop app uses it, so these tests drive it the same way -- run it, read the
BDPROGRESS lines, look at what landed on disk. Everything here is a real
analysis; the fixtures are seconds long, so a whole run costs well under a
second.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import soundfile as sf

import tests._context as ctx  # noqa: F401

import src.config as cfg
from src.pipeline.manifest import FNAME_MANIFEST, read_manifest
from src.pipeline.progress_json import MARKER

MODEL = 'model_general_v3'
FRAMELENGTH = 0.96


def read(path):
    with open(path) as f:
        return f.read()


class Run:
    """One engine invocation and everything it said."""

    def __init__(self, completed):
        self.returncode = completed.returncode
        self.stdout = completed.stdout
        self.stderr = completed.stderr
        self.events = [json.loads(line[len(MARKER):])
                       for line in completed.stdout.splitlines()
                       if line.startswith(MARKER)]

    def of(self, event):
        return [e for e in self.events if e['event'] == event]

    def stages(self):
        return [e['name'] for e in self.of('stage')]


class EngineTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir_audio = os.path.join(self.tmp.name, 'audio')
        self.dir_out = os.path.join(self.tmp.name, 'out')
        os.makedirs(os.path.join(self.dir_audio, 'siteA'))

    def add_wav(self, relpath, seconds=6, samplerate=16000, seed=0):
        path = os.path.join(self.dir_audio, relpath)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rng = np.random.default_rng(seed)
        sf.write(path, (rng.standard_normal(int(seconds * samplerate)) * 0.05).astype('float32'),
                 samplerate)
        return path

    def add_fixture(self, name, relpath=None):
        path = os.path.join(self.dir_audio, relpath or name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        shutil.copyfile(ctx.fixture(name), path)
        return path

    def analyze(self, *extra, stdin='', expect_success=True):
        argv = [sys.executable, 'buzzdetect_cli.py',
                '--modelname', MODEL,
                '--dir_audio', self.dir_audio,
                '--dir_out', self.dir_out,
                '--chunklength', '5',
                '--analyzers_cpu', '1',
                *extra]
        completed = subprocess.run(argv, cwd=ctx.DIR_ENGINE, input=stdin,
                                   capture_output=True, text=True, timeout=300)
        run = Run(completed)
        if expect_success and completed.returncode != 0:
            self.fail(f'engine exited {completed.returncode}\n{completed.stdout}\n{completed.stderr}')
        return run

    def results(self, relpath):
        return os.path.join(self.dir_out, relpath + cfg.SUFFIX_RESULT_COMPLETE)


class TestAnalysis(EngineTestCase):
    def setUp(self):
        super().setUp()
        self.add_wav('siteA/rec.wav', seconds=6)
        self.run1 = self.analyze()

    def test_it_writes_a_completed_result_mirroring_the_audio_tree(self):
        self.assertTrue(os.path.exists(self.results('siteA/rec')))
        self.assertFalse(os.path.exists(
            os.path.join(self.dir_out, 'siteA/rec' + cfg.SUFFIX_RESULT_PARTIAL)))

    def test_the_result_has_a_row_per_frame_from_zero(self):
        df = pd.read_csv(self.results('siteA/rec'))
        self.assertEqual(df['start'][0], 0.0)
        self.assertEqual(list(df['start']), sorted(df['start']))
        self.assertEqual(len(set(df['start'])), len(df))
        self.assertAlmostEqual(df['start'][1] - df['start'][0], FRAMELENGTH, places=6)
        # 6 s of audio at 0.96 s per frame, and never past the end of the file.
        self.assertEqual(len(df), 7)

    def test_activations_are_written_for_every_class(self):
        df = pd.read_csv(self.results('siteA/rec'))
        classes = json.loads(read(os.path.join(cfg.DIR_MODELS, MODEL, 'config_model.json')))['classes']
        self.assertEqual(set(df.columns), {'start'} | {'activation_' + c for c in classes})

    def test_it_claims_the_output_folder_with_a_manifest(self):
        manifest = read_manifest(self.dir_out)
        self.assertEqual(manifest['modelname'], MODEL)
        self.assertEqual(manifest['output_mode'], 'activations')

    def test_it_leaves_a_log_beside_the_results(self):
        self.assertTrue(any(n.endswith('.log') for n in os.listdir(self.dir_out)))

    def test_it_reports_the_startup_stages_in_order(self):
        stages = self.run1.stages()
        for stage in ('starting', 'scanning', 'loading', 'analyzing'):
            self.assertIn(stage, stages)
        self.assertEqual(stages[:1], ['starting'])
        self.assertLess(stages.index('scanning'), stages.index('analyzing'))

    def test_it_announces_the_file_before_it_analyzes_it(self):
        starts = self.run1.of('file_start')
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0]['path'], 'siteA/rec.wav')
        self.assertAlmostEqual(starts[0]['duration'], 6.0, places=2)
        self.assertAlmostEqual(starts[0]['work_seconds'], 6.0, places=2)

    def test_the_discovery_walk_reports_every_file_then_finishes(self):
        paths = [p for e in self.run1.of('manifest') for p in e['paths']]
        self.assertEqual(paths, ['siteA/rec.wav'])
        self.assertEqual(self.run1.of('manifest_done')[0]['count'], 1)
        sizes = [b for e in self.run1.of('manifest') for b in e['bytes']]
        self.assertEqual(sizes, [os.path.getsize(os.path.join(self.dir_audio, 'siteA/rec.wav'))])

    def test_the_chunks_add_up_to_the_work_the_file_start_promised(self):
        # The frontend's progress bar sums chunk lengths against work_seconds;
        # if these disagree a bar either stalls short of the end or overruns.
        work = self.run1.of('file_start')[0]['work_seconds']
        done = sum(e['chunk_end'] - e['chunk_start'] for e in self.run1.of('chunk_done'))
        self.assertAlmostEqual(done, work, places=6)

    def test_exactly_the_last_chunk_is_flagged_done(self):
        flags = [e['done'] for e in self.run1.of('chunk_done')]
        self.assertEqual(flags.count(True), 1)
        self.assertTrue(flags[-1])

    def test_chunks_are_contiguous_and_cover_the_file(self):
        chunks = sorted((e['chunk_start'], e['chunk_end']) for e in self.run1.of('chunk_done'))
        self.assertEqual(chunks[0][0], 0.0)
        self.assertAlmostEqual(chunks[-1][1], 6.0, places=2)
        for a, b in zip(chunks, chunks[1:]):
            self.assertEqual(a[1], b[0])

    def test_no_progress_line_reaches_the_log_pane_as_text(self):
        # Rust routes anything without the marker to the log pane; a progress
        # line that failed to parse would show up there as noise.
        for line in self.run1.stdout.splitlines():
            if 'BDPROGRESS' in line:
                self.assertTrue(line.startswith(MARKER), line)


class TestSkipping(EngineTestCase):
    def test_an_already_analyzed_file_is_skipped_on_a_second_run(self):
        self.add_wav('siteA/rec.wav', seconds=3)
        self.analyze()
        before = read(self.results('siteA/rec'))

        again = self.analyze()
        self.assertEqual([e['reason'] for e in again.of('file_skip')], ['already_analyzed'])
        self.assertEqual(again.of('chunk_done'), [])
        self.assertEqual(read(self.results('siteA/rec')), before)

    def test_a_file_too_small_to_be_audio_is_skipped_not_attempted(self):
        path = os.path.join(self.dir_audio, 'siteA', 'stub.wav')
        with open(path, 'wb') as f:
            f.write(b'RIFF' + b'\0' * 100)
        self.add_wav('siteA/rec.wav', seconds=3)
        run = self.analyze()
        skips = {e['path']: e['reason'] for e in run.of('file_skip')}
        self.assertEqual(skips.get('siteA/stub.wav'), 'too_small')
        self.assertTrue(os.path.exists(self.results('siteA/rec')))

    def test_two_files_whose_results_would_collide(self):
        # rec.wav and rec.mp3 share an ident, so they'd write the same result.
        self.add_wav('siteA/rec.wav', seconds=3)
        self.add_fixture('tiny.mp3', 'siteA/rec.mp3')
        run = self.analyze()
        self.assertIn('name_conflict', [e['reason'] for e in run.of('file_skip')])


class TestResuming(EngineTestCase):
    def test_a_part_analyzed_file_is_finished_without_redoing_the_start(self):
        self.add_wav('siteA/rec.wav', seconds=20)
        self.analyze()

        # Rewind it: keep the first half as a partial, as an interrupted run
        # would have left it.
        df = pd.read_csv(self.results('siteA/rec'))
        keep = df.iloc[:len(df) // 2]
        os.remove(self.results('siteA/rec'))
        partial = os.path.join(self.dir_out, 'siteA/rec' + cfg.SUFFIX_RESULT_PARTIAL)
        keep.to_csv(partial, index=False)

        run = self.analyze()
        self.assertAlmostEqual(run.of('file_start')[0]['duration'], 20.0, places=2)
        self.assertLess(run.of('file_start')[0]['work_seconds'], 20.0)
        # Nothing re-read from the part already covered.
        self.assertGreaterEqual(min(e['chunk_start'] for e in run.of('chunk_done')),
                                keep['start'].max())

        finished = pd.read_csv(self.results('siteA/rec'))
        self.assertFalse(os.path.exists(partial))
        self.assertEqual(len(set(finished['start'])), len(finished))
        self.assertEqual(list(finished['start']), sorted(finished['start']))
        self.assertEqual(len(finished), len(df))


class TestManifestLock(EngineTestCase):
    def setUp(self):
        super().setUp()
        self.add_wav('siteA/rec.wav', seconds=3)
        self.analyze('--classes_out', 'ins_buzz', 'frog')

    def test_matching_settings_just_run(self):
        run = self.analyze('--classes_out', 'frog', 'ins_buzz')
        self.assertNotIn('already contains results', run.stdout)

    def test_a_conflict_is_explained_and_declining_analyzes_nothing(self):
        self.add_wav('siteA/other.wav', seconds=3)
        run = self.analyze('--classes_out', 'ins_buzz', stdin='n\n')
        self.assertIn('already contains results', run.stdout)
        self.assertIn('output classes differ', run.stdout)
        self.assertEqual(run.of('chunk_done'), [])
        self.assertFalse(os.path.exists(self.results('siteA/other')))

    def test_accepting_adopts_the_folders_settings_rather_than_the_requested_ones(self):
        self.add_wav('siteA/other.wav', seconds=3)
        self.analyze('--classes_out', 'ins_buzz', stdin='y\n')
        df = pd.read_csv(self.results('siteA/other'))
        self.assertEqual(set(df.columns), {'start', 'activation_frog', 'activation_ins_buzz'})
        self.assertEqual(read_manifest(self.dir_out)['classes_out'], ['frog', 'ins_buzz'])

    def test_the_manifest_is_not_rewritten_by_a_matching_run(self):
        path = os.path.join(self.dir_out, FNAME_MANIFEST)
        before = read(path)
        self.analyze('--classes_out', 'frog', 'ins_buzz')
        self.assertEqual(read(path), before)


class TestDetections(EngineTestCase):
    def test_precision_switches_the_whole_folder_to_detections(self):
        self.add_wav('siteA/rec.wav', seconds=3)
        run = self.analyze('--precision', '0.95')
        self.assertEqual(run.returncode, 0)
        df = pd.read_csv(self.results('siteA/rec'))
        self.assertEqual(list(df.columns), ['start', 'detections_ins_buzz'])
        self.assertTrue(set(df['detections_ins_buzz']) <= {0, 1})
        self.assertEqual(read_manifest(self.dir_out)['output_mode'], 'detections')


class TestFormats(EngineTestCase):
    def test_an_mp3_is_analyzed_through_our_own_driver(self):
        self.add_fixture('truncating.mp3', 'siteA/rec.mp3')
        run = self.analyze()
        self.assertTrue(os.path.exists(self.results('siteA/rec')))
        df = pd.read_csv(self.results('siteA/rec'))
        self.assertGreater(len(df), 0)
        self.assertAlmostEqual(df['start'].max(),
                               run.of('file_start')[0]['duration'] - FRAMELENGTH, delta=2 * FRAMELENGTH)

    def test_a_stereo_file_is_mixed_down_rather_than_refused(self):
        self.add_fixture('stereo.mp3', 'siteA/stereo.mp3')
        self.analyze()
        self.assertTrue(os.path.exists(self.results('siteA/stereo')))

    def test_a_file_that_is_not_audio_at_all_is_never_offered_to_a_driver(self):
        self.add_wav('siteA/rec.wav', seconds=3)
        with open(os.path.join(self.dir_audio, 'siteA', 'notes.txt'), 'w') as f:
            f.write('x' * 10000)
        run = self.analyze()
        self.assertNotIn('notes.txt', [p for e in run.of('manifest') for p in e['paths']])
        self.assertTrue(os.path.exists(self.results('siteA/rec')))


class TestEmptyInput(EngineTestCase):
    def test_a_folder_with_no_audio_finishes_instead_of_hanging(self):
        run = self.analyze()
        self.assertEqual(run.returncode, 0)
        self.assertEqual(run.of('chunk_done'), [])


class TestProbeGpu(EngineTestCase):
    def test_it_answers_with_one_json_line_and_needs_no_other_arguments(self):
        completed = subprocess.run(
            [sys.executable, 'buzzdetect_cli.py', '--probe_gpu'],
            cwd=ctx.DIR_ENGINE, capture_output=True, text=True, timeout=300)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertIsInstance(payload['gpu_providers'], list)


if __name__ == '__main__':
    unittest.main()
