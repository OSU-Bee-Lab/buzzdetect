"""src/pipeline/progress_json.py, and the wire contract it defines.

These lines are the whole interface between the engine and the desktop app:
Rust picks them out of stdout by prefix (src-tauri/src/lib.rs) and the frontend
store switches on `event` (src/lib/progress.svelte.ts). A change here that
isn't matched there is silent -- the event is simply ignored -- so the shape is
pinned down on this side too.
"""

import io
import json
import re
import unittest
from contextlib import redirect_stdout

import tests._context as ctx  # noqa: F401

from src.pipeline.progress_json import MARKER, emit_progress


def emitted(*args, **kwargs):
    out = io.StringIO()
    with redirect_stdout(out):
        emit_progress(*args, **kwargs)
    return out.getvalue()


class TestEmit(unittest.TestCase):
    def test_one_line_per_event(self):
        self.assertEqual(emitted('manifest_done', count=3).count('\n'), 1)

    def test_marked_and_parseable(self):
        line = emitted('manifest_done', count=3).rstrip('\n')
        self.assertTrue(line.startswith(MARKER))
        self.assertEqual(json.loads(line[len(MARKER):]), {'event': 'manifest_done', 'count': 3})

    def test_marker_matches_the_rust_side(self):
        # src-tauri/src/lib.rs strips this exact prefix, trailing space included.
        self.assertEqual(MARKER, 'BDPROGRESS ')

    def test_fields_are_carried_through(self):
        payload = json.loads(emitted('chunk_done', path='a/b.wav', chunk_start=0.0,
                                     chunk_end=200.0, done=False).rstrip()[len(MARKER):])
        self.assertEqual(payload, {'event': 'chunk_done', 'path': 'a/b.wav',
                                   'chunk_start': 0.0, 'chunk_end': 200.0, 'done': False})

    def test_no_embedded_newline_can_split_an_event(self):
        # A path is user data; a line reader splitting on '\n' must still see
        # one event, so json.dumps' escaping is what has to hold.
        line = emitted('file_start', path='weird\nname.wav', duration=1.0, work_seconds=1.0)
        self.assertEqual(line.count('\n'), 1)
        self.assertEqual(json.loads(line.rstrip()[len(MARKER):])['path'], 'weird\nname.wav')

    def test_an_event_field_overrides_nothing_silently(self):
        payload = json.loads(emitted('stage', name='scanning').rstrip()[len(MARKER):])
        self.assertEqual(payload['event'], 'stage')


class TestWireContract(unittest.TestCase):
    """Every event kind the engine emits, against what the frontend handles."""

    # src/lib/progress.svelte.ts's handleEvent switch, plus the Stage union.
    FRONTEND_EVENTS = {'stage', 'manifest', 'manifest_done', 'file_skip', 'file_start', 'chunk_done'}
    # 'launching' is the frontend's own and is never emitted; 'stopping' is
    # emitted but is handled outside the startup ladder.
    FRONTEND_STAGES = {'starting', 'scanning', 'loading', 'analyzing', 'stopping'}

    def emitted_kinds(self):
        kinds = set()
        for path in ctx.engine_sources():
            with open(path) as f:
                source = f.read()
            for m in re.finditer(r"emit_progress\(\s*'([a-z_]+)'", source):
                kinds.add(m.group(1))
        return kinds

    def emitted_stages(self):
        stages = set()
        for path in ctx.engine_sources():
            with open(path) as f:
                source = f.read()
            for m in re.finditer(r"emit_progress\(\s*'stage'\s*,\s*name='([a-z_]+)'", source):
                stages.add(m.group(1))
        return stages

    def test_every_emitted_event_is_one_the_frontend_handles(self):
        self.assertEqual(self.emitted_kinds() - self.FRONTEND_EVENTS, set())

    def test_every_emitted_stage_is_one_the_frontend_knows(self):
        self.assertEqual(self.emitted_stages() - self.FRONTEND_STAGES, set())


if __name__ == '__main__':
    unittest.main()
