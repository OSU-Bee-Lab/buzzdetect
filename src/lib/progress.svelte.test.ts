// The run store is fed entirely by the engine's BDPROGRESS events (see
// engine/src/pipeline/progress_json.py, and engine/tests/test_cli.py for what
// a real run emits). These tests replay those events and check what the UI
// would draw: the file tree, the segment weights behind each bar, the totals,
// and the rate/ETA.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { formatDuration, run, type TreeDir } from './progress.svelte';

// An engine run, in the order the engine actually emits it.
function discover(files: { path: string; bytes?: number }[]) {
	run.handleEvent({
		event: 'manifest',
		paths: files.map((f) => f.path),
		bytes: files.map((f) => f.bytes ?? 0)
	});
}

function start(path: string, duration: number, workSeconds = duration) {
	run.handleEvent({ event: 'file_start', path, duration, work_seconds: workSeconds });
}

function chunk(path: string, from: number, to: number, done = false) {
	run.handleEvent({ event: 'chunk_done', path, chunk_start: from, chunk_end: to, done });
}

function dir(tree: TreeDir, path: string): TreeDir {
	const found = tree.dirs.find((d) => d.path === path || d.name === path);
	if (!found) throw new Error(`no dir ${path} in ${tree.dirs.map((d) => d.path).join(', ')}`);
	return found;
}

function file(tree: TreeDir, name: string) {
	const found = tree.files.find((f) => f.name === name);
	if (!found) throw new Error(`no file ${name}`);
	return found;
}

beforeEach(() => {
	vi.useFakeTimers();
	run.reset();
});

afterEach(() => {
	run.stop();
	vi.useRealTimers();
});

describe('formatDuration', () => {
	it('gives two adjacent units at most', () => {
		expect(formatDuration(45)).toBe('45 seconds');
		expect(formatDuration(372)).toBe('6 minutes, 12 seconds');
		expect(formatDuration(90000)).toBe('1 day, 1 hour');
	});

	it('drops a zero unit rather than printing it', () => {
		expect(formatDuration(7200)).toBe('2 hours');
	});

	it('singularises', () => {
		expect(formatDuration(61)).toBe('1 minute, 1 second');
	});

	it('never reports a smaller unit than the two at the top', () => {
		// 1 day and 5 minutes is "1 day", not "1 day, 5 minutes".
		expect(formatDuration(86400 + 300)).toBe('1 day');
	});

	it('has something to say about nothing', () => {
		expect(formatDuration(0)).toBe('0 seconds');
		expect(formatDuration(0.4)).toBe('0 seconds');
		expect(formatDuration(NaN)).toBe('0 seconds');
		expect(formatDuration(Infinity)).toBe('0 seconds');
	});
});

describe('discovery', () => {
	it('lists discovered files as pending', () => {
		discover([{ path: 'siteA/rec.wav' }]);
		const f = file(dir(run.tree, 'siteA'), 'rec.wav');
		expect(f.status).toBe('pending');
		expect(f.doneSeconds).toBe(0);
	});

	it('splits a path into a directory and a name', () => {
		discover([{ path: 'a/b/rec.wav' }, { path: 'top.wav' }]);
		expect(file(dir(dir(run.tree, 'a'), 'a/b'), 'rec.wav').path).toBe('a/b/rec.wav');
		expect(file(run.tree, 'top.wav').dir).toBe('');
	});

	it('arrives in batches and never duplicates a file', () => {
		discover([{ path: 'a.wav' }]);
		discover([{ path: 'a.wav' }, { path: 'b.wav' }]);
		expect(run.totals.filesTotal).toBe(2);
	});

	it('does not undo progress if a file is re-announced', () => {
		discover([{ path: 'a.wav' }]);
		start('a.wav', 10);
		chunk('a.wav', 0, 5);
		discover([{ path: 'a.wav' }]);
		expect(file(run.tree, 'a.wav').status).toBe('running');
		expect(run.totals.doneSeconds).toBe(5);
	});

	it('is not final until the walk says so', () => {
		discover([{ path: 'a.wav' }]);
		expect(run.denominatorFinal).toBe(false);
		expect(run.tree.finalized).toBe(false);
		run.handleEvent({ event: 'manifest_done', count: 1 });
		expect(run.denominatorFinal).toBe(true);
		expect(run.tree.finalized).toBe(true);
	});

	it('tolerates a manifest event with no sizes', () => {
		run.handleEvent({ event: 'manifest', paths: ['a.wav'] });
		expect(file(run.tree, 'a.wav').bytes).toBe(0);
	});
});

describe('a file being analyzed', () => {
	beforeEach(() => {
		discover([{ path: 'siteA/rec.wav', bytes: 1000 }]);
	});

	it('keeps the size discovery found when it opens', () => {
		start('siteA/rec.wav', 100);
		expect(file(dir(run.tree, 'siteA'), 'rec.wav').bytes).toBe(1000);
	});

	it('accumulates chunk lengths rather than tracking the last offset', () => {
		// Chunks land out of order across analyzers, and a resumed file only
		// re-analyzes its gaps, so an absolute offset is not a position.
		start('siteA/rec.wav', 100, 50);
		chunk('siteA/rec.wav', 80, 100);
		chunk('siteA/rec.wav', 20, 50);
		expect(run.totals.doneSeconds).toBe(50);
	});

	it('never counts more work than the file promised', () => {
		start('siteA/rec.wav', 100, 10);
		chunk('siteA/rec.wav', 0, 200);
		expect(run.totals.doneSeconds).toBe(10);
	});

	it('is finished by the chunk flagged done, whatever it counted', () => {
		start('siteA/rec.wav', 100);
		chunk('siteA/rec.wav', 0, 30, true);
		const f = file(dir(run.tree, 'siteA'), 'rec.wav');
		expect(f.status).toBe('done');
		expect(f.doneSeconds).toBe(100); // the whole of the work, not the chunk
	});

	it('ignores chunks for a file it never saw start', () => {
		chunk('ghost.wav', 0, 10);
		expect(run.totals.filesTotal).toBe(1);
	});

	it('handles a chunk event with no start offset', () => {
		start('siteA/rec.wav', 100);
		chunk('siteA/rec.wav', undefined as unknown as number, 10);
		expect(run.totals.doneSeconds).toBe(0);
	});
});

describe('bar segments', () => {
	it('charges a resumed file the part an earlier run already did', () => {
		discover([{ path: 'a.wav' }]);
		start('a.wav', 100, 40); // 60 s were analyzed before this run
		chunk('a.wav', 0, 10);
		const w = file(run.tree, 'a.wav').weights;
		expect(w.totalSeconds).toBe(100);
		expect(w.priorSeconds).toBe(60);
		expect(w.activeSeconds).toBe(10);
		expect(w.doneSeconds).toBe(0); // still open: interrupted work, recoloured if the run stops
	});

	it('moves a finished file out of the active segment', () => {
		discover([{ path: 'a.wav' }]);
		start('a.wav', 100, 100);
		chunk('a.wav', 0, 100, true);
		const w = file(run.tree, 'a.wav').weights;
		expect(w.doneSeconds).toBe(100);
		expect(w.activeSeconds).toBe(0);
	});

	it('counts a skipped file as already analyzed', () => {
		discover([{ path: 'a.wav', bytes: 1000 }, { path: 'b.wav', bytes: 1000 }]);
		start('b.wav', 60);
		run.handleEvent({ event: 'file_skip', path: 'a.wav', reason: 'already_analyzed' });
		const w = file(run.tree, 'a.wav').weights;
		expect(w.priorSeconds).toBe(w.totalSeconds);
		expect(w.totalSeconds).toBeGreaterThan(0);
	});

	it('never lets the segments overrun the total', () => {
		discover([{ path: 'a.wav' }]);
		start('a.wav', 100, 40);
		chunk('a.wav', 0, 40);
		const w = file(run.tree, 'a.wav').weights;
		expect(w.priorSeconds + w.doneSeconds + w.activeSeconds).toBeLessThanOrEqual(w.totalSeconds);
	});
});

describe('weighting files nothing has opened', () => {
	it('extrapolates a duration from size, per extension', () => {
		discover([
			{ path: 'opened.wav', bytes: 1000 },
			{ path: 'queued.wav', bytes: 2000 }
		]);
		start('opened.wav', 10);
		expect(file(run.tree, 'queued.wav').weights.totalSeconds).toBe(20);
	});

	it('does not price an mp3 off a wav when it has an mp3 to go on', () => {
		discover([
			{ path: 'a.wav', bytes: 1000 },
			{ path: 'b.mp3', bytes: 1000 },
			{ path: 'c.mp3', bytes: 1000 }
		]);
		start('a.wav', 10); // wav: 100 bytes/s
		start('b.mp3', 100); // mp3: 10 bytes/s
		expect(file(run.tree, 'c.mp3').weights.totalSeconds).toBe(100);
	});

	it('pools every extension for a codec nothing has opened', () => {
		discover([{ path: 'a.wav', bytes: 1000 }, { path: 'b.flac', bytes: 1000 }]);
		start('a.wav', 10);
		expect(file(run.tree, 'b.flac').weights.totalSeconds).toBe(10);
	});

	it('falls back to the mean duration when a file has no size', () => {
		discover([{ path: 'a.wav', bytes: 1000 }, { path: 'b.wav' }]);
		start('a.wav', 10);
		expect(file(run.tree, 'b.wav').weights.totalSeconds).toBe(10);
	});

	it('weights files equally before anything has been opened', () => {
		discover([{ path: 'a.wav', bytes: 1000 }, { path: 'b.wav', bytes: 5000 }]);
		expect(file(run.tree, 'a.wav').weights.totalSeconds).toBe(1);
		expect(file(run.tree, 'b.wav').weights.totalSeconds).toBe(1);
	});
});

describe('the tree', () => {
	beforeEach(() => {
		discover([
			{ path: 'siteB/z.wav', bytes: 1000 },
			{ path: 'siteA/b.wav', bytes: 1000 },
			{ path: 'siteA/a.wav', bytes: 1000 },
			{ path: 'top.wav', bytes: 1000 }
		]);
	});

	it('sorts directories by name and files with the running one first', () => {
		start('siteA/b.wav', 10);
		expect(run.tree.dirs.map((d) => d.name)).toEqual(['siteA', 'siteB']);
		expect(dir(run.tree, 'siteA').files.map((f) => f.name)).toEqual(['b.wav', 'a.wav']);
	});

	it('sorts alphabetically within a status', () => {
		expect(dir(run.tree, 'siteA').files.map((f) => f.name)).toEqual(['a.wav', 'b.wav']);
	});

	it('aggregates counts up the tree', () => {
		start('siteA/a.wav', 10);
		chunk('siteA/a.wav', 0, 10, true);
		run.handleEvent({ event: 'file_skip', path: 'siteA/b.wav', reason: 'already_analyzed' });
		const siteA = dir(run.tree, 'siteA');
		expect(siteA.filesTotal).toBe(2);
		expect(siteA.filesDone).toBe(2);
		expect(run.tree.filesTotal).toBe(4);
		expect(run.tree.filesDone).toBe(2);
	});

	it('aggregates seconds up the tree', () => {
		start('siteA/a.wav', 10);
		chunk('siteA/a.wav', 0, 4);
		start('siteB/z.wav', 20);
		chunk('siteB/z.wav', 0, 5);
		expect(dir(run.tree, 'siteA').activeSeconds).toBe(4);
		expect(run.tree.activeSeconds).toBe(9);
		expect(run.tree.workSeconds).toBe(run.totals.workSeconds);
	});

	it('keeps a file at the audio directory root out of any subdirectory', () => {
		expect(run.tree.files.map((f) => f.name)).toEqual(['top.wav']);
	});

	it('is empty before anything is discovered', () => {
		run.reset();
		expect(run.tree.files).toEqual([]);
		expect(run.tree.dirs).toEqual([]);
		expect(run.tree.totalSeconds).toBe(0);
	});
});

describe('stages', () => {
	it('advances through startup', () => {
		expect(run.stage).toBe('launching');
		run.handleEvent({ event: 'stage', name: 'starting' });
		run.handleEvent({ event: 'stage', name: 'scanning' });
		expect(run.stageLabel).toBe('Finding audio files…');
	});

	it('never goes backwards when a second analyzer comes up', () => {
		run.handleEvent({ event: 'stage', name: 'analyzing' });
		run.handleEvent({ event: 'stage', name: 'loading' });
		expect(run.stage).toBe('analyzing');
	});

	it('takes the engine reporting a stop as a stop, however it was asked for', () => {
		run.handleEvent({ event: 'stage', name: 'stopping' });
		expect(run.stopping).toBe(true);
		expect(run.stage).toBe('launching'); // not a startup stage
	});

	it('ignores a stage name it does not know', () => {
		run.handleEvent({ event: 'stage', name: 'reticulating' });
		expect(run.stage).toBe('launching');
	});

	it('ignores an event kind it does not know', () => {
		run.handleEvent({ event: 'something_new', path: 'a.wav' });
		expect(run.totals.filesTotal).toBe(0);
	});
});

describe('rate and ETA', () => {
	it('are zero before anything has been analyzed', () => {
		expect(run.stats).toEqual({
			priorSeconds: 0,
			remainingSeconds: 0,
			rate: 0,
			etaSeconds: null
		});
	});

	it('divide remaining audio by the trailing rate', () => {
		discover([{ path: 'a.wav' }]);
		start('a.wav', 100);
		vi.advanceTimersByTime(2000);
		chunk('a.wav', 0, 20); // 20 s of audio in 2 s of wall clock
		vi.advanceTimersByTime(2000);
		expect(run.stats.remainingSeconds).toBe(80);
		expect(run.stats.rate).toBeCloseTo(10, 5);
		expect(run.stats.etaSeconds).toBeCloseTo(80 / 5, 0);
	});

	it('decay while the engine says nothing', () => {
		discover([{ path: 'a.wav' }]);
		start('a.wav', 1000);
		vi.advanceTimersByTime(2000);
		chunk('a.wav', 0, 20);
		vi.advanceTimersByTime(2000);
		const early = run.stats.etaSeconds!;
		vi.advanceTimersByTime(20_000);
		expect(run.stats.rate).toBe(0);
		expect(run.stats.etaSeconds!).toBeGreaterThan(early);
	});

	it('do not count audio an earlier run already analyzed', () => {
		discover([{ path: 'a.wav' }]);
		start('a.wav', 100, 10); // 90 s already done elsewhere
		vi.advanceTimersByTime(2000);
		expect(run.stats.priorSeconds).toBe(90);
		expect(run.stats.remainingSeconds).toBe(10);
	});

	it('report no ETA rather than an infinite one', () => {
		discover([{ path: 'a.wav' }]);
		start('a.wav', 100);
		vi.advanceTimersByTime(2000);
		expect(run.stats.etaSeconds).toBeNull();
	});

	it('are published on the tick, not recomputed per chunk', () => {
		discover([{ path: 'a.wav' }]);
		start('a.wav', 100);
		const before = run.stats;
		chunk('a.wav', 0, 20);
		expect(run.stats).toBe(before);
		vi.advanceTimersByTime(2000);
		expect(run.stats).not.toBe(before);
	});
});

describe('starting and stopping', () => {
	it('clears the previous run', () => {
		discover([{ path: 'a.wav' }]);
		run.handleLog('something');
		run.stop('it broke');
		run.reset();
		expect(run.totals.filesTotal).toBe(0);
		expect(run.logLines).toEqual([]);
		expect(run.error).toBeNull();
		expect(run.stopped).toBe(false);
		expect(run.running).toBe(true);
		expect(run.summary).toBeNull();
	});

	it('stays locked while a requested stop is still winding down', () => {
		run.beginStop();
		expect(run.stopping).toBe(true);
		expect(run.running).toBe(true);
	});

	it('summarises the run when the engine exits', () => {
		discover([{ path: 'a.wav' }]);
		start('a.wav', 100);
		chunk('a.wav', 0, 30);
		vi.advanceTimersByTime(10_000);
		run.stop();
		expect(run.running).toBe(false);
		expect(run.stopping).toBe(false);
		expect(run.stopped).toBe(true);
		expect(run.summary).toEqual({ audioSeconds: 30, runtimeSeconds: 10, rate: 3 });
	});

	it('records an error when there is one, and nothing when there is not', () => {
		run.stop();
		expect(run.error).toBeNull();
		run.reset();
		run.stop('engine exited with code 1');
		expect(run.error).toBe('engine exited with code 1');
	});

	it('stops the clock so a finished run does not keep ticking', () => {
		discover([{ path: 'a.wav' }]);
		start('a.wav', 100);
		chunk('a.wav', 0, 30);
		run.stop();
		const settled = run.stats;
		vi.advanceTimersByTime(60_000);
		expect(run.stats).toBe(settled);
		expect(run.stats.rate).toBe(0);
	});
});

describe('the log pane', () => {
	it('keeps lines in order', () => {
		run.handleLog('first');
		run.handleLog('second');
		expect(run.logLines).toEqual(['first', 'second']);
	});

	it('is bounded, keeping the newest', () => {
		for (let i = 0; i < 600; i++) run.handleLog(`line ${i}`);
		expect(run.logLines.length).toBe(500);
		expect(run.logLines[499]).toBe('line 599');
	});
});
