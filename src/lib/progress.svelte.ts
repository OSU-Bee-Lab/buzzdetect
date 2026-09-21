// Reactive state for one analysis run, fed by the "engine-progress" events
// Rust forwards straight from the Python engine's BDPROGRESS lines (see
// engine/src/pipeline/progress_json.py). Event types: manifest (one file
// discovered), manifest_done (discovery walk finished), file_start,
// file_skip, chunk_done (which carries chunk_start/chunk_end as absolute
// offsets within the file, not a done-so-far position), stage --
// startup progress, plus the 'stopping' report, which is not a startup stage
// at all but the engine saying it has begun winding down -- and error, sent
// when a worker thread has died and the analysis cannot continue.

export type FileStatus = 'pending' | 'running' | 'done' | 'skipped';

// Startup stages, in the order the engine passes through them. Between
// clicking Start and the first chunk landing there can be the better part of a
// minute — the sidecar unpacking itself, onnxruntime importing, the audio tree
// being walked, an inference session being built — and all of it used to read
// as one undifferentiated "Analyzing…". 'launching' is the frontend's own:
// it covers the stretch before the engine has said anything at all.
export type Stage = 'launching' | 'starting' | 'scanning' | 'loading' | 'analyzing';

const STAGE_ORDER: Stage[] = ['launching', 'starting', 'scanning', 'loading', 'analyzing'];

// 'launching' covers the subprocess spawn and the frozen sidecar unpacking
// itself; 'starting' is then the engine's import of numpy/pandas/the audio
// stack, which frozen is the single longest stretch of startup (~25s), so it
// gets a label that says something is still loading rather than repeating
// 'engine'. onnxruntime is not in that import — it loads later, per worker,
// under 'loading'.
const STAGE_LABELS: Record<Stage, string> = {
	launching: 'Launching engine…',
	starting: 'Loading engine components…',
	scanning: 'Finding audio files…',
	loading: 'Loading model…',
	analyzing: 'Analyzing…'
};

// Trailing window the ETA uses so it doesn't chase every fluctuation in throughput.
const ETA_WINDOW_MS = 30_000;
// How often the clock driving rate/ETA advances, and the minimum spacing
// between retained samples (bounding the buffer over the ETA window).
const TICK_MS = 2_000;
const SAMPLE_MS = 1_000;

export interface Stats {
	priorSeconds: number;
	remainingSeconds: number;
	rate: number;
	etaSeconds: number | null;
}

export interface RunSummary {
	audioSeconds: number;
	runtimeSeconds: number;
	rate: number;
}

const ZERO_STATS: Stats = {
	priorSeconds: 0,
	remainingSeconds: 0,
	rate: 0,
	etaSeconds: null
};

const UNITS: { label: string; seconds: number }[] = [
	{ label: 'day', seconds: 86400 },
	{ label: 'hour', seconds: 3600 },
	{ label: 'minute', seconds: 60 },
	{ label: 'second', seconds: 1 }
];

// Human duration at two units of precision: "20 days, 5 hours", "6 minutes,
// 12 seconds", "45 seconds". Smaller units are dropped rather than rounded
// into the larger one, which is fine at this precision.
export function formatDuration(seconds: number): string {
	if (!isFinite(seconds) || seconds < 1) return '0 seconds';
	const total = Math.floor(seconds);
	const top = UNITS.findIndex((u) => total >= u.seconds);
	const parts: string[] = [];
	let rest = total;
	// The top unit and the one below it, so precision stays at two adjacent
	// units ("1 day, 3 hours", never "1 day, 5 minutes").
	for (const u of UNITS.slice(top, top + 2)) {
		const n = Math.floor(rest / u.seconds);
		rest -= n * u.seconds;
		if (n > 0) parts.push(`${n} ${u.label}${n === 1 ? '' : 's'}`);
	}
	return parts.join(', ');
}

export interface FileProgress {
	path: string; // path relative to dir_audio, e.g. "siteA/2024-06-01.wav"
	dir: string; // parent dir of path ("" for files at the audio dir's root)
	name: string;
	status: FileStatus;
	bytes: number; // file size from the discovery walk; 0 if unknown
	duration: number; // full audio duration in seconds, once known
	workSeconds: number; // seconds of audio actually needing analysis (duration minus any already-completed portion)
	doneSeconds: number; // work completed so far, summed over finished chunks
}

// How one file (or a whole subtree) splits across the bar's segments:
// green (analyzed by an earlier run), blue (files this session carried to
// completion), work on files still open, and the gray remainder.
// total >= prior + done + active always.
//
// `active` is split out from `done` only so the bar can recolor it if the run
// stops: it's work that was interrupted mid-file rather than work that
// finished a file. Both count as analyzed either way.
export interface Weights {
	totalSeconds: number;
	priorSeconds: number;
	doneSeconds: number;
	activeSeconds: number;
}

function dirOf(path: string): string {
	const idx = path.lastIndexOf('/');
	return idx === -1 ? '' : path.slice(0, idx);
}

function nameOf(path: string): string {
	const idx = path.lastIndexOf('/');
	return idx === -1 ? path : path.slice(idx + 1);
}

// localeCompare builds a collator per call, which dominates sorting thousands
// of names; one shared collator orders them the same way.
const byName = new Intl.Collator().compare;

function extOf(path: string): string {
	const idx = path.lastIndexOf('.');
	return idx === -1 ? '' : path.slice(idx + 1).toLowerCase();
}

// Seconds of audio per byte, pooled over the files already opened this run,
// keyed by extension (a fixed-bitrate codec makes this near-exact; the '' key
// pools every extension as a fallback for a codec nothing has opened yet).
interface Weighting {
	scale: Map<string, { seconds: number; bytes: number }>;
	meanDuration: number; // over opened files; 0 before any file is opened
}

// No size for a file, or nothing opened yet to calibrate against: fall back to
// the mean duration so far, then to weighting files equally.
function fallbackDuration(w: Weighting): number {
	return w.meanDuration > 0 ? w.meanDuration : 1;
}

function pooledFor(ext: string, w: Weighting) {
	const pooled = w.scale.get(ext) ?? w.scale.get('');
	return pooled && pooled.bytes > 0 ? pooled : null;
}

function estimateDuration(f: FileProgress, w: Weighting): number {
	const pooled = pooledFor(extOf(f.path), w);
	if (f.bytes > 0 && pooled) return (f.bytes * pooled.seconds) / pooled.bytes;
	return fallbackDuration(w);
}

// State shared by every node of one run's tree. The calibration is the one
// input that can move a node's numbers without the node itself changing, so
// only estimated numbers read it.
class TreeContext {
	weighting = $state.raw<Weighting>({ scale: new Map(), meanDuration: 0 });
	discoveryDone = $state(false);
}

// A file as the tree hands it to the UI. Its fields are individually reactive,
// so a chunk landing redraws that one row and the bars of its ancestors, and
// nothing else. `weights` is what its bar is drawn from.
export class FileNode implements FileProgress {
	readonly path: string;
	readonly dir: string;
	readonly name: string;
	readonly parent: TreeDir;
	status = $state<FileStatus>('pending');
	bytes = $state(0);
	duration = $state(0);
	workSeconds = $state(0);
	doneSeconds = $state(0);
	#ctx: TreeContext;

	constructor(path: string, parent: TreeDir, ctx: TreeContext) {
		this.path = path;
		this.dir = dirOf(path);
		this.name = nameOf(path);
		this.parent = parent;
		this.#ctx = ctx;
	}

	// A visited file reports its own exact numbers; anything else is estimated
	// from size. A skipped file counts as entirely analyzed-already: the engine
	// only skips files whose results are complete (or that it can't analyze at
	// all, which won't progress either way). Only the estimated branch reads
	// the calibration, so a file that has been opened never recomputes when it
	// moves.
	weights = $derived.by((): Weights => {
		if (this.status === 'running' || this.status === 'done') {
			const totalSeconds = this.duration > 0 ? this.duration : this.workSeconds;
			return {
				totalSeconds,
				priorSeconds: Math.max(0, totalSeconds - this.workSeconds),
				doneSeconds: this.status === 'done' ? this.doneSeconds : 0,
				activeSeconds: this.status === 'done' ? 0 : this.doneSeconds
			};
		}
		const totalSeconds = this.duration > 0 ? this.duration : estimateDuration(this, this.#ctx.weighting);
		return {
			totalSeconds,
			priorSeconds: this.status === 'skipped' ? totalSeconds : 0,
			doneSeconds: 0,
			activeSeconds: 0
		};
	});
}

// What one file adds to every directory above it, split into the part that is
// exact and the part that can only be priced against the calibration. A
// directory keeps running sums of these rather than re-adding its files, so a
// change to one file costs one pass up its ancestors.
interface Contribution {
	workSeconds: number;
	doneSeconds: number;
	activeSeconds: number;
	filesDone: number;
	exactTotal: number;
	exactPrior: number;
	// Set for a file nothing has opened, whose duration is still an estimate.
	estimate: { ext: string; bytes: number; skipped: boolean } | null;
}

function contributionOf(f: FileNode): Contribution {
	const filesDone = f.status === 'done' || f.status === 'skipped' ? 1 : 0;
	if (f.status === 'running' || f.status === 'done') {
		const total = f.duration > 0 ? f.duration : f.workSeconds;
		return {
			workSeconds: f.workSeconds,
			doneSeconds: f.status === 'done' ? f.doneSeconds : 0,
			activeSeconds: f.status === 'done' ? 0 : f.doneSeconds,
			filesDone,
			exactTotal: total,
			exactPrior: Math.max(0, total - f.workSeconds),
			estimate: null
		};
	}
	const skipped = f.status === 'skipped';
	const known = f.duration > 0;
	return {
		workSeconds: f.workSeconds,
		doneSeconds: 0,
		activeSeconds: 0,
		filesDone,
		exactTotal: known ? f.duration : 0,
		exactPrior: known && skipped ? f.duration : 0,
		estimate: known ? null : { ext: extOf(f.path), bytes: f.bytes, skipped }
	};
}

const NO_CONTRIBUTION: Contribution = {
	workSeconds: 0,
	doneSeconds: 0,
	activeSeconds: 0,
	filesDone: 0,
	exactTotal: 0,
	exactPrior: 0,
	estimate: null
};

// Unopened files under a directory, per extension, in the shape the estimate
// needs: bytes to price against that extension's calibration, and counts for
// the files that can only be charged the fallback duration.
interface EstimateBucket {
	pendingBytes: number;
	pendingSized: number;
	pendingUnsized: number;
	skippedBytes: number;
	skippedSized: number;
	skippedUnsized: number;
	files: number;
}

function sameEstimate(a: Contribution['estimate'], b: Contribution['estimate']): boolean {
	if (a === null || b === null) return a === b;
	return a.ext === b.ext && a.bytes === b.bytes && a.skipped === b.skipped;
}

// A directory node in the audio tree. `finalized` means discovery has
// finished, so this subtree's file list and counts are complete; until then
// new files can still appear under it and the bar is drawn provisionally
// (striped).
//
// Every number is a running sum kept up to date as its files change (see
// Contribution), never a walk over the subtree. The weights are partly
// estimated: a file's duration is only known once a streamer opens it, so
// files still queued — and files skipped without ever being opened — are
// charged a duration extrapolated from their byte size (see
// estimateDuration). Every visited file contributes its real numbers, so a
// dir's remaining seconds are exact except for the files nothing has opened
// yet.
//
// `dirs` and `files` are sorted lazily, and only when something reads them —
// in practice, when the directory is expanded.
export class TreeDir implements Weights {
	readonly path: string;
	readonly name: string;
	readonly parent: TreeDir | null;
	workSeconds = $state(0);
	doneSeconds = $state(0);
	activeSeconds = $state(0);
	filesTotal = $state(0);
	filesDone = $state(0);
	#exactTotal = $state(0);
	#exactPrior = $state(0);
	#ctx: TreeContext;

	#children = new Map<string, TreeDir>();
	#files: FileNode[] = [];
	#dirsVersion = $state(0);
	#filesVersion = $state(0);

	#buckets = new Map<string, EstimateBucket>();
	#bucketsVersion = $state(0);

	constructor(path: string, parent: TreeDir | null, ctx: TreeContext) {
		this.path = path;
		this.name = nameOf(path);
		this.parent = parent;
		this.#ctx = ctx;
	}

	get finalized(): boolean {
		return this.#ctx.discoveryDone;
	}

	#estimated = $derived.by(() => {
		this.#bucketsVersion;
		const w = this.#ctx.weighting;
		const fallback = fallbackDuration(w);
		let pending = 0;
		let skipped = 0;
		for (const [ext, b] of this.#buckets) {
			const pooled = pooledFor(ext, w);
			pending += b.pendingUnsized * fallback;
			skipped += b.skippedUnsized * fallback;
			if (pooled) {
				pending += (b.pendingBytes * pooled.seconds) / pooled.bytes;
				skipped += (b.skippedBytes * pooled.seconds) / pooled.bytes;
			} else {
				pending += b.pendingSized * fallback;
				skipped += b.skippedSized * fallback;
			}
		}
		return { pending, skipped };
	});

	totalSeconds = $derived(this.#exactTotal + this.#estimated.pending + this.#estimated.skipped);
	priorSeconds = $derived(this.#exactPrior + this.#estimated.skipped);

	dirs = $derived.by(() => {
		this.#dirsVersion;
		return [...this.#children.values()].sort((a, b) => byName(a.name, b.name));
	});

	#filesByName = $derived.by(() => {
		this.#filesVersion;
		return [...this.#files].sort((a, b) => byName(a.name, b.name));
	});

	// Running files first, then queued ones, then finished, each by name. A
	// stable partition of the name order, so a status change costs a pass over
	// this directory's files, not a sort — and only when it's being drawn.
	files = $derived.by(() => {
		const running: FileNode[] = [];
		const pending: FileNode[] = [];
		const finished: FileNode[] = [];
		for (const f of this.#filesByName) {
			if (f.status === 'running') running.push(f);
			else if (f.status === 'pending') pending.push(f);
			else finished.push(f);
		}
		return running.concat(pending, finished);
	});

	child(name: string): TreeDir {
		let d = this.#children.get(name);
		if (!d) {
			d = new TreeDir(this.path ? `${this.path}/${name}` : name, this, this.#ctx);
			this.#children.set(name, d);
			this.#dirsVersion += 1;
		}
		return d;
	}

	addFile(f: FileNode) {
		this.#files.push(f);
		this.#filesVersion += 1;
		for (let d: TreeDir | null = this; d; d = d.parent) d.filesTotal += 1;
		this.adjust(NO_CONTRIBUTION, contributionOf(f));
	}

	// Moves this directory and every one above it from one file's old
	// contribution to its new one.
	adjust(before: Contribution, after: Contribution) {
		const work = after.workSeconds - before.workSeconds;
		const done = after.doneSeconds - before.doneSeconds;
		const active = after.activeSeconds - before.activeSeconds;
		const filesDone = after.filesDone - before.filesDone;
		const total = after.exactTotal - before.exactTotal;
		const prior = after.exactPrior - before.exactPrior;
		const moved = !sameEstimate(before.estimate, after.estimate);
		for (let d: TreeDir | null = this; d; d = d.parent) {
			d.#shift(work, done, active, filesDone, total, prior);
			if (moved) d.#rebucket(before.estimate, after.estimate);
		}
	}

	// Svelte rewrites `this.#field` to its signal, but not another instance's
	// `d.#field`, so the walk above goes through methods.
	#shift(work: number, done: number, active: number, filesDone: number, total: number, prior: number) {
		if (work) this.workSeconds += work;
		if (done) this.doneSeconds += done;
		if (active) this.activeSeconds += active;
		if (filesDone) this.filesDone += filesDone;
		if (total) this.#exactTotal += total;
		if (prior) this.#exactPrior += prior;
	}

	#rebucket(before: Contribution['estimate'], after: Contribution['estimate']) {
		if (before) this.#bucket(before, -1);
		if (after) this.#bucket(after, 1);
		this.#bucketsVersion += 1;
	}

	#bucket(e: NonNullable<Contribution['estimate']>, sign: 1 | -1) {
		let b = this.#buckets.get(e.ext);
		if (!b) {
			b = {
				pendingBytes: 0,
				pendingSized: 0,
				pendingUnsized: 0,
				skippedBytes: 0,
				skippedSized: 0,
				skippedUnsized: 0,
				files: 0
			};
			this.#buckets.set(e.ext, b);
		}
		b.files += sign;
		if (e.skipped) {
			if (e.bytes > 0) {
				b.skippedBytes += sign * e.bytes;
				b.skippedSized += sign;
			} else b.skippedUnsized += sign;
		} else if (e.bytes > 0) {
			b.pendingBytes += sign * e.bytes;
			b.pendingSized += sign;
		} else b.pendingUnsized += sign;
		if (b.files === 0) this.#buckets.delete(e.ext);
	}
}

// What an opened file adds to the size -> duration calibration.
interface Calibrant {
	ext: string;
	seconds: number;
	bytes: number;
}

function calibrantOf(f: FileNode): Calibrant | null {
	if (f.status !== 'running' && f.status !== 'done') return null;
	if (f.duration <= 0) return null;
	return { ext: extOf(f.path), seconds: f.duration, bytes: f.bytes };
}

function sameCalibrant(a: Calibrant | null, b: Calibrant | null): boolean {
	if (a === null || b === null) return a === b;
	return a.ext === b.ext && a.seconds === b.seconds && a.bytes === b.bytes;
}

// Calibrates size -> duration from the files already opened this run, so
// files nothing has opened can be weighted by their byte size. Kept as running
// sums and republished only when an opened file joins or changes.
class Calibration {
	private scale = new Map<string, { seconds: number; bytes: number; files: number }>();
	private durations = 0;
	private opened = 0;

	add(c: Calibrant, sign: 1 | -1) {
		this.durations += sign * c.seconds;
		this.opened += sign;
		if (c.bytes > 0) {
			this.pool(c.ext, c, sign);
			this.pool('', c, sign);
		}
	}

	private pool(key: string, c: Calibrant, sign: 1 | -1) {
		let p = this.scale.get(key);
		if (!p) {
			p = { seconds: 0, bytes: 0, files: 0 };
			this.scale.set(key, p);
		}
		p.seconds += sign * c.seconds;
		p.bytes += sign * c.bytes;
		p.files += sign;
		if (p.files === 0) this.scale.delete(key);
	}

	snapshot(): Weighting {
		const scale: Weighting['scale'] = new Map();
		for (const [k, p] of this.scale) scale.set(k, { seconds: p.seconds, bytes: p.bytes });
		return { scale, meanDuration: this.opened === 0 ? 0 : this.durations / this.opened };
	}
}

class AnalysisRun {
	logLines = $state<string[]>([]);
	running = $state(false);
	error = $state<string | null>(null);
	// True once a run has stopped, whether cleanly finished, cancelled, or
	// errored — gates which of the run-finished rows below render.
	stopped = $state(false);
	// True if the run was stopped via beginStop() (the user's stop button)
	// rather than running to completion or dying on its own. Distinguishes
	// the "Stopped" header from "Analysis complete!" on a clean finish.
	cancelled = $state(false);
	startedAt = $state<number | null>(null);
	summary = $state<RunSummary | null>(null);
	// How far through startup the engine has reported getting. Only ever
	// advances: 'analyzing' is emitted once per analyzer, and a second
	// analyzer coming up must not drag the header back a step.
	stage = $state<Stage>('launching');
	// Rolling audio-seconds-processed samples, used to compute trailing ETA rate.
	// Not $state: nothing renders from it directly, only from the tick's snapshot.
	private rateSamples: { t: number; doneSeconds: number }[] = [];
	private lastPollTime = Date.now();
	private lastPollDoneSeconds = 0;
	private instantaneousRate = 0;
	// Wall clock, advanced on the tick so rate/ETA keep updating (and decay)
	// between engine events rather than freezing at the last one.
	private now = Date.now();
	private ticker: ReturnType<typeof setInterval> | null = null;
	private statsSnapshot = $state<Stats>(ZERO_STATS);

	// The audio tree, mutated in place as events arrive: an event updates the
	// one file it names and the running sums of the directories above it, so
	// its cost is the file's depth rather than the size of the tree, and only
	// rows whose numbers changed redraw. Replaced wholesale only by reset().
	private ctx = new TreeContext();
	tree = $state.raw(new TreeDir('', null, this.ctx));
	// Every file by path, for the events that name one.
	private index = new Map<string, FileNode>();
	private dirIndex = new Map<string, TreeDir>();
	private calibration = new Calibration();

	private dirFor(path: string): TreeDir {
		let d = this.dirIndex.get(path);
		if (!d) {
			d = path === '' ? this.tree : this.dirFor(dirOf(path)).child(nameOf(path));
			this.dirIndex.set(path, d);
		}
		return d;
	}

	private fileFor(path: string): FileNode {
		let f = this.index.get(path);
		if (!f) {
			const dir = this.dirFor(dirOf(path));
			f = new FileNode(path, dir, this.ctx);
			this.index.set(path, f);
			dir.addFile(f);
		}
		return f;
	}

	// Applies a change to one file and carries it up the tree and into the
	// calibration.
	private update(f: FileNode, change: (f: FileNode) => void) {
		const before = contributionOf(f);
		const calBefore = calibrantOf(f);
		change(f);
		f.parent.adjust(before, contributionOf(f));
		const calAfter = calibrantOf(f);
		if (!sameCalibrant(calBefore, calAfter)) {
			if (calBefore) this.calibration.add(calBefore, -1);
			if (calAfter) this.calibration.add(calAfter, 1);
			this.ctx.weighting = this.calibration.snapshot();
		}
	}

	// True once the engine's directory walk has reported every file it's
	// going to (manifest_done). Before that, the file list itself is
	// incomplete, on top of individual files' work not yet being known.
	get discoveryDone(): boolean {
		return this.ctx.discoveryDone;
	}

	// Run-wide sums, read straight off the root rather than recounted.
	get totals(): { workSeconds: number; doneSeconds: number; filesDone: number; filesTotal: number } {
		const t = this.tree;
		return {
			workSeconds: t.workSeconds,
			doneSeconds: t.doneSeconds + t.activeSeconds,
			filesDone: t.filesDone,
			filesTotal: t.filesTotal
		};
	}

	get stageLabel(): string {
		return STAGE_LABELS[this.stage];
	}

	// Whether the file list is complete — see TreeDir.finalized.
	get denominatorFinal(): boolean {
		return this.discoveryDone;
	}

	// Instantaneous realtime multiple: audio seconds analyzed since last poll
	// divided by elapsed wall-clock seconds.
	get rate(): number {
		return this.instantaneousRate;
	}

	private rateOver(windowMs: number): number {
		if (this.rateSamples.length < 2) return 0;
		const now = Math.max(this.now, this.rateSamples[this.rateSamples.length - 1].t);
		const cutoff = now - windowMs;
		// Baseline: the newest sample at or before the window start, so the
		// window covers the full span even when events are sparse.
		let first = this.rateSamples[0];
		for (const s of this.rateSamples) {
			if (s.t > cutoff) break;
			first = s;
		}
		const last = this.rateSamples[this.rateSamples.length - 1];
		const dt = (now - first.t) / 1000;
		if (dt <= 0) return 0;
		return Math.max(0, (last.doneSeconds - first.doneSeconds) / dt);
	}

	// Headline numbers for the run: everything but `priorSeconds` covers only
	// this session's work, so the ETA never counts audio an earlier run
	// already analyzed. Remaining is partly estimated — files nothing has
	// opened are charged a duration extrapolated from their size.
	//
	// Published as a snapshot refreshed on the tick rather than a live getter:
	// every chunk_done mutates `files` and the samples, so a getter would
	// recompute — and redraw these numbers — many times a second.
	get stats(): Stats {
		return this.statsSnapshot;
	}

	private computeStats(): Stats {
		const t = this.tree;
		const analyzed = t.doneSeconds + t.activeSeconds;
		const remainingSeconds = Math.max(0, t.totalSeconds - t.priorSeconds - analyzed);
		// The ETA runs off a trailing window so a momentary spike or dip
		// in throughput shouldn't swing it.
		const etaRate = this.rateOver(ETA_WINDOW_MS);
		return {
			priorSeconds: t.priorSeconds,
			remainingSeconds,
			rate: this.rate,
			etaSeconds: etaRate > 0 ? remainingSeconds / etaRate : null
		};
	}

	// Highest `seq` applied so far. Rust stamps one counter across a run's
	// events and log lines; a page re-attaching to a run replays a snapshot and
	// then has to skip the live events that snapshot already covered.
	private lastSeq = -1;

	private isStale(seq: unknown): boolean {
		if (typeof seq !== 'number') return false;
		if (seq <= this.lastSeq) return true;
		this.lastSeq = seq;
		return false;
	}

	// `startedAt` is passed when re-attaching, so runtime counts from the
	// engine's launch rather than from the page reload.
	reset(startedAt?: number) {
		this.lastSeq = -1;
		// The context is kept rather than replaced: the page's reads of
		// discoveryDone are subscribed to this one's signals.
		this.ctx.discoveryDone = false;
		this.ctx.weighting = { scale: new Map(), meanDuration: 0 };
		this.calibration = new Calibration();
		this.index = new Map();
		this.dirIndex = new Map();
		this.tree = new TreeDir('', null, this.ctx);
		this.logLines = [];
		this.error = null;
		this.stopped = false;
		this.cancelled = false;
		this.stopping = false;
		this.summary = null;
		this.stage = 'launching';
		const now = Date.now();
		this.now = now;
		this.rateSamples = [{ t: now, doneSeconds: 0 }];
		this.startedAt = startedAt ?? now;
		this.lastPollTime = now;
		this.lastPollDoneSeconds = 0;
		this.instantaneousRate = 0;
		this.running = true;
		this.statsSnapshot = ZERO_STATS;
		if (this.ticker === null && typeof setInterval === 'function') {
			this.ticker = setInterval(() => this.tick(), TICK_MS);
		}
	}

	private tick() {
		const now = Date.now();
		this.now = now;
		const currentDoneSeconds = this.totals.doneSeconds;
		const dt = (now - this.lastPollTime) / 1000;
		if (dt > 0) {
			const dDone = currentDoneSeconds - this.lastPollDoneSeconds;
			this.instantaneousRate = Math.max(0, dDone / dt);
		}
		this.lastPollTime = now;
		this.lastPollDoneSeconds = currentDoneSeconds;
		this.statsSnapshot = this.computeStats();
	}

	// Cancel has been requested but the engine hasn't exited yet. The run stays
	// locked while this is true: the engine is still analysing, still writing
	// results, and still sending progress, so releasing the UI here would let
	// the user start a second run on top of the first.
	stopping = $state(false);

	beginStop() {
		this.stopping = true;
	}

	stop(error?: string) {
		this.cancelled = this.stopping;
		this.running = false;
		this.stopping = false;
		this.stopped = true;
		if (error) this.error = error;
		const now = Date.now();
		this.now = now;
		this.instantaneousRate = 0;
		this.statsSnapshot = this.computeStats();
		if (this.startedAt !== null) {
			const runtimeSeconds = Math.max(0, (now - this.startedAt) / 1000);
			const audioSeconds = this.totals.doneSeconds;
			const rate = runtimeSeconds > 0 ? audioSeconds / runtimeSeconds : 0;
			this.summary = { audioSeconds, runtimeSeconds, rate };
		}
		if (this.ticker !== null) {
			clearInterval(this.ticker);
			this.ticker = null;
		}
	}

	private touchRate() {
		const now = Date.now();
		const sample = { t: now, doneSeconds: this.totals.doneSeconds };
		const samples = this.rateSamples;
		// Chunks can land many times a second; collapse those into one sample
		// per SAMPLE_MS so the buffer stays small over the ETA window.
		const last = samples[samples.length - 1];
		if (last && now - last.t < SAMPLE_MS && samples.length > 1) samples[samples.length - 1] = sample;
		else samples.push(sample);
		// Drop samples that fall entirely out of the trailing window, keeping
		// one baseline before it so the rate covers the whole window.
		const cutoff = now - ETA_WINDOW_MS;
		while (samples.length > 2 && samples[1].t <= cutoff) samples.shift();
	}

	handleEvent(payload: any) {
		if (this.isStale(payload.seq)) return;
		switch (payload.event) {
			case 'stage': {
				// Not a startup stage: the engine has begun winding down, which
				// it reports whether the stop came from this app's button, a
				// Ctrl-C in a terminal, or a signal from elsewhere. Only the
				// first of those has already flipped `stopping` locally.
				if (payload.name === 'stopping') {
					this.beginStop();
					break;
				}
				const next = payload.name as Stage;
				const rank = STAGE_ORDER.indexOf(next);
				if (rank > STAGE_ORDER.indexOf(this.stage)) this.stage = next;
				break;
			}
			case 'manifest': {
				const sizes = (payload.bytes ?? []) as number[];
				(payload.paths as string[]).forEach((path, i) => {
					if (this.index.has(path)) return;
					const f = this.fileFor(path);
					if (sizes[i]) this.update(f, (f) => (f.bytes = sizes[i]));
				});
				break;
			}
			case 'manifest_done': {
				this.ctx.discoveryDone = true;
				break;
			}
			case 'file_skip': {
				this.update(this.fileFor(payload.path), (f) => {
					f.status = 'skipped';
					f.workSeconds = 0;
					f.doneSeconds = 0;
				});
				break;
			}
			case 'file_start': {
				this.update(this.fileFor(payload.path), (f) => {
					f.status = 'running';
					f.duration = payload.duration;
					f.workSeconds = payload.work_seconds;
					f.doneSeconds = 0;
				});
				break;
			}
			case 'chunk_done': {
				const existing = this.index.get(payload.path);
				if (existing) {
					// chunk_start/chunk_end are absolute offsets in the file, so
					// they can't be used as a done-so-far position: a resumed file
					// only re-analyzes the gaps its previous run left, and chunks
					// can complete out of order across analyzers. Accumulate chunk
					// lengths instead, which is what work_seconds counts.
					const chunkWork = payload.chunk_end - (payload.chunk_start ?? payload.chunk_end);
					this.update(existing, (f) => {
						f.doneSeconds = payload.done
							? f.workSeconds
							: Math.min(f.workSeconds, f.doneSeconds + chunkWork);
						f.status = payload.done ? 'done' : 'running';
					});
				}
				this.touchRate();
				break;
			}
			case 'error': {
				this.stop(payload.message);
				break;
			}
		}
	}

	handleLog(line: string, seq?: number) {
		if (this.isStale(seq)) return;
		this.logLines.push(line);
		if (this.logLines.length > 500) this.logLines.shift();
	}

	// Applies engine output in arrival order. Events and log lines share one
	// seq counter, so they have to be applied interleaved as they came.
	handleOutput(items: ({ kind: 'event'; payload: any } | { kind: 'log'; line: string; seq?: number })[]) {
		for (const it of items) {
			if (it.kind === 'event') this.handleEvent(it.payload);
			else this.handleLog(it.line, it.seq);
		}
	}
}

export const run = new AnalysisRun();
