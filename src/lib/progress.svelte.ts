// Reactive state for one analysis run, fed by the "engine-progress" events
// Rust forwards straight from the Python engine's BDPROGRESS lines (see
// engine/src/pipeline/progress_json.py). Event types: manifest (one file
// discovered), manifest_done (discovery walk finished), file_start,
// file_skip, chunk_done (which carries chunk_start/chunk_end as absolute
// offsets within the file, not a done-so-far position).

export type FileStatus = 'pending' | 'running' | 'done' | 'skipped';

// Trailing window the displayed realtime rate averages over, and the longer
// one the ETA uses so it doesn't chase every fluctuation in throughput.
const RATE_WINDOW_MS = 30_000;
const ETA_WINDOW_MS = 300_000;
// How often the clock driving rate/ETA advances, and the minimum spacing
// between retained samples (bounding the buffer over the ETA window).
const TICK_MS = 2_000;
const SAMPLE_MS = 1_000;

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

// A file as the tree hands it to the UI: its own progress plus the segment
// weights the bar is drawn from, resolved once per tree build.
export interface FileNode extends FileProgress {
	weights: Weights;
}

// A directory node in the audio tree, aggregated recursively from its
// children. `finalized` means discovery has finished, so this subtree's file
// list and counts are complete; until then new files can still appear under
// it and the bar is drawn provisionally (striped).
//
// The weights are partly estimated: a file's duration is only known once a
// streamer opens it, so files still queued — and files skipped without ever
// being opened — are charged a duration extrapolated from their byte size
// (see estimateDuration). Every visited file contributes its real numbers,
// so a dir's remaining seconds are exact except for the files nothing has
// opened yet.
export interface TreeDir extends Weights {
	path: string;
	name: string;
	dirs: TreeDir[];
	files: FileNode[];
	workSeconds: number;
	filesTotal: number;
	filesDone: number;
	finalized: boolean;
}

function dirOf(path: string): string {
	const idx = path.lastIndexOf('/');
	return idx === -1 ? '' : path.slice(0, idx);
}

function nameOf(path: string): string {
	const idx = path.lastIndexOf('/');
	return idx === -1 ? path : path.slice(idx + 1);
}

function statusRank(s: FileStatus): number {
	return s === 'running' ? 0 : s === 'pending' ? 1 : 2;
}

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

function estimateDuration(f: FileProgress, w: Weighting): number {
	const pooled = w.scale.get(extOf(f.path)) ?? w.scale.get('');
	if (f.bytes > 0 && pooled && pooled.bytes > 0) {
		return (f.bytes * pooled.seconds) / pooled.bytes;
	}
	// No size for this file, or nothing opened yet to calibrate against: fall
	// back to the mean duration so far, then to weighting files equally.
	return w.meanDuration > 0 ? w.meanDuration : 1;
}

// Splits one file across the bar's segments. A visited file reports its own
// exact numbers; anything else is estimated from size. A skipped file counts
// as entirely analyzed-already: the engine only skips files whose results are
// complete (or that it can't analyze at all, which won't progress either way).
function fileWeights(f: FileProgress, w: Weighting): Weights {
	if (f.status === 'running' || f.status === 'done') {
		const totalSeconds = f.duration > 0 ? f.duration : f.workSeconds;
		return {
			totalSeconds,
			priorSeconds: Math.max(0, totalSeconds - f.workSeconds),
			doneSeconds: f.status === 'done' ? f.doneSeconds : 0,
			activeSeconds: f.status === 'done' ? 0 : f.doneSeconds
		};
	}
	const totalSeconds = f.duration > 0 ? f.duration : estimateDuration(f, w);
	return {
		totalSeconds,
		priorSeconds: f.status === 'skipped' ? totalSeconds : 0,
		doneSeconds: 0,
		activeSeconds: 0
	};
}

interface MutableNode {
	path: string;
	name: string;
	dirs: Map<string, MutableNode>;
	files: FileProgress[];
}

class AnalysisRun {
	files = $state<Map<string, FileProgress>>(new Map());
	logLines = $state<string[]>([]);
	running = $state(false);
	error = $state<string | null>(null);
	// True once a run has stopped, whether cleanly finished, cancelled, or
	// errored — drives the "Stopped" header without needing an error message
	// (a user-initiated cancel has nothing to say in the error paragraph).
	stopped = $state(false);
	startedAt = $state<number | null>(null);
	// True once the engine's directory walk has reported every file it's
	// going to (manifest_done). Before that, the file list itself is
	// incomplete, on top of individual files' work not yet being known.
	discoveryDone = $state(false);
	// Rolling audio-seconds-processed samples, used to compute a live
	// realtime-multiple rate instead of an average since the run started
	// (which would understate current speed after a slow startup). $state so
	// the rate getter re-runs in the UI as samples arrive.
	private rateSamples = $state<{ t: number; doneSeconds: number }[]>([]);
	// Wall clock, ticked while a run is live so rate/ETA keep updating (and
	// decay) between engine events rather than freezing at the last one.
	private now = $state(Date.now());
	private ticker: ReturnType<typeof setInterval> | null = null;

	get tree(): TreeDir {
		const root: MutableNode = { path: '', name: '', dirs: new Map(), files: [] };
		for (const f of this.files.values()) {
			const segments = f.dir === '' ? [] : f.dir.split('/');
			let node = root;
			let pathSoFar = '';
			for (const seg of segments) {
				pathSoFar = pathSoFar ? `${pathSoFar}/${seg}` : seg;
				let child = node.dirs.get(seg);
				if (!child) {
					child = { path: pathSoFar, name: seg, dirs: new Map(), files: [] };
					node.dirs.set(seg, child);
				}
				node = child;
			}
			node.files.push(f);
		}

		const discoveryDone = this.discoveryDone;
		const weighting = this.weighting;
		const build = (node: MutableNode): TreeDir => {
			const dirs = [...node.dirs.values()].map(build).sort((a, b) => a.name.localeCompare(b.name));
			const files: FileNode[] = [...node.files]
				.sort((a, b) => statusRank(a.status) - statusRank(b.status) || a.name.localeCompare(b.name))
				.map((f) => ({ ...f, weights: fileWeights(f, weighting) }));

			let workSeconds = 0;
			let totalSeconds = 0;
			let priorSeconds = 0;
			let doneSeconds = 0;
			let activeSeconds = 0;
			let filesTotal = 0;
			let filesDone = 0;
			for (const d of dirs) {
				workSeconds += d.workSeconds;
				totalSeconds += d.totalSeconds;
				priorSeconds += d.priorSeconds;
				doneSeconds += d.doneSeconds;
				activeSeconds += d.activeSeconds;
				filesTotal += d.filesTotal;
				filesDone += d.filesDone;
			}
			for (const f of files) {
				workSeconds += f.workSeconds;
				totalSeconds += f.weights.totalSeconds;
				priorSeconds += f.weights.priorSeconds;
				doneSeconds += f.weights.doneSeconds;
				activeSeconds += f.weights.activeSeconds;
				filesTotal += 1;
				if (f.status === 'done' || f.status === 'skipped') filesDone += 1;
			}

			return {
				path: node.path,
				name: node.name,
				dirs,
				files,
				workSeconds,
				totalSeconds,
				priorSeconds,
				doneSeconds,
				activeSeconds,
				filesTotal,
				filesDone,
				finalized: discoveryDone
			};
		};

		return build(root);
	}

	// Calibrates size -> duration from the files already opened this run, so
	// files nothing has opened can be weighted by their byte size.
	private get weighting(): Weighting {
		const scale: Weighting['scale'] = new Map();
		let durations = 0;
		let n = 0;
		const add = (key: string, seconds: number, bytes: number) => {
			const pooled = scale.get(key);
			if (pooled) {
				pooled.seconds += seconds;
				pooled.bytes += bytes;
			} else {
				scale.set(key, { seconds, bytes });
			}
		};
		for (const f of this.files.values()) {
			if (f.status !== 'running' && f.status !== 'done') continue;
			if (f.duration <= 0) continue;
			durations += f.duration;
			n += 1;
			if (f.bytes > 0) {
				add(extOf(f.path), f.duration, f.bytes);
				add('', f.duration, f.bytes);
			}
		}
		return { scale, meanDuration: n === 0 ? 0 : durations / n };
	}

	get totals(): { workSeconds: number; doneSeconds: number; filesDone: number; filesTotal: number } {
		let workSeconds = 0;
		let doneSeconds = 0;
		let filesDone = 0;
		let filesTotal = 0;
		for (const f of this.files.values()) {
			workSeconds += f.workSeconds;
			doneSeconds += f.doneSeconds;
			filesTotal += 1;
			if (f.status === 'done' || f.status === 'skipped') filesDone += 1;
		}
		return { workSeconds, doneSeconds, filesDone, filesTotal };
	}

	// Whether the file list is complete — see TreeDir.finalized.
	get denominatorFinal(): boolean {
		return this.discoveryDone;
	}

	// Realtime multiple over the trailing window: audio seconds analyzed per
	// wall-clock second, computed here rather than taken from the engine (the
	// analyzers report their own throughput, not this end-to-end rate). The
	// window ends at `now`, not at the last event, so a stall decays the rate
	// instead of freezing it at the last burst.
	get rate(): number {
		return this.rateOver(RATE_WINDOW_MS);
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
	get stats(): {
		priorSeconds: number;
		remainingSeconds: number;
		rate: number;
		etaSeconds: number | null;
	} {
		const t = this.tree;
		const analyzed = t.doneSeconds + t.activeSeconds;
		const remainingSeconds = Math.max(0, t.totalSeconds - t.priorSeconds - analyzed);
		// The ETA runs off a much longer window than the displayed rate: it's a
		// projection over the whole remaining run, so a momentary spike or dip
		// in throughput shouldn't swing it.
		const etaRate = this.rateOver(ETA_WINDOW_MS);
		return {
			priorSeconds: t.priorSeconds,
			remainingSeconds,
			rate: this.rate,
			etaSeconds: etaRate > 0 ? remainingSeconds / etaRate : null
		};
	}

	reset() {
		this.files = new Map();
		this.logLines = [];
		this.error = null;
		this.stopped = false;
		this.discoveryDone = false;
		const now = Date.now();
		this.now = now;
		this.rateSamples = [{ t: now, doneSeconds: 0 }];
		this.startedAt = now;
		this.running = true;
		if (this.ticker === null && typeof setInterval === 'function') {
			this.ticker = setInterval(() => (this.now = Date.now()), TICK_MS);
		}
	}

	stop(error?: string) {
		this.running = false;
		this.stopped = true;
		if (error) this.error = error;
		if (this.ticker !== null) {
			clearInterval(this.ticker);
			this.ticker = null;
		}
	}

	private touchRate() {
		const now = Date.now();
		this.now = now;
		const sample = { t: now, doneSeconds: this.totals.doneSeconds };
		const samples = [...this.rateSamples];
		// Chunks can land many times a second; collapse those into one sample
		// per SAMPLE_MS so the buffer stays small over the ETA window.
		const last = samples[samples.length - 1];
		if (last && now - last.t < SAMPLE_MS && samples.length > 1) samples[samples.length - 1] = sample;
		else samples.push(sample);
		// Drop samples that fall entirely out of the trailing window, keeping
		// one baseline before it so the rate covers the whole window.
		const cutoff = now - ETA_WINDOW_MS;
		while (samples.length > 2 && samples[1].t <= cutoff) samples.shift();
		this.rateSamples = samples;
	}

	handleEvent(payload: any) {
		switch (payload.event) {
			case 'manifest': {
				const files = new Map(this.files);
				const sizes = (payload.bytes ?? []) as number[];
				(payload.paths as string[]).forEach((path, i) => {
					if (!files.has(path)) {
						files.set(path, {
							path,
							dir: dirOf(path),
							name: nameOf(path),
							status: 'pending',
							bytes: sizes[i] ?? 0,
							duration: 0,
							workSeconds: 0,
							doneSeconds: 0
						});
					}
				});
				this.files = files;
				break;
			}
			case 'manifest_done': {
				this.discoveryDone = true;
				break;
			}
			case 'file_skip': {
				const files = new Map(this.files);
				const existing = files.get(payload.path);
				files.set(payload.path, {
					path: payload.path,
					dir: dirOf(payload.path),
					name: nameOf(payload.path),
					status: 'skipped',
					bytes: existing?.bytes ?? 0,
					duration: existing?.duration ?? 0,
					workSeconds: 0,
					doneSeconds: 0
				});
				this.files = files;
				break;
			}
			case 'file_start': {
				const files = new Map(this.files);
				files.set(payload.path, {
					path: payload.path,
					dir: dirOf(payload.path),
					name: nameOf(payload.path),
					status: 'running',
					bytes: files.get(payload.path)?.bytes ?? 0,
					duration: payload.duration,
					workSeconds: payload.work_seconds,
					doneSeconds: 0
				});
				this.files = files;
				break;
			}
			case 'chunk_done': {
				const files = new Map(this.files);
				const existing = files.get(payload.path);
				if (existing) {
					// chunk_start/chunk_end are absolute offsets in the file, so
					// they can't be used as a done-so-far position: a resumed file
					// only re-analyzes the gaps its previous run left, and chunks
					// can complete out of order across analyzers. Accumulate chunk
					// lengths instead, which is what work_seconds counts.
					const chunkWork = payload.chunk_end - (payload.chunk_start ?? payload.chunk_end);
					const doneSeconds = payload.done
						? existing.workSeconds
						: Math.min(existing.workSeconds, existing.doneSeconds + chunkWork);
					files.set(payload.path, {
						...existing,
						doneSeconds,
						status: payload.done ? 'done' : 'running'
					});
					this.files = files;
				}
				this.touchRate();
				break;
			}
		}
	}

	handleLog(line: string) {
		this.logLines.push(line);
		if (this.logLines.length > 500) this.logLines.shift();
	}
}

export const run = new AnalysisRun();
