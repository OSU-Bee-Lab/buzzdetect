// Reactive state for one analysis run, fed by the "engine-progress" events
// Rust forwards straight from the Python engine's BDPROGRESS lines (see
// engine/src/pipeline/progress_json.py). Event types: manifest (one file
// discovered), manifest_done (discovery walk finished), file_start,
// file_skip, chunk_done (which carries chunk_start/chunk_end as absolute
// offsets within the file, not a done-so-far position).

export type FileStatus = 'pending' | 'running' | 'done' | 'skipped';

export interface FileProgress {
	path: string; // path relative to dir_audio, e.g. "siteA/2024-06-01.wav"
	dir: string; // parent dir of path ("" for files at the audio dir's root)
	name: string;
	status: FileStatus;
	duration: number; // full audio duration in seconds, once known
	workSeconds: number; // seconds of audio actually needing analysis (duration minus any already-completed portion)
	doneSeconds: number; // work completed so far, summed over finished chunks
}

// A directory node in the audio tree, aggregated recursively from its
// children. `finalized` means discovery has finished, so this subtree's file
// list and counts are complete; until then new files can still appear under
// it and the bar is drawn provisionally (striped).
//
// A file's real work_seconds is only known once a streamer reaches it, so
// `workSeconds` covers visited files only. `estWorkSeconds` additionally
// charges each not-yet-visited file the mean work of the files seen so far,
// which keeps the percentage roughly stable across the run instead of having
// it collapse every time a new file's duration lands in the denominator.
export interface TreeDir {
	path: string;
	name: string;
	dirs: TreeDir[];
	files: FileProgress[];
	workSeconds: number;
	estWorkSeconds: number;
	doneSeconds: number;
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
	// (which would understate current speed after a slow startup).
	private rateSamples: { t: number; doneSeconds: number }[] = [];

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
		const meanWork = this.meanKnownWork;
		const build = (node: MutableNode): TreeDir => {
			const dirs = [...node.dirs.values()].map(build).sort((a, b) => a.name.localeCompare(b.name));
			const files = [...node.files].sort(
				(a, b) => statusRank(a.status) - statusRank(b.status) || a.name.localeCompare(b.name)
			);

			let workSeconds = 0;
			let estWorkSeconds = 0;
			let doneSeconds = 0;
			let filesTotal = 0;
			let filesDone = 0;
			for (const d of dirs) {
				workSeconds += d.workSeconds;
				estWorkSeconds += d.estWorkSeconds;
				doneSeconds += d.doneSeconds;
				filesTotal += d.filesTotal;
				filesDone += d.filesDone;
			}
			for (const f of files) {
				workSeconds += f.workSeconds;
				estWorkSeconds += f.status === 'pending' ? meanWork : f.workSeconds;
				doneSeconds += f.doneSeconds;
				filesTotal += 1;
				if (f.status === 'done' || f.status === 'skipped') filesDone += 1;
			}

			return {
				path: node.path,
				name: node.name,
				dirs,
				files,
				workSeconds,
				estWorkSeconds,
				doneSeconds,
				filesTotal,
				filesDone,
				finalized: discoveryDone
			};
		};

		return build(root);
	}

	// Mean work_seconds over the files a streamer has already reached, used to
	// charge not-yet-visited files a plausible weight. Skipped files are
	// excluded: their zero work says nothing about the files still queued.
	private get meanKnownWork(): number {
		let total = 0;
		let n = 0;
		for (const f of this.files.values()) {
			if (f.status === 'running' || f.status === 'done') {
				total += f.workSeconds;
				n += 1;
			}
		}
		return n === 0 ? 0 : total / n;
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

	// Realtime multiple over the trailing window, e.g. 12.4 means the engine
	// is processing 12.4 seconds of audio per wall-clock second.
	get rate(): number {
		if (this.rateSamples.length < 2) return 0;
		const first = this.rateSamples[0];
		const last = this.rateSamples[this.rateSamples.length - 1];
		const dt = (last.t - first.t) / 1000;
		if (dt <= 0) return 0;
		return (last.doneSeconds - first.doneSeconds) / dt;
	}

	reset() {
		this.files = new Map();
		this.logLines = [];
		this.error = null;
		this.stopped = false;
		this.discoveryDone = false;
		this.rateSamples = [];
		this.startedAt = Date.now();
		this.running = true;
	}

	stop(error?: string) {
		this.running = false;
		this.stopped = true;
		if (error) this.error = error;
	}

	private touchRate() {
		const now = Date.now();
		this.rateSamples.push({ t: now, doneSeconds: this.totals.doneSeconds });
		// Keep ~30s of samples so the rate reflects recent speed, not the
		// whole run (long-idle-then-burst wouldn't average sensibly otherwise).
		const cutoff = now - 30_000;
		while (this.rateSamples.length > 2 && this.rateSamples[0].t < cutoff) {
			this.rateSamples.shift();
		}
	}

	handleEvent(payload: any) {
		switch (payload.event) {
			case 'manifest': {
				const files = new Map(this.files);
				for (const path of payload.paths as string[]) {
					if (!files.has(path)) {
						files.set(path, {
							path,
							dir: dirOf(path),
							name: nameOf(path),
							status: 'pending',
							duration: 0,
							workSeconds: 0,
							doneSeconds: 0
						});
					}
				}
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
