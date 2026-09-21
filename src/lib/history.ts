// What the past-runs window shows. The entries come from Rust's list_history
// (see history_entry in lib.rs).

export interface HistorySettings {
	modelname: string;
	dir_audio: string;
	dir_out: string;
	classes_out: string[];
	chunklength?: number;
	analyzers_cpu?: number;
	analyzers_gpu?: number;
	gpu_fp16?: boolean;
	n_streamers?: number | null;
	stream_buffer_depth?: number | null;
	verbosity_print?: string;
	verbosity_log?: string;
	log_progress?: boolean;
}

export interface HistoryEntry {
	started_at: number; // unix seconds
	status: 'running' | 'completed' | 'stopped' | 'errored' | 'interrupted';
	manifest: {
		modelname: string;
		dir_audio?: string;
		dir_out?: string;
		classes_out: string[] | null;
	};
	// What the run got done; absent while it runs, or if it ended before analyzing anything.
	result?: { audio_seconds: number; runtime_seconds: number; rate: number };
	// Absent on runs recorded before the full settings were kept.
	settings?: HistorySettings;
}

/** "September 21, 2026 3:49 PM" -- the locale's own "at" is left out. */
export function formatRunTime(unixSeconds: number, locale = 'en-US'): string {
	const d = new Date(unixSeconds * 1000);
	const day = d.toLocaleDateString(locale, { dateStyle: 'long' });
	const time = d.toLocaleTimeString(locale, { timeStyle: 'short' });
	return `${day} ${time}`;
}

/** "30x realtime", or null when the run recorded no rate. */
export function formatRate(e: HistoryEntry): string | null {
	const r = e.result?.rate;
	if (!r || r <= 0) return null;
	return `${r >= 100 ? Math.round(r) : r.toFixed(1)}x realtime`;
}

/** The settings to preview: the full record if there is one, else what the manifest kept. */
export function previewSettings(e: HistoryEntry): [string, string][] {
	const s = e.settings;
	const m = e.manifest;
	const rows: [string, string][] = [
		['Status', e.status],
		...(e.result
			? ([
					['Analysis rate', formatRate(e) ?? '?'],
					['Audio analyzed (s)', String(Math.round(e.result.audio_seconds))],
					['Run time (s)', String(Math.round(e.result.runtime_seconds))]
				] as [string, string][])
			: []),
		['Model', s?.modelname ?? m.modelname],
		['Audio directory', s?.dir_audio ?? m.dir_audio ?? '?'],
		['Output directory', s?.dir_out ?? m.dir_out ?? '?'],
		['Classes out', (s?.classes_out ?? m.classes_out ?? []).join(', ') || 'all']
	];
	if (!s) return rows;
	const auto = (v: number | null | undefined) => (v == null ? 'auto' : String(v));
	rows.push(
		['Chunk length (s)', String(s.chunklength ?? '')],
		['CPU analyzers', String(s.analyzers_cpu ?? '')],
		['GPU analyzers', String(s.analyzers_gpu ?? '')],
		['Reduced precision (fp16)', s.gpu_fp16 ? 'yes' : 'no'],
		['Concurrent streamers', auto(s.n_streamers)],
		['Stream buffer depth', auto(s.stream_buffer_depth)],
		['Console verbosity', s.verbosity_print ?? ''],
		['Log file verbosity', s.verbosity_log ?? ''],
		['Log progress statements', s.log_progress ? 'yes' : 'no']
	);
	return rows;
}
