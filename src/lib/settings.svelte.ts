// Persisted analysis settings. Basic settings (model/audio dir/output dir)
// live directly on the page; everything else sits behind Advanced. Whole
// object is round-tripped through localStorage so the app reopens with
// whatever was last used (see buzzdetect's old guisettings.json cache, which
// this replaces).

export const LOGLEVELS = ['NOTSET', 'DEBUG', 'PROGRESS', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'];

export interface Settings {
	modelname: string;
	dirAudio: string;
	dirOut: string;
	// Once the user edits dirOut by hand, model changes stop overwriting it.
	dirOutTouched: boolean;
	classesOut: string[]; // must be non-empty to start a run; defaults to all of the selected model's classes
	chunklength: number;
	analyzersCpu: number;
	analyzersGpu: number;
	nStreamers: number | null;
	streamBufferDepth: number | null;
	verbosityPrint: string;
	verbosityLog: string;
	logProgress: boolean;
}

const STORAGE_KEY = 'buzzdetect2.settings';

function defaults(): Settings {
	return {
		modelname: '',
		dirAudio: '',
		dirOut: '',
		dirOutTouched: false,
		classesOut: [],
		chunklength: 200,
		analyzersCpu: 2,
		analyzersGpu: 0,
		nStreamers: null,
		streamBufferDepth: null,
		verbosityPrint: 'PROGRESS',
		verbosityLog: 'DEBUG',
		logProgress: false
	};
}

function load(): Settings {
	if (typeof localStorage === 'undefined') return defaults();
	try {
		const raw = localStorage.getItem(STORAGE_KEY);
		if (!raw) return defaults();
		return { ...defaults(), ...JSON.parse(raw) };
	} catch {
		return defaults();
	}
}

class SettingsStore {
	value = $state<Settings>(load());

	save() {
		if (typeof localStorage === 'undefined') return;
		try {
			localStorage.setItem(STORAGE_KEY, JSON.stringify(this.value));
		} catch {
			// best-effort; a private window or full storage just means no persistence
		}
	}
}

export const settings = new SettingsStore();
