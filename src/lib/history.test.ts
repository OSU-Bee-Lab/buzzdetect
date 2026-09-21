import { describe, expect, it } from 'vitest';
import { formatRunTime, previewSettings, type HistoryEntry } from './history';

const entry: HistoryEntry = {
	started_at: 0,
	status: 'completed',
	manifest: { modelname: 'm', dir_audio: '/a', dir_out: '/o', classes_out: ['ins_buzz'] }
};

describe('formatRunTime', () => {
	it('reads as a date then a time, with no "at"', () => {
		const t = Date.UTC(2026, 8, 21, 15, 49) / 1000;
		const out = formatRunTime(t);
		expect(out).toMatch(/^September 2[12], 2026 \d{1,2}:\d{2}\s?[AP]M$/);
	});
});

describe('previewSettings', () => {
	it('falls back to the manifest keys for an old entry', () => {
		const rows = Object.fromEntries(previewSettings(entry));
		expect(rows['Model']).toBe('m');
		expect(rows['Classes out']).toBe('ins_buzz');
		expect(rows['Chunk length (s)']).toBeUndefined();
	});
	it('lists every launch setting when they were recorded', () => {
		const rows = Object.fromEntries(
			previewSettings({
				...entry,
				settings: {
					modelname: 'm',
					dir_audio: '/a',
					dir_out: '/o',
					classes_out: ['ins_buzz'],
					chunklength: 200,
					n_streamers: null,
					gpu_fp16: true
				}
			})
		);
		expect(rows['Chunk length (s)']).toBe('200');
		expect(rows['Concurrent streamers']).toBe('auto');
		expect(rows['Reduced precision (fp16)']).toBe('yes');
	});
});
