// The settings store round-trips through localStorage, so each test loads a
// fresh module against a fresh fake store.

import { beforeEach, describe, expect, it, vi } from 'vitest';

const KEY = 'buzzdetect.settings';

function fakeStorage(initial: Record<string, string> = {}) {
	const data = new Map(Object.entries(initial));
	return {
		getItem: (k: string) => data.get(k) ?? null,
		setItem: (k: string, v: string) => void data.set(k, v),
		removeItem: (k: string) => void data.delete(k),
		clear: () => data.clear(),
		key: (i: number) => [...data.keys()][i] ?? null,
		get length() {
			return data.size;
		}
	};
}

async function load(stored?: unknown) {
	vi.resetModules();
	const storage =
		stored === undefined ? fakeStorage() : fakeStorage({ [KEY]: JSON.stringify(stored) });
	vi.stubGlobal('localStorage', storage);
	const mod = await import('./settings.svelte');
	return { settings: mod.settings, storage, LOGLEVELS: mod.LOGLEVELS };
}

beforeEach(() => {
	vi.unstubAllGlobals();
});

describe('defaults', () => {
	it('start with nothing chosen and a run that can be configured', async () => {
		const { settings } = await load();
		expect(settings.value.modelname).toBe('');
		expect(settings.value.dirAudio).toBe('');
		expect(settings.value.dirOut).toBe('');
		expect(settings.value.dirOutTouched).toBe(false);
		expect(settings.value.classesOut).toEqual([]);
		expect(settings.value.chunklength).toBeGreaterThan(0);
		expect(settings.value.analyzersCpu).toBeGreaterThan(0);
		expect(settings.value.analyzersGpu).toBe(0);
		expect(settings.value.gpuFp16).toBe(false);
	});

	it('name log levels the engine accepts', async () => {
		const { settings, LOGLEVELS } = await load();
		expect(LOGLEVELS).toContain(settings.value.verbosityPrint);
		expect(LOGLEVELS).toContain(settings.value.verbosityLog);
	});
});

describe('loading what was stored', () => {
	it('restores a previous session', async () => {
		const { settings } = await load({
			modelname: 'model_general_v3',
			dirAudio: '/data/audio',
			dirOut: '/data/out',
			dirOutTouched: true,
			classesOut: ['ins_buzz']
		});
		expect(settings.value.modelname).toBe('model_general_v3');
		expect(settings.value.classesOut).toEqual(['ins_buzz']);
	});

	it('fills in settings a stored object predates', async () => {
		const { settings } = await load({ modelname: 'm' });
		expect(settings.value.chunklength).toBeGreaterThan(0);
		expect(settings.value.verbosityLog).toBeTruthy();
	});

	it('falls back to defaults on unreadable storage', async () => {
		vi.resetModules();
		vi.stubGlobal('localStorage', {
			getItem: () => '{ not json',
			setItem: () => {}
		});
		const { settings } = await import('./settings.svelte');
		expect(settings.value.modelname).toBe('');
	});

	it('works with no storage at all', async () => {
		vi.resetModules();
		vi.stubGlobal('localStorage', undefined);
		const { settings } = await import('./settings.svelte');
		expect(settings.value.modelname).toBe('');
	});
});

describe('the relative output path an older build could have stored', () => {
	// The engine resolves a relative path against its own working directory,
	// which in an installed app is read-only inside the bundle.
	it('is dropped so the default can be re-derived', async () => {
		const { settings } = await load({ dirOut: 'output/model_general_v3', dirOutTouched: true });
		expect(settings.value.dirOut).toBe('');
		expect(settings.value.dirOutTouched).toBe(false);
	});

	it('keeps a unix absolute path', async () => {
		const { settings } = await load({ dirOut: '/home/luke/results', dirOutTouched: true });
		expect(settings.value.dirOut).toBe('/home/luke/results');
		expect(settings.value.dirOutTouched).toBe(true);
	});

	it('keeps a windows path, drive letter or UNC', async () => {
		expect((await load({ dirOut: 'C:\\Users\\luke\\results' })).settings.value.dirOut).toBe(
			'C:\\Users\\luke\\results'
		);
		expect((await load({ dirOut: 'C:/Users/luke/results' })).settings.value.dirOut).toBe(
			'C:/Users/luke/results'
		);
		expect((await load({ dirOut: '\\\\server\\share\\results' })).settings.value.dirOut).toBe(
			'\\\\server\\share\\results'
		);
	});

	it('leaves an empty path alone rather than treating it as relative', async () => {
		const { settings } = await load({ dirOut: '', dirOutTouched: true });
		expect(settings.value.dirOut).toBe('');
	});
});

describe('saving', () => {
	it('round-trips through storage', async () => {
		const { settings, storage } = await load();
		settings.value.modelname = 'model_general_v3';
		settings.value.classesOut = ['ins_buzz', 'frog'];
		settings.save();
		expect(JSON.parse(storage.getItem(KEY)!)).toMatchObject({
			modelname: 'model_general_v3',
			classesOut: ['ins_buzz', 'frog']
		});
	});

	it('is best-effort: a private window is not a crash', async () => {
		vi.resetModules();
		vi.stubGlobal('localStorage', {
			getItem: () => null,
			setItem: () => {
				throw new Error('QuotaExceededError');
			}
		});
		const { settings } = await import('./settings.svelte');
		expect(() => settings.save()).not.toThrow();
	});
});
