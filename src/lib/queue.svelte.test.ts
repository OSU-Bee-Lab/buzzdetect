// The queue store round-trips through localStorage, so each test loads a
// fresh module against a fresh fake store. What it decides -- which run goes
// next, what can be edited -- is the part worth pinning down; the page only
// starts whatever it hands back.

import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { Settings } from './settings.svelte';

const KEY = 'buzzdetect.queue';

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
	const mod = await import('./queue.svelte');
	return { ...mod, storage };
}

function s(dirAudio: string): Settings {
	return { dirAudio } as Settings;
}

beforeEach(() => {
	vi.unstubAllGlobals();
});

describe('order', () => {
	it('runs from the top, then moves forward only', async () => {
		const { queue } = await load();
		const a = queue.add(s('a'));
		const b = queue.add(s('b'));
		const c = queue.add(s('c'));
		expect(queue.next()?.id).toBe(a.id);
		queue.setStatus(a.id, 'running');
		queue.finish(a.id, { status: 'error', weights: null, summary: null, error: 'boom' });
		// An error moves on rather than retrying or halting.
		expect(queue.next(a.id)?.id).toBe(b.id);
		queue.setStatus(b.id, 'done');
		expect(queue.next(b.id)?.id).toBe(c.id);
		queue.setStatus(c.id, 'done');
		expect(queue.next(c.id)).toBeUndefined();
		// From the top again, the errored run is retried.
		expect(queue.next()?.id).toBe(a.id);
	});

	it('picks up an item added behind the one running', async () => {
		const { queue } = await load();
		const a = queue.add(s('a'));
		queue.setStatus(a.id, 'running');
		const b = queue.add(s('b'));
		expect(queue.next(a.id)?.id).toBe(b.id);
	});

	it('treats stopped and errored runs as editable, finished and running ones not', async () => {
		const { queue, isEditable } = await load();
		const a = queue.add(s('a'));
		const status = (st: Parameters<typeof queue.setStatus>[1]) => {
			queue.setStatus(a.id, st);
			return isEditable(queue.items[0]);
		};
		expect(status('pending')).toBe(true);
		expect(status('stopped')).toBe(true);
		expect(status('error')).toBe(true);
		expect(status('running')).toBe(false);
		expect(status('done')).toBe(false);
	});
});

describe('remove', () => {
	it('refuses the running item and drops the selection of a removed one', async () => {
		const { queue } = await load();
		const a = queue.add(s('a'));
		const b = queue.add(s('b'));
		queue.setStatus(a.id, 'running');
		queue.remove(a.id);
		expect(queue.items).toHaveLength(2);
		queue.selectedId = b.id;
		queue.remove(b.id);
		expect(queue.items.map((i) => i.id)).toEqual([a.id]);
		expect(queue.selectedId).toBeNull();
	});
});

describe('persistence', () => {
	it('copies settings rather than aliasing them', async () => {
		const { queue } = await load();
		const src = s('a');
		queue.add(src);
		src.dirAudio = 'changed';
		expect(queue.items[0].settings.dirAudio).toBe('a');
	});

	it('survives a reload, with fresh ids continuing past the stored ones', async () => {
		const first = await load();
		first.queue.add(s('a'));
		first.queue.add(s('b'));
		const raw = first.storage.getItem(KEY)!;
		const { queue } = await load(JSON.parse(raw));
		expect(queue.items.map((i) => i.settings.dirAudio)).toEqual(['a', 'b']);
		const c = queue.add(s('c'));
		expect(new Set(queue.items.map((i) => i.id)).size).toBe(3);
		expect(c.id).toBeGreaterThan(Math.max(...queue.items.slice(0, 2).map((i) => i.id)));
	});

	it('marks a run nobody is attached to any more as stopped and halts', async () => {
		const { queue } = await load({
			active: true,
			items: [{ id: 1, settings: s('a'), status: 'running', weights: null, summary: null, error: null }]
		});
		expect(queue.active).toBe(true);
		queue.orphaned();
		expect(queue.items[0].status).toBe('stopped');
		expect(queue.active).toBe(false);
	});

	it('falls back to an empty queue on garbage', async () => {
		vi.resetModules();
		vi.stubGlobal('localStorage', fakeStorage({ [KEY]: '{not json' }));
		const { queue } = await import('./queue.svelte');
		expect(queue.items).toEqual([]);
	});
});
