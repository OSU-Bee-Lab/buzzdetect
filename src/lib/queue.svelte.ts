// A queue of runs, each with its own settings, executed one after another.
// The engine still runs one analysis at a time; the page starts the next
// queued item when the previous one's engine exits.
//
// An item can be edited until it starts. Once it has run to completion it is
// frozen; one that was stopped or errored can be edited and run again (the
// engine resumes whatever it left unfinished). An error doesn't halt the
// queue -- the point is to leave a batch running unattended -- only Stop does.
//
// Persisted to localStorage, so a queue survives the app being closed and the
// webview reloading under a running engine.

import type { Settings } from './settings.svelte';
import type { RunSummary, Weights } from './progress.svelte';

export type QueueStatus = 'pending' | 'running' | 'done' | 'stopped' | 'error';

export interface QueueItem {
	id: number;
	settings: Settings;
	status: QueueStatus;
	// The bar as the run left it; the running item's bar is drawn live instead.
	weights: Weights | null;
	summary: RunSummary | null;
	error: string | null;
}

const STORAGE_KEY = 'buzzdetect.queue';

/** Pending, stopped and errored items can be edited, deleted and run. */
export function isEditable(item: QueueItem): boolean {
	return item.status !== 'running' && item.status !== 'done';
}

interface Stored {
	items: QueueItem[];
	active: boolean;
}

function load(): Stored {
	if (typeof localStorage === 'undefined') return { items: [], active: false };
	try {
		const raw = localStorage.getItem(STORAGE_KEY);
		if (!raw) return { items: [], active: false };
		const s = JSON.parse(raw) as Stored;
		return { items: Array.isArray(s.items) ? s.items : [], active: !!s.active };
	} catch {
		return { items: [], active: false };
	}
}

class RunQueue {
	#stored = load();
	items = $state<QueueItem[]>(this.#stored.items);
	// True while the queue is advancing on its own. Cleared by Stop, by an
	// empty queue, and by a start the engine refused.
	active = $state(this.#stored.active);
	selectedId = $state<number | null>(null);
	#nextId = Math.max(0, ...this.items.map((i) => i.id)) + 1;

	get selected(): QueueItem | undefined {
		return this.items.find((i) => i.id === this.selectedId);
	}

	get current(): QueueItem | undefined {
		return this.items.find((i) => i.status === 'running');
	}

	get hasRunnable(): boolean {
		return this.items.some(isEditable);
	}

	add(settings: Settings, status: QueueStatus = 'pending'): QueueItem {
		this.items.push({
			id: this.#nextId++,
			settings: structuredClone(settings),
			status,
			weights: null,
			summary: null,
			error: null
		});
		this.save();
		return this.items[this.items.length - 1];
	}

	remove(id: number) {
		const item = this.items.find((i) => i.id === id);
		if (!item || item.status === 'running') return;
		this.items = this.items.filter((i) => i.id !== id);
		if (this.selectedId === id) this.selectedId = null;
		if (!this.hasRunnable && !this.current) this.active = false;
		this.save();
	}

	/**
	 * The next item to run: the first runnable one after `afterId` in queue
	 * order, or from the top when `afterId` is omitted. Advancing only forward
	 * keeps a run that just errored from being retried in a loop.
	 */
	next(afterId?: number): QueueItem | undefined {
		const from = afterId === undefined ? 0 : this.items.findIndex((i) => i.id === afterId) + 1;
		return this.items.slice(from).find(isEditable);
	}

	finish(id: number, result: Pick<QueueItem, 'status' | 'weights' | 'summary' | 'error'>) {
		const item = this.items.find((i) => i.id === id);
		if (!item) return;
		Object.assign(item, result);
		this.save();
	}

	setStatus(id: number, status: QueueStatus, error: string | null = null) {
		const item = this.items.find((i) => i.id === id);
		if (!item) return;
		item.status = status;
		item.error = error;
		this.save();
	}

	/**
	 * The page found no engine running, so whatever the queue thought was
	 * running ended while nobody was listening.
	 */
	orphaned() {
		for (const i of this.items) if (i.status === 'running') i.status = 'stopped';
		this.active = false;
		this.save();
	}

	save() {
		if (typeof localStorage === 'undefined') return;
		try {
			localStorage.setItem(
				STORAGE_KEY,
				JSON.stringify({ items: $state.snapshot(this.items), active: this.active })
			);
		} catch {
			// best-effort, as with settings
		}
	}
}

export const queue = new RunQueue();
