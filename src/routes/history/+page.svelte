<script lang="ts">
	// The past-runs window, opened by the main window's history button
	// (open_history). Left: one row per run. Right: the settings that run used.
	import { invoke } from '@tauri-apps/api/core';
	import { emit, listen } from '@tauri-apps/api/event';
	import { getCurrentWindow } from '@tauri-apps/api/window';
	import { onMount } from 'svelte';
	import { baseName } from '$lib/paths';
	import { formatRate, formatRunTime, previewSettings, type HistoryEntry } from '$lib/history';

	let entries = $state<HistoryEntry[]>([]);
	let selectedId = $state<number | null>(null);

	const selected = $derived(entries.find((e) => e.started_at === selectedId) ?? null);
	// list_history only leaves "running" on the newest entry while a run is live.
	const runLive = $derived(entries[0]?.status === 'running');
	const rows = $derived(selected ? previewSettings(selected) : []);

	async function load() {
		try {
			entries = await invoke<HistoryEntry[]>('list_history');
		} catch {
			entries = [];
		}
		if (!entries.some((e) => e.started_at === selectedId)) selectedId = null;
	}

	async function clear() {
		if (!confirm('Clear the list of past runs? Results already written are not touched.')) return;
		await invoke('clear_history');
		await load();
	}

	// The main window owns the settings, so this hands them over, then closes.
	async function useSettings(e: HistoryEntry) {
		await emit('history-use', e.settings ?? { ...e.manifest });
		await getCurrentWindow().close();
	}

	onMount(() => {
		load();
		const unlisten = listen('engine-exit', load);
		const unlistenUpdated = listen('history-updated', load);
		window.addEventListener('focus', load);
		return () => {
			unlisten.then((f) => f());
			unlistenUpdated.then((f) => f());
			window.removeEventListener('focus', load);
		};
	});
</script>

<svelte:head><title>History</title></svelte:head>

<div class="window">
	<nav>
		{#each entries as e (e.started_at)}
			<button class="run" class:active={e.started_at === selectedId} onclick={() => (selectedId = e.started_at)}>
				<span class="time {e.status}">{formatRunTime(e.started_at)}</span>
				<span class="detail">{e.manifest.modelname}</span>
				<span class="detail">in: {baseName(e.manifest.dir_audio ?? '?')}</span>
				<span class="detail">out: {baseName(e.manifest.dir_out ?? '?')}</span>
				{#if formatRate(e)}<span class="detail">{formatRate(e)}</span>{/if}
			</button>
		{:else}
			<p class="empty">No runs in history.</p>
		{/each}
		{#if entries.length}
			<button class="clear" onclick={clear}>Clear history</button>
		{/if}
	</nav>

	<section>
		{#if selected}
			<dl>
				{#each rows as [k, v]}
					<dt>{k}</dt>
					<dd>{v}</dd>
				{/each}
			</dl>
			<div class="use-wrap">
				<button class="use" disabled={runLive} onclick={() => useSettings(selected)}>Use these settings</button>
				{#if runLive}<span class="use-note">Run in progress</span>{/if}
			</div>
		{:else if entries.length}
			<p class="empty">Select a run to see its settings.</p>
		{/if}
	</section>
</div>

<style>
	:global(html),
	:global(body) {
		margin: 0;
		font-family: Inter, Avenir, Helvetica, Arial, sans-serif;
		color-scheme: light dark;
	}

	.window {
		display: grid;
		grid-template-columns: 17rem 1fr;
		height: 100vh;
	}

	nav {
		overflow-y: auto;
		border-right: 1px solid rgba(127, 127, 127, 0.3);
		display: flex;
		flex-direction: column;
	}

	.run {
		display: flex;
		flex-direction: column;
		gap: 0.1rem;
		text-align: left;
		padding: 0.6rem 0.75rem;
		border: none;
		border-bottom: 1px solid rgba(127, 127, 127, 0.2);
		border-radius: 0;
		background: transparent;
		color: inherit;
		font: inherit;
		cursor: pointer;
	}

	.run.active {
		background: rgba(127, 127, 127, 0.18);
	}

	.time {
		font-weight: 600;
		font-size: 0.85rem;
	}

	.time.completed {
		color: light-dark(#2f6fe0, #4c8dff);
	}

	.time.errored {
		color: light-dark(#c0392b, #e5695c);
	}

	.time.interrupted {
		color: light-dark(#b26a00, #e0a030);
	}

	.detail {
		font-size: 0.8rem;
		opacity: 0.7;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.clear {
		margin: 0.75rem;
		align-self: flex-start;
		font: inherit;
		font-size: 0.8rem;
		padding: 0.25rem 0.5rem;
	}

	section {
		display: flex;
		flex-direction: column;
		overflow-y: auto;
		padding: 1rem 1.25rem;
	}

	dl {
		display: grid;
		grid-template-columns: max-content 1fr;
		gap: 0.4rem 1rem;
		margin: 0 0 1rem;
		font-size: 0.85rem;
	}

	dt {
		opacity: 0.6;
	}

	dd {
		margin: 0;
		word-break: break-all;
	}

	.use-wrap {
		margin-top: auto;
		align-self: flex-start;
		position: sticky;
		bottom: 0;
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}

	.use-note {
		font-size: 0.75rem;
		opacity: 0.6;
	}

	.use {
		font: inherit;
		padding: 0.4rem 0.75rem;
		cursor: pointer;
	}

	.empty {
		opacity: 0.6;
		font-size: 0.85rem;
		padding: 0 0.75rem;
	}
</style>
