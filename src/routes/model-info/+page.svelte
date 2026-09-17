<script lang="ts">
	// The model info window (opened by the main window's Info button through
	// the open_model_info command). A reader, not an editor: the README as
	// rendered markdown, with the model's suggested thresholds from
	// config_model.json above it.
	import { invoke } from '@tauri-apps/api/core';
	import { listen } from '@tauri-apps/api/event';
	import { openUrl } from '@tauri-apps/plugin-opener';
	import { marked } from 'marked';
	import DOMPurify from 'dompurify';
	import { onMount } from 'svelte';
	import {
		classifyLink,
		fprTarget,
		thresholdRows,
		type ModelDetails,
		type ModelInfo
	} from '$lib/modelInfo';

	let models = $state<ModelInfo[]>([]);
	let selected = $state(new URLSearchParams(location.search).get('model') ?? '');
	let details = $state<ModelDetails | null>(null);
	let error = $state<string | null>(null);
	let body: HTMLElement | undefined = $state();

	const rows = $derived(details ? thresholdRows(details) : []);
	const target = $derived(details ? fprTarget(details) : null);
	const hasStats = $derived(rows.some((r) => r.ci95 || r.folds || r.events !== null));
	// README content comes from model folders, including ones a collaborator
	// sent, and this webview can invoke app commands -- so it's sanitized.
	const html = $derived(
		details?.readme ? DOMPurify.sanitize(marked.parse(details.readme, { async: false })) : ''
	);

	async function load(name: string) {
		selected = name;
		error = null;
		history.replaceState(null, '', `?model=${encodeURIComponent(name)}`);
		try {
			details = await invoke<ModelDetails>('model_details', { modelname: name });
		} catch (e) {
			details = null;
			error = String(e);
		}
		body?.scrollTo(0, 0);
	}

	function onClick(e: MouseEvent) {
		const a = (e.target as HTMLElement).closest('a');
		if (!a) return;
		const link = classifyLink(a.getAttribute('href') ?? '');
		if (link.kind === 'anchor') return;
		e.preventDefault();
		const opened =
			link.kind === 'external'
				? openUrl(link.url)
				: invoke('open_model_file', { modelname: selected, rel: link.rel });
		opened.catch((err) => (error = String(err)));
	}

	const fmt = (v: number) => v.toFixed(3);

	onMount(() => {
		invoke<ModelInfo[]>('list_models').then((list) => (models = list));
		if (selected) load(selected);
		const unlisten = listen<string>('model-info-select', (e) => load(e.payload));
		return () => {
			unlisten.then((f) => f());
		};
	});
</script>

<svelte:head><title>{selected ? `${selected} — model info` : 'Model info'}</title></svelte:head>

<div class="window" class:single={models.length <= 1}>
	{#if models.length > 1}
		<nav>
			{#each models as m}
				<button
					type="button"
					class:active={m.name === selected}
					class:no-readme={!m.has_readme}
					onclick={() => load(m.name)}>{m.name}</button
				>
			{/each}
		</nav>
	{/if}
	<main bind:this={body}>
		<h1>{selected}</h1>
		{#if details?.description}
			<p class="description">{details.description}</p>
		{/if}
		{#if error}
			<p class="error">{error}</p>
		{/if}

		{#if rows.length}
			<section class="thresholds">
				<h2>
					Recommended thresholds{#if target !== null}<span class="hint">
							at {(target * 100).toFixed(1)}% false positive rate</span
						>{/if}
				</h2>
				<table>
					<thead>
						<tr>
							<th>Class</th>
							<th>Threshold</th>
							{#if hasStats}
								<th>95% CI</th>
								<th>SD</th>
								<th>Deployments</th>
								<th>Events</th>
							{/if}
						</tr>
					</thead>
					<tbody>
						{#each rows as r}
							<tr>
								<td>{r.cls}</td>
								<td class="num">{fmt(r.threshold)}</td>
								{#if hasStats}
									<td class="num">{r.ci95 ? `${fmt(r.ci95[0])} to ${fmt(r.ci95[1])}` : ''}</td>
									<td class="num">{r.sd !== null ? fmt(r.sd) : ''}</td>
									<td class="num">{r.folds ?? ''}</td>
									<td class="num">{r.events ?? ''}</td>
								{/if}
							</tr>
						{/each}
					</tbody>
				</table>
			</section>
		{/if}

		{#if details && !details.readme}
			<p class="hint">This model has no README.</p>
		{:else if html}
			<!-- eslint-disable-next-line svelte/no-at-html-tags -- sanitized above -->
			<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_noninteractive_element_interactions -->
			<article class="readme" onclick={onClick}>{@html html}</article>
		{/if}
	</main>
</div>

<style>
	:root {
		font-family: Inter, Avenir, Helvetica, Arial, sans-serif;
		color-scheme: light dark;
	}

	:global(html),
	:global(body) {
		margin: 0;
		height: 100%;
		overflow: hidden;
	}

	.window {
		display: grid;
		grid-template-columns: 200px 1fr;
		height: 100vh;
	}

	.window.single {
		grid-template-columns: 1fr;
	}

	nav {
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
		padding: 1rem 0.5rem;
		border-right: 1px solid rgba(127, 127, 127, 0.2);
		overflow-y: auto;
	}

	nav button {
		font: inherit;
		font-size: 0.9rem;
		text-align: left;
		padding: 0.35rem 0.6rem;
		border: none;
		border-radius: 6px;
		background: none;
		color: inherit;
		cursor: pointer;
		overflow-wrap: anywhere;
	}

	nav button:hover {
		background: rgba(127, 127, 127, 0.12);
	}

	nav button.active {
		background: rgba(127, 127, 127, 0.22);
		font-weight: 600;
	}

	nav button.no-readme {
		opacity: 0.6;
	}

	main {
		overflow-y: auto;
		padding: 1.25rem 1.75rem 2rem;
		min-width: 0;
	}

	h1 {
		margin: 0 0 0.25rem;
		font-size: 1.4rem;
		overflow-wrap: anywhere;
	}

	.description {
		margin: 0 0 1rem;
		opacity: 0.8;
	}

	.hint {
		opacity: 0.6;
		font-size: 0.85rem;
		font-weight: normal;
		margin-left: 0.5rem;
	}

	p.hint {
		margin-left: 0;
	}

	.error {
		color: #d33;
	}

	.thresholds {
		margin: 0.5rem 0 1.5rem;
		padding: 0.75rem 1rem;
		border: 1px solid rgba(127, 127, 127, 0.3);
		border-radius: 8px;
		overflow-x: auto;
	}

	.thresholds h2 {
		margin: 0 0 0.5rem;
		font-size: 1rem;
	}

	table {
		border-collapse: collapse;
		font-size: 0.9rem;
	}

	th,
	td {
		padding: 0.25rem 0.75rem 0.25rem 0;
		text-align: left;
		white-space: nowrap;
	}

	th {
		font-weight: 600;
		border-bottom: 1px solid rgba(127, 127, 127, 0.3);
	}

	td.num {
		font-variant-numeric: tabular-nums;
	}

	.readme {
		line-height: 1.55;
		max-width: 75ch;
	}

	.readme :global(h1),
	.readme :global(h2),
	.readme :global(h3) {
		margin: 1.4em 0 0.4em;
		line-height: 1.25;
	}

	.readme :global(h1) {
		font-size: 1.3rem;
	}

	.readme :global(h2) {
		font-size: 1.15rem;
		padding-bottom: 0.2em;
		border-bottom: 1px solid rgba(127, 127, 127, 0.25);
	}

	.readme :global(h3) {
		font-size: 1rem;
	}

	.readme :global(a) {
		color: #3b82c4;
	}

	.readme :global(code) {
		font-size: 0.88em;
		padding: 0.1em 0.3em;
		border-radius: 4px;
		background: rgba(127, 127, 127, 0.15);
	}

	.readme :global(pre) {
		padding: 0.75rem;
		border-radius: 6px;
		background: rgba(127, 127, 127, 0.12);
		overflow-x: auto;
	}

	.readme :global(pre code) {
		padding: 0;
		background: none;
	}

	.readme :global(table) {
		display: block;
		overflow-x: auto;
		margin: 0.75rem 0;
	}

	.readme :global(td),
	.readme :global(th) {
		border-bottom: 1px solid rgba(127, 127, 127, 0.2);
	}

	.readme :global(img) {
		max-width: 100%;
	}

	.readme :global(blockquote) {
		margin: 0.75rem 0;
		padding-left: 0.9rem;
		border-left: 3px solid rgba(127, 127, 127, 0.4);
		opacity: 0.85;
	}
</style>
