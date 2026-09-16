<script lang="ts">
	import { run, type FileNode } from './progress.svelte';
	import ProgressBar from './ProgressBar.svelte';

	// Directories can hold thousands of recordings, and mounting a row (plus
	// its bar) for each one at once stalls the webview for seconds. Rows are
	// drawn a page at a time instead; the rest wait behind a button.
	const PAGE = 200;

	let {
		files,
		depth,
		pct
	}: {
		files: FileNode[];
		depth: number;
		pct: (done: number, total: number) => number;
	} = $props();

	let limit = $state(PAGE);
	const shown = $derived(files.length > limit ? files.slice(0, limit) : files);
	const hidden = $derived(files.length - shown.length);
</script>

{#each shown as f (f.path)}
	{@const w = f.weights}
	{@const filePct = pct(w.priorSeconds + w.doneSeconds + w.activeSeconds, w.totalSeconds)}
	<div class="tree-row" style="padding-left: {depth * 1.25}rem">
		<div class="row">
			<span class="disclosure"></span>
			<span class="name">{f.name}</span>
			<ProgressBar weights={w} />
			{#if f.status === 'done' || f.status === 'skipped'}
				<span class="count check" class:session={f.status === 'done'}>✓</span>
			{:else}
				<!-- Red marks a file the run left part-analyzed. -->
				<span class="count" class:interrupted={run.stopped && f.status === 'running'}
					>{filePct}%</span
				>
			{/if}
		</div>
	</div>
{/each}
{#if hidden > 0}
	<div class="tree-row" style="padding-left: {depth * 1.25}rem">
		<button class="more" onclick={() => (limit += PAGE)}>
			Show {Math.min(PAGE, hidden)} more ({hidden} hidden)
		</button>
	</div>
{/if}

<style>
	.tree-row {
		min-width: 0;
	}

	.row {
		width: 100%;
		display: grid;
		grid-template-columns: 1.2em minmax(0, 1fr) 100px 3.5em;
		align-items: center;
		gap: 0.5rem;
		padding: 0.35rem 0.6rem;
		text-align: left;
		font-size: 0.85rem;
		box-sizing: border-box;
		cursor: default;
	}

	.disclosure {
		opacity: 0.6;
		text-align: center;
	}

	.name {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		min-width: 0;
	}

	.count {
		text-align: right;
		font-variant-numeric: tabular-nums;
		opacity: 0.7;
	}

	.count.check {
		opacity: 1;
		color: #4caf50;
	}

	.count.check.session {
		color: #4c8dff;
	}

	.count.interrupted {
		opacity: 1;
		color: #e05a4f;
	}

	.more {
		border: none;
		background: none;
		font: inherit;
		font-size: 0.8rem;
		opacity: 0.7;
		cursor: pointer;
		padding: 0.35rem 0.6rem 0.35rem calc(0.6rem + 1.2em + 0.5rem);
	}

	.more:hover {
		opacity: 1;
		text-decoration: underline;
	}
</style>
