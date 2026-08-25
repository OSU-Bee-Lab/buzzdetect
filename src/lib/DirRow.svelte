<script lang="ts">
	import type { TreeDir } from './progress.svelte';
	import Self from './DirRow.svelte';
	import ProgressBar from './ProgressBar.svelte';

	let {
		node,
		depth,
		expanded,
		pct
	}: {
		node: TreeDir;
		depth: number;
		expanded: Set<string>;
		pct: (done: number, total: number) => number;
	} = $props();

	const isOpen = $derived(expanded.has(node.path));
	const isDone = $derived(node.finalized && node.filesTotal > 0 && node.filesDone === node.filesTotal);
	// Counts everything analyzed, whether by this session or an earlier run,
	// so the number agrees with how much of the bar is filled.
	const percent = $derived(pct(node.priorSeconds + node.doneSeconds, node.totalSeconds));

	function toggle() {
		if (isOpen) expanded.delete(node.path);
		else expanded.add(node.path);
	}
</script>

<div class="tree-row" style="padding-left: {depth * 1.25}rem">
	<button class="row" class:done={isDone} onclick={toggle}>
		<span class="disclosure">{isOpen ? '▾' : '▸'}</span>
		<span class="name">{node.name}</span>
		<ProgressBar weights={node} provisional={!node.finalized} />
		<span class="count">{isDone ? '✓' : `${percent}%`}</span>
	</button>
</div>

{#if isOpen}
	{#each node.dirs as child (child.path)}
		<Self node={child} depth={depth + 1} {expanded} {pct} />
	{/each}
	{#each node.files as f (f.path)}
		{@const w = f.weights}
		{@const filePct = pct(w.priorSeconds + w.doneSeconds, w.totalSeconds)}
		{@const fileDone = f.status === 'done' || f.status === 'skipped'}
		<div class="tree-row" style="padding-left: {(depth + 1) * 1.25}rem">
			<div class="row static" class:done={fileDone}>
				<span class="disclosure"></span>
				<span class="name">{f.name}</span>
				<ProgressBar weights={w} />
				<span class="count">{fileDone ? '✓' : `${filePct}%`}</span>
			</div>
		</div>
	{/each}
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
		border: none;
		background: none;
		text-align: left;
		font-size: 0.85rem;
		box-sizing: border-box;
	}

	.row.static {
		cursor: default;
	}

	button.row {
		cursor: pointer;
		font: inherit;
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

	.row.done {
		background: rgba(76, 141, 255, 0.16);
		border-radius: 6px;
	}

	.count {
		text-align: right;
		font-variant-numeric: tabular-nums;
		opacity: 0.7;
	}
</style>
