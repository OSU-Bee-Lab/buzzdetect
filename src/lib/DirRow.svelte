<script lang="ts">
	import { run, type TreeDir } from './progress.svelte';
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
	const percent = $derived(
		pct(node.priorSeconds + node.doneSeconds + node.activeSeconds, node.totalSeconds)
	);
	// Green check means nothing here needed analyzing this session; blue means
	// this run did some of the work.
	const bySession = $derived(node.doneSeconds + node.activeSeconds > 0);

	function toggle() {
		if (isOpen) expanded.delete(node.path);
		else expanded.add(node.path);
	}
</script>

<div class="tree-row" style="padding-left: {depth * 1.25}rem">
	<button class="row" onclick={toggle}>
		<span class="disclosure">{isOpen ? '▾' : '▸'}</span>
		<span class="name">{node.name}</span>
		<ProgressBar weights={node} provisional={!node.finalized} />
		{#if isDone}
			<span class="count check" class:session={bySession}>✓</span>
		{:else}
			<span class="count">{percent}%</span>
		{/if}
	</button>
</div>

{#if isOpen}
	{#each node.dirs as child (child.path)}
		<Self node={child} depth={depth + 1} {expanded} {pct} />
	{/each}
	{#each node.files as f (f.path)}
		{@const w = f.weights}
		{@const filePct = pct(w.priorSeconds + w.doneSeconds + w.activeSeconds, w.totalSeconds)}
		<div class="tree-row" style="padding-left: {(depth + 1) * 1.25}rem">
			<div class="row static">
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
</style>
