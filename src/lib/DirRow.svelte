<script lang="ts">
	import type { TreeDir } from './progress.svelte';
	import Self from './DirRow.svelte';

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
	const percent = $derived(pct(node.doneSeconds, node.estWorkSeconds));

	function toggle() {
		if (isOpen) expanded.delete(node.path);
		else expanded.add(node.path);
	}
</script>

<div class="tree-row" style="padding-left: {depth * 1.25}rem">
	<button class="row" class:done={isDone} class:provisional={!node.finalized} onclick={toggle}>
		<span class="disclosure">{isOpen ? '▾' : '▸'}</span>
		<span class="name">{node.name}</span>
		<span class="bar"><span class="fill" style="width: {percent}%"></span></span>
		<span class="count">{isDone ? '✓' : `${percent}%`}</span>
	</button>
</div>

{#if isOpen}
	{#each node.dirs as child (child.path)}
		<Self node={child} depth={depth + 1} {expanded} {pct} />
	{/each}
	{#each node.files as f (f.path)}
		{@const filePct = f.status === 'skipped' ? 100 : pct(f.doneSeconds, f.workSeconds || f.duration || 1)}
		{@const fileDone = f.status === 'done' || f.status === 'skipped'}
		<div class="tree-row" style="padding-left: {(depth + 1) * 1.25}rem">
			<div class="row static" class:done={fileDone}>
				<span class="disclosure"></span>
				<span class="name">{f.name}</span>
				<span class="bar"><span class="fill" style="width: {filePct}%"></span></span>
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

	.bar {
		position: relative;
		height: 0.5rem;
		border-radius: 4px;
		background: rgba(127, 127, 127, 0.25);
		overflow: hidden;
	}

	.bar .fill {
		position: absolute;
		inset: 0;
		width: 0;
		background: #4c8dff;
	}

	.row.provisional .bar .fill {
		background: repeating-linear-gradient(
			45deg,
			#c99b3f,
			#c99b3f 6px,
			#dcb35f 6px,
			#dcb35f 12px
		);
	}

	.row.done {
		background: rgba(76, 141, 255, 0.16);
		border-radius: 6px;
	}

	.row.done .bar .fill {
		background: #4caf50;
	}

	.count {
		text-align: right;
		font-variant-numeric: tabular-nums;
		opacity: 0.7;
	}
</style>
