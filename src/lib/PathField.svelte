<script lang="ts">
	import { splitPath } from './paths';

	// A path shown so its final folder is always readable: a long parent is
	// abbreviated from the left, the leaf stays whole. Clicking it swaps in a
	// plain input for typing or pasting a path.
	let {
		value = $bindable(''),
		placeholder = '',
		oninput
	}: { value: string; placeholder?: string; oninput?: () => void } = $props();

	let editing = $state(false);
	let input = $state<HTMLInputElement | null>(null);
	const parts = $derived(splitPath(value));

	$effect(() => {
		if (editing) input?.focus();
	});
</script>

{#if editing || !value}
	<input bind:this={input} bind:value {placeholder} {oninput} onblur={() => (editing = false)} />
{:else}
	<button type="button" class="path-display" data-tooltip={value} onclick={() => (editing = true)}>
		<!-- rtl + ellipsis clips the parent from its left edge; <bdi> keeps the text itself reading left to right. -->
		{#if parts.parent}<span class="parent"><bdi>{parts.parent}</bdi></span>{/if}<span class="leaf" class:solo={!parts.parent}>{parts.leaf}</span>
	</button>
{/if}

<style>
	input,
	.path-display {
		flex: 1;
		min-width: 0;
		font: inherit;
		padding: 0.4rem 0.6rem;
		border-radius: 6px;
		border: 1px solid rgba(127, 127, 127, 0.4);
		box-sizing: border-box;
	}

	.path-display {
		display: flex;
		align-items: baseline;
		text-align: left;
		background: transparent;
		color: inherit;
		white-space: nowrap;
		cursor: text;
	}

	.parent {
		flex: 0 100 auto;
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		direction: rtl;
		opacity: 0.6;
	}

	.leaf {
		flex: 0 1 auto;
		min-width: 3ch;
		overflow: hidden;
		text-overflow: ellipsis;
		font-weight: 600;
	}

	.leaf.solo {
		font-weight: inherit;
	}
</style>
