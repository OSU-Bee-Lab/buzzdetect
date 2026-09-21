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
		<!-- rtl + ellipsis clips from the left edge, so the end of the path stays visible; <bdi> keeps the text itself reading left to right. -->
		<span class="path"><bdi>{#if parts.parent}<span class="parent">{parts.parent}</span>{/if}<span>{parts.leaf}</span></bdi></span>
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

	.path {
		flex: 1;
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		direction: rtl;
	}

	.parent {
		opacity: 0.6;
	}
</style>
