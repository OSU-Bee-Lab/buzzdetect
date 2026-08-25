<script lang="ts">
	import type { Weights } from './progress.svelte';

	// Segments left to right: work a previous run finished (green), work this
	// session did (blue, split into finished files and files still open), then
	// the gray remainder. `provisional` stripes the filled part to say the
	// total can still grow, i.e. discovery is still walking the audio
	// directory; each segment stripes in its own color so the green/blue
	// distinction survives that state.
	let {
		weights,
		provisional = false,
		large = false
	}: { weights: Weights; provisional?: boolean; large?: boolean } = $props();

	const share = $derived((v: number) =>
		weights.totalSeconds > 0 ? Math.max(0, Math.min(100, (v / weights.totalSeconds) * 100)) : 0
	);
	const priorPct = $derived(share(weights.priorSeconds));
	const donePct = $derived(Math.min(100 - priorPct, share(weights.doneSeconds)));
	const activePct = $derived(Math.min(100 - priorPct - donePct, share(weights.activeSeconds)));
</script>

<span class="bar" class:provisional class:large>
	<span class="seg prior" style="width: {priorPct}%"></span>
	<span class="seg done" style="width: {donePct}%"></span>
	<span class="seg active" style="width: {activePct}%"></span>
</span>

<style>
	.bar {
		display: flex;
		height: 0.5rem;
		border-radius: 4px;
		background: rgba(127, 127, 127, 0.25);
		overflow: hidden;
	}

	.bar.large {
		height: 1rem;
		border-radius: 6px;
		background: rgba(127, 127, 127, 0.2);
		flex-shrink: 0;
	}

	.seg {
		transition: width 0.2s ease;
	}

	.seg.prior {
		background: #4caf50;
	}

	.seg.done,
	.seg.active {
		background: #4c8dff;
	}

	.bar.provisional .seg.prior {
		background: repeating-linear-gradient(45deg, #3f8a43, #3f8a43 6px, #5fbf64 6px, #5fbf64 12px);
	}

	.bar.provisional .seg.done,
	.bar.provisional .seg.active {
		background: repeating-linear-gradient(45deg, #3d74d4, #3d74d4 6px, #6ba1ff 6px, #6ba1ff 12px);
	}
</style>
