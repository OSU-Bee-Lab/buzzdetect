<script lang="ts">
	import type { Weights } from './progress.svelte';

	// Segments left to right: work a previous run finished (green), files this
	// session finished (blue), work on files still open (blue while the run is
	// live, red once it stops — that work was interrupted mid-file), then the
	// gray remainder. `provisional` stripes the filled part to say the total
	// can still grow, i.e. discovery is still walking the audio directory.
	let {
		weights,
		provisional = false,
		large = false,
		stopped = false
	}: { weights: Weights; provisional?: boolean; large?: boolean; stopped?: boolean } = $props();

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
	<span class="seg active" class:stopped style="width: {activePct}%"></span>
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

	.seg.active.stopped {
		background: #e05a4f;
	}

	.bar.provisional .seg.prior {
		background: repeating-linear-gradient(45deg, #3f8a43, #3f8a43 6px, #5fbf64 6px, #5fbf64 12px);
	}

	/* Interrupted work keeps its solid red — that state is worth reading at a
	   glance even if discovery never finished. */
	.bar.provisional .seg.done,
	.bar.provisional .seg.active:not(.stopped) {
		background: repeating-linear-gradient(45deg, #c99b3f, #c99b3f 6px, #dcb35f 6px, #dcb35f 12px);
	}
</style>
