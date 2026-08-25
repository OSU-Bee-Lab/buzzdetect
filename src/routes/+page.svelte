<script lang="ts">
	import { invoke } from '@tauri-apps/api/core';
	import { listen } from '@tauri-apps/api/event';
	import { open } from '@tauri-apps/plugin-dialog';
	import { onMount } from 'svelte';
	import { run, type TreeDir } from '$lib/progress.svelte';
	import { settings, LOGLEVELS } from '$lib/settings.svelte';
	import DirRow from '$lib/DirRow.svelte';
	import ProgressBar from '$lib/ProgressBar.svelte';

	interface Manifest {
		modelname: string;
		classes_out: string[] | null;
		framehop_prop: number | null;
	}

	let models = $state<string[]>([]);
	let availableClasses = $state<string[]>([]);
	let startError = $state<string | null>(null);
	let settingsWidth = $state(300);
	let resizing = false;
	let expanded = $state<Set<string>>(new Set());
	let hasAutoExpanded = false;
	let hasStarted = $state(false);
	let manifest = $state<Manifest | null>(null);

	const modelMismatch = $derived(
		manifest !== null && manifest.modelname !== settings.value.modelname
	);
	const manifestLocked = $derived(manifest !== null && !modelMismatch);

	function startResize(e: PointerEvent) {
		resizing = true;
		(e.target as HTMLElement).setPointerCapture(e.pointerId);
	}
	function onResize(e: PointerEvent) {
		if (!resizing) return;
		settingsWidth = Math.min(600, Math.max(220, e.clientX));
	}
	function stopResize() {
		resizing = false;
	}

	onMount(() => {
		const unlistenProgress = listen<any>('engine-progress', (e) => run.handleEvent(e.payload));
		const unlistenLog = listen<{ line: string; stderr: boolean }>('engine-log', (e) =>
			run.handleLog(e.payload.line)
		);
		const unlistenExit = listen<{ code: number | null }>('engine-exit', (e) => {
			run.stop(e.payload.code !== 0 ? `engine exited with code ${e.payload.code}` : undefined);
		});

		invoke<string[]>('list_models').then((list) => {
			models = list;
			if (!settings.value.modelname || !list.includes(settings.value.modelname)) {
				settings.value.modelname = list[0] ?? '';
			}
			onModelChange();
		});
		checkManifest();

		return () => {
			unlistenProgress.then((f) => f());
			unlistenLog.then((f) => f());
			unlistenExit.then((f) => f());
		};
	});

	// Auto-expand the first top-level directory once a run's file tree
	// starts filling in, so there's always something visible by default.
	$effect(() => {
		const topDirs = run.tree.dirs;
		if (!hasAutoExpanded && topDirs.length > 0) {
			hasAutoExpanded = true;
			expanded.add(topDirs[0].path);
		}
	});

	// buzzdetect locks schema-defining settings (output classes, framehop) to
	// match an output folder's existing manifest, so a resumed run can't
	// silently write incompatible results into it — see buzzdetect_gui.py's
	// _apply_manifest_lock, which this mirrors. The model itself is never
	// forced to match; a mismatch is surfaced as an error instead (see
	// modelMismatch), since forcing it out from under the user is surprising.
	async function checkManifest() {
		if (!settings.value.dirOut) {
			manifest = null;
			return;
		}
		try {
			manifest = await invoke<Manifest | null>('read_manifest', { dirOut: settings.value.dirOut });
		} catch {
			manifest = null;
		}
		if (!manifest || manifest.modelname !== settings.value.modelname) return;
		if (manifest.classes_out) settings.value.classesOut = manifest.classes_out;
		if (manifest.framehop_prop !== null) settings.value.framehopProp = manifest.framehop_prop;
		settings.save();
	}

	// Re-derive the class list whenever the model changes, and auto-fill
	// dirOut to the model's output dir unless the user has typed their own.
	async function onModelChange() {
		if (!settings.value.modelname) return;
		if (!settings.value.dirOutTouched) {
			settings.value.dirOut = `models/${settings.value.modelname}/output`;
			await checkManifest();
		}
		try {
			const classes = await invoke<string[]>('get_model_classes', {
				modelname: settings.value.modelname
			});
			availableClasses = classes;
			// Keep only still-valid selections from a prior model; if that
			// leaves nothing selected (fresh model, or the prior selection no
			// longer applies), default to everything rather than leaving the
			// run blocked on an empty selection.
			const kept = settings.value.classesOut.filter((c) => classes.includes(c));
			settings.value.classesOut = kept.length > 0 ? kept : [...classes];
		} catch {
			availableClasses = [];
		}
		settings.save();
	}

	function onDirOutInput() {
		settings.value.dirOutTouched = true;
		settings.save();
		checkManifest();
	}

	async function browseDirAudio() {
		const dir = await open({ directory: true, defaultPath: settings.value.dirAudio || undefined });
		if (typeof dir === 'string') {
			settings.value.dirAudio = dir;
			settings.save();
		}
	}

	async function browseDirOut() {
		const dir = await open({ directory: true, defaultPath: settings.value.dirOut || undefined });
		if (typeof dir === 'string') {
			settings.value.dirOut = dir;
			onDirOutInput();
		}
	}

	function toggleClass(cls: string) {
		const set = new Set(settings.value.classesOut);
		if (set.has(cls)) set.delete(cls);
		else set.add(cls);
		settings.value.classesOut = [...set];
		settings.save();
	}

	function toggleAllClasses() {
		settings.value.classesOut =
			settings.value.classesOut.length === availableClasses.length ? [] : [...availableClasses];
		settings.save();
	}

	async function start() {
		startError = null;
		if (settings.value.classesOut.length === 0) {
			startError = 'Select at least one class to output.';
			return;
		}
		run.reset();
		hasStarted = true;
		hasAutoExpanded = false;
		expanded = new Set();
		try {
			await invoke('start_analysis', {
				settings: {
					modelname: settings.value.modelname,
					dir_audio: settings.value.dirAudio,
					dir_out: settings.value.dirOut,
					classes_out: settings.value.classesOut,
					chunklength: settings.value.chunklength,
					analyzers_cpu: settings.value.analyzersCpu,
					analyzers_gpu: settings.value.analyzersGpu,
					framehop_prop: settings.value.framehopProp,
					n_streamers: settings.value.nStreamers,
					stream_buffer_depth: settings.value.streamBufferDepth,
					verbosity_print: settings.value.verbosityPrint,
					verbosity_log: settings.value.verbosityLog,
					log_progress: settings.value.logProgress
				}
			});
		} catch (e) {
			startError = String(e);
			run.stop(startError);
		}
	}

	async function cancel() {
		await invoke('cancel_analysis');
		run.stop();
	}

	function allDirPaths(): string[] {
		const all: string[] = [];
		const walk = (d: TreeDir) => {
			all.push(d.path);
			d.dirs.forEach(walk);
		};
		tree.dirs.forEach(walk);
		return all;
	}

	function toggleExpandAll() {
		const all = allDirPaths();
		const allOpen = all.length > 0 && all.every((p) => expanded.has(p));
		expanded = allOpen ? new Set() : new Set(all);
	}

	const allExpanded = $derived.by(() => {
		const all = allDirPaths();
		return all.length > 0 && all.every((p) => expanded.has(p));
	});

	// Rounds down so a run only shows 100%/a checkmark once truly finished,
	// never early from rounding (e.g. 99.98% should read 99%, not 100%).
	function pct(done: number, total: number): number {
		if (total <= 0) return 0;
		return Math.min(100, Math.floor((done / total) * 100));
	}

	function resetDirOut() {
		if (!settings.value.modelname) return;
		settings.value.dirOutTouched = false;
		settings.value.dirOut = `models/${settings.value.modelname}/output`;
		checkManifest();
		settings.save();
	}

	const tree = $derived(run.tree);
	const missingDirs = $derived(!settings.value.dirAudio || !settings.value.dirOut);
</script>

<main class="app">
<div class="panels" style="grid-template-columns: {settingsWidth}px 6px 1fr">
	<section class="settings">
		<h2>Settings</h2>
		<fieldset class="settings-fields" disabled={run.running}>

		{#if modelMismatch}
			<p class="error">
				Results have already been written to this output folder with model "{manifest?.modelname}".
				Select that model to continue, or choose a different output folder.
			</p>
		{/if}

		<label>
			<span class="label-text">Model <span class="qmark" title="Select a model to use for analysis.">?</span></span>
			<select
				bind:value={settings.value.modelname}
				onchange={() => {
					if (manifest) {
						// Changing model away from a results folder's existing
						// manifest can't be reconciled in place, so fall back to
						// the new model's default output dir instead.
						settings.value.dirOutTouched = false;
					}
					onModelChange();
					settings.save();
				}}
			>
				{#each models as m}
					<option value={m}>{m}</option>
				{/each}
			</select>
		</label>
		<label>
			<span class="label-text">Audio directory <span class="qmark" title="Input folder containing audio files to analyze.">?</span></span>
			<span class="path-row">
				<input
					bind:value={settings.value.dirAudio}
					oninput={() => settings.save()}
					placeholder="/path/to/audio_in"
				/>
				<button type="button" onclick={browseDirAudio}>Browse…</button>
			</span>
		</label>
		<label>
			<span class="label-text">Output directory <span class="qmark" title="Output folder for analysis results.">?</span></span>
			<span class="path-row">
				<input bind:value={settings.value.dirOut} oninput={onDirOutInput} />
				<button type="button" onclick={browseDirOut}>Browse…</button>
				<button
					type="button"
					class="icon-btn"
					title="Reset to the model's default output folder"
					onclick={resetDirOut}
					aria-label="Reset output directory"
				>
					<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
						<path d="M3 11.5 12 4l9 7.5" stroke-linecap="round" stroke-linejoin="round" />
						<path d="M5.5 10v9a1 1 0 0 0 1 1H9a1 1 0 0 0 1-1v-4a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1v4a1 1 0 0 0 1 1h2.5a1 1 0 0 0 1-1v-9" stroke-linecap="round" stroke-linejoin="round" />
					</svg>
				</button>
			</span>
			{#if manifest && !modelMismatch}
				<span class="found-hint">Existing results found</span>
			{/if}
		</label>

		<details class="advanced">
			<summary>Advanced settings</summary>

			<fieldset>
				<legend class:locked={manifestLocked}>
					Classes to output
					{#if manifestLocked}
						<span class="lock-icon" title="Locked to match existing results in this output folder">
							<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2">
								<rect x="5" y="11" width="14" height="9" rx="1.5" />
								<path d="M8 11V7a4 4 0 0 1 8 0v4" stroke-linecap="round" />
							</svg>
						</span>
					{/if}
				</legend>
				<button
					type="button"
					class="toggle-all"
					disabled={manifestLocked}
					onclick={toggleAllClasses}>Select All/None</button
				>
				<div class="classes">
					{#each availableClasses as cls}
						<label class="checkbox" class:locked={manifestLocked}>
							<input
								type="checkbox"
								disabled={manifestLocked}
								checked={settings.value.classesOut.includes(cls)}
								onchange={() => toggleClass(cls)}
							/>
							{cls}
						</label>
					{/each}
				</div>
			</fieldset>

			<label>
				<span class="label-text" class:locked={manifestLocked}>
					Framehop
					{#if manifestLocked}
						<span class="lock-icon" title="Locked to match existing results in this output folder">
							<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2">
								<rect x="5" y="11" width="14" height="9" rx="1.5" />
								<path d="M8 11V7a4 4 0 0 1 8 0v4" stroke-linecap="round" />
							</svg>
						</span>
					{/if}
					<span
						class="qmark"
						title="The spacing between frames, expressed as a proportion of the frame length.
E.g., a framehop of 1 produces contiguous frames, 0.50 produces frames with 50% overlap."
					>
						?
					</span>
				</span>
				<input
					type="number"
					step="0.05"
					min="0"
					disabled={manifestLocked}
					bind:value={settings.value.framehopProp}
					oninput={() => settings.save()}
				/>
			</label>

			<label>
				<span class="label-text">Chunk length (s) <span class="qmark" title="The length of each chunk in seconds.">?</span></span>
				<input type="number" min="1" bind:value={settings.value.chunklength} oninput={() => settings.save()} />
			</label>
			<label>
				<span class="label-text">
					CPU analyzers
					<span
						class="qmark"
						title="The number of CPU-based workers to launch.
Usually, 1 worker will efficiently use your system's resources, but try adding more."
					>
						?
					</span>
				</span>
				<input type="number" min="0" bind:value={settings.value.analyzersCpu} oninput={() => settings.save()} />
			</label>
			<label>
				<span class="label-text">
					GPU analyzers
					<span
						class="qmark"
						title="The number of GPU-based workers to launch.
If you're using GPU, you probably don't want any CPU analyzers."
					>
						?
					</span>
				</span>
				<input type="number" min="0" bind:value={settings.value.analyzersGpu} oninput={() => settings.save()} />
			</label>
			<label>
				<span class="label-text">
					Concurrent streamers
					<span
						class="qmark"
						title="How many parallel audio streamers should be launched?
If you run into buffer bottlenecks, try increasing this number.
Leave blank for automatic assignment."
					>
						?
					</span>
				</span>
				<input
					type="number"
					min="1"
					value={settings.value.nStreamers ?? ''}
					oninput={(e) => {
						const v = (e.target as HTMLInputElement).value;
						settings.value.nStreamers = v === '' ? null : Number(v);
						settings.save();
					}}
					placeholder="auto"
				/>
			</label>
			<label>
				<span class="label-text">
					Stream buffer depth
					<span
						class="qmark"
						title="How many audio chunks should be buffered in memory?
Leave blank for automatic assignment."
					>
						?
					</span>
				</span>
				<input
					type="number"
					min="1"
					value={settings.value.streamBufferDepth ?? ''}
					oninput={(e) => {
						const v = (e.target as HTMLInputElement).value;
						settings.value.streamBufferDepth = v === '' ? null : Number(v);
						settings.save();
					}}
					placeholder="auto"
				/>
			</label>
			<label>
				<span class="label-text">Console verbosity <span class="qmark" title="How verbose should the console output be?">?</span></span>
				<select bind:value={settings.value.verbosityPrint} onchange={() => settings.save()}>
					{#each LOGLEVELS as lvl}
						<option value={lvl}>{lvl}</option>
					{/each}
				</select>
			</label>
			<label>
				<span class="label-text">Log file verbosity <span class="qmark" title="How verbose should the log file output be?">?</span></span>
				<select bind:value={settings.value.verbosityLog} onchange={() => settings.save()}>
					{#each LOGLEVELS as lvl}
						<option value={lvl}>{lvl}</option>
					{/each}
				</select>
			</label>
			<label class="checkbox">
				<input type="checkbox" bind:checked={settings.value.logProgress} onchange={() => settings.save()} />
				Log progress statements to file
				<span
					class="qmark"
					title="Should progress statements (e.g., reports from analyzers) be written to the log file?
Can produce very large log files."
				>
					?
				</span>
			</label>
		</details>

		{#if settings.value.classesOut.length === 0}
			<p class="error">Select at least one class to output.</p>
		{/if}
		{#if startError}
			<p class="error">{startError}</p>
		{/if}
		</fieldset>

		{#if !run.running}
			<div class="settings-actions">
				<button
					onclick={start}
					disabled={!settings.value.dirAudio ||
						!settings.value.modelname ||
						settings.value.classesOut.length === 0 ||
						modelMismatch}>Launch Analysis</button
				>
			</div>
		{/if}
	</section>

	<div
		class="resize-handle"
		role="separator"
		aria-orientation="vertical"
		onpointerdown={startResize}
		onpointermove={onResize}
		onpointerup={stopResize}
	></div>

	<section class="run">
		<div class="header">
			<h2>{run.running ? 'Analyzing…' : run.stopped ? 'Stopped' : 'Ready'}</h2>
			{#if run.rate > 0}
				<span class="rate">{run.rate.toFixed(1)}x realtime</span>
			{/if}
		</div>
		{#if !run.running && !hasStarted && missingDirs}
			<p class="hint">Set audio and output directories to begin.</p>
		{/if}
		{#if run.error}
			<p class="error">{run.error}</p>
		{/if}
		<ProgressBar weights={tree} provisional={!run.denominatorFinal} stopped={run.stopped} large />

		<div class="tree-toolbar">
			<button
				type="button"
				class="icon-btn"
				title={allExpanded ? 'Collapse All' : 'Expand All'}
				aria-label={allExpanded ? 'Collapse All' : 'Expand All'}
				onclick={toggleExpandAll}
			>
				{#if allExpanded}
					<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
						<path d="M6 10 12 5l6 5" stroke-linecap="round" stroke-linejoin="round" />
						<path d="M6 17 12 12l6 5" stroke-linecap="round" stroke-linejoin="round" />
					</svg>
				{:else}
					<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
						<path d="M6 7 12 12l6-5" stroke-linecap="round" stroke-linejoin="round" />
						<path d="M6 14 12 19l6-5" stroke-linecap="round" stroke-linejoin="round" />
					</svg>
				{/if}
			</button>
		</div>

		<div class="tree">
			{#each tree.files as f (f.path)}
				{@const w = f.weights}
				{@const filePct = pct(w.priorSeconds + w.doneSeconds + w.activeSeconds, w.totalSeconds)}
				<div class="tree-row">
					<div class="row static">
						<span class="disclosure"></span>
						<span class="name">{f.name}</span>
						<ProgressBar weights={w} stopped={run.stopped} />
						{#if f.status === 'done' || f.status === 'skipped'}
							<span class="count check" class:session={f.status === 'done'}>✓</span>
						{:else}
							<span class="count">{filePct}%</span>
						{/if}
					</div>
				</div>
			{/each}
			{#each tree.dirs as d (d.path)}
				<DirRow node={d} depth={0} {expanded} {pct} />
			{/each}
		</div>

		<details class="log">
			<summary>Log ({run.logLines.length})</summary>
			<pre>{run.logLines.join('\n')}</pre>
		</details>

		{#if run.running}
			<div class="run-actions">
				<button class="danger" onclick={cancel}>Stop Analysis</button>
			</div>
		{/if}
	</section>
</div>
</main>

<style>
	:root {
		font-family: Inter, Avenir, Helvetica, Arial, sans-serif;
		color-scheme: light dark;
	}

	:global(html),
	:global(body) {
		margin: 0;
		overflow-x: hidden;
	}

	.app {
		display: flex;
		flex-direction: column;
		gap: 1rem;
		padding: 1.5rem;
		height: 100vh;
		box-sizing: border-box;
		overflow: hidden;
	}

	.panels {
		display: grid;
		gap: 0;
		flex: 1;
		min-height: 0;
	}

	.settings {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
		overflow-y: auto;
		overflow-x: hidden;
		padding-right: 1rem;
		min-width: 0;
	}

	.settings-fields {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
		border: none;
		padding: 0;
		margin: 0;
		min-width: 0;
	}

	.settings-fields:disabled {
		opacity: 0.55;
	}

	.settings-actions {
		display: flex;
		justify-content: flex-end;
		flex-shrink: 0;
		margin-top: 0.5rem;
	}

	.run-actions {
		display: flex;
		justify-content: flex-end;
		flex-shrink: 0;
	}

	.tree-toolbar {
		display: flex;
		justify-content: flex-end;
		gap: 0.5rem;
		flex-shrink: 0;
	}

	.icon-btn {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		padding: 0.4rem;
		line-height: 0;
	}

	.qmark {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 1.1em;
		height: 1.1em;
		border-radius: 50%;
		border: 1px solid rgba(127, 127, 127, 0.6);
		font-size: 0.7em;
		line-height: 1;
		opacity: 0.75;
		cursor: help;
		vertical-align: middle;
	}

	.lock-icon {
		display: inline-flex;
		align-items: center;
		opacity: 0.7;
		vertical-align: middle;
	}

	.label-text {
		display: inline-flex;
		align-items: center;
		gap: 0.3rem;
	}

	.label-text.locked,
	legend.locked {
		color: rgba(127, 127, 127, 0.9);
	}

	label.checkbox.locked {
		color: rgba(127, 127, 127, 0.9);
	}

	.found-hint {
		font-size: 0.75rem;
		opacity: 0.6;
	}

	.resize-handle {
		cursor: col-resize;
		touch-action: none;
	}

	.resize-handle::after {
		content: '';
		display: block;
		width: 2px;
		height: 100%;
		margin: 0 auto;
		background: rgba(127, 127, 127, 0.3);
	}

	.path-row {
		display: flex;
		gap: 0.4rem;
		min-width: 0;
	}

	.path-row input {
		flex: 1;
		min-width: 0;
	}

	.toggle-all {
		margin: 0.5rem 0.75rem 0;
	}

	.settings label {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
		font-size: 0.85rem;
		opacity: 0.85;
	}

	.settings label.checkbox {
		flex-direction: row;
		align-items: center;
		gap: 0.4rem;
	}

	.advanced {
		border: 1px solid rgba(127, 127, 127, 0.3);
		border-radius: 6px;
		padding: 0.5rem 0.75rem;
	}

	.advanced summary {
		cursor: pointer;
		font-weight: 600;
	}

	.advanced label {
		margin-top: 0.6rem;
	}

	fieldset {
		border: 1px solid rgba(127, 127, 127, 0.3);
		border-radius: 6px;
		margin-top: 0.6rem;
	}

	legend {
		display: flex;
		align-items: center;
		gap: 0.35rem;
	}

	.classes {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 0.2rem 0.5rem;
		max-height: 160px;
		overflow-y: auto;
	}

	input,
	button,
	select {
		font: inherit;
		padding: 0.4rem 0.6rem;
		border-radius: 6px;
		border: 1px solid rgba(127, 127, 127, 0.4);
		box-sizing: border-box;
	}

	input[type='number'] {
		width: 100%;
	}

	button {
		cursor: pointer;
	}

	button.danger {
		border-color: #d33;
		color: #d33;
	}

	.error {
		color: #d33;
	}

	.hint {
		opacity: 0.6;
		font-size: 0.85rem;
	}

	.run {
		display: flex;
		flex-direction: column;
		gap: 1rem;
		min-width: 0;
		overflow: hidden;
	}

	.header {
		display: flex;
		align-items: baseline;
		gap: 0.75rem;
	}

	.rate {
		opacity: 0.7;
		font-variant-numeric: tabular-nums;
	}

	.tree {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
		overflow-x: hidden;
		border: 1px solid rgba(127, 127, 127, 0.2);
		border-radius: 6px;
	}

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

	.disclosure {
		opacity: 0.6;
		text-align: center;
	}

	.row .name {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		min-width: 0;
	}

	.row .count.check {
		opacity: 1;
		color: #4caf50;
	}

	.row .count.check.session {
		color: #4c8dff;
	}

	.row .count {
		text-align: right;
		font-variant-numeric: tabular-nums;
		opacity: 0.7;
	}

	.log {
		font-size: 0.8rem;
		flex-shrink: 0;
	}

	.log pre {
		max-height: 200px;
		overflow: auto;
		background: rgba(127, 127, 127, 0.1);
		padding: 0.5rem;
		border-radius: 6px;
		white-space: pre-wrap;
		word-break: break-all;
	}
</style>
