<script lang="ts">
	import { invoke } from '@tauri-apps/api/core';
	import { listen } from '@tauri-apps/api/event';
	import { open } from '@tauri-apps/plugin-dialog';
	import { onMount } from 'svelte';
	import { run } from '$lib/progress.svelte';
	import { settings, LOGLEVELS } from '$lib/settings.svelte';
	import DirRow from '$lib/DirRow.svelte';

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

	const manifestLocked = $derived(manifest !== null);

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

	// buzzdetect locks schema-defining settings (model, output classes,
	// framehop) to match an output folder's existing manifest, so a resumed
	// run can't silently write incompatible results into it — see
	// buzzdetect_gui.py's _apply_manifest_lock, which this mirrors.
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
		if (!manifest) return;
		if (models.includes(manifest.modelname) && settings.value.modelname !== manifest.modelname) {
			settings.value.modelname = manifest.modelname;
			await onModelChange();
		}
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

	function newAnalysis() {
		hasStarted = false;
		run.clear();
	}

	async function cancel() {
		await invoke('cancel_analysis');
		run.stop('cancelled');
	}

	function pct(done: number, total: number): number {
		if (total <= 0) return 0;
		return Math.min(100, Math.round((done / total) * 100));
	}

	const tree = $derived(run.tree);
	const missingDirs = $derived(!settings.value.dirAudio || !settings.value.dirOut);
</script>

<main class="app" style="grid-template-columns: {settingsWidth}px 6px 1fr">
	<section class="settings">
		<h2>Settings</h2>

		{#if manifestLocked}
			<p class="warning">
				Results have already been written to this output folder. Model, output classes, and framehop
				are locked to match existing results. Choose a different output folder to use different
				settings.
			</p>
		{/if}

		<label title="Select a model to use for analysis.">
			Model
			<select
				bind:value={settings.value.modelname}
				disabled={manifestLocked}
				onchange={() => {
					onModelChange();
					settings.save();
				}}
			>
				{#each models as m}
					<option value={m}>{m}</option>
				{/each}
			</select>
		</label>
		<label title="Input folder containing audio files to analyze.">
			Audio directory
			<span class="path-row">
				<input
					bind:value={settings.value.dirAudio}
					oninput={() => settings.save()}
					placeholder="/path/to/audio_in"
				/>
				<button type="button" onclick={browseDirAudio}>Browse…</button>
			</span>
		</label>
		<label title="Output folder for analysis results.">
			Output directory
			<span class="path-row">
				<input bind:value={settings.value.dirOut} oninput={onDirOutInput} />
				<button type="button" onclick={browseDirOut}>Browse…</button>
			</span>
		</label>

		<details class="advanced">
			<summary>Advanced settings</summary>

			<fieldset>
				<legend>Classes to output</legend>
				<button
					type="button"
					class="toggle-all"
					disabled={manifestLocked}
					onclick={toggleAllClasses}>Select All/None</button
				>
				<div class="classes">
					{#each availableClasses as cls}
						<label class="checkbox">
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

			<label title="The length of each chunk in seconds.">
				Chunk length (s)
				<input type="number" min="1" bind:value={settings.value.chunklength} oninput={() => settings.save()} />
			</label>
			<label
				title="The number of CPU-based workers to launch.
Usually, 1 worker will efficiently use your system's resources, but try adding more."
			>
				CPU analyzers
				<input type="number" min="0" bind:value={settings.value.analyzersCpu} oninput={() => settings.save()} />
			</label>
			<label
				title="The number of GPU-based workers to launch.
If you're using GPU, you probably don't want any CPU analyzers."
			>
				GPU analyzers
				<input type="number" min="0" bind:value={settings.value.analyzersGpu} oninput={() => settings.save()} />
			</label>
			<label
				title="The spacing between frames, expressed as a proportion of the frame length.
E.g., a framehop of 1 produces contiguous frames, 0.50 produces frames with 50% overlap."
			>
				Framehop
				<input
					type="number"
					step="0.05"
					min="0"
					disabled={manifestLocked}
					bind:value={settings.value.framehopProp}
					oninput={() => settings.save()}
				/>
			</label>
			<label
				title="How many parallel audio streamers should be launched?
If you run into buffer bottlenecks, try increasing this number.
Leave blank for automatic assignment."
			>
				Concurrent streamers
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
			<label
				title="How many audio chunks should be buffered in memory?
Leave blank for automatic assignment."
			>
				Stream buffer depth
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
			<label title="How verbose should the console output be?">
				Console verbosity
				<select bind:value={settings.value.verbosityPrint} onchange={() => settings.save()}>
					{#each LOGLEVELS as lvl}
						<option value={lvl}>{lvl}</option>
					{/each}
				</select>
			</label>
			<label title="How verbose should the log file output be?">
				Log file verbosity
				<select bind:value={settings.value.verbosityLog} onchange={() => settings.save()}>
					{#each LOGLEVELS as lvl}
						<option value={lvl}>{lvl}</option>
					{/each}
				</select>
			</label>
			<label
				class="checkbox"
				title="Should progress statements (e.g., reports from analyzers) be written to the log file?
Can produce very large log files."
			>
				<input type="checkbox" bind:checked={settings.value.logProgress} onchange={() => settings.save()} />
				Log progress statements to file
			</label>
		</details>

		<div class="actions">
			{#if !run.running}
				{#if hasStarted}
					<button onclick={start}>Restart</button>
					<button onclick={newAnalysis}>New Analysis</button>
				{:else}
					<button
						onclick={start}
						disabled={!settings.value.dirAudio ||
							!settings.value.modelname ||
							settings.value.classesOut.length === 0}>Start Analysis</button
					>
				{/if}
			{/if}
		</div>
		{#if settings.value.classesOut.length === 0}
			<p class="error">Select at least one class to output.</p>
		{/if}
		{#if startError}
			<p class="error">{startError}</p>
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
			<h2>{run.running ? 'Analyzing…' : run.error ? 'Stopped' : 'Ready'}</h2>
			{#if run.rate > 0}
				<span class="rate">{run.rate.toFixed(1)}x realtime</span>
			{/if}
			{#if run.running}
				<button class="danger" onclick={cancel}>Cancel</button>
			{/if}
		</div>
		{#if !run.running && !hasStarted && missingDirs}
			<p class="hint">Set audio and output directories to begin.</p>
		{/if}
		{#if run.error}
			<p class="error">{run.error}</p>
		{/if}
		<div class="overall-bar" class:provisional={!run.denominatorFinal}>
			<div class="fill" style="width: {pct(tree.doneSeconds, tree.workSeconds)}%"></div>
		</div>

		<div class="tree">
			{#each tree.files as f (f.path)}
				{@const filePct = f.status === 'skipped' ? 100 : pct(f.doneSeconds, f.workSeconds || f.duration || 1)}
				<div class="tree-row">
					<div class="row static" class:done={f.status === 'done' || f.status === 'skipped'}>
						<span class="disclosure"></span>
						<span class="name">{f.name}</span>
						<span class="bar"><span class="fill" style="width: {filePct}%"></span></span>
						<span class="count">{f.status === 'skipped' ? 'skipped' : `${filePct}%`}</span>
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
	</section>
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
		display: grid;
		gap: 0;
		padding: 1.5rem;
		height: 100vh;
		box-sizing: border-box;
		overflow: hidden;
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

	.warning {
		background: rgba(217, 119, 6, 0.15);
		border: 1px solid rgba(217, 119, 6, 0.4);
		border-radius: 6px;
		padding: 0.5rem 0.75rem;
		font-size: 0.8rem;
		margin: 0;
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

	.overall-bar {
		position: relative;
		height: 1rem;
		border-radius: 6px;
		background: rgba(127, 127, 127, 0.2);
		overflow: hidden;
		flex-shrink: 0;
	}

	.overall-bar .fill {
		position: absolute;
		inset: 0;
		width: 0;
		background: #4c8dff;
		transition: width 0.2s ease;
	}

	.overall-bar.provisional .fill {
		background: repeating-linear-gradient(45deg, #c99b3f, #c99b3f 6px, #dcb35f 6px, #dcb35f 12px);
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

	.row .bar {
		position: relative;
		height: 0.5rem;
		border-radius: 4px;
		background: rgba(127, 127, 127, 0.25);
		overflow: hidden;
	}

	.row .bar .fill {
		position: absolute;
		inset: 0;
		width: 0;
		background: #4c8dff;
	}

	.row.done {
		background: rgba(76, 141, 255, 0.16);
		border-radius: 6px;
	}

	.row.done .bar .fill {
		background: #4caf50;
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
