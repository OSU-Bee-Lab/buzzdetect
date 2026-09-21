<script lang="ts">
	import { invoke } from '@tauri-apps/api/core';
	import { listen } from '@tauri-apps/api/event';
	import { open } from '@tauri-apps/plugin-dialog';
	import { documentDir, join } from '@tauri-apps/api/path';
	import { onMount } from 'svelte';
	import { SvelteSet } from 'svelte/reactivity';
	import { run, formatDuration, type TreeDir } from '$lib/progress.svelte';
	import { settings, LOGLEVELS } from '$lib/settings.svelte';
	import DirRow from '$lib/DirRow.svelte';
	import FileRows from '$lib/FileRows.svelte';
	import ProgressBar from '$lib/ProgressBar.svelte';
	import PathField from '$lib/PathField.svelte';
	import type { HistorySettings } from '$lib/history';
	import type { ModelInfo } from '$lib/modelInfo';

	interface Manifest {
		modelname: string;
		classes_out: string[] | null;
	}

	let models = $state<ModelInfo[]>([]);
	let modelActionError = $state<string | null>(null);
	let availableClasses = $state<string[]>([]);
	let startError = $state<string | null>(null);
	// Wide enough that the two-column class checklist doesn't clip the longest
	// shipped class name ("ambient_background"/"mech_hum_chainsaw") — sized to
	// that text, not to the unbounded audio/output path fields below it.
	let settingsWidth = $state(380);
	let resizing = false;
	// A SvelteSet, not $state(new Set()): $state doesn't proxy a Set, so the
	// per-folder toggles' add/delete would change nothing on screen.
	const expanded = new SvelteSet<string>();
	let hasAutoExpanded = false;
	let hasStarted = $state(false);
	let manifest = $state<Manifest | null>(null);
	// Assumed true until checked, so a valid folder doesn't flash the warning.
	let dirOutExists = $state(true);
	// The settings the last launch was given, to tell whether a stopped run can
	// be restarted as it was.
	let startedKey = $state<string | null>(null);
	// What the engine reports about GPU support. Null until the probe answers,
	// which is a real wait -- it spawns the engine and asks onnxruntime to build
	// a session -- so the controls show a checking state rather than flickering
	// from absent to present.
	type GpuStatus = {
		supported: boolean;
		usable: boolean;
		providers: string[];
		detail: string | null;
	};
	let gpu = $state<GpuStatus | null>(null);

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

	interface RunSnapshot {
		running: boolean;
		started_at_ms: number;
		events: any[];
		logs: { line: string; seq: number }[];
	}

	type Output = Parameters<typeof run.handleOutput>[0][number];
	const FLUSH_MS = 100;
	let pending: Output[] = [];
	let flushTimer: ReturnType<typeof setTimeout> | null = null;

	function queueOutput(item: Output) {
		pending.push(item);
		flushTimer ??= setTimeout(flushOutput, FLUSH_MS);
	}

	function flushOutput() {
		if (flushTimer !== null) clearTimeout(flushTimer);
		flushTimer = null;
		const items = pending;
		pending = [];
		run.handleOutput(items);
	}

	onMount(() => {
		// Live events are held back until the attach below has had its say, so
		// they apply on top of the snapshot rather than before it.
		let held: (() => void)[] | null = [];
		const deliver = (f: () => void) => (held ? held.push(f) : f());

		// The engine announces every audio file as its own event, and a large
		// tree means thousands of them in the first seconds. Applying each as it
		// arrives rebuilt the whole tree per event and starved the UI thread --
		// including the stop button. They are queued and applied together.
		const unlistenProgress = listen<any>('engine-progress', (e) =>
			deliver(() => queueOutput({ kind: 'event', payload: e.payload }))
		);
		const unlistenLog = listen<{ line: string; stderr: boolean; seq: number }>('engine-log', (e) =>
			deliver(() => queueOutput({ kind: 'log', line: e.payload.line, seq: e.payload.seq }))
		);
		const unlistenUse = listen<HistorySettings>('history-use', (e) => useHistorySettings(e.payload));
		const unlistenExit = listen<{ code: number | null }>('engine-exit', (e) =>
			deliver(() => {
				flushOutput();
				// The run may have created the output folder.
				checkManifest();
				if (!run.running) return;
				// A cancelled engine is killed, so it exits by signal (null code) or
				// non-zero -- expected, not an error worth showing.
				const cancelled = run.stopping;
				run.stop(!cancelled && e.payload.code !== 0 ? `engine exited with code ${e.payload.code}` : undefined);
				const sum = run.summary;
				if (sum && sum.audioSeconds > 0) {
					invoke('record_run_result', {
						audioSeconds: sum.audioSeconds,
						runtimeSeconds: sum.runtimeSeconds
					}).catch(() => {});
				}
			})
		);

		// An engine can outlive the page that started it (the webview reloads,
		// the run doesn't), so pick up whatever is already running.
		Promise.all([unlistenProgress, unlistenLog, unlistenExit])
			.then(() => invoke<RunSnapshot>('attach_analysis'))
			.then((snap) => {
				if (!snap.running) return;
				run.reset(snap.started_at_ms);
				hasStarted = true;
				hasAutoExpanded = false;
				expanded.clear();
				for (const l of snap.logs) run.handleLog(l.line, l.seq);
				for (const ev of snap.events) run.handleEvent(ev);
			})
			.catch(() => {})
			.finally(() => {
				const queued = held ?? [];
				held = null;
				queued.forEach((f) => f());
			});

		invoke<GpuStatus>('gpu_status')
			.then((status) => {
				gpu = status;
				if (!status.usable && settings.value.analyzersGpu !== 0) {
					settings.value.analyzersGpu = 0;
					settings.save();
				}
				// fp16 only affects Apple's Neural Engine (see the setting's
				// tooltip), so default it on for a CoreML-capable machine unless
				// the user has already made a choice of their own.
				if (
					status.usable &&
					!settings.value.gpuFp16Touched &&
					status.providers.includes('CoreMLExecutionProvider')
				) {
					settings.value.gpuFp16 = true;
					settings.save();
				}
			})
			.catch((e) => {
				gpu = {
					supported: true,
					usable: false,
					providers: [],
					detail: `Couldn't check this machine for a GPU: ${e}`
				};
			});

		invoke<ModelInfo[]>('list_models').then((list) => {
			models = list;
			if (!settings.value.modelname || !list.some((m) => m.name === settings.value.modelname)) {
				settings.value.modelname = list[0]?.name ?? '';
			}
			onModelChange();
		});
		checkManifest();

		return () => {
			unlistenProgress.then((f) => f());
			unlistenLog.then((f) => f());
			unlistenExit.then((f) => f());
			unlistenUse.then((f) => f());
			if (flushTimer !== null) clearTimeout(flushTimer);
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

	// The log pane follows new lines only while the user is already at the
	// bottom; scrolling up to read something has to pin the view there, or the
	// next report yanks it away. `logStick` is deliberately not $state — it's
	// read inside the effect below, and making it reactive would re-run the
	// effect (and re-scroll) every time a scroll event flipped it.
	let logPre = $state<HTMLPreElement | null>(null);
	let logDetails = $state<HTMLDetailsElement | null>(null);
	let logStick = true;
	// Slack rather than an exact match: momentum scrolling and sub-pixel
	// layout leave the bottom a pixel or two short of exact.
	const LOG_STICK_SLOP = 24;

	function onLogScroll() {
		if (!logPre) return;
		logStick = logPre.scrollHeight - logPre.scrollTop - logPre.clientHeight <= LOG_STICK_SLOP;
	}

	function scrollLogToBottom() {
		if (!logPre) return;
		logStick = true;
		logPre.scrollTop = logPre.scrollHeight;
	}

	// Runs after the DOM has been updated with the new lines, so scrollHeight
	// already accounts for them.
	$effect(() => {
		run.logLines.length;
		if (!logPre || !logDetails?.open || !logStick) return;
		logPre.scrollTop = logPre.scrollHeight;
	});

	// buzzdetect locks schema-defining settings (output classes) to
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
		dirOutExists = await invoke<boolean>('dir_exists', { path: settings.value.dirOut }).catch(
			() => true
		);
		try {
			manifest = await invoke<Manifest | null>('read_manifest', { dirOut: settings.value.dirOut });
		} catch {
			manifest = null;
		}
		if (!manifest || manifest.modelname !== settings.value.modelname) return;
		if (manifest.classes_out) settings.value.classesOut = manifest.classes_out;
		settings.save();
	}

	// Where results go when the user hasn't chosen somewhere themselves. Has
	// to be an absolute path in a writable location: the engine resolves
	// relative paths against its own working directory, which in an installed
	// build is the read-only resource directory inside the app bundle.
	async function defaultDirOut(modelname: string): Promise<string> {
		return await join(await documentDir(), 'buzzdetect', modelname);
	}

	// Re-derive the class list whenever the model changes. dirOut is only
	// filled (with a per-model default) when empty; changing model never moves
	// it, but the folder's manifest is re-checked against the new model.
	async function onModelChange() {
		if (!settings.value.modelname) return;
		if (!settings.value.dirOut) {
			settings.value.dirOut = await defaultDirOut(settings.value.modelname);
		}
		await checkManifest();
		try {
			const classes = await invoke<string[]>('get_model_classes', {
				modelname: settings.value.modelname
			});
			availableClasses = classes;
			// Keep only still-valid selections from a prior model; if that
			// leaves nothing selected (fresh install, fresh model, or the prior
			// selection no longer applies), default to just ins_buzz rather than
			// leaving the run blocked on an empty selection — falling back to
			// everything only if this model has no ins_buzz class at all.
			const kept = settings.value.classesOut.filter((c) => classes.includes(c));
			settings.value.classesOut =
				kept.length > 0 ? kept : classes.includes('ins_buzz') ? ['ins_buzz'] : [...classes];
		} catch {
			availableClasses = [];
		}
		settings.save();
	}

	const currentModel = $derived(models.find((m) => m.name === settings.value.modelname));
	let currentModelRemovable = $derived(currentModel?.removable ?? false);

	function openModelInfo() {
		modelActionError = null;
		invoke('open_model_info', { modelname: settings.value.modelname }).catch(
			(e) => (modelActionError = String(e))
		);
	}

	async function reloadModels(select?: string) {
		models = await invoke<ModelInfo[]>('list_models');
		if (select && models.some((m) => m.name === select)) {
			settings.value.modelname = select;
		} else if (!models.some((m) => m.name === settings.value.modelname)) {
			settings.value.modelname = models[0]?.name ?? '';
		}
		await onModelChange();
	}

	// Import a model (a .zip of a folder holding model.onnx + config_model.json,
	// or the folder itself) into the per-user store, outside the app bundle, so
	// it survives updates and needs no admin rights. The engine picks it up by
	// name on the next run.
	async function importModel() {
		modelActionError = null;
		const picked = await open({
			title: 'Select a model .zip',
			filters: [{ name: 'Model bundle', extensions: ['zip'] }]
		});
		if (typeof picked !== 'string') return;
		try {
			const info = await invoke<ModelInfo>('import_model', { src: picked });
			await reloadModels(info.name);
		} catch (e) {
			modelActionError = String(e);
		}
	}

	async function removeCurrentModel() {
		modelActionError = null;
		const name = settings.value.modelname;
		if (!confirm(`Delete the imported model "${name}"? Its files will be deleted.`)) return;
		try {
			await invoke('remove_model', { name });
			await reloadModels();
		} catch (e) {
			modelActionError = String(e);
		}
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

	// Refill the settings from a past run (sent by the past-runs window).
	// dirOut counts as touched so the model change below doesn't swap it for
	// the per-model default.
	async function useHistorySettings(h: HistorySettings) {
		if (run.running || run.stopping) return;
		const v = settings.value;
		v.modelname = h.modelname;
		if (h.dir_audio) v.dirAudio = h.dir_audio;
		if (h.dir_out) {
			v.dirOut = h.dir_out;
			v.dirOutTouched = true;
		}
		if (h.classes_out) v.classesOut = h.classes_out;
		if (h.chunklength !== undefined) v.chunklength = h.chunklength;
		if (h.analyzers_cpu !== undefined) v.analyzersCpu = h.analyzers_cpu;
		// A run recorded on a machine with a GPU shouldn't ask for one that isn't here.
		if (h.analyzers_gpu !== undefined) v.analyzersGpu = gpu?.usable ? h.analyzers_gpu : 0;
		if (h.gpu_fp16 !== undefined) v.gpuFp16 = h.gpu_fp16;
		if (h.n_streamers !== undefined) v.nStreamers = h.n_streamers;
		if (h.stream_buffer_depth !== undefined) v.streamBufferDepth = h.stream_buffer_depth;
		if (h.verbosity_print) v.verbosityPrint = h.verbosity_print;
		if (h.verbosity_log) v.verbosityLog = h.verbosity_log;
		if (h.log_progress !== undefined) v.logProgress = h.log_progress;
		settings.save();
		await onModelChange();
	}

	function launchSettings() {
		return {
			modelname: settings.value.modelname,
			dir_audio: settings.value.dirAudio,
			dir_out: settings.value.dirOut,
			classes_out: settings.value.classesOut,
			chunklength: settings.value.chunklength,
			analyzers_cpu: settings.value.analyzersCpu,
			analyzers_gpu: settings.value.analyzersGpu,
			gpu_fp16: settings.value.gpuFp16,
			n_streamers: settings.value.nStreamers,
			stream_buffer_depth: settings.value.streamBufferDepth,
			verbosity_print: settings.value.verbosityPrint,
			verbosity_log: settings.value.verbosityLog,
			log_progress: settings.value.logProgress
		};
	}

	const launchKey = $derived(JSON.stringify(launchSettings()));
	// A run the user stopped can be picked up again, but only as it was:
	// change anything and it is a new launch.
	const canRestart = $derived(run.stopped && run.cancelled && startedKey === launchKey);

	function openHistory() {
		invoke('open_history').catch((e) => (startError = String(e)));
	}

	async function start() {
		startError = null;
		if (settings.value.classesOut.length === 0) {
			startError = 'Select at least one class to output.';
			return;
		}
		pending = [];
		run.reset();
		hasStarted = true;
		hasAutoExpanded = false;
		expanded.clear();
		startedKey = launchKey;
		try {
			await invoke('start_analysis', { settings: launchSettings() });
		} catch (e) {
			startError = String(e);
			run.stop(startError);
		}
	}

	async function cancel() {
		// A second click, while it's already winding down, kills it outright.
		if (run.stopping) {
			try {
				await invoke('kill_analysis');
			} catch (e) {
				run.stop(String(e));
			}
			return;
		}
		// Only marks the run as stopping. run.stop() is left to the engine-exit
		// listener, so the UI stays locked until the engine has actually gone
		// rather than while it's still analysing in the background.
		run.beginStop();
		try {
			await invoke('cancel_analysis');
		} catch (e) {
			run.stop(String(e));
		}
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
		expanded.clear();
		if (!allOpen) all.forEach((p) => expanded.add(p));
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

	async function resetDirOut() {
		if (!settings.value.modelname) return;
		settings.value.dirOutTouched = false;
		settings.value.dirOut = await defaultDirOut(settings.value.modelname);
		await checkManifest();
		settings.save();
	}

	const tree = $derived(run.tree);
	// Tauri's WKWebView doesn't render native `title` tooltips on hover, so
	// `[data-tooltip]` elements are shown via this single fixed-position
	// tooltip instead. Fixed positioning (rather than a CSS ::after anchored
	// to the element) is required so the tooltip can escape the settings
	// panel's `overflow-x: hidden`, which would otherwise clip it.
	let tooltipText = $state<string | null>(null);
	let tooltipX = $state(0);
	let tooltipY = $state(0);
	let tooltipMaxWidth = $state(352);

	function showTooltip(e: Event) {
		const target = (e.target as HTMLElement).closest<HTMLElement>('[data-tooltip]');
		if (!target) return;
		const text = target.getAttribute('data-tooltip');
		if (!text) return;
		const rect = target.getBoundingClientRect();
		tooltipText = text;
		// Anchored to the icon's left edge (not centered) so the tooltip never
		// needs to spill past the left edge of the window; `tooltipMaxWidth`
		// shrinks to whatever room remains so it can't overflow the right
		// edge either.
		tooltipX = Math.max(rect.left, 12);
		tooltipY = rect.top - 6;
		tooltipMaxWidth = Math.min(352, window.innerWidth - tooltipX - 12);
	}

	function hideTooltip(e: Event) {
		const related = (e as FocusEvent).relatedTarget as HTMLElement | null;
		if (related?.closest('[data-tooltip]')) return;
		tooltipText = null;
	}
</script>

<!-- svelte-ignore a11y_mouse_events_have_key_events -->
<main class="app" onmouseover={showTooltip} onmouseout={hideTooltip} onfocusin={showTooltip} onfocusout={hideTooltip}>
{#if tooltipText}
	<div
		class="tooltip-popup"
		style="left: {tooltipX}px; top: {tooltipY}px; max-width: {tooltipMaxWidth}px;"
	>{tooltipText}</div>
{/if}
<div class="panels" style="grid-template-columns: {settingsWidth}px 6px 1fr">
	<section class="settings">
		<h2>Settings</h2>
		<div class="settings-body">
			<fieldset class="settings-fields" disabled={run.running || run.stopping}>

		<!-- for= rather than nesting alone: the Info button is the label's first
		     labelable descendant, so without it a click on "Model" would press it. -->
		<label class:field-error={modelMismatch} for="model-select">
			<span class="label-text">Model <span class="qmark" data-tooltip="Select a model to use for analysis.">?</span>
				<button
					type="button"
					class="info-btn"
					disabled={!currentModel?.has_readme}
					data-tooltip={currentModel?.has_readme ? undefined : 'This model has no README.'}
					onclick={openModelInfo}>Info</button
				></span>
			<span class="model-row">
				<select
					id="model-select"
					bind:value={settings.value.modelname}
					onchange={() => {
						onModelChange();
						settings.save();
					}}
				>
					{#each models as m}
						<option value={m.name}>{m.name}</option>
					{/each}
				</select>
				{#if currentModelRemovable}
					<button type="button" class="delete-btn" onclick={removeCurrentModel}>Delete</button>
				{/if}
			</span>
			{#if currentModel?.description}
				<span class="model-description">{currentModel.description}</span>
			{/if}
			{#if modelActionError}
				<span class="error">{modelActionError}</span>
			{/if}
		</label>
		<label class:field-error={!settings.value.dirAudio}>
			<span class="label-text">Audio directory <span class="qmark" data-tooltip="Input folder containing audio files to analyze.">?</span></span>
			<span class="path-row">
				<PathField
					bind:value={settings.value.dirAudio}
					oninput={() => settings.save()}
					placeholder="/path/to/audio_in"
				/>
				<button type="button" onclick={browseDirAudio}>Browse…</button>
			</span>
		</label>
		<label class:field-error={modelMismatch || !settings.value.dirOut}>
			<span class="label-text">Output directory <span class="qmark" data-tooltip="Output folder for analysis results.">?</span></span>
			<span class="path-row">
				<PathField bind:value={settings.value.dirOut} oninput={onDirOutInput} />
				<button type="button" onclick={browseDirOut}>Browse…</button>
				<button
					type="button"
					class="icon-btn"
					data-tooltip="Reset to the model's default output folder"
					onclick={resetDirOut}
					aria-label="Reset output directory"
				>
					<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
						<path d="M3 11.5 12 4l9 7.5" stroke-linecap="round" stroke-linejoin="round" />
						<path d="M5.5 10v9a1 1 0 0 0 1 1H9a1 1 0 0 0 1-1v-4a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1v4a1 1 0 0 0 1 1h2.5a1 1 0 0 0 1-1v-9" stroke-linecap="round" stroke-linejoin="round" />
					</svg>
				</button>
			</span>
			{#if settings.value.dirOut && !dirOutExists}
				<span class="found-hint">Output directory does not exist yet. It will be created upon analysis.</span>
			{:else if manifest && !modelMismatch}
				<span class="found-hint">Previous settings found in output folder</span>
			{/if}
		</label>

		<fieldset class:field-error={settings.value.classesOut.length === 0}>
			<legend class:locked={manifestLocked}>
				<span class="legend-text">
					Classes out
					{#if manifestLocked}
						<span class="lock-icon" data-tooltip="Locked to match existing results in this output folder">
							<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2">
								<rect x="5" y="11" width="14" height="9" rx="1.5" />
								<path d="M8 11V7a4 4 0 0 1 8 0v4" stroke-linecap="round" />
							</svg>
						</span>
					{/if}
				</span>
				<button
					type="button"
					class="toggle-all"
					disabled={manifestLocked}
					onclick={toggleAllClasses}
					>{availableClasses.length > 0 && settings.value.classesOut.length === availableClasses.length
						? 'Select None'
						: 'Select All'}</button
				>
			</legend>
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

		<details class="advanced">
			<summary>Advanced settings</summary>

			<label>
				<span class="label-text">Chunk length (s) <span class="qmark" data-tooltip="The length of each chunk in seconds.">?</span></span>
				<input type="number" min="1" bind:value={settings.value.chunklength} oninput={() => settings.save()} />
			</label>
			<label>
				<span class="label-text">
					CPU analyzers
					<span
						class="qmark"
						data-tooltip="The number of CPU-based workers to launch.
Usually, 1 worker will efficiently use your system's resources, but try adding more."
					>
						?
					</span>
				</span>
				<input type="number" min="0" bind:value={settings.value.analyzersCpu} oninput={() => settings.save()} />
			</label>
			{#if gpu === null}
				<p class="hint checking">
					<span class="spinner" aria-hidden="true"></span>
					Checking this machine for a usable GPU&hellip;
				</p>
			{:else if gpu.supported}
			<label>
				<span class="label-text">
					GPU analyzers
					<span
						class="qmark"
						data-tooltip="The number of GPU-based workers to launch.
If you're using GPU, you probably don't want any CPU analyzers."
					>
						?
					</span>
				</span>
				<input
					type="number"
					min="0"
					disabled={!gpu.usable}
					bind:value={settings.value.analyzersGpu}
					oninput={() => settings.save()}
				/>
			</label>

			{#if !gpu.usable && gpu.detail}
				<p class="hint warn">{gpu.detail}</p>
			{/if}

			<label class="checkbox-setting">
				<input
					type="checkbox"
					disabled={!gpu.usable}
					bind:checked={settings.value.gpuFp16}
					onchange={() => {
						settings.value.gpuFp16Touched = true;
						settings.save();
					}}
				/>
				<span class="label-text">
					Reduced precision (fp16)
					<span
						class="qmark"
						data-tooltip="Runs the model at half precision on Apple's Neural Engine, which is about twice as fast but shifts results by roughly 0.015 against a full-precision run.
Results from a reduced-precision run are not directly comparable with full-precision ones near a detection threshold. Currently affects Apple GPUs only."
					>
						?
					</span>
				</span>
			</label>
			{/if}
			<label>
				<span class="label-text">
					Concurrent streamers
					<span
						class="qmark"
						data-tooltip="How many parallel audio streamers should be launched?
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
						data-tooltip="How many audio chunks should be buffered in memory?
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
				<span class="label-text">Console verbosity <span class="qmark" data-tooltip="How verbose should the console output be?">?</span></span>
				<select bind:value={settings.value.verbosityPrint} onchange={() => settings.save()}>
					{#each LOGLEVELS as lvl}
						<option value={lvl}>{lvl}</option>
					{/each}
				</select>
			</label>
			<label>
				<span class="label-text">Log file verbosity <span class="qmark" data-tooltip="How verbose should the log file output be?">?</span></span>
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
					data-tooltip="Should progress statements (e.g., reports from analyzers) be written to the log file?
Can produce very large log files."
				>
					?
				</span>
			</label>
			<label>
				<span class="label-text">Models</span>
				<span class="model-actions">
					<button type="button" onclick={importModel}>Import model (.zip)…</button>
				</span>
				{#if modelActionError}
					<span class="error">{modelActionError}</span>
				{/if}
			</label>
		</details>

		</fieldset>
		</div>

		<div class="settings-actions">
			<div class="action-row">
				<button type="button" class="history-btn" onclick={openHistory}>History</button>
				{#if run.running || run.stopping}
					<button class="danger" onclick={cancel}>
						{run.stopping ? 'Force Stop' : 'Stop Analysis'}
					</button>
				{:else}
					<button
						onclick={start}
						disabled={!settings.value.dirAudio ||
							!settings.value.modelname ||
							settings.value.classesOut.length === 0 ||
							modelMismatch}>{canRestart ? 'Restart Analysis' : 'Launch Analysis'}</button
					>
				{/if}
			</div>
			{#if !(run.running || run.stopping)}
				{#if modelMismatch}
					<p class="error">
						Results have already been written to this output folder with model "{manifest?.modelname}".
						Select that model to continue, or choose a different output folder.
					</p>
				{/if}
				{#if settings.value.classesOut.length === 0}
					<p class="error">Select at least one class to output.</p>
				{/if}
				{#if !settings.value.dirAudio || !settings.value.dirOut}
					<p class="hint">Set audio and output directories to begin.</p>
				{/if}
				{#if startError}
					<p class="error">{startError}</p>
				{/if}
			{/if}
		</div>
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
			<h2>
				{#if run.stopping}
					Stopping<span class="ellipsis" aria-hidden="true"></span>
				{:else if run.running}
					<!-- The startup stages can each sit for many seconds (see
					     progress.svelte.ts); the animated dots are what says the
					     app hasn't hung while a stage that can't report finer
					     progress runs. -->
					{run.stageLabel.replace(/…$/, '')}<span class="ellipsis" aria-hidden="true"
					></span>
				{:else if run.stopped}
					{run.error ? 'Error — see log' : run.cancelled ? 'Stopped' : 'Analysis complete!'}
				{:else}
					Ready
				{/if}
			</h2>
		</div>
		{#if hasStarted}
			{#if run.running}
				{@const s = run.stats}
				<!-- Stacked rows with a fixed-width label column: values change
				     length constantly, so nothing may share a line with them. -->
				<dl class="stats">
					<!-- Rate and ETA are kept in place with a placeholder until the
					     first samples land, so the rows don't jump once they do. -->
					<dt>Rate:</dt>
					<dd>{s.rate > 0 ? `${s.rate.toFixed(1)}x realtime` : '—'}</dd>
					<dt>Audio remaining:</dt>
					<dd>{formatDuration(s.remainingSeconds)}</dd>
					<dt>ETA:</dt>
					<dd>{s.etaSeconds === null ? '—' : formatDuration(s.etaSeconds)}</dd>
				</dl>
			{:else if run.summary}
				{@const sum = run.summary}
				{@const runtimeStr = sum.runtimeSeconds < 10 && sum.runtimeSeconds > 0
					? `${sum.runtimeSeconds.toFixed(1)}s`
					: formatDuration(sum.runtimeSeconds)}
				<p class="summary">
					Analyzed {formatDuration(sum.audioSeconds)} of audio in {runtimeStr} ({sum.rate.toFixed(1)}x)
				</p>
			{/if}
		{/if}
		{#if run.error}
			<p class="error">{run.error}</p>
		{/if}
		<ProgressBar weights={tree} provisional={!run.denominatorFinal} large />

		<div class="tree-toolbar">
			<button
				type="button"
				class="icon-btn"
				data-tooltip={allExpanded ? 'Collapse All' : 'Expand All'}
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
			<FileRows files={tree.files} depth={0} {pct} />
			{#each tree.dirs as d (d.path)}
				<DirRow node={d} depth={0} {expanded} {pct} />
			{/each}
		</div>

		<details class="log" bind:this={logDetails} ontoggle={scrollLogToBottom}>
			<summary>Log ({run.logLines.length})</summary>
			<pre bind:this={logPre} onscroll={onLogScroll}>{run.logLines.join('\n')}</pre>
		</details>
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
		height: 100%;
		min-height: 0;
		min-width: 0;
		padding-right: 1rem;
		box-sizing: border-box;
	}

	.settings h2 {
		margin: 0 0 0.5rem 0;
		flex-shrink: 0;
	}

	.settings-body {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
		overflow-x: hidden;
		padding-right: 0.25rem;
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
		flex-direction: column;
		gap: 0.4rem;
		flex-shrink: 0;
		padding-top: 0.75rem;
		margin-top: 0.5rem;
		border-top: 1px solid rgba(127, 127, 127, 0.2);
	}

	.action-row {
		display: flex;
		gap: 0.4rem;
		align-items: stretch;
	}

	.settings-actions button {
		flex: 1;
		font-weight: 600;
	}

	.settings-actions p {
		margin: 0;
		font-size: 0.85rem;
	}

	.field-error {
		outline: 1px solid #d33;
		outline-offset: 2px;
		border-radius: 6px;
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

	.tooltip-popup {
		position: fixed;
		transform: translateY(-100%);
		width: max-content;
		white-space: pre-line;
		background: #2a2a2a;
		color: #fff;
		font-size: 0.75rem;
		line-height: 1.35;
		padding: 0.4rem 0.6rem;
		border-radius: 4px;
		box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
		pointer-events: none;
		z-index: 1000;
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

	.toggle-all {
		flex-shrink: 0;
		padding: 0.15rem 0.5rem;
		font-size: 0.75rem;
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
		justify-content: space-between;
		gap: 0.5rem;
		width: 100%;
	}

	.legend-text {
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

	.info-btn {
		margin-left: auto;
		padding: 0.05rem 0.45rem;
		font-size: 0.75rem;
	}

	.info-btn:disabled {
		opacity: 0.45;
		cursor: default;
	}

	.model-description {
		display: block;
		margin-top: 0.3rem;
		font-size: 0.8rem;
		opacity: 0.7;
		line-height: 1.3;
	}

	.model-actions {
		display: flex;
		gap: 0.4rem;
		margin-top: 0.3rem;
	}

	.model-row {
		display: flex;
		gap: 0.4rem;
		align-items: center;
	}

	.model-row select {
		flex: 1;
		min-width: 0;
	}

	.delete-btn {
		flex-shrink: 0;
		padding: 0.15rem 0.5rem;
		font-size: 0.75rem;
		border-color: #d33;
		color: #d33;
	}

	.settings-actions .history-btn {
		flex: 0 0 auto;
		font-size: 0.8rem;
		font-weight: 400;
	}

	.model-actions button {
		padding: 0.25rem 0.5rem;
		font-size: 0.85rem;
	}

	.hint {
		opacity: 0.6;
		font-size: 0.85rem;
	}

	.hint.warn {
		opacity: 0.9;
		color: #c98a2b;
		margin: -0.25rem 0 0.25rem;
	}

	.hint.checking {
		display: flex;
		align-items: center;
		gap: 0.4rem;
	}

	.spinner {
		width: 0.8em;
		height: 0.8em;
		border: 2px solid currentColor;
		border-top-color: transparent;
		border-radius: 50%;
		animation: spin 0.7s linear infinite;
		flex: none;
	}

	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}

	/* A disabled control still has to read as a control, not as absent. */
	input:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.checkbox-setting {
		flex-direction: row;
		align-items: center;
		gap: 0.4rem;
	}

	.checkbox-setting input {
		width: auto;
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

	/* Reveals "..." one dot at a time by widening a clipped box, so it animates
	   without JS and without a timer in the store. steps(4, jump-none) walks
	   0 -> 1 across four frames: 0, 1, 2, 3 dots. */
	.ellipsis {
		display: inline-block;
		width: 1.5ch;
		overflow: hidden;
		vertical-align: bottom;
		white-space: pre;
		animation: ellipsis 1.6s steps(4, jump-none) infinite;
	}

	.ellipsis::after {
		content: '...';
	}

	@keyframes ellipsis {
		from {
			width: 0;
		}
		to {
			width: 1.5ch;
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.ellipsis {
			animation: none;
			width: 1.5ch;
		}
	}

	.stats {
		display: grid;
		/* Fixed label column: "Previously analyzed:" is the longest label, so
		   rows keep their positions whether or not it is shown. */
		grid-template-columns: 11em minmax(0, 1fr);
		gap: 0.15rem 0.5rem;
		margin: -0.5rem 0 0;
		font-size: 0.85rem;
		opacity: 0.7;
		font-variant-numeric: tabular-nums;
	}

	.stats dd {
		margin: 0;
	}

	.summary {
		margin: -0.5rem 0 0;
		font-size: 0.85rem;
		opacity: 0.85;
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
