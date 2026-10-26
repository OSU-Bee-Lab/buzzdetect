<script lang="ts">
	// The Models window: every installed model and every model in the catalog
	// (src-tauri/src/catalog.rs), with the selected one's README
	// on the right. Opened by the main window's View Models button through
	// open_models. Downloads, imports, removals and ignores all happen here;
	// each emits models-changed, which the main window listens for too.
	import { getVersion } from '@tauri-apps/api/app';
	import { invoke } from '@tauri-apps/api/core';
	import { emitTo, listen } from '@tauri-apps/api/event';
	import { getCurrentWindow } from '@tauri-apps/api/window';
	import { open } from '@tauri-apps/plugin-dialog';
	import { openUrl } from '@tauri-apps/plugin-opener';
	import { marked } from 'marked';
	import DOMPurify from 'dompurify';
	import { onMount } from 'svelte';
	import {
		BUZZDETECT,
		citeModel,
		classifyLink,
		describeVia,
		formatSize,
		groupModels,
		inheritedTerms,
		type ModelDetails,
		type ModelInfo,
		type ModelRow,
		type ModelsOverview
	} from '$lib/modelInfo';

	let overview = $state<ModelsOverview | null>(null);
	let selected = $state(new URLSearchParams(location.search).get('model') ?? '');
	let details = $state<ModelDetails | null>(null);
	let loadingDetails = $state(false);
	let error = $state<string | null>(null);
	let downloading = $state<Record<string, boolean>>({});
	let body: HTMLElement | undefined = $state();

	// The Attribution view: how to cite the selected model and what it's built
	// from, from config_model.json's `metadata` and `licenses` blocks.
	let view = $state<'readme' | 'attribution'>('readme');
	let appVersion = $state<string | null>(null);
	let sha = $state<{ name: string; value: string | null } | null>(null);
	let licenseTexts = $state<Record<string, string>>({});
	let openLicense = $state<string | null>(null);
	let copied = $state<string | null>(null);
	const citation = $derived(
		details && selected ? citeModel(selected, details, appVersion, sha?.value ?? null) : null
	);
	const components = $derived(details?.licenses?.components ?? []);
	const terms = $derived(inheritedTerms(details?.licenses ?? null));
	const provenance = $derived.by(() => {
		const md = details?.metadata ?? {};
		const own = details?.licenses?.model;
		const rows: [string, string | null | undefined, boolean?][] = [
			['Model', selected],
			['Trained as', md.modelname_internal],
			['Trained', md.trained_date],
			['Training set', md.set],
			['Embedder', md.embeddername],
			['Distilled from', md.teacher],
			['Ensemble of', md.ensemble_of?.join(', ')],
			['Training branch', md.branch],
			['Model license', own?.spdx],
			['model.onnx SHA-256', sha?.name === selected ? (sha.value ?? 'unavailable') : 'computing…', true],
			['buzzdetect version', appVersion]
		];
		return rows.filter(([, v]) => v);
	});

	// An installed model's hash is computed on request; a catalog one comes
	// with its details.
	$effect(() => {
		const name = selected;
		if (view !== 'attribution' || !details || sha?.name === name) return;
		if (details.sha256 || !row?.installed) {
			sha = { name, value: details.sha256 };
			return;
		}
		invoke<string>('model_sha256', { modelname: name })
			.then((value) => {
				if (selected === name) sha = { name, value };
			})
			.catch(() => {
				if (selected === name) sha = { name, value: null };
			});
	});

	async function toggleLicense(file: string) {
		if (openLicense === file) {
			openLicense = null;
			return;
		}
		const name = selected;
		if (!(file in licenseTexts)) {
			try {
				const text = await (row?.installed
					? invoke<string>('model_license_text', { modelname: name, file })
					: invoke<string>('catalog_license_text', { name, file }));
				if (selected !== name) return;
				licenseTexts = { ...licenseTexts, [file]: text };
			} catch (e) {
				error = String(e);
				return;
			}
		}
		openLicense = file;
	}

	async function copy(key: string, text: string) {
		try {
			await navigator.clipboard.writeText(text);
			copied = key;
			setTimeout(() => {
				if (copied === key) copied = null;
			}, 1500);
		} catch (e) {
			error = String(e);
		}
	}

	const groups = $derived(overview ? groupModels(overview.models) : []);
	const row = $derived(overview?.models.find((m) => m.name === selected) ?? null);
	// README content comes from model folders, including ones a collaborator
	// sent, and from the catalog; this webview can invoke app commands -- so
	// it's sanitized.
	const html = $derived(
		details?.readme ? DOMPurify.sanitize(marked.parse(details.readme, { async: false })) : ''
	);
	const descriptionHtml = $derived.by(() => {
		const d = details?.description ?? row?.description;
		return d ? DOMPurify.sanitize(marked.parseInline(d, { async: false })) : '';
	});

	let navWidth = $state(200);
	let resizing = false;
	function startResize(e: PointerEvent) {
		resizing = true;
		(e.target as HTMLElement).setPointerCapture(e.pointerId);
	}
	function onResize(e: PointerEvent) {
		if (!resizing) return;
		navWidth = Math.min(420, Math.max(140, e.clientX));
	}
	function stopResize() {
		resizing = false;
	}

	async function refresh(fromNetwork = false) {
		overview = await invoke<ModelsOverview>('models_overview', { refresh: fromNetwork });
		if (!overview.models.some((m) => m.name === selected)) {
			const first = overview.models.find((m) => m.notify) ?? overview.models[0];
			if (first) await load(first.name);
			else selected = '';
		}
	}

	// The details shown for a model: an installed one's own config and README,
	// with the catalog's README swapped in when there is one, since that's the
	// copy that gets corrected; a model not installed yet, entirely from the
	// catalog.
	let loadToken = 0;
	async function load(name: string) {
		const token = ++loadToken;
		if (name !== selected) {
			sha = null;
			licenseTexts = {};
			openLicense = null;
		}
		selected = name;
		error = null;
		history.replaceState(null, '', `?model=${encodeURIComponent(name)}`);
		const r = overview?.models.find((m) => m.name === name);
		loadingDetails = true;
		try {
			let d: ModelDetails | null = null;
			if (!r || r.installed) {
				d = await invoke<ModelDetails>('model_details', { modelname: name });
				if (token !== loadToken) return;
				details = d;
				loadingDetails = false;
			}
			if (r?.in_catalog) {
				const remote = await invoke<ModelDetails>('catalog_model_details', { name }).catch(
					(e) => {
						if (!d) throw e;
						return null;
					}
				);
				if (token !== loadToken) return;
				details = d ? { ...d, readme: remote?.readme ?? d.readme } : remote;
			}
		} catch (e) {
			if (token !== loadToken) return;
			details = null;
			error = String(e);
		} finally {
			if (token === loadToken) loadingDetails = false;
		}
		body?.scrollTo(0, 0);
	}

	async function act(f: () => Promise<unknown>) {
		error = null;
		try {
			await f();
		} catch (e) {
			error = String(e);
		}
	}

	async function download(name: string) {
		downloading = { ...downloading, [name]: true };
		await act(() => invoke('download_model', { name }));
		downloading = { ...downloading, [name]: false };
	}

	const setIgnored = (name: string, ignored: boolean) =>
		act(() => invoke('set_model_ignored', { name, ignored }));

	const setDisabled = (name: string, disabled: boolean) =>
		act(() => invoke('set_model_disabled', { name, disabled }));

	// The main window owns the settings; it picks the model up from this.
	async function useModel(name: string) {
		await act(async () => {
			await emitTo('main', 'models-use', name);
			await getCurrentWindow().close();
		});
	}

	function remove(name: string) {
		if (!confirm(`Delete the model "${name}"? Its files will be deleted.`)) return;
		act(() => invoke('remove_model', { name }));
	}

	// A .zip of a folder holding model.onnx + config_model.json, copied into
	// the per-user store, outside the app bundle, so it survives updates and
	// needs no admin rights.
	async function importModel() {
		error = null;
		const picked = await open({
			title: 'Select a model .zip',
			filters: [{ name: 'Model bundle', extensions: ['zip'] }]
		});
		if (typeof picked !== 'string') return;
		await act(async () => {
			const info = await invoke<ModelInfo>('import_model', { src: picked });
			await refresh();
			await load(info.name);
		});
	}

	function onClick(e: MouseEvent) {
		const a = (e.target as HTMLElement).closest('a');
		if (!a) return;
		const link = classifyLink(a.getAttribute('href') ?? '');
		if (link.kind === 'anchor') return;
		e.preventDefault();
		const opened =
			link.kind === 'external'
				? openUrl(link.url)
				: invoke('open_model_file', { modelname: selected, rel: link.rel });
		opened.catch((err) => (error = String(err)));
	}

	const status = (r: ModelRow) =>
		r.update
			? 'Update available'
			: !r.installed && !r.compatible
				? `Needs buzzdetect ${r.min_app_version} or newer`
				: r.bundled
					? r.disabled
						? 'Comes with buzzdetect; hidden from the model picker'
						: 'Comes with buzzdetect'
					: r.installed
						? r.in_catalog
							? 'Downloaded'
							: 'Imported'
						: r.download_size
							? formatSize(r.download_size)
							: '';

	onMount(() => {
		getVersion()
			.then((v) => (appVersion = v))
			.catch(() => {});
		refresh(true)
			.then(() => (selected ? load(selected) : undefined))
			.catch((e) => (error = String(e)));
		const unlistenSelect = listen<string>('models-select', (e) => load(e.payload));
		const unlistenChanged = listen('models-changed', async () => {
			await refresh();
			if (selected) await load(selected);
		});
		return () => {
			unlistenSelect.then((f) => f());
			unlistenChanged.then((f) => f());
		};
	});
</script>

<svelte:head><title>{selected ? `${selected} — Models` : 'Models'}</title></svelte:head>

<div class="window" style="grid-template-columns: {navWidth}px 6px 1fr">
	<nav>
		{#each groups as g}
			<h3>{g.title}</h3>
			{#each g.rows as m}
				<button
					type="button"
					class:active={m.name === selected}
					class:dim={m.ignored || m.disabled || (!m.installed && !m.compatible)}
					onclick={() => load(m.name)}
					><span class="name">{m.name}</span>{#if m.notify}<span
							class="badge"
							aria-label={m.update ? 'update available' : 'new'}
						></span>{/if}</button
				>
			{/each}
		{/each}
		<div class="nav-foot">
			<button type="button" class="import" onclick={importModel}>Import from .zip…</button>
			{#if overview?.catalog_error}
				<p class="hint" title={overview.catalog_error}>Couldn't reach the model catalog.</p>
			{/if}
		</div>
	</nav>
	<div
		class="resize-handle"
		role="separator"
		aria-orientation="vertical"
		onpointerdown={startResize}
		onpointermove={onResize}
		onpointerup={stopResize}
	></div>
	<main>
		<div class="content" bind:this={body}>
		{#if selected}
			<h1>{selected}</h1>
			{#if descriptionHtml}
				<!-- eslint-disable-next-line svelte/no-at-html-tags -- sanitized above -->
				<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_noninteractive_element_interactions -->
				<p class="description" onclick={onClick}>{@html descriptionHtml}</p>
			{/if}
		{/if}
		{#if error}
			<p class="error">{error}</p>
		{/if}

		{#if loadingDetails && !details}
			<p class="hint">Loading…</p>
		{/if}

		{#if view === 'attribution' && details && citation}
			<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_noninteractive_element_interactions -->
			<article class="readme attribution" onclick={onClick}>
				<h2>Citing this model</h2>
				<p>Cite this model as:</p>
				<div class="cite">
					<pre>{citation.text}</pre>
					<button type="button" onclick={() => copy('model', citation.text)}
						>{copied === 'model' ? 'Copied' : 'Copy'}</button
					>
				</div>
				<details>
					<summary>BibTeX</summary>
					<div class="cite">
						<pre>{citation.bibtex}</pre>
						<button type="button" onclick={() => copy('model-bib', citation.bibtex)}
							>{copied === 'model-bib' ? 'Copied' : 'Copy'}</button
						>
					</div>
				</details>
				<p>
					And cite the method, buzzdetect, as described in the <a
						href="https://doi.org/10.1093/jisesa/ieaf104">paper</a
					>:
				</p>
				<div class="cite">
					<pre>{BUZZDETECT.paper}</pre>
					<button type="button" onclick={() => copy('paper', BUZZDETECT.paper)}
						>{copied === 'paper' ? 'Copied' : 'Copy'}</button
					>
				</div>
				<details>
					<summary>BibTeX</summary>
					<div class="cite">
						<pre>{BUZZDETECT.paperBibtex}</pre>
						<button type="button" onclick={() => copy('paper-bib', BUZZDETECT.paperBibtex)}
							>{copied === 'paper-bib' ? 'Copied' : 'Copy'}</button
						>
					</div>
				</details>
				<p>
					Every buzzdetect release is archived on Zenodo: <a
						href="https://doi.org/{BUZZDETECT.zenodoDoi}">doi:{BUZZDETECT.zenodoDoi}</a
					>. The SHA-256 below pins exactly which model file you ran.
				</p>

				<h3>Identifying this model</h3>
				<table>
					<tbody>
						{#each provenance as [label, value, mono]}
							<tr>
								<th scope="row">{label}</th>
								<td>{#if mono}<code class="hash">{value}</code>{:else}{value}{/if}</td>
							</tr>
						{/each}
					</tbody>
				</table>
				{#each terms as t}
					<p class="terms">{t}</p>
				{/each}

				<h2>The sources for this model</h2>
				{#if components.length}
					<p>
						This model contains, or was derived from, the weights of the model{components.length > 1
							? 's'
							: ''} below. Each stays under its own license: credit {components.length > 1
							? 'each'
							: 'it'} when you cite or share this model.
					</p>
					{#each components as c}
						<section class="source">
							<h3>{c.name}</h3>
							<dl>
								{#if c.copyright}<dt>Authors</dt><dd>{c.copyright}</dd>{/if}
								<dt>License</dt>
								<dd>
									{c.spdx}
									{#if c.files?.length}
										<button type="button" class="link" onclick={() => toggleLicense(c.files![0].name)}
											>{openLicense === c.files[0].name ? 'Hide license' : 'View license'}</button
										>
									{/if}
								</dd>
								{#if c.source}<dt>Source</dt><dd><a href={c.source}>{c.source}</a></dd>{/if}
								{#if c.weights}<dt>Weights</dt><dd><a href={c.weights}>{c.weights}</a></dd>{/if}
								{#if c.revision}<dt>Revision</dt><dd><code>{c.revision}</code></dd>{/if}
								{#if c.paper}<dt>Paper</dt><dd><a href={c.paper}>{c.paper}</a></dd>{/if}
								{#if describeVia(c.via)}<dt>In this model</dt><dd>{describeVia(c.via)}</dd>{/if}
							</dl>
							{#if c.note}<p class="note">{c.note}</p>{/if}
							{#if (c.files?.length ?? 0) > 1}
								<p class="extra">
									Also included:
									{#each c.files!.slice(1) as f}
										<button type="button" class="link" onclick={() => toggleLicense(f.name)}
											>{openLicense === f.name ? `Hide ${f.label}` : f.label}</button
										>
									{/each}
								</p>
							{/if}
							{#if openLicense && c.files?.some((f) => f.name === openLicense)}
								<pre class="license">{licenseTexts[openLicense]}</pre>
							{/if}
						</section>
					{/each}
				{:else}
					<p class="hint">
						This model was exported before buzzdetect models recorded their sources, so there is
						nothing to list here. Its README may say what it was built from.
					</p>
				{/if}
			</article>
		{:else if details && !details.readme}
			<p class="hint">This model has no README.</p>
		{:else if html}
			<!-- eslint-disable-next-line svelte/no-at-html-tags -- sanitized above -->
			<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_noninteractive_element_interactions -->
			<article class="readme" onclick={onClick}>{@html html}</article>
		{/if}
		</div>
		{#if row}
			<footer>
				<span class="status" class:notify={row.update}>{status(row)}</span>
				{#if downloading[row.name]}<span class="spinner" aria-hidden="true"></span>{/if}
				<span class="actions">
					{#if details}
						<button
							type="button"
							onclick={() => {
								view = view === 'attribution' ? 'readme' : 'attribution';
								body?.scrollTo(0, 0);
							}}>{view === 'attribution' ? 'View README' : 'View Attribution'}</button
						>
					{/if}
					{#if row.installed}
						{#if row.bundled}
							<button type="button" onclick={() => setDisabled(row.name, !row.disabled)}
								>{row.disabled ? 'Enable' : 'Disable'}</button
							>
						{:else}
							<button type="button" class="danger" onclick={() => remove(row.name)}>Delete</button>
						{/if}
						{#if row.update}
							<button
								type="button"
								disabled={downloading[row.name]}
								onclick={() => download(row.name)}
								>{downloading[row.name] ? 'Updating…' : 'Update'}</button
							>
						{/if}
						{#if !row.disabled}
							<button type="button" class="primary" onclick={() => useModel(row.name)}
								>Use this model</button
							>
						{/if}
					{:else}
						<button type="button" onclick={() => setIgnored(row.name, !row.ignored)}
							>{row.ignored ? 'Un-ignore' : 'Ignore'}</button
						>
						<button
							type="button"
							class="primary"
							disabled={!row.compatible || downloading[row.name]}
							onclick={() => download(row.name)}
							>{downloading[row.name] ? 'Downloading…' : 'Download'}</button
						>
					{/if}
				</span>
			</footer>
		{/if}
	</main>
</div>

<style>
	:root {
		font-family: Inter, Avenir, Helvetica, Arial, sans-serif;
		color-scheme: light dark;
	}

	:global(html),
	:global(body) {
		margin: 0;
		height: 100%;
		overflow: hidden;
	}

	.window {
		display: grid;
		height: 100vh;
	}

	nav {
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
		padding: 1rem 0.5rem;
		overflow-y: auto;
		min-width: 0;
	}

	.resize-handle {
		cursor: col-resize;
		touch-action: none;
	}

	.resize-handle::after {
		content: '';
		display: block;
		width: 1px;
		height: 100%;
		margin: 0 auto;
		background: rgba(127, 127, 127, 0.25);
	}

	nav button {
		font: inherit;
		font-size: 0.9rem;
		text-align: left;
		padding: 0.35rem 0.6rem;
		border: none;
		border-radius: 6px;
		background: none;
		color: inherit;
		cursor: pointer;
		overflow-wrap: anywhere;
	}

	nav button:hover {
		background: rgba(127, 127, 127, 0.12);
	}

	nav button.active {
		background: rgba(127, 127, 127, 0.22);
		font-weight: 600;
	}

	nav button {
		display: flex;
		align-items: center;
		gap: 0.4rem;
	}

	nav button .name {
		flex: 1;
		min-width: 0;
	}

	nav button.dim {
		opacity: 0.55;
	}

	nav h3 {
		margin: 0.75rem 0.6rem 0.2rem;
		font-size: 0.72rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		opacity: 0.55;
	}

	nav h3:first-child {
		margin-top: 0;
	}

	.badge {
		flex: none;
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: #2f7de1;
	}

	.nav-foot {
		margin-top: auto;
		padding: 1rem 0.3rem 0;
	}

	.nav-foot .import {
		width: 100%;
		text-align: center;
		border: 1px solid rgba(127, 127, 127, 0.35);
	}

	.nav-foot .hint {
		margin: 0.5rem 0.3rem 0;
	}

	footer {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.6rem 1.75rem;
		border-top: 1px solid rgba(127, 127, 127, 0.2);
	}

	.actions {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin-left: auto;
	}

	.actions button.danger {
		border-color: #d33;
		color: #d33;
		background: none;
	}

	.actions button {
		font: inherit;
		font-size: 0.9rem;
		padding: 0.3rem 0.8rem;
		border-radius: 6px;
		border: 1px solid rgba(127, 127, 127, 0.4);
		background: rgba(127, 127, 127, 0.08);
		color: inherit;
		cursor: pointer;
	}

	.actions button.primary {
		background: #2f7de1;
		border-color: #2f7de1;
		color: white;
	}

	.actions button:disabled {
		opacity: 0.5;
		cursor: default;
	}

	.status {
		font-size: 0.85rem;
		opacity: 0.65;
	}

	.status.notify {
		color: #2f7de1;
		opacity: 1;
	}

	.spinner {
		width: 12px;
		height: 12px;
		border: 2px solid rgba(127, 127, 127, 0.3);
		border-top-color: #2f7de1;
		border-radius: 50%;
		animation: spin 0.8s linear infinite;
	}

	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}

	main {
		display: grid;
		grid-template-rows: 1fr auto;
		min-width: 0;
		min-height: 0;
	}

	.content {
		overflow-y: auto;
		padding: 1.25rem 1.75rem 2rem;
		min-height: 0;
	}

	h1 {
		margin: 0 0 0.25rem;
		font-size: 1.4rem;
		overflow-wrap: anywhere;
	}

	.description {
		margin: 0 0 1rem;
		opacity: 0.8;
	}

	.hint {
		opacity: 0.6;
		font-size: 0.85rem;
		font-weight: normal;
		margin-left: 0.5rem;
	}

	p.hint {
		margin-left: 0;
	}

	.error {
		color: #d33;
	}

	.readme {
		line-height: 1.55;
		max-width: 75ch;
	}

	.readme :global(h1),
	.readme :global(h2),
	.readme :global(h3) {
		margin: 1.4em 0 0.4em;
		line-height: 1.25;
	}

	.readme :global(h1) {
		font-size: 1.3rem;
	}

	.readme :global(h2) {
		font-size: 1.15rem;
		padding-bottom: 0.2em;
		border-bottom: 1px solid rgba(127, 127, 127, 0.25);
	}

	.readme :global(h3) {
		font-size: 1rem;
	}

	.readme :global(a) {
		color: #3b82c4;
	}

	.readme :global(code) {
		font-size: 0.88em;
		padding: 0.1em 0.3em;
		border-radius: 4px;
		background: rgba(127, 127, 127, 0.15);
	}

	.readme :global(pre) {
		padding: 0.75rem;
		border-radius: 6px;
		background: rgba(127, 127, 127, 0.12);
		overflow-x: auto;
	}

	.readme :global(pre code) {
		padding: 0;
		background: none;
	}

	.readme :global(table) {
		display: block;
		overflow-x: auto;
		margin: 0.75rem 0;
	}

	.readme :global(td),
	.readme :global(th) {
		border-bottom: 1px solid rgba(127, 127, 127, 0.2);
	}

	.readme :global(img) {
		max-width: 100%;
	}

	.attribution .cite {
		display: flex;
		align-items: flex-start;
		gap: 0.5rem;
		margin: 0.4rem 0 0.8rem;
	}

	.attribution .cite pre {
		flex: 1;
		margin: 0;
		white-space: pre-wrap;
		overflow-wrap: anywhere;
		font-size: 0.85rem;
	}

	.attribution button {
		font: inherit;
		font-size: 0.8rem;
		padding: 0.2rem 0.6rem;
		border-radius: 6px;
		border: 1px solid rgba(127, 127, 127, 0.4);
		background: rgba(127, 127, 127, 0.08);
		color: inherit;
		cursor: pointer;
	}

	.attribution button.link {
		margin-left: 0.4rem;
	}

	.attribution details summary {
		cursor: pointer;
		font-size: 0.85rem;
		opacity: 0.75;
	}

	.attribution table {
		display: table;
	}

	.attribution th {
		text-align: left;
		font-weight: 600;
		padding: 0.25rem 1rem 0.25rem 0;
		border-bottom: 1px solid rgba(127, 127, 127, 0.2);
		white-space: nowrap;
	}

	.attribution td {
		padding: 0.25rem 0;
		overflow-wrap: anywhere;
	}

	.attribution .hash {
		font-size: 0.8em;
	}

	.attribution .terms {
		padding: 0.5rem 0.75rem;
		border-left: 3px solid #d98a1c;
		background: rgba(217, 138, 28, 0.1);
	}

	.attribution .source {
		margin-bottom: 1.25rem;
	}

	.attribution dl {
		display: grid;
		grid-template-columns: max-content 1fr;
		gap: 0.3rem 1rem;
		margin: 0.5rem 0;
	}

	.attribution dt {
		font-weight: 600;
	}

	.attribution dd {
		margin: 0;
		overflow-wrap: anywhere;
	}

	.attribution .note,
	.attribution .extra {
		font-size: 0.9rem;
		opacity: 0.85;
	}

	.attribution pre.license {
		max-height: 24rem;
		overflow: auto;
		white-space: pre-wrap;
		font-size: 0.8rem;
	}

	.readme :global(blockquote) {
		margin: 0.75rem 0;
		padding-left: 0.9rem;
		border-left: 3px solid rgba(127, 127, 127, 0.4);
		opacity: 0.85;
	}
</style>
