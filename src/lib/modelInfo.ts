// What the model info window shows, minus the rendering: the shapes
// `list_models` and `model_details` return.

export interface ModelInfo {
	name: string;
	removable: boolean;
	description: string | null;
	has_readme: boolean;
	has_fp16: boolean;
	disabled: boolean;
}

export interface ModelDetails {
	name: string;
	description: string | null;
	readme: string | null;
	/** config_model.json's provenance block; null for a model exported without one. */
	metadata: ModelMetadata | null;
	/** config_model.json's licence block; null for a model exported before
	 * buzzdetect-training wrote one (2026-10). */
	licenses: ModelLicenses | null;
	/** model.onnx's SHA-256 when known up front (a catalog model); an installed
	 * model's comes from the model_sha256 command. */
	sha256: string | null;
}

export interface ModelMetadata {
	modelname_internal?: string;
	trained_date?: string;
	embeddername?: string;
	set?: string;
	branch?: string;
	teacher?: string;
	ensemble_of?: string[];
	[key: string]: unknown;
}

/** Written by buzzdetect-training's embedders/licensing.py. */
export interface ModelLicenses {
	model?: { authors?: string[]; publisher?: string; spdx?: string; source?: string };
	commercial_use?: boolean;
	share_alike?: string[];
	components?: LicenseComponent[];
}

export interface LicenseComponent {
	id: string;
	name: string;
	spdx: string;
	copyright?: string;
	source?: string;
	weights?: string;
	revision?: string;
	paper?: string;
	note?: string;
	/** How the weights got into the model: 'embedder', 'teacher <name>', 'init'. */
	via?: string[];
	commercial_use?: boolean;
	share_alike?: boolean;
	/** The licence text first, then anything else that travels with it. */
	files?: { name: string; label: string }[];
}

// buzzdetect itself: what a model citation points readers to.
export const BUZZDETECT = {
	repo: 'https://github.com/OSU-Bee-Lab/buzzdetect',
	// The Zenodo concept DOI: resolves to every archived version.
	zenodoDoi: '10.5281/zenodo.15537954',
	paper:
		'Hearon, L. E., Johnson, L. H. P., Underwood, J., Lin, C.-H., & Johnson, R. M. (2025). ' +
		'buzzdetect: an open-source deep learning tool for automated bioacoustic pollinator ' +
		'monitoring. Journal of Insect Science, 25(6). https://doi.org/10.1093/jisesa/ieaf104',
	paperBibtex: `@article{hearon2025buzzdetect,
  title = {buzzdetect: an open-source deep learning tool for automated bioacoustic pollinator monitoring},
  author = {Hearon, Luke E. and Johnson, Lillian H. P. and Underwood, James and Lin, Chia-Hua and Johnson, Reed M.},
  journal = {Journal of Insect Science},
  volume = {25},
  number = {6},
  year = {2025},
  doi = {10.1093/jisesa/ieaf104}
}`
};

export interface Citation {
	text: string;
	bibtex: string;
}

/** How to cite a model: which model (its name, the name it was trained
 * under, its date and the hash of its graph, which pins it exactly), who made
 * it, and the buzzdetect version it was run with. */
export function citeModel(
	name: string,
	details: Pick<ModelDetails, 'metadata' | 'licenses'>,
	appVersion: string | null,
	sha256: string | null
): Citation {
	const md = details.metadata ?? {};
	const own = details.licenses?.model ?? {};
	const authors = own.authors ?? [];
	const internal = md.modelname_internal && md.modelname_internal !== name ? md.modelname_internal : null;
	const date = md.trained_date ?? null;
	const year = date?.slice(0, 4) ?? null;
	const version = appVersion ? `v${appVersion}` : null;
	const shortSha = sha256 ? sha256.slice(0, 12) : null;

	const lead = [authors.join('; '), own.publisher && !authors.length ? own.publisher : null]
		.filter(Boolean)
		.join(' ');
	const text =
		[
			lead ? `${lead}${year ? ` (${year})` : ''}.` : year ? `(${year}).` : null,
			`${name}${internal ? ` (trained as ${internal})` : ''} [buzzdetect model${date ? `, trained ${date}` : ''}].`,
			own.publisher && authors.length ? `${own.publisher}.` : null,
			`Run with buzzdetect${version ? ` ${version}` : ''}, ${BUZZDETECT.repo}, https://doi.org/${BUZZDETECT.zenodoDoi}.`,
			shortSha ? `model.onnx SHA-256 ${shortSha}.` : null
		]
			.filter(Boolean)
			.join(' ');

	const key = `buzzdetect_${name}`.replace(/[^A-Za-z0-9_]/g, '_');
	const note = [
		internal ? `trained as ${internal}` : null,
		version ? `run with buzzdetect ${version}` : null,
		sha256 ? `model.onnx SHA-256 ${sha256}` : null,
		own.spdx ? `license ${own.spdx}` : null
	].filter(Boolean);
	const fields: [string, string | null | undefined][] = [
		['title', `${name}: a buzzdetect model`],
		['author', authors.length ? authors.join(' and ') : null],
		['organization', own.publisher],
		['year', year],
		['date', date],
		['version', name],
		['url', own.source ?? BUZZDETECT.repo],
		['doi', BUZZDETECT.zenodoDoi],
		['note', note.length ? note.join('; ') : null]
	];
	const bibtex =
		`@software{${key},\n` +
		fields
			.filter(([, v]) => v)
			.map(([k, v]) => `  ${k} = {${v}}`)
			.join(',\n') +
		'\n}';
	return { text, bibtex };
}

/** Where a source's weights came into the model, in words. */
export function describeVia(via: string[] | undefined): string | null {
	if (!via?.length) return null;
	const words = via.map((v) =>
		v === 'embedder'
			? 'its embedder runs these weights'
			: v === 'init'
				? 'initialised from these weights'
				: v.startsWith('teacher ')
					? `distilled from ${v.slice('teacher '.length)}, which runs these weights`
					: v
	);
	return words.join('; ');
}

/** The terms a model inherits from its sources, as sentences; empty when none
 * restrict it beyond attribution. */
export function inheritedTerms(licenses: ModelLicenses | null): string[] {
	if (!licenses) return [];
	const comps = licenses.components ?? [];
	const terms: string[] = [];
	if (licenses.commercial_use === false) {
		const from = comps.filter((c) => c.commercial_use === false).map((c) => `${c.name} (${c.spdx})`);
		terms.push(`Non-commercial use only${from.length ? `, inherited from ${from.join(', ')}` : ''}.`);
	}
	if (licenses.share_alike?.length) {
		terms.push(
			`Share-alike: redistribute this model, or anything adapted from it, under ${licenses.share_alike.join(' and ')}.`
		);
	}
	return terms;
}

export type LinkTarget =
	| { kind: 'external'; url: string }
	| { kind: 'anchor' }
	| { kind: 'model-file'; rel: string };

/** Where a link in a README goes. Relative links name files in the model's
 * folder (e.g. tests/metrics.svg), which the app opens with the system viewer. */
export function classifyLink(href: string): LinkTarget {
	if (/^(https?:|mailto:)/i.test(href)) return { kind: 'external', url: href };
	if (href.startsWith('#') || href === '') return { kind: 'anchor' };
	return { kind: 'model-file', rel: decodeURIComponent(href.replace(/^\.\//, '').split('#')[0]) };
}

// The Models window's list: installed models plus the catalog's (see
// src-tauri/src/catalog.rs, which decides every flag here).
export interface ModelRow {
	name: string;
	description: string | null;
	installed: boolean;
	bundled: boolean;
	in_catalog: boolean;
	compatible: boolean;
	min_app_version: string | null;
	update: boolean;
	ignored: boolean;
	disabled: boolean;
	notify: boolean;
	download_size: number | null;
}

export interface ModelsOverview {
	models: ModelRow[];
	catalog_error: string | null;
}

export interface ModelGroup {
	title: string;
	rows: ModelRow[];
}

/** Available (not installed or ignored) models, then installed ones, then
 * disabled and ignored ones; within each, anything badged first. Empty groups
 * are dropped. */
export function groupModels(rows: ModelRow[]): ModelGroup[] {
	const badgedFirst = (list: ModelRow[]) =>
		[...list].sort((a, b) => Number(b.notify) - Number(a.notify));
	return [
		{ title: 'Available', rows: badgedFirst(rows.filter((r) => !r.installed && !r.ignored)) },
		{ title: 'Installed', rows: badgedFirst(rows.filter((r) => r.installed && !r.disabled)) },
		{ title: 'Disabled', rows: rows.filter((r) => r.installed && r.disabled) },
		{ title: 'Ignored', rows: rows.filter((r) => !r.installed && r.ignored) }
	].filter((g) => g.rows.length > 0);
}

export function formatSize(bytes: number): string {
	if (bytes < 1e6) return `${Math.max(1, Math.round(bytes / 1e3))} KB`;
	return `${(bytes / 1e6).toFixed(bytes < 1e7 ? 1 : 0)} MB`;
}
