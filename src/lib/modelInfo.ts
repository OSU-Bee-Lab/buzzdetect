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
