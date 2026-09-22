// What the model info window shows, minus the rendering: the shapes
// `model_details` returns, and the rows of the thresholds table built from
// them. `thresholds` and `threshold_stats` are written by buzzdetect-training's
// 03_train/thresholds.py; a model exported before that, or by hand, may carry
// thresholds with no stats, or neither. config_model.json is the only place
// this table lives -- there used to be a duplicate rendering of it in the
// model's README, which 03_train/thresholds.py no longer generates.

export interface ModelInfo {
	name: string;
	removable: boolean;
	description: string | null;
	has_readme: boolean;
}

export interface ThresholdStat {
	fpr_target?: number;
	folds?: number;
	folds_total?: number;
	events?: number;
	frames?: number;
	sensitivity?: number;
	sensitivity_exclquiet?: number;
}

export interface ModelDetails {
	name: string;
	description: string | null;
	readme: string | null;
	thresholds: Record<string, unknown> | null;
	threshold_stats: Record<string, ThresholdStat> | null;
}

export interface ThresholdRow {
	cls: string;
	threshold: number;
	sensitivity: number | null;
	sensitivityExclQuiet: number | null;
	folds: string | null;
	events: number | null;
}

const num = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

/** One row per class with a numeric threshold, buzz first, then by name. */
export function thresholdRows(details: ModelDetails): ThresholdRow[] {
	const stats = details.threshold_stats ?? {};
	return Object.entries(details.thresholds ?? {})
		.filter((e): e is [string, number] => num(e[1]))
		.map(([cls, threshold]) => {
			const s = stats[cls] ?? {};
			return {
				cls,
				threshold,
				sensitivity: num(s.sensitivity) ? s.sensitivity : null,
				sensitivityExclQuiet: num(s.sensitivity_exclquiet) ? s.sensitivity_exclquiet : null,
				folds: num(s.folds)
					? num(s.folds_total)
						? `${s.folds}/${s.folds_total}`
						: String(s.folds)
					: null,
				events: num(s.events) ? s.events : null
			} satisfies ThresholdRow;
		})
		.sort((a, b) =>
			a.cls === 'ins_buzz' ? -1 : b.cls === 'ins_buzz' ? 1 : a.cls.localeCompare(b.cls)
		);
}

/** The FPR the suggestions were set at, if every stat agrees on one. */
export function fprTarget(details: ModelDetails): number | null {
	const targets = new Set(
		Object.values(details.threshold_stats ?? {})
			.map((s) => s.fpr_target)
			.filter(num)
	);
	return targets.size === 1 ? [...targets][0] : null;
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
 * ignored ones; within each, anything badged first. Empty groups are dropped. */
export function groupModels(rows: ModelRow[]): ModelGroup[] {
	const badgedFirst = (list: ModelRow[]) =>
		[...list].sort((a, b) => Number(b.notify) - Number(a.notify));
	return [
		{ title: 'Available', rows: badgedFirst(rows.filter((r) => !r.installed && !r.ignored)) },
		{ title: 'Installed', rows: badgedFirst(rows.filter((r) => r.installed)) },
		{ title: 'Ignored', rows: rows.filter((r) => !r.installed && r.ignored) }
	].filter((g) => g.rows.length > 0);
}

export function formatSize(bytes: number): string {
	if (bytes < 1e6) return `${Math.max(1, Math.round(bytes / 1e3))} KB`;
	return `${(bytes / 1e6).toFixed(bytes < 1e7 ? 1 : 0)} MB`;
}
