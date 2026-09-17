// What the model info window shows, minus the rendering: the shapes
// `model_details` returns, and the rows of the thresholds table built from
// them. `thresholds` and `threshold_stats` are written by buzzdetect-training's
// 03_train/thresholds.py; a model exported before that, or by hand, may carry
// thresholds with no stats, or neither.

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
	sd?: number;
	ci95_low?: number;
	ci95_high?: number;
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
	ci95: [number, number] | null;
	sd: number | null;
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
				ci95: num(s.ci95_low) && num(s.ci95_high) ? [s.ci95_low, s.ci95_high] : null,
				sd: num(s.sd) ? s.sd : null,
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
