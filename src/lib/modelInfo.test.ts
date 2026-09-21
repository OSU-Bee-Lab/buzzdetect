import { describe, expect, it } from 'vitest';
import { classifyLink, fprTarget, thresholdRows, type ModelDetails } from './modelInfo';

function details(over: Partial<ModelDetails> = {}): ModelDetails {
	return {
		name: 'm',
		description: null,
		readme: null,
		thresholds: null,
		threshold_stats: null,
		...over
	};
}

describe('thresholdRows', () => {
	it('is empty for a model that suggests nothing', () => {
		expect(thresholdRows(details())).toEqual([]);
	});

	it('shows a bare threshold with no stats', () => {
		expect(thresholdRows(details({ thresholds: { ins_buzz: -1.2 } }))).toEqual([
			{
				cls: 'ins_buzz',
				threshold: -1.2,
				sensitivity: null,
				sensitivityExclQuiet: null,
				folds: null,
				events: null
			}
		]);
	});

	it('pairs each threshold with its stats and puts ins_buzz first', () => {
		const rows = thresholdRows(
			details({
				thresholds: { frog: 0.1, ins_buzz: -0.6 },
				threshold_stats: {
					ins_buzz: { sensitivity: 0.4, sensitivity_exclquiet: 0.5, folds: 7, folds_total: 8, events: 400 },
					frog: { folds: 2, events: 5 }
				}
			})
		);
		expect(rows.map((r) => r.cls)).toEqual(['ins_buzz', 'frog']);
		expect(rows[0]).toMatchObject({
			sensitivity: 0.4,
			sensitivityExclQuiet: 0.5,
			folds: '7/8',
			events: 400
		});
		expect(rows[1]).toMatchObject({ sensitivity: null, sensitivityExclQuiet: null, folds: '2', events: 5 });
	});

	it('skips a class whose threshold is not a number', () => {
		expect(thresholdRows(details({ thresholds: { a: 'high', b: 1 } }))).toHaveLength(1);
	});
});

describe('fprTarget', () => {
	it('is the shared target, or null when there is none or they disagree', () => {
		expect(fprTarget(details())).toBeNull();
		expect(
			fprTarget(details({ threshold_stats: { a: { fpr_target: 0.005 }, b: { fpr_target: 0.005 } } }))
		).toBe(0.005);
		expect(
			fprTarget(details({ threshold_stats: { a: { fpr_target: 0.005 }, b: { fpr_target: 0.01 } } }))
		).toBeNull();
	});
});

describe('classifyLink', () => {
	it('sends web links out and model-relative links to the model folder', () => {
		expect(classifyLink('https://doi.org/x')).toEqual({ kind: 'external', url: 'https://doi.org/x' });
		expect(classifyLink('#training')).toEqual({ kind: 'anchor' });
		expect(classifyLink('./tests/metrics%20buzz.svg')).toEqual({
			kind: 'model-file',
			rel: 'tests/metrics buzz.svg'
		});
	});
});
