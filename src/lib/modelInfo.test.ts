import { describe, expect, it } from 'vitest';
import {
	classifyLink,
	formatSize,
	groupModels,
	type ModelRow
} from './modelInfo';

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

function row(name: string, over: Partial<ModelRow> = {}): ModelRow {
	return {
		name,
		description: null,
		installed: false,
		bundled: false,
		in_catalog: true,
		compatible: true,
		min_app_version: null,
		update: false,
		ignored: false,
		disabled: false,
		notify: false,
		download_size: null,
		...over
	};
}

describe('groupModels', () => {
	it('puts available, installed, disabled, then ignored, badged first within each', () => {
		const groups = groupModels([
			row('installed', { installed: true }),
			row('stale', { installed: true, update: true, notify: true }),
			row('too_new', { compatible: false }),
			row('new', { notify: true }),
			row('ignored', { ignored: true }),
			row('bundled_off', { installed: true, bundled: true, disabled: true })
		]);
		expect(groups.map((g) => [g.title, g.rows.map((r) => r.name)])).toEqual([
			['Available', ['new', 'too_new']],
			['Installed', ['stale', 'installed']],
			['Disabled', ['bundled_off']],
			['Ignored', ['ignored']]
		]);
	});

	it('drops empty groups', () => {
		expect(groupModels([row('a', { installed: true })]).map((g) => g.title)).toEqual(['Installed']);
	});
});

describe('formatSize', () => {
	it('reads in KB or MB', () => {
		expect(formatSize(578)).toBe('1 KB');
		expect(formatSize(7_610_503)).toBe('7.6 MB');
		expect(formatSize(21_617_645)).toBe('22 MB');
	});
});
