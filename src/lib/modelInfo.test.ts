import { describe, expect, it } from 'vitest';
import {
	citeModel,
	classifyLink,
	describeVia,
	formatSize,
	groupModels,
	inheritedTerms,
	type ModelLicenses,
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

const licensed: ModelLicenses = {
	model: {
		authors: ['Hearon, Luke E.'],
		publisher: 'OSU Bee Lab',
		spdx: 'MIT',
		source: 'https://github.com/OSU-Bee-Lab/buzzdetect'
	},
	commercial_use: true,
	share_alike: [],
	components: [{ id: 'yamnet', name: 'YAMNet', spdx: 'Apache-2.0', via: ['embedder'], commercial_use: true }]
};

describe('citeModel', () => {
	it('names the model, the name it was trained under, its authors, app version and hash', () => {
		const c = citeModel(
			'heavy_v1',
			{
				metadata: { modelname_internal: 'v4-ft-ps-e60-moderate', trained_date: '2026-09-28' },
				licenses: licensed
			},
			'2.1.0',
			'abcdef0123456789'
		);
		expect(c.text).toBe(
			'Hearon, Luke E. (2026). heavy_v1 (trained as v4-ft-ps-e60-moderate) ' +
				'[buzzdetect model, trained 2026-09-28]. OSU Bee Lab. Run with buzzdetect v2.1.0, ' +
				'https://github.com/OSU-Bee-Lab/buzzdetect, https://doi.org/10.5281/zenodo.15537954. ' +
				'model.onnx SHA-256 abcdef012345.'
		);
		expect(c.bibtex).toContain('@software{buzzdetect_heavy_v1,');
		expect(c.bibtex).toContain('author = {Hearon, Luke E.}');
		expect(c.bibtex).toContain('model.onnx SHA-256 abcdef0123456789');
		expect(c.bibtex).toContain('run with buzzdetect v2.1.0');
	});

	it('still cites a model exported before it recorded any of that', () => {
		const c = citeModel('model_general_v3', { metadata: null, licenses: null }, null, null);
		expect(c.text).toBe(
			'model_general_v3 [buzzdetect model]. Run with buzzdetect, ' +
				'https://github.com/OSU-Bee-Lab/buzzdetect, https://doi.org/10.5281/zenodo.15537954.'
		);
		expect(c.bibtex).not.toContain('author');
	});
});

describe('licence terms', () => {
	it('says nothing beyond attribution for permissive sources', () => {
		expect(inheritedTerms(licensed)).toEqual([]);
		expect(inheritedTerms(null)).toEqual([]);
	});

	it('names the source a restriction comes from', () => {
		const terms = inheritedTerms({
			commercial_use: false,
			share_alike: ['CC-BY-NC-SA-4.0'],
			components: [
				{ id: 'aves2_beats', name: 'AVES2', spdx: 'CC-BY-NC-SA-4.0', commercial_use: false }
			]
		});
		expect(terms[0]).toBe('Non-commercial use only, inherited from AVES2 (CC-BY-NC-SA-4.0).');
		expect(terms[1]).toContain('under CC-BY-NC-SA-4.0');
	});

	it('explains how a source got into a student', () => {
		expect(describeVia(['teacher heavy_v1', 'init'])).toBe(
			'distilled from heavy_v1, which runs these weights; initialised from these weights'
		);
		expect(describeVia([])).toBeNull();
	});
});
