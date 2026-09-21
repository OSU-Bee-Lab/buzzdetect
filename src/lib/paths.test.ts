import { describe, expect, it } from 'vitest';
import { baseName, splitPath } from './paths';

describe('splitPath', () => {
	it('separates the final folder from its parent', () => {
		expect(splitPath('/Users/luke/Documents/audio')).toEqual({
			parent: '/Users/luke/Documents/',
			leaf: 'audio'
		});
	});
	it('ignores a trailing separator', () => {
		expect(splitPath('/data/site A/')).toEqual({ parent: '/data/', leaf: 'site A' });
	});
	it('handles Windows paths', () => {
		expect(splitPath('C:\\Users\\luke\\out')).toEqual({ parent: 'C:\\Users\\luke\\', leaf: 'out' });
	});
	it('has no parent for a bare name or empty path', () => {
		expect(splitPath('audio')).toEqual({ parent: '', leaf: 'audio' });
		expect(splitPath('')).toEqual({ parent: '', leaf: '' });
	});
	it('keeps the root as the leaf of "/"', () => {
		expect(baseName('/')).toBe('/');
	});
});
