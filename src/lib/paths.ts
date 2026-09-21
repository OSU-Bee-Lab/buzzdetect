// Path display helpers. Paths here can use either separator: the user may be
// on Windows, and an output folder may have been typed by hand.

/** The last component of a path, ignoring any trailing separator. */
export function baseName(path: string): string {
	return splitPath(path).leaf;
}

/**
 * Splits a path into the part that can be abbreviated (`parent`, which keeps
 * its trailing separator) and the final component (`leaf`), which is usually
 * what identifies the folder.
 */
export function splitPath(path: string): { parent: string; leaf: string } {
	const trimmed = path.replace(/[\\/]+$/, '');
	const i = Math.max(trimmed.lastIndexOf('/'), trimmed.lastIndexOf('\\'));
	if (i < 0) return { parent: '', leaf: trimmed || path };
	return { parent: trimmed.slice(0, i + 1), leaf: trimmed.slice(i + 1) || trimmed };
}
