#!/usr/bin/env node
/**
 * Rebuild src-tauri/engine-payload/ if it's missing or older than the engine
 * sources it was built from, so a local `npm run tauri build` never silently
 * bundles a stale frozen engine.
 *
 *   node scripts/check-engine-fresh.mjs
 *
 * Wired into tauri.conf.json's beforeBuildCommand. CI builds engine-payload/
 * explicitly (.github/workflows/release.yml) right before `tauri build`, so
 * this is a fast no-op there -- it exists for whoever runs `tauri build`
 * locally without remembering to run `npm run build:engine` first, or after
 * pulling engine changes without rebuilding.
 *
 * A CUDA payload rebuilds itself as CPU here, since this script has no way to
 * know a stale payload was ever a CUDA one. Pass --cuda by hand if that's
 * what you need: `node scripts/build-engine.mjs --cuda`.
 */
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const ENGINE = join(ROOT, 'engine');
const PAYLOAD = join(ROOT, 'src-tauri', 'engine-payload');
const LAUNCHER = join(
	PAYLOAD,
	'engine-bin',
	process.platform === 'win32' ? 'buzzdetect-engine.exe' : 'buzzdetect-engine'
);

// Mirrors build-engine.mjs's MODEL_FILES and readShippedModels(). Not
// imported from there: that script runs the whole freeze+assemble+install
// pipeline as soon as it's loaded, so importing it would build the engine
// just to ask whether the engine needs building.
const MODEL_FILES = ['model.onnx', 'model.fp16.onnx', 'config_model.json', 'translation.csv', 'weights.csv', 'README.md'];

function shippedModelNames() {
	const path = join(ROOT, 'shipped-models.txt');
	if (!existsSync(path)) return [];
	return readFileSync(path, 'utf8')
		.split(/\r?\n/)
		.map((line) => line.replace(/#.*/, '').trim())
		.filter(Boolean);
}

/** Newest mtime under any of `paths`, recursing into directories. 0 if none exist. */
function newestMtime(paths) {
	let newest = 0;
	for (const p of paths) {
		if (!existsSync(p)) continue;
		const stat = statSync(p);
		if (stat.isDirectory()) {
			const children = readdirSync(p)
				.filter((name) => name !== '__pycache__')
				.map((name) => join(p, name));
			newest = Math.max(newest, newestMtime(children));
		} else {
			newest = Math.max(newest, stat.mtimeMs);
		}
	}
	return newest;
}

const sources = [
	join(ENGINE, 'src'),
	join(ENGINE, 'embedders'),
	join(ENGINE, 'buzzdetect_cli.py'),
	join(ENGINE, 'buzzdetect.spec'),
	join(ENGINE, 'requirements.txt'),
	join(ENGINE, 'requirements-onnx.txt'),
	join(ENGINE, 'requirements-onnx-cuda.txt'),
	join(ROOT, 'shipped-models.txt'),
	...shippedModelNames().flatMap((name) =>
		MODEL_FILES.map((file) => join(ENGINE, 'models', name, file))
	)
];

const sourceMtime = newestMtime(sources);
const payloadMtime = existsSync(LAUNCHER) ? statSync(LAUNCHER).mtimeMs : 0;

if (payloadMtime === 0) {
	console.log('engine-payload/ missing; building it (npm run build:engine)');
} else if (sourceMtime > payloadMtime) {
	console.log('engine-payload/ is older than the engine sources it was built from; rebuilding');
} else {
	console.log('engine-payload/ is up to date');
	process.exit(0);
}

execFileSync('node', [join(ROOT, 'scripts', 'build-engine.mjs')], { stdio: 'inherit', cwd: ROOT });
