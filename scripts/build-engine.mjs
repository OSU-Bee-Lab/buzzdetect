#!/usr/bin/env node
/**
 * Freeze the Python engine into a sidecar binary the desktop app can spawn,
 * plus the data payload that has to sit beside it.
 *
 *   node scripts/build-engine.mjs
 *
 * Outputs, both gitignored and both consumed by tauri.conf.json:
 *
 *   src-tauri/binaries/buzzdetect-<target-triple>   (bundle.externalBin)
 *   src-tauri/engine-payload/                       (bundle.resources)
 *
 * The payload holds the parts buzzdetect reads off disk at runtime rather than
 * importing: the ONNX models and embedder (loaded by path via importlib in
 * src/inference) and the stream drivers (src/stream/audio.py builds its driver
 * map by listing that directory). The app runs the sidecar with the payload as
 * its working directory, which is what makes engine/src/config.py's relative
 * paths -- 'models', 'embedders', 'src/stream/drivers' -- resolve.
 *
 * Requires uv (https://docs.astral.sh/uv/) and a Rust toolchain on PATH.
 */

import { execFileSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, readdirSync, rmSync, chmodSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const ENGINE = join(ROOT, 'engine');
const VENV = join(ENGINE, '.venv-build');
const IS_WINDOWS = process.platform === 'win32';
const VENV_BIN = join(VENV, IS_WINDOWS ? 'Scripts' : 'bin');
const PYTHON = join(VENV_BIN, IS_WINDOWS ? 'python.exe' : 'python3');

const OUT_BIN_DIR = join(ROOT, 'src-tauri', 'binaries');
const OUT_PAYLOAD = join(ROOT, 'src-tauri', 'engine-payload');

function run(cmd, args, opts = {}) {
	console.log(`$ ${cmd} ${args.join(' ')}`);
	execFileSync(cmd, args, { stdio: 'inherit', ...opts });
}

/** The triple Tauri expects appended to an externalBin filename. */
function targetTriple() {
	if (process.env.ENGINE_TARGET_TRIPLE) return process.env.ENGINE_TARGET_TRIPLE;
	const out = execFileSync('rustc', ['-Vv'], { encoding: 'utf8' });
	const match = out.match(/^host:\s*(\S+)$/m);
	if (!match) throw new Error('could not read host triple from `rustc -Vv`');
	return match[1];
}

function setupVenv() {
	if (!existsSync(PYTHON)) {
		run('uv', ['venv', '--python', '3.13', VENV], { cwd: ENGINE });
	}
	run('uv', [
		'pip',
		'install',
		'--python',
		PYTHON,
		'-r',
		join(ENGINE, 'requirements-onnx.txt'),
		'pyinstaller'
	]);
}

function freeze() {
	rmSync(join(ENGINE, 'build'), { recursive: true, force: true });
	rmSync(join(ENGINE, 'dist'), { recursive: true, force: true });
	run(PYTHON, ['-m', 'PyInstaller', '--noconfirm', '--clean', 'buzzdetect.spec'], {
		cwd: ENGINE
	});

	const built = join(ENGINE, 'dist', IS_WINDOWS ? 'buzzdetect.exe' : 'buzzdetect');
	if (!existsSync(built)) throw new Error(`pyinstaller produced no binary at ${built}`);

	mkdirSync(OUT_BIN_DIR, { recursive: true });
	const dest = join(OUT_BIN_DIR, `buzzdetect-${targetTriple()}${IS_WINDOWS ? '.exe' : ''}`);
	cpSync(built, dest);
	if (!IS_WINDOWS) chmodSync(dest, 0o755);
	console.log(`\nsidecar -> ${dest}`);
	return dest;
}

function assemblePayload() {
	rmSync(OUT_PAYLOAD, { recursive: true, force: true });

	// Only the ONNX models: a tensorflow model directory would be dead weight
	// in a bundle whose engine has no tensorflow to run it with. dereference
	// because models/ is commonly a tree of symlinks during development.
	const modelsSrc = join(ENGINE, 'models');
	const modelsOut = join(OUT_PAYLOAD, 'models');
	mkdirSync(modelsOut, { recursive: true });
	const shipped = readdirSync(modelsSrc).filter((name) => {
		if (!name.endsWith('_onnx')) return false;
		return existsSync(join(modelsSrc, name, 'model.onnx'));
	});
	if (shipped.length === 0) {
		throw new Error(
			`no ONNX models found in ${modelsSrc}. Run ` +
				`.venv/bin/python3 tools/onnxify_model.py <modelname> in engine/ first.`
		);
	}
	for (const name of shipped) {
		cpSync(join(modelsSrc, name), join(modelsOut, name), {
			recursive: true,
			dereference: true
		});
	}

	cpSync(join(ENGINE, 'embedders', 'yamnet_onnx'), join(OUT_PAYLOAD, 'embedders', 'yamnet_onnx'), {
		recursive: true,
		dereference: true,
		filter: (src) => !src.includes('__pycache__')
	});

	// src/stream/audio.py only lists this directory; the modules themselves are
	// frozen into the binary (see buzzdetect.spec's collect_submodules).
	cpSync(join(ENGINE, 'src', 'stream', 'drivers'), join(OUT_PAYLOAD, 'src', 'stream', 'drivers'), {
		recursive: true,
		dereference: true,
		filter: (src) => !src.includes('__pycache__')
	});

	console.log(`payload -> ${OUT_PAYLOAD}`);
	console.log(`  models: ${shipped.join(', ')}`);
}

setupVenv();
freeze();
assemblePayload();
console.log('\nengine build complete');
