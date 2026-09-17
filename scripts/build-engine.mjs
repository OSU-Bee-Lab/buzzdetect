#!/usr/bin/env node
/**
 * Freeze the Python engine into a sidecar binary the desktop app can spawn,
 * plus the data payload that has to sit beside it.
 *
 *   node scripts/build-engine.mjs            # CPU build (onnxruntime)
 *   node scripts/build-engine.mjs --cuda     # CUDA build (onnxruntime-gpu)
 *
 * The CUDA build bundles the NVIDIA runtime, so it needs no system CUDA -- but
 * it is roughly a gigabyte larger and is only produced for Linux and Windows.
 *
 * Output, gitignored and consumed by tauri.conf.json as bundle.resources:
 *
 *   src-tauri/engine-payload/
 *     engine-bin/          the frozen engine (PyInstaller onedir: the launcher
 *                          plus _internal/)
 *     models/  src/  ...   the data payload
 *
 * The payload holds the parts buzzdetect reads off disk at runtime rather than
 * importing: the ONNX models and embedder (loaded by path via importlib in
 * src/inference) and the stream drivers (src/stream/audio.py builds its driver
 * map by listing that directory). The app runs the sidecar with the payload as
 * its working directory, which is what makes engine/src/config.py's relative
 * paths -- 'models', 'src/stream/drivers' -- resolve.
 *
 * onedir rather than a onefile externalBin: a onefile sidecar re-extracts its
 * libraries to a temp dir on every launch, which measured ~25s of frozen
 * window on the packaged app. The cost of onedir is that the frozen engine is
 * a directory, so it rides along in the resource payload instead of being a
 * Tauri externalBin, and src-tauri/src/lib.rs spawns engine-bin/buzzdetect-engine.
 *
 * Requires uv (https://docs.astral.sh/uv/) and a Rust toolchain on PATH.
 */

import { execFileSync } from 'node:child_process';
import {
	cpSync,
	existsSync,
	readFileSync,
	mkdirSync,
	readdirSync,
	renameSync,
	rmSync,
	chmodSync,
	statSync,
	writeFileSync
} from 'node:fs';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const ENGINE = join(ROOT, 'engine');

const SHIPLIST = 'shipped-models.txt';

// The frozen engine: PyInstaller's onedir output is dist/buzzdetect-engine/
// with a launcher of the same name inside it. Not plain 'buzzdetect' so it
// stays distinct from the app executable in logs and process lists.
const SIDECAR = 'buzzdetect-engine';

// What a shipped model directory consists of. model.onnx and config_model.json
// are required (checked below); the CSVs are carried for whoever reads the
// results and are copied when present. Notably absent: the TensorFlow weights,
// which the sidecar has no TensorFlow to run.
const MODEL_FILES = [
	'model.onnx',
	// Optional: the reduced-precision sibling, used only when a run asks for it
	// (BUZZDETECT_GPU_FP16) on a provider that can act on it.
	'model.fp16.onnx',
	'config_model.json',
	'translation.csv',
	'weights.csv',
	// Shown in the app's model info window.
	'README.md'
];

const CUDA = process.argv.includes('--cuda');

/**
 * Names from shipped-models.txt: one per line, # comments and blanks ignored.
 *
 * Split on either line ending and strip the comment unanchored: a Windows
 * runner checks the file out with CRLF, and `.` does not match `\r` while JS's
 * `$` (no `m` flag) only matches the true end of the string -- so an anchored
 * /#.*$/ silently matches nothing there and every comment line comes back as a
 * model name.
 */
function readShippedModels() {
	const path = join(ROOT, SHIPLIST);
	if (!existsSync(path)) throw new Error(`no ${SHIPLIST} at ${path}`);
	const names = readFileSync(path, 'utf8')
		.split(/\r?\n/)
		.map((line) => line.replace(/#.*/, '').trim())
		.filter((line) => line.length > 0);
	if (names.length === 0) {
		throw new Error(`${SHIPLIST} names no models, so the bundle would ship none.`);
	}
	return names;
}
const REQUIREMENTS = join(ENGINE, CUDA ? 'requirements-onnx-cuda.txt' : 'requirements-onnx.txt');
// Separate venvs: onnxruntime and onnxruntime-gpu install the same module and
// cannot coexist, so sharing one would silently freeze whichever was installed
// last.
const VENV = join(ENGINE, CUDA ? '.venv-build-cuda' : '.venv-build');
const IS_WINDOWS = process.platform === 'win32';
const VENV_BIN = join(VENV, IS_WINDOWS ? 'Scripts' : 'bin');
const PYTHON = join(VENV_BIN, IS_WINDOWS ? 'python.exe' : 'python3');

const OUT_PAYLOAD = join(ROOT, 'src-tauri', 'engine-payload');
// The frozen engine directory lands here, inside the payload.
const OUT_ENGINE_BIN = join(OUT_PAYLOAD, 'engine-bin');

function run(cmd, args, opts = {}) {
	console.log(`$ ${cmd} ${args.join(' ')}`);
	execFileSync(cmd, args, { stdio: 'inherit', ...opts });
}

function setupVenv() {
	if (!existsSync(PYTHON)) {
		run('uv', ['venv', '--python', '3.13', VENV], { cwd: ENGINE });
	}
	run('uv', ['pip', 'install', '--python', PYTHON, '-r', REQUIREMENTS, 'pyinstaller']);
}

/**
 * Freeze the engine. Returns the path to PyInstaller's onedir output
 * (engine/dist/buzzdetect-engine/); installEngineBin moves it into the payload
 * after assemblePayload has rebuilt that directory.
 */
function freeze() {
	rmSync(join(ENGINE, 'build'), { recursive: true, force: true });
	rmSync(join(ENGINE, 'dist'), { recursive: true, force: true });
	run(PYTHON, ['-m', 'PyInstaller', '--noconfirm', '--clean', 'buzzdetect.spec'], {
		cwd: ENGINE
	});

	const builtDir = join(ENGINE, 'dist', SIDECAR);
	const launcher = join(builtDir, IS_WINDOWS ? `${SIDECAR}.exe` : SIDECAR);
	if (!existsSync(launcher)) {
		throw new Error(`pyinstaller produced no launcher at ${launcher}`);
	}
	return builtDir;
}

/**
 * Move the frozen engine directory into the payload. Separate from freeze()
 * because assemblePayload() wipes and rebuilds OUT_PAYLOAD, so this has to run
 * after it. PyInstaller's build/ and dist/ are deleted right after: between
 * them they are most of a CUDA build's disk footprint and the Linux CI runner
 * has under 14GB for the whole job.
 */
function installEngineBin(builtDir) {
	rmSync(OUT_ENGINE_BIN, { recursive: true, force: true });
	renameSync(builtDir, OUT_ENGINE_BIN);
	if (!IS_WINDOWS) chmodSync(join(OUT_ENGINE_BIN, SIDECAR), 0o755);
	rmSync(join(ENGINE, 'build'), { recursive: true, force: true });
	rmSync(join(ENGINE, 'dist'), { recursive: true, force: true });
	console.log(`\nengine -> ${OUT_ENGINE_BIN} (${humanSize(OUT_ENGINE_BIN)})`);
}

/** Size of a file or directory tree, for the build log. */
function humanSize(path) {
	const bytes = statSync(path).isDirectory() ? dirBytes(path) : statSync(path).size;
	const mb = bytes / 1024 / 1024;
	return mb >= 1024 ? `${(mb / 1024).toFixed(2)} GB` : `${mb.toFixed(0)} MB`;
}

function dirBytes(dir) {
	let total = 0;
	for (const entry of readdirSync(dir, { withFileTypes: true })) {
		const p = join(dir, entry.name);
		total += entry.isDirectory() ? dirBytes(p) : statSync(p).size;
	}
	return total;
}

/**
 * Copy the NVIDIA runtime out of the build venv and into the payload.
 *
 * buzzdetect.spec deliberately keeps these out of the frozen executable -- see
 * strip_nvidia() there for why -- so they travel as loose files in the Tauri
 * resource directory instead, and src-tauri/src/lib.rs puts that directory on
 * the sidecar's library search path when it spawns it.
 *
 * Flattened into one directory: onnxruntime dlopen()s them by bare soname, so
 * keeping the nvidia/<component>/lib layout would mean a search path entry per
 * component. The sonames are distinct, so nothing collides.
 */
function copyNvidiaRuntime() {
	const scan = [
		'import glob, json, os, site',
		'found = []',
		'for sp in site.getsitepackages():',
		"    root = os.path.join(sp, 'nvidia')",
		'    if not os.path.isdir(root): continue',
		"    for pattern in ('*/lib/*.so*', '*/bin/*.dll'):",
		'        found += [p for p in glob.glob(os.path.join(root, pattern)) if os.path.isfile(p)]',
		'print(json.dumps(found))'
	].join('\n');
	const libraries = JSON.parse(execFileSync(PYTHON, ['-c', scan], { encoding: 'utf8' }).trim());

	// Fail here rather than ship a CUDA installer with no CUDA in it. Without
	// this the build succeeds, the app runs, and the only symptom is a GPU
	// analyzer quietly falling back to the CPU on the user's machine.
	if (libraries.length === 0) {
		throw new Error(
			`CUDA build requested but no nvidia-* shared libraries were found in ${VENV}. ` +
				'Check that requirements-onnx-cuda.txt installed the cuda/cudnn extras ' +
				'into this venv.'
		);
	}

	const out = join(OUT_PAYLOAD, 'nvidia');
	mkdirSync(out, { recursive: true });
	let bytes = 0;
	for (const lib of libraries) {
		cpSync(lib, join(out, basename(lib)), { dereference: true });
		bytes += statSync(lib).size;
	}
	const gb = (bytes / 1024 / 1024 / 1024).toFixed(2);
	console.log(`  nvidia runtime: ${libraries.length} libraries, ${gb} GB`);
}

function assemblePayload() {
	rmSync(OUT_PAYLOAD, { recursive: true, force: true });

	// engine/models/ is the only model directory -- what the engine reads from
	// a checkout is what ships. shipped-models.txt is the whole difference
	// between a model that's merely present and one that goes in the bundle,
	// so adding a model to a release is editing one line rather than copying
	// files between directories.
	const modelsSrc = join(ENGINE, 'models');
	const modelsOut = join(OUT_PAYLOAD, 'models');
	mkdirSync(modelsOut, { recursive: true });

	const shipped = readShippedModels();
	for (const name of shipped) {
		const dir = join(modelsSrc, name);
		if (!existsSync(dir)) {
			throw new Error(
				`${SHIPLIST} names '${name}', which is not in ${modelsSrc}.`
			);
		}
		// The engine runs ONNX and nothing else, so a model directory that
		// carries only Keras weights would build a bundle that fails at
		// analysis time on the user's machine rather than here.
		if (!existsSync(join(dir, 'model.onnx'))) {
			throw new Error(
				`${SHIPLIST} names '${name}', which has no model.onnx. Only ONNX ` +
					`builds can ship; convert it with buzzdetect-training's ` +
					`04_deploy/export_onnx.py.`
			);
		}
		if (!existsSync(join(dir, 'config_model.json'))) {
			throw new Error(
				`${SHIPLIST} names '${name}', which has no config_model.json ` +
					`(the class list and framing parameters). Re-export it with ` +
					`buzzdetect-training's 04_deploy/export_onnx.py.`
			);
		}
		// An allowlist, not the directory. engine/models/ is a working
		// directory: a model there sits next to its analysis output, its
		// tests, its training history and its TensorFlow weights, none of
		// which the bundle can use and one of which (output/) routinely runs
		// to tens of gigabytes. Copy only the files the engine reads.
		const dirOut = join(modelsOut, name);
		mkdirSync(dirOut, { recursive: true });
		for (const file of MODEL_FILES) {
			const from = join(dir, file);
			if (existsSync(from)) cpSync(from, join(dirOut, file), { dereference: true });
		}
	}

	// src/stream/audio.py only lists this directory; the modules themselves are
	// frozen into the binary (see buzzdetect.spec's collect_submodules).
	cpSync(join(ENGINE, 'src', 'stream', 'drivers'), join(OUT_PAYLOAD, 'src', 'stream', 'drivers'), {
		recursive: true,
		dereference: true,
		filter: (src) => !src.includes('__pycache__')
	});

	if (CUDA) copyNvidiaRuntime();

	// What this build can actually accelerate on. The app reads it to decide
	// whether to offer a GPU analyzer at all, rather than letting someone pick
	// one that silently runs on the CPU. Asked of the venv that was just
	// frozen, so it can't drift from what shipped.
	const gpuProviders = JSON.parse(
		execFileSync(
			PYTHON,
			['-c', 'import json,sys; sys.path.insert(0, "."); from src.inference.onnx import gpu_providers_available; print(json.dumps(gpu_providers_available()))'],
			{ cwd: ENGINE, encoding: 'utf8' }
		).trim()
	);
	writeFileSync(
		join(OUT_PAYLOAD, 'gpu-providers.json'),
		JSON.stringify({ gpu_providers: gpuProviders }, null, 2) + '\n'
	);

	console.log(`payload -> ${OUT_PAYLOAD}`);
	console.log(`  gpu providers: ${gpuProviders.length ? gpuProviders.join(', ') : 'none (CPU only)'}`);
	console.log(`  models: ${shipped.join(', ')}`);
}

console.log(`building the ${CUDA ? 'CUDA' : 'CPU'} engine`);
setupVenv();
const builtDir = freeze();
assemblePayload();
installEngineBin(builtDir);
console.log('\nengine build complete');
