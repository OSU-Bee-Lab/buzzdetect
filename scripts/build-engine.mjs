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
import {
	cpSync,
	existsSync,
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

const CUDA = process.argv.includes('--cuda');
const REQUIREMENTS = join(ENGINE, CUDA ? 'requirements-onnx-cuda.txt' : 'requirements-onnx.txt');
// Separate venvs: onnxruntime and onnxruntime-gpu install the same module and
// cannot coexist, so sharing one would silently freeze whichever was installed
// last.
const VENV = join(ENGINE, CUDA ? '.venv-build-cuda' : '.venv-build');
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
	run('uv', ['pip', 'install', '--python', PYTHON, '-r', REQUIREMENTS, 'pyinstaller']);
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
	// Moved rather than copied, and PyInstaller's staging directory deleted
	// right after: between them they were most of a CUDA build's disk
	// footprint, and the Linux CI runner has under 14GB for the whole job.
	rmSync(dest, { force: true });
	renameSync(built, dest);
	rmSync(join(ENGINE, 'build'), { recursive: true, force: true });
	rmSync(join(ENGINE, 'dist'), { recursive: true, force: true });
	if (!IS_WINDOWS) chmodSync(dest, 0o755);
	console.log(`\nsidecar -> ${dest} (${humanSize(dest)})`);
	return dest;
}

/** Size of a file, for the build log. */
function humanSize(path) {
	const mb = statSync(path).size / 1024 / 1024;
	return mb >= 1024 ? `${(mb / 1024).toFixed(2)} GB` : `${mb.toFixed(0)} MB`;
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

	// The repo's own models/, not engine/models: the latter is whatever the
	// developer happens to have locally -- often symlinks, often TensorFlow
	// models this engine has no TensorFlow to run -- while models/ is the
	// deliberate list of what ships. See models/README.md.
	const modelsSrc = join(ROOT, 'models');
	const modelsOut = join(OUT_PAYLOAD, 'models');
	mkdirSync(modelsOut, { recursive: true });
	const shipped = readdirSync(modelsSrc).filter((name) =>
		existsSync(join(modelsSrc, name, 'model.onnx'))
	);
	if (shipped.length === 0) {
		throw new Error(
			`no ONNX models found in ${modelsSrc}. Convert one with ` +
				`engine/tools/onnxify_model.py and copy it there; see models/README.md.`
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
freeze();
assemblePayload();
console.log('\nengine build complete');
