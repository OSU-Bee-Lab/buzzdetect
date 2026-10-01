#!/usr/bin/env node
/**
 * Publish a model to the catalog the app's Models window reads.
 *
 *   node scripts/publish-model.mjs <model_dir>                 # publish or republish
 *   node scripts/publish-model.mjs <model_dir> --readme        # just the README
 *   node scripts/publish-model.mjs <model_dir> --min-app 2.1.0 # needs a newer app
 *   node scripts/publish-model.mjs <model_dir> --dry-run       # print, don't run
 *
 * Each model is its own GitHub release, tagged `model-<name>`, whose assets are
 * the model directory's files, flat. The catalog, models.json, is the only
 * asset of a release tagged `models`; the app fetches it from there. This
 * script keeps the committed copy of models.json at the repo root and uploads
 * it after the model's files, so the catalog never names a file that isn't up
 * yet. Commit models.json afterwards.
 *
 * A model's name is its identity. Republishing a name replaces its files, and
 * the app offers the new ones as an update to anyone whose installed hashes
 * differ -- that's how a bad binary gets fixed. The README carries no hash, so
 * `--readme` changes what everyone sees without prompting anyone to update.
 *
 * Every release here is created with --latest=false. The app updater reads
 * releases/latest/download/latest.json, so if a model release became "latest",
 * update checks would break for every installed app.
 *
 * Requires the gh CLI, authenticated with write access to the repo.
 */

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO = 'OSU-Bee-Lab/buzzdetect';
const CATALOG_TAG = 'models';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const catalogPath = join(root, 'models.json');

// The files the app copies out of a model directory; mirrors IMPORT_FILES in
// src-tauri/src/lib.rs, which also refuses any other name in the catalog.
const MODEL_FILES = [
	'model.onnx',
	'model.fp16.onnx',
	'config_model.json',
	'translation.csv',
	'weights.csv',
	'README.md'
];
// Mirrors REQUIRED_CONFIG_KEYS in lib.rs and engine/src/inference/models.py.
const REQUIRED_CONFIG_KEYS = [
	'classes',
	'samplerate',
	'framelength_s',
	'digits_time',
	'digits_results',
	'samples_hop',
	'samples_min'
];
const UNHASHED = new Set(['README.md']);

function usage(msg) {
	if (msg) console.error(`error: ${msg}\n`);
	console.error('usage: node scripts/publish-model.mjs <model_dir> [--readme] [--min-app <version>] [--dry-run]');
	process.exit(1);
}

const args = process.argv.slice(2);
const flag = (f) => {
	const i = args.indexOf(f);
	if (i < 0) return false;
	args.splice(i, 1);
	return true;
};
const option = (f) => {
	const i = args.indexOf(f);
	if (i < 0) return undefined;
	const v = args[i + 1];
	if (!v || v.startsWith('--')) usage(`${f} needs a value`);
	args.splice(i, 2);
	return v;
};
const readmeOnly = flag('--readme');
const dryRun = flag('--dry-run');
const minApp = option('--min-app');
if (args.length !== 1 || args[0].startsWith('--')) usage();

const dir = resolve(args[0]);
const name = basename(dir);
const tag = `model-${name}`;

if (!/^[A-Za-z0-9_][A-Za-z0-9_.-]*$/.test(name)) usage(`'${name}' is not a usable model name`);
if (minApp && !/^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$/.test(minApp)) usage(`--min-app '${minApp}' isn't semver`);

// Validate the model the way the app will when it installs it.
const configPath = join(dir, 'config_model.json');
if (!existsSync(join(dir, 'model.onnx'))) usage(`${dir} has no model.onnx`);
if (!existsSync(configPath)) usage(`${dir} has no config_model.json`);
const config = JSON.parse(readFileSync(configPath, 'utf8'));
const missing = REQUIRED_CONFIG_KEYS.filter((k) => !(k in config));
if (missing.length) usage(`config_model.json is missing ${missing.join(', ')}`);

const files = MODEL_FILES.filter((f) => existsSync(join(dir, f)));
// A README is optional: a model can go out undocumented and get one later
// with --readme, which adds it to the catalog entry.
const readmePath = join(dir, 'README.md');
const hasReadme = existsSync(readmePath);
if (readmeOnly && !hasReadme) usage(`${dir} has no README.md`);
const notes = hasReadme
	? ['--notes-file', readmePath]
	: ['--notes', (typeof config.description === 'string' && config.description.trim()) || 'No README yet.'];
const sha256 = (path) => createHash('sha256').update(readFileSync(path)).digest('hex');

function run(cmd, cmdArgs, { check = true } = {}) {
	const shown = [cmd, ...cmdArgs].map((a) => (/\s/.test(a) ? JSON.stringify(a) : a)).join(' ');
	if (dryRun) {
		console.log(`$ ${shown}`);
		return true;
	}
	console.log(`$ ${shown}`);
	try {
		execFileSync(cmd, cmdArgs, { stdio: ['ignore', 'inherit', check ? 'inherit' : 'ignore'] });
		return true;
	} catch (e) {
		if (check) process.exit(e.status ?? 1);
		return false;
	}
}

function releaseExists(t) {
	try {
		execFileSync('gh', ['release', 'view', t, '--repo', REPO], { stdio: 'ignore' });
		return true;
	} catch {
		return false;
	}
}

// Update the committed catalog.
const catalog = existsSync(catalogPath)
	? JSON.parse(readFileSync(catalogPath, 'utf8'))
	: { models: [] };
const existing = catalog.models.find((m) => m.name === name);
if (readmeOnly && !existing) usage(`${name} isn't in models.json yet; publish it without --readme first`);

const entry = {
	name,
	...(typeof config.description === 'string' && config.description.trim()
		? { description: config.description.trim() }
		: {}),
	...((minApp ?? existing?.min_app_version) ? { min_app_version: minApp ?? existing.min_app_version } : {}),
	url: `https://github.com/${REPO}/releases/download/${tag}/`,
	files: Object.fromEntries(
		files.map((f) => {
			const path = join(dir, f);
			return [f, UNHASHED.has(f) ? {} : { sha256: sha256(path), size: statSync(path).size }];
		})
	)
};

function uploadCatalog(text) {
	const scratch = mkdtempSync(join(tmpdir(), 'buzzdetect-catalog-'));
	const upload = join(scratch, 'models.json');
	writeFileSync(upload, text);
	if (!dryRun && !releaseExists(CATALOG_TAG)) {
		run('gh', [
			'release', 'create', CATALOG_TAG,
			'--title', 'Model catalog',
			'--notes', 'The list of downloadable models the buzzdetect app reads. Each model is its own model-<name> release; see scripts/publish-model.mjs.',
			'--latest=false',
			'--repo', REPO
		]);
	}
	run('gh', ['release', 'upload', CATALOG_TAG, upload, '--clobber', '--repo', REPO]);
	rmSync(scratch, { recursive: true, force: true });
}

if (readmeOnly) {
	// Only the README moves; the hashed files stay as they were.
	run('gh', ['release', 'upload', tag, readmePath, '--clobber', '--repo', REPO]);
	run('gh', ['release', 'edit', tag, ...notes, '--repo', REPO]);
	if (existing.files['README.md']) {
		console.log(`\nUpdated ${name}'s README. The catalog is unchanged.`);
		process.exit(0);
	}
	// First README for a model published without one: the app only fetches
	// files the entry names, so the catalog has to learn about it.
	existing.files['README.md'] = {};
	const text = JSON.stringify(catalog, null, 2) + '\n';
	if (!dryRun) writeFileSync(catalogPath, text);
	uploadCatalog(text);
	console.log(`\nAdded ${name}'s README. Commit models.json.`);
	process.exit(0);
}

// 1. The model's own release.
const paths = files.map((f) => join(dir, f));
if (!dryRun && releaseExists(tag)) {
	run('gh', ['release', 'upload', tag, ...paths, '--clobber', '--repo', REPO]);
	run('gh', ['release', 'edit', tag, ...notes, '--latest=false', '--repo', REPO]);
} else {
	run('gh', [
		'release', 'create', tag, ...paths,
		'--title', name,
		...notes,
		'--latest=false',
		'--repo', REPO
	]);
}

// 2. The catalog, after the files it names.
if (existing) catalog.models[catalog.models.indexOf(existing)] = entry;
else catalog.models.push(entry);
const text = JSON.stringify(catalog, null, 2) + '\n';
if (!dryRun) writeFileSync(catalogPath, text);

uploadCatalog(text);

if (dryRun) console.log(`\nmodels.json would become:\n${text}`);
else console.log(`\nPublished ${name}. Commit models.json.`);
