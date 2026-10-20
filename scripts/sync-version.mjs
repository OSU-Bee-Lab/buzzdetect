#!/usr/bin/env node
// Run by npm's "version" lifecycle hook so `npm version <x>` moves the version
// everywhere at once. tauri.conf.json's version is what ends up in the built
// installers' filenames and in the app's About box, and Cargo wants its own
// copy, so all three have to agree or the release assets are misnamed.
import { readFileSync, writeFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const { version } = JSON.parse(readFileSync(resolve(root, 'package.json'), 'utf8'));

const tauriConfPath = resolve(root, 'src-tauri/tauri.conf.json');
const tauriConf = JSON.parse(readFileSync(tauriConfPath, 'utf8'));
tauriConf.version = version;
writeFileSync(tauriConfPath, JSON.stringify(tauriConf, null, 2) + '\n');

const cargoPath = resolve(root, 'src-tauri/Cargo.toml');
writeFileSync(
	cargoPath,
	readFileSync(cargoPath, 'utf8').replace(/^version = ".*"/m, `version = "${version}"`)
);

// Only this package's own entry in the lockfile, not any dependency's.
const cargoLockPath = resolve(root, 'src-tauri/Cargo.lock');
writeFileSync(
	cargoLockPath,
	readFileSync(cargoLockPath, 'utf8').replace(
		/(\[\[package\]\]\nname = "buzzdetect"\nversion = ").*(")/,
		`$1${version}$2`
	)
);

// CITATION.cff carries the version and release date too; Zenodo reads it when
// the release is published. The date is the day `npm version` runs, which is
// the day the tag is made. The regexes leave the trailing comments alone.
const citationPath = resolve(root, 'CITATION.cff');
const today = new Date().toLocaleDateString('en-CA'); // YYYY-MM-DD, local time
writeFileSync(
	citationPath,
	readFileSync(citationPath, 'utf8')
		.replace(/^version: ".*?"/m, `version: "${version}"`)
		.replace(/^date-released: ".*?"/m, `date-released: "${today}"`)
);

console.log(`Synced version ${version} -> tauri.conf.json, Cargo.toml, Cargo.lock, CITATION.cff`);
