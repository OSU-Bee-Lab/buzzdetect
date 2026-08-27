"""Assemble a symlink corpus of N audio files for the end-to-end runs.

The source tree holds one ~5.8h file per subdirectory, so pointing --dir_audio
at a single one would leave 11 of 12 streamers idle. Symlinks because os.walk
lists symlinked files normally (search_dir in src/utils.py), and because
copying 90 x 120MB would be absurd.
"""
import argparse, os, sys

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True)
ap.add_argument('--dest', required=True)
ap.add_argument('--n', type=int, default=90)
args = ap.parse_args()

paths = []
for root, dirs, files in os.walk(args.src):
    dirs.sort()
    for f in sorted(files):
        if f.lower().endswith('.mp3'):
            paths.append(os.path.join(root, f))
paths.sort()

if len(paths) < args.n:
    sys.exit(f'wanted {args.n} files, found {len(paths)}')

os.makedirs(args.dest, exist_ok=True)
for i, p in enumerate(paths[:args.n]):
    # Flat, with an index prefix: the source basenames are not unique across
    # subdirectories, and buzzdetect keys results on the path under dir_audio.
    link = os.path.join(args.dest, f'{i:03d}_{os.path.basename(p)}')
    if not os.path.islink(link):
        os.symlink(p, link)

print(f'{args.n} symlinks in {args.dest} (of {len(paths)} available)')
