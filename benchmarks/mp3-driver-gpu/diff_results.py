"""Do two analysis output folders contain the same results, file for file?

The point of the tail-scan read path is that it changes how the audio is read
and not what comes out, so the binding check is not a sample comparison but
this: run the same corpus through both read paths and diff the CSVs. The
analyzer is deterministic for a given input, so anything other than byte-for-byte
equality is a difference in what was read.

    python3 diff_results.py <dir_a> <dir_b>
"""

import filecmp
import os
import sys


def results(root):
    found = {}
    for folder, _, names in os.walk(root):
        for name in names:
            if name.endswith('.csv'):
                path = os.path.join(folder, name)
                found[os.path.relpath(path, root)] = path
    return found


def main():
    a_root, b_root = sys.argv[1], sys.argv[2]
    a, b = results(a_root), results(b_root)

    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    shared = sorted(set(a) & set(b))

    identical, differing = [], []
    for name in shared:
        if filecmp.cmp(a[name], b[name], shallow=False):
            identical.append(name)
        else:
            differing.append(name)

    print(f'{os.path.basename(a_root)} vs {os.path.basename(b_root)}')
    print(f'  {len(identical)} identical, {len(differing)} differing, '
          f'{len(only_a)} only in the first, {len(only_b)} only in the second')
    for name in differing[:20]:
        size_a, size_b = os.path.getsize(a[name]), os.path.getsize(b[name])
        with open(a[name]) as fa, open(b[name]) as fb:
            for line_no, (la, lb) in enumerate(zip(fa, fb), 1):
                if la != lb:
                    print(f'    {name}: first differs at line {line_no}')
                    print(f'      a: {la.rstrip()}')
                    print(f'      b: {lb.rstrip()}')
                    break
            else:
                print(f'    {name}: one is longer ({size_a:,} vs {size_b:,} bytes)')
    for name in only_a[:10]:
        print(f'    only in the first:  {name}')
    for name in only_b[:10]:
        print(f'    only in the second: {name}')

    sys.exit(1 if differing or only_a or only_b else 0)


if __name__ == '__main__':
    main()
