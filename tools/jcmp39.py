#!/usr/bin/env python3
"""jcmp39: compare two JSON files numerically (reads only).
python3 jcmp39.py NEW REF [--common]   (--common: compare only keys present in both)"""
import json, math, sys

new, ref = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
common = '--common' in sys.argv
diffs, n = [], [0]


def walk(a, b, path):
    n[0] += 1
    if isinstance(a, dict) and isinstance(b, dict):
        ka, kb = set(a), set(b)
        if not common:
            for k in sorted(ka ^ kb, key=str):
                diffs.append(f'{path}/{k}: only in {"NEW" if k in ka else "REF"}')
        for k in sorted(ka & kb, key=str):
            walk(a[k], b[k], f'{path}/{k}')
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f'{path}: length {len(a)} vs {len(b)}')
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f'{path}[{i}]')
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (a == b or (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-9 * max(1.0, abs(b))):
            diffs.append(f'{path}: {a} vs {b}')
    elif a != b:
        diffs.append(f'{path}: {str(a)[:60]!r} vs {str(b)[:60]!r}')


walk(new, ref, '')
print(f'{"EQUAL" if not diffs else "DIFF"} nodes={n[0]} diffs={len(diffs)}')
for d in diffs[:12]:
    print('  ' + d)
