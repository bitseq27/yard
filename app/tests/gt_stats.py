"""Ground-truth stats from the origin session's segmentation (seg.npy + seg_legend.json).

Tells us how many "real" objects the courtyard has, so the M2 auto-seg acceptance
criterion (>=80% of objects, <=60 suggestions) has a concrete denominator.
"""
import collections
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OBJ_RE = re.compile(r'^(plant|lily_pad|pot|carpet)_(\d+)$')


def load():
    seg = np.load(os.path.join(ROOT, 'seg.npy'))
    leg = json.load(open(os.path.join(ROOT, 'seg_legend.json')))
    return seg, leg


def objects(min_px=25):
    seg, leg = load()
    out = []
    for k, v in leg.items():
        m = OBJ_RE.match(v['name'])
        if not m:
            continue
        cid = int(k)
        px = int((seg == cid).sum())
        if px < min_px:
            continue
        ys, xs = np.nonzero(seg == cid)
        out.append({'cid': cid, 'name': v['name'], 'kind': m.group(1), 'px': px,
                    'bbox': (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))})
    return seg, out


if __name__ == '__main__':
    seg, leg = load()
    by = collections.defaultdict(list)
    for k, v in leg.items():
        m = OBJ_RE.match(v['name'])
        if not m:
            continue
        by[m.group(1)].append(int((seg == int(k)).sum()))
    print(f'{"class":10s} {"ids":>4s} {"0px":>4s} {">=10":>5s} {">=25":>5s} {">=50":>5s} {">=100":>6s} {"maxpx":>6s}')
    tot = collections.Counter()
    for t, v in sorted(by.items()):
        a = np.array(v)
        print(f'{t:10s} {len(a):4d} {(a == 0).sum():4d} {(a >= 10).sum():5d} {(a >= 25).sum():5d} '
              f'{(a >= 50).sum():5d} {(a >= 100).sum():6d} {a.max():6d}')
        for thr in (10, 25, 50, 100):
            tot[thr] += int((a >= thr).sum())
    print('TOTALS  >=10px:', tot[10], ' >=25px:', tot[25], ' >=50px:', tot[50], ' >=100px:', tot[100])
    _, objs = objects(25)
    print('\nobjects >=25px:', len(objs))
    for o in sorted(objs, key=lambda o: -o['px'])[:15]:
        print(f"  {o['name']:14s} {o['px']:6d}px bbox={o['bbox']}")
