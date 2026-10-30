"""Per-object coverage diagnostic: for every ground-truth object, which suggestion covers
it best and by how much. Separates "dropped by the cap" from "wrong class / over-merged".
"""
import collections
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import imaging as im          # noqa: E402
from engine import segment                # noqa: E402
import gt_stats                           # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(cap=60, cfg=None, show=25):
    nat = np.asarray(Image.open(os.path.join(ROOT, 'native.png')))
    res = segment.auto_segment(nat, max_suggestions=cap, cfg=cfg)
    sug = res['suggestions']
    seg, gt = gt_stats.objects(25)

    kinds = collections.Counter(s.get('cls', s['name']) + '/' + s['kind'] for s in sug)
    print(f'cap={cap} suggestions={len(sug)} {dict(kinds)}  '
          f'candidates={res["n_objects"]}obj+{res["n_regions"]}reg dropped={res["dropped"]}')

    rows = []
    for o in gt:
        gtm = seg == o['cid']
        best = (0.0, 0.0, None)
        for s in sug:
            m = s['mask']
            cov = (m & gtm).sum() / o['px']
            iou = (m & gtm).sum() / max(1, (m | gtm).sum())
            if max(cov, iou / 0.5) > best[0]:
                best = (max(cov, iou / 0.5), cov, s['name'])
        ok = any(((s['mask'] & gtm).sum() / o['px'] >= 0.5) or
                 ((s['mask'] & gtm).sum() / max(1, (s['mask'] | gtm).sum()) >= 0.3) for s in sug)
        rows.append((o['name'], o['kind'], o['px'], best[2], round(best[1], 2), ok))
    hit = [r for r in rows if r[5]]
    print(f'recall {len(hit)}/{len(rows)} = {len(hit) / len(rows) * 100:.1f}%\n')

    byk = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        byk[r[1]][1] += 1
        byk[r[1]][0] += bool(r[5])
    for k, (h, t) in sorted(byk.items()):
        print(f'  {k:9s} {h:3d}/{t:3d} = {h / t * 100:5.1f}%')

    miss = sorted([r for r in rows if not r[5]], key=lambda r: -r[2])
    print(f'\nworst misses ({len(miss)}):')
    for r in miss[:show]:
        print(f'  {r[0]:14s} {r[1]:9s} {r[2]:6d}px best={str(r[3]):14s} cov={r[4]}')
    return res, rows


if __name__ == '__main__':
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    res, rows = run(cap)
    # uncapped candidate census
    full = segment.auto_segment(np.asarray(Image.open(os.path.join(ROOT, 'native.png'))),
                                max_suggestions=10 ** 6)
    c = collections.Counter()
    px = collections.defaultdict(list)
    for s in full['suggestions']:
        k = s.get('cls', s['name']) + '/' + s['kind']
        c[k] += 1
        px[k].append(s['px'])
    print('\nUNCAPPED census:')
    for k in sorted(c):
        v = sorted(px[k], reverse=True)
        print(f'  {k:20s} n={c[k]:3d}  px {v[:6]}{" ..." if len(v) > 6 else ""}')
