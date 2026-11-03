"""Diagnostics: where do auto-seg's spurious objects come from? Dumps class totals,
component-count/size histograms per base name, timing, and per-class debug PNGs."""
import collections
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import imaging as im          # noqa: E402
from engine import segment                # noqa: E402
import gt_stats                           # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'app', 'tests')


def main():
    nat = np.asarray(Image.open(os.path.join(ROOT, 'native.png')))
    H, W = nat.shape[:2]

    t = time.time()
    cls, names, qrgb, brown_id, d = segment.classify(nat, {})
    print(f'classify {1000 * (time.time() - t):.0f} ms  palette entries -> {names}')

    # per-class component histogram
    for n in names:
        m = cls == (names.index(n) + 1)
        px = int(m.sum())
        if not px:
            continue
        t = time.time()
        comps = im.components(m, 1, 8)
        sizes = sorted((c['px'] for c in comps), reverse=True)
        big = [c for c in comps if c['px'] >= 25]
        print(f'{n:8s} px={px:6d} ({px / (H * W) * 100:4.1f}%) comps={len(comps):4d} '
              f'>=25px={len(big):4d} top10={sizes[:10]} [{1000 * (time.time() - t):.0f} ms]')

    t = time.time()
    res = segment.auto_segment(nat, max_suggestions=10 ** 6)   # no cap: see everything
    print(f'\nauto_segment (uncapped) {1000 * (time.time() - t):.0f} ms')
    hist = collections.defaultdict(list)
    for s in res['suggestions']:
        hist[(s['kind'], s['name'].rsplit('_', 1)[0])].append(s['px'])
    for k in sorted(hist):
        v = sorted(hist[k], reverse=True)
        print(f'  {k[0]:6s} {k[1]:10s} n={len(v):4d} px>=25:{sum(1 for x in v if x >= 25):4d} '
              f'top={v[:8]}')
    print(f"  TOTAL suggestions={len(res['suggestions'])} "
          f"objects={sum(1 for s in res['suggestions'] if s['kind'] == 'object')}")

    # class debug image
    rgb = np.zeros((H, W, 3), np.uint8)
    for i, n in enumerate(names):
        rgb[cls == i + 1] = im.golden_color(i * 7)
    Image.fromarray(rgb).resize((W * 2, H * 2), Image.NEAREST).save(os.path.join(OUT, 'diag_classes.png'))
    Image.fromarray(segment.render_debug(res, (H, W))).resize(
        (W * 2, H * 2), Image.NEAREST).save(os.path.join(OUT, 'diag_suggestions.png'))

    # ground truth: which classes does each GT object's pixels fall in?
    seg, gt = gt_stats.objects(25)
    print('\nGT object pixels by my class (leak check):')
    for o in sorted(gt, key=lambda o: -o['px'])[:14]:
        gtm = seg == o['cid']
        vals, cnts = np.unique(cls[gtm], return_counts=True)
        top = ', '.join(f'{names[v - 1] if v else "none"}:{c * 100 // o["px"]}%'
                        for v, c in sorted(zip(vals, cnts), key=lambda t: -t[1])[:4])
        print(f'  {o["name"]:14s} {o["px"]:6d}px  ->  {top}')


if __name__ == '__main__':
    main()
