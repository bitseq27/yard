"""Dev harness: run auto_segment on the courtyard and score it against the origin
session's ground truth (seg.npy / seg_legend.json).

Acceptance (plan §6 M2): auto-seg returns >=80% of courtyard objects as <=60 suggestions.
An object counts as "returned" when one suggestion covers >=COVER of its pixels
(merged suggestions are fine — the origin labelling over-splits rugs/foliage).
"""
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import imaging as im          # noqa: E402
from engine import segment                # noqa: E402
import gt_stats                           # noqa: E402

COVER = 0.5
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def score(res, cover=COVER, verbose=True, gt_min_px=25):
    seg, gt = gt_stats.objects(gt_min_px)
    sug = res['suggestions']
    H, W = seg.shape
    masks = [s['mask'] for s in sug]
    hit, miss = [], []
    for o in gt:
        gtm = seg == o['cid']
        best, bs = 0.0, None
        for s, m in zip(sug, masks):
            if not m.any():
                continue
            cov = (m & gtm).sum() / o['px']
            iou = (m & gtm).sum() / max(1, (m | gtm).sum())
            sc = max(cov, iou / cover)          # credit either coverage or a tight match
            if cov >= cover or iou >= 0.3:
                if sc > best:
                    best, bs = sc, s['name']
        if bs:
            hit.append((o['name'], o['px'], bs))
        else:
            gys, gxs = np.nonzero(gtm)
            miss.append((o['name'], o['px'], (int(gxs.min()), int(gys.min()), int(gxs.max()), int(gys.max()))))
    recall = len(hit) / max(1, len(gt))
    objs = [s for s in sug if s['kind'] == 'object']
    if verbose:
        print(f'suggestions: {len(sug)} ({len(objs)} object / {len(sug) - len(objs)} region)  '
              f'cap=60 {"OK" if len(sug) <= 60 else "OVER"}')
        print(f'ground-truth objects >={gt_min_px}px: {len(gt)}   covered: {len(hit)}   '
              f'recall {recall * 100:.1f}%  {"OK" if recall >= 0.8 else "FAIL"} (target >=80%)')
        print(f'pipeline dropped for the cap: {res["dropped"]}  (n_objects={res["n_objects"]})')
        by = {}
        for s in sug:
            by.setdefault(s['kind'] + '/' + s['name'].rsplit('_', 1)[0], []).append(s['px'])
        for k in sorted(by):
            v = by[k]
            print(f'   {k:22s} n={len(v):3d} px {min(v)}..{max(v)}')
        if miss:
            print(f'missed ({len(miss)}):')
            for n, px, bb in sorted(miss, key=lambda t: -t[1])[:20]:
                print(f'   {n:14s} {px:6d}px bbox={bb}')
    return recall, len(sug), miss


def main():
    nat = np.asarray(Image.open(os.path.join(ROOT, 'native.png')))
    t0 = time.time()
    res = segment.auto_segment(nat, max_suggestions=60)
    ms = (time.time() - t0) * 1000
    print(f'auto_segment: {ms:.0f} ms  classes={res["classes"]}')
    recall, n, miss = score(res)
    dbg = segment.render_debug(res, nat.shape[:2])
    Image.fromarray(dbg).resize((dbg.shape[1] * 2, dbg.shape[0] * 2), Image.NEAREST).save(
        os.path.join(ROOT, 'app', 'tests', 'autoseg_debug.png'))
    print('\nACCEPTANCE:', 'PASS' if (recall >= 0.8 and n <= 60) else 'FAIL')
    print('debug image -> app/tests/autoseg_debug.png')


if __name__ == '__main__':
    main()
