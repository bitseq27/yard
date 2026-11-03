"""Parameter sweep for auto_segment, scored against the courtyard ground truth.

Reports recall / suggestion count / largest object (a product-soundness guard: a suggestion
covering 12% of the frame is not an editable object) / mean fragments per suggestion.
"""
import itertools
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import segment                # noqa: E402
import gt_stats                           # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NAT = np.asarray(Image.open(os.path.join(ROOT, 'native.png')))
SEG, GT = gt_stats.objects(25)
GTM = [(o['name'], o['px'], SEG == o['cid']) for o in GT]


def recall_of(sug):
    hit = 0
    for name, px, gtm in GTM:
        for s in sug:
            m = s['mask']
            if (m & gtm).sum() / px >= 0.5 or (m & gtm).sum() / max(1, (m | gtm).sum()) >= 0.3:
                hit += 1
                break
    return hit / len(GTM)


def evaluate(cfg, cap=60):
    res = segment.auto_segment(NAT, max_suggestions=cap, cfg=cfg)
    sug = res['suggestions']
    objs = [s for s in sug if s['kind'] == 'object']
    if not objs:
        return None
    biggest = max(s['px'] for s in objs) / (NAT.shape[0] * NAT.shape[1])
    return dict(recall=recall_of(sug), n=len(sug), nobj=len(objs),
                biggest=biggest,
                frags=sum(s.get('nfrag', 1) for s in objs) / len(objs),
                kinds=len({s['name'].rsplit('_', 1)[0] for s in objs}),
                ms=res['ms'], dropped=res['dropped'])


GRID = dict(
    min_core_px=[12, 25],
    pot_dir=['any', 'above'],
    merge_k=[1, 2],
    absorb_gap=[6, 8, 14],
)

if __name__ == '__main__':
    keys = list(GRID)
    rows = []
    for combo in itertools.product(*(GRID[k] for k in keys)):
        cfg = dict(zip(keys, combo))
        r = evaluate(cfg)
        if r is None:
            continue
        rows.append((cfg, r))
    rows.sort(key=lambda t: (-t[1]['recall'], t[1]['biggest']))
    print(f'{"recall":>7s} {"n":>3s} {"obj":>4s} {"big%":>6s} {"frags":>6s} {"drop":>5s} {"ms":>6s}  cfg')
    for cfg, r in rows:
        flag = 'PASS' if (r['recall'] >= 0.8 and r['n'] <= 60) else '    '
        print(f"{r['recall'] * 100:6.1f}% {r['n']:3d} {r['nobj']:4d} {r['biggest'] * 100:5.1f}% "
              f"{r['frags']:6.2f} {r['dropped']:5d} {r['ms']:6d}  {flag} "
              + ' '.join(f'{k}={cfg[k]}' for k in keys))
