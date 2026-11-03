"""What do GT lily pads / pool water / plants actually look like in HSV?
Used to settle the green-vs-teal class boundary (pads were being eaten by `water`).
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import imaging as im          # noqa: E402
import gt_stats                           # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
nat = np.asarray(Image.open(os.path.join(ROOT, 'native.png')))
seg, leg = gt_stats.load()
H, W = seg.shape
hue, sat, val = im.hsv_maps(nat)
R, G, B = nat[..., 0].astype(int), nat[..., 1].astype(int), nat[..., 2].astype(int)


def report(label, m):
    if m.sum() < 5:
        return
    h, s, v = hue[m], sat[m], val[m]
    gb = (G - B)[m]
    print(f'{label:18s} n={m.sum():6d}  hue p5/p50/p95={np.percentile(h, 5):6.1f}/'
          f'{np.percentile(h, 50):6.1f}/{np.percentile(h, 95):6.1f}  sat p50={np.percentile(s, 50):.2f} '
          f' val p50={np.percentile(v, 50):.2f}  (G-B) p10/p50/p90='
          f'{np.percentile(gb, 10):5.0f}/{np.percentile(gb, 50):5.0f}/{np.percentile(gb, 90):5.0f}')


groups = {'pool_water': [], 'lily_pad': [], 'plant': [], 'floor_tiles': [], 'carpet': [], 'pot': []}
for k, v in leg.items():
    base = v['name'].rsplit('_', 1)[0]
    if base in groups and int((seg == int(k)).sum()) >= 25:
        groups[base].append(seg == int(k))
for base, ms in groups.items():
    if ms:
        m = np.logical_or.reduce(ms)
        report(base, m)
        # per-instance spread for pads
        if base == 'lily_pad':
            for i, mm in enumerate(ms):
                report(f'   pad inst {i}', mm)
