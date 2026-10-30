"""Probe one region: why did lily-pad detection fail? Dumps the class composition of the
GT pads, the pool/water masks there, and every suggestion overlapping the pool bbox.
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import imaging as im          # noqa: E402
from engine import segment                # noqa: E402
import gt_stats                           # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
nat = np.asarray(Image.open(os.path.join(ROOT, 'native.png')))
seg, leg = gt_stats.load()
H, W = seg.shape

cls, names, d = segment.classify(nat, {})
print('classes:', names)

# GT pads + the pool quad area
pad_ids = [int(k) for k, v in leg.items() if v['name'].startswith('lily_pad') and (seg == int(k)).sum() >= 25]
padm = np.isin(seg, pad_ids)
ys, xs = np.nonzero(padm)
print(f'GT pads: ids={pad_ids} px={padm.sum()} bbox=({xs.min()},{ys.min()})-({xs.max()},{ys.max()})')

vals, cnts = np.unique(cls[padm], return_counts=True)
print('GT pad pixels by my class:',
      {names[v - 1] if v else 'none': f'{c * 100 // padm.sum()}%' for v, c in zip(vals, cnts)})

water = segment._mask(cls, names, 'water')
print(f'water class px={water.sum()}')
wcomps = im.components(water, 8, 8)
print('water comps (top 12): px, bbox, touches_border')
for c in wcomps[:12]:
    print(f"   {c['px']:6d} {c['bbox']} border={segment._touches_border(c['bbox'], (H, W))}")

# pool as the segmenter sees it
pool = np.zeros((H, W), bool)
for c in wcomps:
    if not segment._touches_border(c['bbox'], (H, W)) and c['px'] <= 0.30 * H * W:
        pool |= im.mask_from_runs(c['runs'], (H, W))
print(f'pool px={pool.sum()}  pad-in-pool fraction={pool[padm].mean():.2f} '
      f'(dilated3={im.dilate4(pool.copy(), 3)[padm].mean():.2f})')

# plant class inside the pool neighbourhood
plant = segment._mask(cls, names, 'plant')
pool_d = im.dilate4(pool.copy(), 3)
print(f'plant px total={plant.sum()}  plant px inside pool_d={(plant & pool_d).sum()}')
pc = im.close4(plant.copy(), 1)
for c in im.components(pc, 25, 8):
    m = im.mask_from_runs(c['runs'], (H, W))
    if not (m & pool_d).any():
        continue
    frac = pool_d[m].mean()
    print(f"   plant comp px={c['px']:5d} bbox={c['bbox']} in_pool={frac:.2f} "
          f"-> {'PAD' if frac >= 0.55 and c['px'] <= 2000 else 'plant'} "
          f"sheet={segment._is_sheet(c['px'], c['bbox'], H * W, d)}")

res = segment.auto_segment(nat, max_suggestions=10 ** 6)
print('\nsuggestions overlapping the pool bbox:')
for s in res['suggestions']:
    x0, y0, x1, y1 = s['bbox']
    if x1 >= xs.min() - 20 and x0 <= xs.max() + 20 and y1 >= ys.min() - 20 and y0 <= ys.max() + 20:
        cov = (s['mask'] & padm).sum() / padm.sum()
        print(f"   {s['name']:14s} {s['kind']:6s} px={s['px']:6d} bbox={s['bbox']} pad_cov={cov:.2f}")
