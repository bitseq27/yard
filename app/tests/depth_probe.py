"""M6a dev probe — is the depth/orientation field faithful, and does picking the
fill axis from it actually fill better?

Run:  python3 app/tests/depth_probe.py

1. field on the courtyard, compared against the origin session's depth.npy
2. detected walls listed with bboxes
3. punch-hole reconstruction: boxes punched into clean wall/floor surface, refilled
   with each strategy, error against the real art that was underneath
4. writes app/tests/depth_debug.png (closeness luminance + orientation tint)
"""
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'app'))
sys.path.insert(0, os.path.join(ROOT, 'app', 'engine'))

from engine import animate, depth, imaging as im          # noqa: E402

HOLE = (14, 22)          # (h, w) punched boxes


def punch(nat, cy, cx):
    h, w = HOLE
    m = np.zeros(nat.shape[:2], bool)
    m[cy - h // 2:cy - h // 2 + h, cx - w // 2:cx - w // 2 + w] = True
    return m


def fills(nat, hole, orient):
    out = {}
    p = np.array(nat)
    animate._hfill(p, hole)
    out['h'] = p.copy()
    p2 = np.array(nat)
    lens = animate._hfill(p2, hole)
    bi, ok = animate._bifill(nat, hole & (lens > animate.VBLEND_LONG_RUN))
    p2[hole & (lens > animate.VBLEND_LONG_RUN) & ok] = \
        bi[hole & (lens > animate.VBLEND_LONG_RUN) & ok]
    out['m4'] = p2
    p3 = np.array(nat)
    animate._vfill(p3, hole)
    out['v'] = p3
    p4 = np.array(nat)
    if depth.hole_axis(orient, hole) == depth.WALL:
        animate._vfill(p4, hole)
    else:
        animate._hfill(p4, hole)
        bi, ok = animate._bifill(nat, hole)
        p4[hole & ok] = bi[hole & ok]
    out['axis'] = p4
    return out


def err(a, b, m):
    return float(np.abs(a[m].astype(int) - b[m].astype(int)).mean())


def sample_boxes(mask, n=8, shrink=5):
    """Box centres well inside `mask` (punched holes keep their boundary on-mask)."""
    m = im.erode4(mask.copy(), shrink)
    out = []
    for c in im.components(m, HOLE[0] * HOLE[1] * 2, 8):
        x0, y0, x1, y1 = c['bbox']
        h, w = HOLE
        ys = range(y0 + h // 2 + 1, y1 - h // 2, max(1, (y1 - y0) // 3))
        xs = range(x0 + w // 2 + 1, x1 - w // 2, max(1, (x1 - x0) // 3))
        for y in ys:
            for x in xs:
                out.append((y, x))
                if len(out) >= n:
                    return out
    return out


def main():
    nat = np.asarray(Image.open(os.path.join(ROOT, 'native.png'))).astype(np.uint8)
    objs = []
    if os.path.exists(os.path.join(ROOT, 'masks.npy')):
        M = np.load(os.path.join(ROOT, 'masks.npy'))
        objs = [M[0], M[1]]
    f = depth.field(nat, objects=objs)
    print(f"field {nat.shape} walls={f['walls_px']} ms={f['ms']}")
    if f['horizon'] is not None:
        h = f['horizon']
        print(f"   horizon cols: min={h.min():.0f} median={np.median(h):.0f} "
              f"max={h.max():.0f}  (rows above it are standing surface)")

    gt = os.path.join(ROOT, 'depth.npy')
    if os.path.exists(gt):
        o = np.load(gt).astype(np.float32)
        a = f['depth'].astype(np.float32)
        # both are min-max normalised closeness: compare ranks, not absolute bits
        ra = np.argsort(np.argsort(a.ravel())).reshape(a.shape).astype(np.float64)
        ro = np.argsort(np.argsort(o.ravel())).reshape(o.shape).astype(np.float64)
        rho = np.corrcoef(ra.ravel(), ro.ravel())[0, 1]
        print(f'  origin depth.npy: spearman rho={rho:.3f}  mean|delta|='
              f'{np.abs(a - o).mean():.1f}/255')

    wallm = f['orient'] == depth.WALL
    floor = (f['orient'] == depth.GROUND)
    print(f'  orient px: wall={int(wallm.sum())} ground={int(floor.sum())} '
          f'object={int((f["orient"] == depth.OBJECT).sum())} '
          f'none={int((f["orient"] == depth.NONE).sum())}')

    print('\npunch-hole reconstruction error (mean abs vs the art underneath)')
    for label, region in (('wall', wallm), ('floor', floor)):
        boxes = sample_boxes(region)
        if not boxes:
            print(f'  {label}: no clean sample area')
            continue
        tot = {k: 0.0 for k in ('h', 'm4', 'v', 'axis')}
        for cy, cx in boxes:
            hole = punch(nat, cy, cx)
            for k, p in fills(nat, hole, f['orient']).items():
                tot[k] += err(p, nat, hole)
        n = len(boxes)
        print(f'  {label} n={n}: ' + '  '.join(f'{k}={v / n:.2f}' for k, v in tot.items()))

    Image.fromarray(depth.render_debug(f)).resize(
        (nat.shape[1] * 2, nat.shape[0] * 2), Image.NEAREST).save(
        os.path.join(ROOT, 'app', 'tests', 'depth_debug.png'))
    print('wrote app/tests/depth_debug.png')


if __name__ == '__main__':
    main()
