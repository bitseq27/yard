"""M6e depth bake-off — the M6a closeness field reads FLAT next to the origin's
depth.npy (smooth plates + cut-out silhouettes vs dense facade relief), so this
harness scores alternative depth solutions against that ground truth:

  classical   m6a (shipped baseline) · iso3 (per-pixel iso-correct height ramp)
              · sfs (shape-from-shading: Poisson-integrated M6c normals)
              · lum (box-mean luminance, the naive proxy)
              · hybrid λ (m6a or iso3 + λ·high-pass(sfs): layout from the parse,
                relief from the shading)
  models      Depth Anything V2 small (onnx, fp32 + int8), MiDaS v3.1 small if
              present — evaluation only, NEVER in the engine (numpy+PIL rule)

Metrics vs depth.npy (higher = nearer for all of them):
  rho      global spearman                      conc    pairwise order concordance
  rho_in   mean within-object spearman (does the INSIDE of an object vary right?)
  gradrho  spearman of |grad| maps (is detail where the truth has detail?)
  wallrho  spearman on the facade band (rows above the median horizon)
  wstd     mean within-object std (the flatness complaint; gt = 12.8)
  ms       compute time

Run:  python3 app/tests/depth_alts.py [--models] [--montage OUT.png]
"""
import os
import sys
import time

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'app'))
from engine import depth, imaging as im          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ORT = os.path.join(HERE, '_ort')
MODELS = os.path.join(HERE, 'models')
SEED = 6


# ---------------------------------------------------------------- helpers
def ranks(a):
    a = np.asarray(a, np.float64).ravel()
    return np.argsort(np.argsort(a)).astype(np.float64)


def spear(a, b):
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def box(a, r):
    return depth._box_mean(np.asarray(a, np.float32), r)


def poisson(gx, gy):
    """Periodic least-squares integration of a gradient field (numpy FFT only)."""
    H, W = gx.shape
    div = (np.roll(gx, -1, axis=1) - gx) + (np.roll(gy, -1, axis=0) - gy)
    F = np.fft.rfft2(div)
    xx = np.fft.rfftfreq(W)[None, :]
    yy = np.fft.fftfreq(H)[:, None]
    lap = 2 * (np.cos(2 * np.pi * xx) - 1) + 2 * np.cos(2 * np.pi * yy) - 2
    lap[0, 0] = 1.0
    Z = F / lap
    Z[0, 0] = 0.0
    z = np.fft.irfft2(Z, s=(H, W))
    return z - z.mean()


def sfs_height(nat, bump=3.2):
    """Shape-from-shading: integrate the M6c bump normals back into a height field."""
    nm = depth.normal_map(nat, bump=bump).astype(np.float64) / 255.0 * 2.0 - 1.0
    nz = np.maximum(nm[..., 2], 0.2)
    hx = -nm[..., 0] / nz          # encoding is n ∝ (-dh/dx, dh/dy, 1)
    hy = nm[..., 1] / nz
    return poisson(hx, hy)


def highpass(z, r=9):
    return z - box(z, r)


def znorm(a):
    a = np.asarray(a, np.float64)
    return (a - a.mean()) / (a.std() + 1e-9)


# ---------------------------------------------------------------- models
def _ort_session(path):
    if not os.path.isdir(ORT):
        return None
    sys.path.insert(0, ORT)
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.inter_op_num_threads = 4
    so.intra_op_num_threads = 4
    return ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])


def _prep(img, size, mean, std):
    a = img.resize((size, size), Image.BILINEAR)
    a = np.asarray(a, np.float32) / 255.0
    a = (a - np.array(mean, np.float32)) / np.array(std, np.float32)
    return np.transpose(a, (2, 0, 1))[None].astype(np.float32)


IMNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def run_model(path, nat, size=518, norm=IMNET):
    sess = _ort_session(path)
    if sess is None:
        return None
    inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
    img = Image.fromarray(nat).convert('RGB')
    x = _prep(img, size, *norm)
    t0 = time.time()
    y = sess.run([out.name], {inp.name: x})[0]
    ms = (time.time() - t0) * 1000
    d = np.asarray(y[0])
    if d.ndim == 3:
        d = d[None]
    d = d[0]
    d = np.asarray(Image.fromarray(d).resize((nat.shape[1], nat.shape[0]),
                                             Image.BILINEAR), np.float64)
    return d, ms


# ---------------------------------------------------------------- candidates
def candidates(nat, f, with_models, seg=None):
    base = f['base_y'].astype(np.float64)
    ys = np.mgrid[0:nat.shape[0], 0:nat.shape[1]][0]
    m6a = f['depth'].astype(np.float64)
    iso3 = 3.0 * base - ys
    t0 = time.time(); z = sfs_height(nat); ms_sfs = (time.time() - t0) * 1000
    lum = box(np.asarray(Image.fromarray(nat).convert('L'), np.float64), 2)
    hp = highpass(z)
    L = lambda a: a / (a.std() + 1e-9)          # std-normalised part
    out = [('m6a (shipped)', m6a, 0.0, 'ground-row closeness, objects flat at feet'),
           ('iso3', iso3, 0.0, 'per-pixel iso depth 3*base_y - y (height ramp inside objects)'),
           ('sfs', z, 0.0, 'shape-from-shading: Poisson-integrated M6c normals'),
           ('lum', lum, 0.0, 'box-mean luminance (naive height proxy)')]
    for lam in (0.5, 1.0, 2.0, 4.0):
        out.append((f'm6a+{lam:g}*hp(sfs)', L(m6a) * 40 + lam * L(hp) * 8, ms_sfs,
                    f'layout m6a + {lam:g}x shading relief'))
    da = mid = None
    if with_models:
        for name, fn, kw in (('midas-v21-small', 'midas_v21_small.onnx', {'size': 256}),
                             ('da-v2-small-int8', 'da_v2_small_quant.onnx', {}),
                             ('da-v2-small', 'da_v2_small.onnx', {})):
            pth = os.path.join(MODELS, fn)
            if not os.path.exists(pth):
                continue
            r = run_model(pth, nat, **kw)
            if not r:
                continue
            d, ms = r
            # orient without peeking at gt: in iso art depth grows toward the bottom
            if np.corrcoef(d.ravel(), ys.ravel())[0, 1] < 0:
                d = -d
            out.append((name, d, ms, 'onnx monocular depth, evaluation only'))
            if name == 'da-v2-small-int8':
                da = d
            if name == 'midas-v21-small':
                mid = d
        if da is not None:
            hpd = highpass(da)
            for w in (0.3, 0.5, 0.7):
                out.append((f'fuse{w:g}', w * znorm(da) + (1 - w) * znorm(m6a), 0.0,
                            f'rank-free blend: {w:g} DA + {1-w:g} m6a (std-normalised)'))
            for lam in (0.5, 1.0, 2.0):
                out.append((f'm6a+{lam:g}*hp(DA)', L(m6a) * 40 + lam * L(hpd) * 8, 0.0,
                            f'm6a layout + {lam:g}x DA relief (facade detail)'))
            for lam in (0.5, 1.0):
                out.append((f'iso3+{lam:g}*hp(DA)', L(iso3) * 40 + lam * L(hpd) * 8, 0.0,
                            f'iso3 layout + {lam:g}x DA relief'))
            if mid is not None:
                hpm = highpass(mid)
                for w in (0.5, 0.7):
                    out.append((f'fuseM{w:g}', w * znorm(mid) + (1 - w) * znorm(m6a), 0.0,
                                f'blend: {w:g} MiDaS + {1-w:g} m6a (fast model path)'))
                for lam in (1.0, 2.0):
                    out.append((f'm6a+{lam:g}*hp(MiDaS)', L(m6a) * 40 - lam * L(hpm) * 8,
                                0.0, f'm6a layout + {lam:g}x MiDaS relief (sign-flipped: '
                                     'its facade relief correlates negative)'))
            if seg is not None:
                dof = da.copy()
                for lab in range(1, int(seg.max()) + 1):
                    mk = seg == lab
                    if mk.sum() < 150:
                        continue
                    dof[mk] += np.median(m6a[mk]) - np.median(da[mk])
                out.append(('DA+feet-offset', dof, 0.0,
                            'DA layout, per-mask feet offset from the parse (runtime-realistic)'))
                out.append(('DA+feet+hp(sfs)', dof + 1.0 * L(hp) * 8 / (L(dof).std() + 1e-9) * 0,
                            0.0, 'placeholder'))
                out.pop()
    return out


# ---------------------------------------------------------------- scoring
def score(name, d, gt, seg, horizon, ms):
    d = np.asarray(d, np.float64)
    rho = spear(d, gt)
    flip = rho < 0
    if flip:
        d = -d
        rho = -rho
    rng = np.random.RandomState(SEED)
    n = 200000
    i = rng.randint(0, gt.size, n)
    j = rng.randint(0, gt.size, n)
    keep = np.abs(gt.ravel()[i] - gt.ravel()[j]) > 8
    i, j = i[keep], j[keep]
    dv = d.ravel()
    conc = float(np.mean(np.sign(dv[i] - dv[j]) == np.sign(gt.ravel()[i] - gt.ravel()[j])))
    rin, cnt = [], 0
    for lab in range(1, int(seg.max()) + 1):
        mk = seg == lab
        if mk.sum() < 400 or gt[mk].std() < 6:
            continue
        if d[mk].std() < 1e-9:
            rin.append(0.0); cnt += 1; continue
        rin.append(spear(d[mk], gt[mk])); cnt += 1
    gd = np.hypot(*np.gradient(d))
    gg = np.hypot(*np.gradient(gt))
    wall = ys_band(gt.shape, horizon)
    wv = [d[seg == lab].std() for lab in range(1, int(seg.max()) + 1)
          if (seg == lab).sum() >= 150]
    wstd = float(np.mean(wv) / (d.std() + 1e-9))
    return dict(name=name, rho=rho, flip=flip, conc=conc,
                rin=float(np.mean(rin)) if rin else float('nan'), nobj=cnt,
                gradrho=spear(gd, gg), wallrho=spear(d[wall], gt[wall]),
                wstd=float(wstd), ms=ms)


def ys_band(shape, horizon):
    ys = np.mgrid[0:shape[0], 0:shape[1]][0]
    return ys < int(np.median(horizon))


def main():
    with_models = '--models' in sys.argv
    mont = None
    if '--montage' in sys.argv:
        mont = sys.argv[sys.argv.index('--montage') + 1]
    nat = np.asarray(Image.open(os.path.join(ROOT, 'native.png'))).astype(np.uint8)
    gt = np.load(os.path.join(ROOT, 'depth.npy')).astype(np.float64)
    seg = np.load(os.path.join(ROOT, 'seg.npy'))
    M = np.load(os.path.join(ROOT, 'masks.npy'))
    f = depth.field(nat, objects=[M[0], M[1]])
    print('courtyard %dx%d, gt std %.1f, field %d ms'
          % (nat.shape[0], nat.shape[1], gt.std(), f['ms']))
    rows, maps = [], {}
    for name, d, ms, note in candidates(nat, f, with_models, seg):
        s = score(name, d, gt, seg, f['horizon'], ms)
        s['note'] = note
        rows.append(s)
        maps[name] = np.asarray(d, np.float64)
        print('  %-18s rho %+.3f%s conc %.3f  rho_in %+.3f (n=%d)  gradrho %+.3f  '
              'wallrho %+.3f  wflat %.2f  %6.0f ms'
              % (name, s['rho'], '*' if s['flip'] else ' ', s['conc'], s['rin'],
                 s['nobj'], s['gradrho'], s['wallrho'], s['wstd'], s['ms']))
    gtv = [gt[seg == lab].std() for lab in range(1, int(seg.max()) + 1)
           if (seg == lab).sum() >= 150]
    print('\n  gt reference     wflat %.2f  (within-object std / overall std;'
          ' gt raw wstd %.1f)' % (np.mean(gtv) / gt.std(), np.mean(gtv)))
    best = max(rows, key=lambda r: (r['rho'] + r['conc'] + r['rin'] + r['gradrho']))
    print('\nbest combined: %s' % best['name'])
    if mont:
        comb = lambda r: r['rho'] + r['conc'] + r['rin'] + r['gradrho']
        order = [r['name'] for r in sorted(rows, key=lambda r: -comb(r))[:6]]
        panels = [nat, gt] + [maps[k] for k in order]
        normed = []
        for p in panels:
            p = np.asarray(p, np.float64)
            p = (p - p.min()) / (p.ptp() + 1e-9) * 255
            normed.append(np.repeat(p[..., None].astype(np.uint8), 3, 2)
                          if p.ndim == 2 else p.astype(np.uint8))
        gap = np.full((nat.shape[0], 4, 3), 255, np.uint8)
        strip = []
        for i, p in enumerate(normed):
            if i:
                strip.append(gap)
            strip.append(p)
        Image.fromarray(np.concatenate(strip, 1)).save(mont)
        print('montage ->', mont, ' panels: art, gt, ' + ', '.join(order))
    return rows


if __name__ == '__main__':
    main()
