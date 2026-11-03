"""M7 acceptance test — depth-band layers for game-scene prep (plan §8b).

Synthetic ramp scene (exact answers) + the courtyard (regression) + the server
routes that build/serve/bundle the plates:
  M7 bands      1-D k-means cuts monotone, every pixel in exactly one band,
                deterministic for the same field + k
  M7 plates     far→near recomposition is PIXEL-EXACT on non-object pixels;
                farther bands stay transparent on nearer plates; nearer
                footprints are inpainted (disocclusion backing, opaque)
  M7 objects    band comes from the FEET (97th row percentile), continuous
                depth_n kept for y-sort inside a band
  M7 fused      with the accelerator installed: DA3 is the shipped source and
                the fused map scores rho >= 0.85 vs the origin depth.npy; the
                parse path stays the no-model fallback
  M7 server     POST /api/layers builds + caches; plates/preview/overlay are
                served; /api/layers_regions offers bands as regions; the game
                bundle ships layers/L*.png + depthmap.png + scene.json layers

Run:  python3 app/tests/test_m7.py
"""
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'app'))
from engine import depth, depth_ml, imaging as im, layers        # noqa: E402

PORT = int(os.environ.get('M7_TEST_PORT', 8147))
BASE = f'http://127.0.0.1:{PORT}'
FAILS = []


def check(name, ok, detail=''):
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'  — {detail}' if detail else ''))
    if not ok:
        FAILS.append(name)


def req(path, body=None, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    r = urllib.request.Request(BASE + path, data=data,
                               method='POST' if data is not None else 'GET',
                               headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(r, timeout=300) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def jreq(path, body=None):
    code, raw = req(path, body)
    return code, json.loads(raw.decode())


def ranks(a):
    a = np.asarray(a, np.float64).ravel()
    return np.argsort(np.argsort(a)).astype(np.float64)


def spear(a, b):
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


# ------------------------------------------------------------------ fixtures
def synth_scene():
    """64² three-strip scene with a far pole and a near pot as objects."""
    H, W = 64, 64
    nat = np.zeros((H, W, 3), np.uint8)
    nat[:] = (40, 60, 90)
    nat[32:] = (90, 120, 80)
    nat[48:] = (140, 110, 70)
    nat[16:40, 10:14] = (200, 200, 60)          # pole, feet row 39
    nat[40:56, 40:52] = (220, 90, 40)           # pot,  feet row 55
    d = np.clip(np.mgrid[0:H, 0:W][0] * 4, 0, 255).astype(np.uint8)
    pole = np.zeros((H, W), bool)
    pole[16:40, 10:14] = True
    pot = np.zeros((H, W), bool)
    pot[40:56, 40:52] = True
    objs = [{'id': 'a', 'name': 'pole', 'mask': pole},
            {'id': 'b', 'name': 'pot', 'mask': pot}]
    return nat, d, objs


def recompose_ok(nat, res, obj_any):
    c = layers.composite(res)
    return bool((c[~obj_any] == nat[~obj_any]).all())


# ------------------------------------------------------------------ engine
def main():
    print('M7 layers — banding math (synthetic ramp)')
    d = np.clip(np.mgrid[0:64, 0:64][0] * 4, 0, 255).astype(np.uint8)
    cuts, centers = layers.bands(d, 3)
    check('k=3 on a row ramp gives two monotone cuts', len(cuts) == 2 and
          cuts[0] < cuts[1], f'cuts={cuts} centers={centers}')
    check('banding is deterministic', layers.bands(d, 3)[0].tolist() == cuts.tolist())
    bm = layers.band_map(d, cuts)
    rm = layers.region_masks(bm, 3)
    check('every pixel lands in exactly one band',
          sum(int(m.sum()) for m in rm) == d.size and
          int((sum(m.astype(int) for m in rm) == 1).all()),
          f'px={[int(m.sum()) for m in rm]}')
    check('bands are contiguous in depth (row ramp => row strips)',
          all(bool((np.diff(np.nonzero(m.any(axis=1))[0]) == 1).all()) for m in rm))
    k6 = layers.bands(d, 9)[0]                       # clamped to CFG max
    check('k clamps to the configured range', len(k6) + 1 == layers.CFG['max_k'],
          f'k={len(k6) + 1}')

    print('\nM7 layers — plates + recomposition (synthetic scene)')
    nat, d, objs = synth_scene()
    res = layers.separate(nat, d, objects=objs, k=3)
    obj_any = res['obj_any']
    check('three plates, far to near, with pixel counts', res['k'] == 3 and
          all(b['plate'].shape == nat.shape[:2] + (4,) for b in res['bands']),
          f"px={[b['px'] for b in res['bands']]}")
    check('far→near recomposition is pixel-exact off objects',
          recompose_ok(nat, res, obj_any))
    far, near = res['bands'][0]['plate'], res['bands'][2]['plate']
    plate_free = ~obj_any            # object pixels carry backing on every plate
    check('nearer plates keep farther non-object pixels transparent',
          int(near[..., 3][(bm == 0) & plate_free].max()) == 0 and
          int(res['bands'][1]['plate'][..., 3][(bm == 0) & plate_free].max()) == 0,
          f"near-over-far alpha max="
          f"{int(near[..., 3][(bm == 0) & plate_free].max())}")
    check('nearer footprints are inpainted opaque on farther plates (backing)',
          int(far[..., 3][obj_any].min()) == 255 and
          int(far[..., 3][bm == 2].min()) == 255)
    ob = {o['id']: o for o in res['objects']}
    check('objects are banded at their feet (pole mid, pot near)',
          ob['a']['band'] == 1 and ob['b']['band'] == 2,
          f"pole band={ob['a']['band']} foot={ob['a']['foot_y']}, "
          f"pot band={ob['b']['band']} foot={ob['b']['foot_y']}")
    check('continuous foot depth kept for y-sort inside a band',
          0.0 <= ob['b']['depth_n'] <= 1.0 and
          ob['b']['depth_n'] > ob['a']['depth_n'],
          f"pole {ob['a']['depth_n']} pot {ob['b']['depth_n']}")
    cut = layers.separate(nat, d, objects=objs, k=3, fill=False)
    check('fill=False gives pure cut-out bands (no backing)',
          int(cut['bands'][0]['plate'][..., 3][obj_any].max()) == 0)

    print('\nM7 layers — courtyard regression (parse field, no model needed)')
    natp = os.path.join(ROOT, 'native.png')
    nat = np.asarray(Image.open(natp).convert('RGB'))
    fld = depth.field(nat)
    res = layers.separate(nat, fld['depth'], k=3)
    check('courtyard bands are all populated', res['k'] == 3 and
          all(b['px'] > 1000 for b in res['bands']),
          f"cuts={res['cuts']} px={[b['px'] for b in res['bands']]}")
    check('courtyard recomposition is pixel-exact (no objects)',
          bool((layers.composite(res) == nat).all()))
    ov = layers.overlay(nat, layers.band_map(fld['depth'], res['cuts']), res['k'])
    check('overlay preview is art-sized RGB', ov.shape == nat.shape)
    st = layers.strip(res)
    check('strip preview stacks the plates', st.shape[0] == nat.shape[0] and
          st.shape[1] > nat.shape[1] * 2, f'{st.shape}')

    print('\nM7 fused — DA3 accelerator (skips without the models installed)')
    src = depth_ml.source()
    if depth_ml.available():
        gt = np.load(os.path.join(ROOT, 'depth.npy')).astype(np.float64)
        u8, meta = depth_ml.fuse(nat, fld['depth'])
        check('the shipped source is DA3', src.startswith('da3'), src)
        check('fused depth agrees with the origin depth.npy (rho >= 0.85)',
              u8 is not None and spear(u8, gt) >= 0.85,
              f"rho={spear(u8, gt):.3f}" if u8 is not None else 'no fuse')
        if u8 is not None:
            r3 = layers.separate(nat, u8, k=3)
            check('fused-field layers recompose pixel-exactly',
                  bool((layers.composite(r3) == nat).all()))
    else:
        print('  [SKIP] no depth models installed — parse-only path is the rule')

    print('\nM7 server — build, serve, region-ify, bundle')
    srv = subprocess.Popen([sys.executable, 'server.py', str(PORT)],
                           cwd=os.path.join(ROOT, 'app'),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(BASE + '/api/ping', timeout=2)
                break
            except OSError:
                time.sleep(0.25)
        with open(natp, 'rb') as fh:
            code, raw = req('/api/upload', raw=fh.read())
        doc = json.loads(raw.decode())
        check('upload ok', code == 200, str(code))
        pid = doc['id']
        t0 = time.time()
        code, meta = jreq('/api/layers', {'pid': pid, 'k': 3})
        dt = time.time() - t0
        check('POST /api/layers builds three bands', code == 200 and
              meta.get('k') == 3 and len(meta.get('bands', [])) == 3,
              f"{code} source={meta.get('source')} {dt:.1f}s")
        check('layer meta names the depth source', bool(meta.get('source')),
              str(meta.get('source')))
        pdir = os.path.join(ROOT, 'app', 'projects', pid)
        check('plates + previews written to the project dir',
              all(os.path.exists(os.path.join(pdir, f)) for f in
                  ('layers/L0.png', 'layers/L2.png', 'layers/bands.png',
                   'layers_preview.png', 'layers_overlay.png', 'layers.json')))
        code, _r = jreq('/api/layers', {'pid': pid, 'k': 3})
        check('same field + k answers from cache', _r.get('cached') is True, str(code))
        for f in ('layers/L0.png', 'layers_preview.png', 'layers_overlay.png'):
            code, raw = req(f'/api/image/{pid}/{f}')
            check(f'GET /api/image/<pid>/{f}', code == 200 and
                  raw[:8] == b'\x89PNG\r\n\x1a\n', str(code))
        code, raw = req(f'/api/image/{pid}/layers/L7.png')
        check('unknown plate index 404s', code == 404, str(code))
        code, reg = jreq('/api/layers_regions', {'pid': pid})
        check('bands offered as region suggestions', code == 200 and
              len(reg.get('suggestions', [])) == 3 and
              all(s['kind'] == 'region' for s in reg['suggestions']),
              f"{code} n={len(reg.get('suggestions', []))}")
        if code == 200 and reg.get('suggestions'):
            s0 = reg['suggestions'][0]
            m = im.rle_decode(s0['rle'], (doc['native_h'], doc['native_w']))
            check('region rle decodes to the band mask', int(m.sum()) == s0['px'],
                  f"px={s0['px']}")
            code2, _m = jreq('/api/masks', {'pid': pid, 'ops': [
                {'op': 'upsert', 'id': s0['id'], 'name': s0['name'],
                 'kind': 'region', 'color': s0['color'], 'rle': s0['rle']}]})
            check('a band imports into the mask list like any suggestion',
                  code2 == 200 and any(o['id'] == s0['id'] for o in _m['objects']),
                  str(code2))
        code, bun = jreq('/api/bundle', {'pid': pid, 'N': 8, 'fps': 12})
        check('bundle builds with layers', code == 200 and bun.get('layers') == 3,
              f"{code} layers={bun.get('layers')}")
        code, raw = req(f'/api/out/{pid}/bundle.zip')
        z = zipfile.ZipFile(io.BytesIO(raw))
        names = set(z.namelist())
        check('bundle zip ships the layer plates + depthmap',
              {'layers/L0.png', 'layers/L1.png', 'layers/L2.png',
               'depthmap.png'} <= names, f'{len(names)} files')
        scene = json.loads(z.read('scene.json').decode())
        check('scene.json carries the layer contract',
              scene.get('layers', {}).get('k') == 3 and
              scene['layers']['bands'][0]['file'] == 'layers/L0.png' and
              scene.get('depthmap', {}).get('file') == 'depthmap.png')
        code, bun2 = jreq('/api/bundle', {'pid': pid, 'N': 8, 'fps': 12,
                                          'layers': 0})
        code2, raw2 = req(f'/api/out/{pid}/bundle.zip')
        z2 = zipfile.ZipFile(io.BytesIO(raw2))
        check('layers=0 omits them (opt-out holds)', bun2.get('layers') == 0 and
              not any(n.startswith('layers/') for n in z2.namelist()))
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=10)
        except subprocess.TimeoutExpired:
            srv.kill()

    print('\n%s' % ('ALL PASS' if not FAILS else f'FAILURES: {FAILS}'))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
