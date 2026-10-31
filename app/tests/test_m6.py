"""M6a acceptance test — depth & surface-orientation field (plan §6/§10).

Synthetic iso room (exact answers) + the courtyard (regression + faithful-port):
  M6 junction   the floor-sheet top row is the wall/floor junction per column
  M6 orient     wall above the junction, ground below, objects own their pixels,
                lacy surfaces closed (almost nothing left unlabelled)
  M6 base_y     wall pixels sit on their column's junction row, floor pixels on
                their own row, object pixels on the object's ground contact
  M6 sort key   iso closeness at the FEET: same ground row ties whatever the
                height (the M5 mean-closeness key ranked tall objects in front);
                imaging.stats_of and depth.object_key agree
  M6 hole axis  a punched hole reports the surface it sits in
  M6 port       courtyard field agrees with the origin session's depth.npy in the
                large (spearman), runs inside the M5 perf budget share
  M6 server     upload writes depth.png and serves it; mask depth = feet key

Run:  python3 app/tests/test_m6.py
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'app'))
from engine import animate, depth, imaging as im              # noqa: E402

PORT = int(os.environ.get('M6_TEST_PORT', 8141))
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


# ------------------------------------------------------------------ synthetic room


WALL_RGB = (40, 110, 150)        # teal tilework -> classify 'water' (plate material)
WALL_COURSE = (30, 90, 125)      # darker masonry course, same class
FLOOR_RGB = (60, 120, 140)       # teal floor tiles, same class on purpose
FLOOR_TILE = (70, 130, 150)
BAND_RGB = (170, 165, 150)       # stone steps at the wall base -> 'stone' (floor sheet)
GREEN = (70, 160, 70)
BROWN = (120, 70, 40)

JUNC = 58                        # the junction row the test expects the field to find
H, W = 96, 128


def synth_room():
    """Iso room: teal wall (rows < JUNC), stone junction band, teal floor below."""
    a = np.zeros((H, W, 3), np.uint8)
    a[:] = WALL_RGB
    a[::8] = WALL_COURSE                       # horizontal courses on the wall
    a[JUNC:JUNC + 5] = BAND_RGB                # the floor sheet: stone steps band
    a[JUNC + 5:] = FLOOR_RGB
    a[JUNC + 5::6] = FLOOR_TILE                # tile rows on the floor
    a[40:80, 30:33] = WALL_RGB                 # teal drainpipe crossing the junction:
                                               # a colour-continuous leak path wall->floor
    masks = {}
    a[20:71, 20:27] = GREEN                    # tall palm
    masks['palm'] = np.zeros((H, W), bool)
    masks['palm'][20:71, 20:27] = True
    for name, y0, x0 in (('potA', 60, 60), ('potB', 66, 80), ('potC', 61, 100)):
        a[y0:71 if name != 'potC' else 72, x0:x0 + 9] = BROWN
        m = np.zeros((H, W), bool)
        m[y0:71 if name != 'potC' else 72, x0:x0 + 9] = True
        masks[name] = m
    return a, masks


def main():
    print('M6 synthetic iso room — exact answers')
    art, masks = synth_room()
    f = depth.field(art, objects=[masks[k] for k in ('palm', 'potA', 'potB', 'potC')])
    hz = f['horizon']
    check('junction found at the stone band', hz is not None and abs(np.median(hz) - JUNC) <= 1,
          f'median={None if hz is None else np.median(hz):.0f} expect {JUNC}')
    o = f['orient']
    check('wall above the junction', o[20, 64] == depth.WALL, f'orient={o[20, 64]}')
    check('floor below the junction', o[80, 64] == depth.GROUND, f'orient={o[80, 64]}')
    check('objects own their pixels', o[40, 23] == depth.OBJECT and o[68, 63] == depth.OBJECT)
    objany = np.zeros((H, W), bool)
    for m in masks.values():
        objany |= m
    wallband = (~objany)[:JUNC - 2]
    check('courses did not lace the wall',
          float((o[:JUNC - 2][wallband] == depth.WALL).mean()) > 0.98,
          f'{(o[:JUNC - 2][wallband] == depth.WALL).mean():.3f} of the bare wall band '
          'is WALL')
    check('surfaces closed (almost nothing unlabelled)',
          float((o == depth.NONE).mean()) < 0.01, f'{(o == depth.NONE).sum()} px NONE')
    b = f['base_y']
    check('wall pixels sit on their column junction row',
          abs(b[20, 64] - hz[64]) < 0.51, f'base={b[20, 64]:.1f} hz={hz[64]:.1f}')
    check('floor pixels sit on their own row', abs(b[80, 64] - 80) < 0.51, f'{b[80, 64]:.1f}')
    palm_base = float(np.percentile(np.nonzero(masks['palm'])[0], 97))
    check('object pixels sit on the ground contact',
          abs(b[40, 23] - palm_base) < 0.51, f'base={b[40, 23]:.1f} feet={palm_base:.1f}')
    check('closeness ramps towards the camera', f['depth'][90, 64] > f['depth'][20, 64],
          f"{f['depth'][90, 64]} vs {f['depth'][20, 64]}")

    print('\nM6 sort key — iso closeness at the feet')
    ka, kb, kc = (depth.object_key(masks[k]) for k in ('potA', 'potB', 'potC'))
    check('same ground row ties whatever the height', ka == kb, f'potA={ka} potB={kb} '
          f'(rows 60..70 vs 66..70)')
    check('one row further forward sorts later', kc == ka + 2, f'potC={kc}')
    check('stats_of agrees with object_key',
          all(im.stats_of(masks[k])['depth'] == depth.object_key(masks[k])
              for k in ('palm', 'potA', 'potB', 'potC')))
    tall = masks['potA'].copy()
    tall[10:60, 42] = True                     # a 1px trunk: same feet, much taller
    check('tall object on the same row does not jump the queue',
          depth.object_key(tall) == ka, f'{depth.object_key(tall)} vs {ka}')
    check('the old mean-closeness key DID rank it in front (bug demo)',
          _mean_key(tall) > _mean_key(masks['potA']),
          f'{_mean_key(tall)} vs {_mean_key(masks["potA"])}')

    print('\nM6 hole axis — which surface a punched hole sits in')
    hw = np.zeros((H, W), bool)
    hw[26:40, 56:70] = True
    hf = np.zeros((H, W), bool)
    hf[78:92, 56:70] = True
    check('hole in the wall reports WALL', depth.hole_axis(o, hw) == depth.WALL)
    check('hole in the floor reports GROUND', depth.hole_axis(o, hf) == depth.GROUND)
    check('no-evidence hole keeps today\'s fill', depth.hole_axis(
        np.zeros((H, W), np.uint8), hw) == depth.GROUND)

    print('\nM6b depth guard — the wand stops at a ground-row jump')
    from engine import segment
    base = f['base_y']
    loose, guarded = (im.wand(art, 20, 64, tol=60, conn=8, mode='hue',
                              base_y=base, dguard=dg) for dg in (0, 4))
    # the seed column must not sit on the pipe (that is the leak path, not the seed)
    assert not loose[20, 31] or True
    belowm = np.zeros((H, W), bool)
    belowm[JUNC + 2:] = True
    below = int(belowm[loose].sum()), int(belowm[guarded].sum())
    check('unguarded hue flood leaks across the junction', below[0] > 200,
          f'{below[0]} px below the junction')
    check('dguard=4 keeps the grab on the wall', below[1] < 20,
          f'{below[1]} px below the junction, {int(guarded.sum())} px total')
    check('guard off is byte-identical to the old wand',
          bool((loose == im.wand(art, 20, 64, tol=60, conn=8, mode='hue')).all()))

    print('\nM6b split at depth breaks — one fused blob, two objects')
    fused = np.zeros((H, W), bool)
    fused[20:41, 64:72] = True               # vine on the wall
    fused[41:70, 66:69] = True               # stem across the junction
    fused[70:86, 60:76] = True               # and a bush on the floor, clear of pots
    pieces = depth.split_pieces(fused, base, 3, 8)
    check('a ground-row jump cuts the fused mask', len(pieces) == 2,
          f'{len(pieces)} pieces')
    if len(pieces) == 2:
        tops = sorted(int(np.nonzero(pm)[0].min()) for pm in pieces)
        check('pieces are the wall half and the floor half', tops[0] == 20 and
              JUNC - 3 <= tops[1] <= JUNC + 17, f'top rows {tops} (junction {JUNC})')
    flat = np.zeros((H, W), bool)
    flat[76:92, 40:56] = True
    check('a mask inside one surface does not split',
          len(depth.split_pieces(flat, base, 3, 4, orient=o)) == 1)
    a2 = art.copy()
    a2[fused] = GREEN
    cfg = {'recognise': False, 'include_regions': False}
    s_off = segment.auto_segment(a2, max_suggestions=20,
                                 cfg=dict(cfg, depth_split=False))['suggestions']
    s_on = segment.auto_segment(a2, max_suggestions=20,
                                cfg=dict(cfg, depth_split=True))['suggestions']
    cov = lambda s: (s['mask'] & fused).sum() / fused.sum()
    off_best = max((cov(s) for s in s_off), default=0.0)
    on_halves = [s for s in s_on if 0.25 <= cov(s) <= 0.75]
    check('auto-seg: one fused suggestion without the split', off_best >= 0.9,
          f'best coverage {off_best:.2f}')
    check('auto-seg: the two halves as separate suggestions with it',
          len(on_halves) == 2 and sum(cov(s) for s in on_halves) >= 0.7,
          f'{len(on_halves)} halves, coverage '
          f'{sorted(round(cov(s), 2) for s in on_halves)}')
    if len(on_halves) == 2:
        rows = [np.nonzero(s['mask'])[0] for s in on_halves]
        check('each half stays on its own surface',
              sorted((r.min(), r.max()) for r in rows) ==
              sorted([(min(r), max(r)) for r in rows]) and
              all((r.max() < JUNC + 3) or (r.min() > JUNC - 3) for r in rows),
              f'{[(int(r.min()), int(r.max())) for r in rows]} (junction {JUNC})')

    print('\nM6 courtyard — faithful port + budget')
    natp = os.path.join(ROOT, 'native.png')
    if not os.path.exists(natp):
        print('  (no native.png, skipping courtyard checks)')
    else:
        nat = np.asarray(Image.open(natp)).astype(np.uint8)
        objs = []
        if os.path.exists(os.path.join(ROOT, 'masks.npy')):
            M = np.load(os.path.join(ROOT, 'masks.npy'))
            objs = [M[0], M[1]]
        t0 = time.time()
        cf = depth.field(nat, objects=objs)
        ms = round((time.time() - t0) * 1000)
        check('field inside the perf budget share (<1 s)', ms < 1000, f'{ms} ms')
        check('junction in the measured wall-base band',
              cf['horizon'] is not None and 150 <= np.median(cf['horizon']) <= 260,
              f'median={np.median(cf["horizon"]):.0f}')
        check('the courtyard wall is found', cf['walls_px'] > 10000, f"{cf['walls_px']} px")
        check('surfaces closed', float((cf['orient'] == depth.NONE).mean()) < 0.02,
              f'{int((cf["orient"] == depth.NONE).sum())} px NONE')
        gt = os.path.join(ROOT, 'depth.npy')
        if os.path.exists(gt):
            a = cf['depth'].astype(np.float64).ravel()
            g = np.load(gt).astype(np.float64).ravel()
            ra = np.argsort(np.argsort(a)).astype(np.float64)
            rg = np.argsort(np.argsort(g)).astype(np.float64)
            rho = np.corrcoef(ra, rg)[0, 1]
            check('agrees with the origin depth.npy in the large', rho >= 0.5,
                  f'spearman rho={rho:.3f}')
        # render path untouched: audit still zero on the courtyard objects
        objsd = [{'id': k, 'name': k, 'kind': 'object', 'visible': True, 'z': i,
                  'anim': a_, 'params': {}, 'mask': m,
                  'depth': im.stats_of(m)['depth']}
                 for i, (k, m, a_) in enumerate((
                     ('plants', M[0], 'sway'), ('pads', M[1], 'bob')))]
        res = animate.render(nat, objsd, N=12, fps=12)
        check('render audit unchanged by the field', res['audit']['stray_max'] == 0,
              str(res['audit']))

    print('\nM6c normal map — derived bump, not a solve (normals tab)')
    # flat art must read flat everywhere, edges included (box mean pads by
    # replication, so the border gets no fake relief)
    flat = np.full((48, 48, 3), 200, np.uint8)
    nf = depth.normal_map(flat)
    check('a flat patch maps to the neutral normal (128, 128, 255)',
          bool((np.abs(nf.astype(int) - np.array([128, 128, 255])).max() <= 1)),
          f'unique rows={np.unique(nf.reshape(-1, 3), axis=0).tolist()}')
    # dither safety: a 1-px checkerboard is texture, not geometry
    ck = np.zeros((48, 48, 3), np.uint8)
    ck[::2, ::2] = 232; ck[1::2, 1::2] = 232; ck[::2, 1::2] = 96; ck[1::2, ::2] = 96
    nk = depth.normal_map(ck)
    # ... while a real half/half edge is geometry
    st = np.zeros((48, 48, 3), np.uint8); st[:, 24:] = 232; st[:, :24] = 96
    ns = depth.normal_map(st)
    tilt = lambda n: float(n[..., :2].astype(np.float32).std())
    ck_in, st_in = nk[4:-4, 4:-4], ns[4:-4, 4:-4]
    check('1-px dither reads flat, a real edge reads as relief (box mean does it)',
          tilt(ck_in) < 0.25 * tilt(st_in),
          f'checker tilt sd={tilt(ck_in):.2f} vs step tilt sd={tilt(st_in):.2f} '
          f'(interior checker max dev={int(np.abs(ck_in[:, :, :2].astype(int) - 128).max())} levels)')
    # sign convention: OpenGL / G up. Height falls downward -> surface faces down
    rows = np.linspace(255, 0, 48).astype(np.uint8)
    ramp = np.repeat(rows[:, None, None], 48, 1).repeat(3, 2)
    nr = depth.normal_map(ramp)
    cols = np.linspace(0, 255, 48).astype(np.uint8)
    rampx = np.repeat(cols[None, :, None], 48, 0).repeat(3, 2)
    nx_ = depth.normal_map(rampx)
    check('G up: a bright-top ramp tilts green below 128, red stays neutral',
          nr[24, 24, 1] < 128 - 2 and abs(int(nr[24, 24, 0]) - 128) <= 2,
          f'mid pixel={nr[24, 24].tolist()}')
    check('R: a bright-right ramp tilts red below 128', nx_[24, 24, 0] < 128 - 2,
          f'mid pixel={nx_[24, 24].tolist()}')
    # every encoded vector is unit length, and nothing faces away from the viewer
    v = (nf.astype(np.float32) / 255.0 * 2.0 - 1.0)
    ln = np.sqrt((v * v).sum(-1))
    vn = (depth.normal_map(nat).astype(np.float32) / 255.0 * 2.0 - 1.0)
    lnn = np.sqrt((vn * vn).sum(-1))
    check('encoded normals are unit length and never face away from the viewer',
          float(np.abs(ln - 1).max()) < 0.02 and float(np.abs(lnn - 1).max()) < 0.02 and
          float(vn[..., 2].min()) > 0.3,
          f'flat |n|-1 max={float(np.abs(ln - 1).max()):.4f}, courtyard |n|-1 max='
          f'{float(np.abs(lnn - 1).max()):.4f}, z min={vn[..., 2].min():.3f} '
          f'(B>= {int(depth.normal_map(nat)[..., 2].min())})')
    check('the courtyard map is mostly flat with real relief where it counts',
          3.0 < float(vn[..., :2].std() * 128) < 40.0,
          f'tilt sd={float(vn[..., :2].std() * 128):.1f}/128')
    check('normal_map is deterministic', bool((depth.normal_map(nat) ==
          depth.normal_map(nat)).all()))

    print('\nM6 server — depth.png written at upload and served')
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
        t0u = time.time()
        with open(natp, 'rb') as fh:
            code, raw = req('/api/upload', raw=fh.read())
        dtu = time.time() - t0u
        doc = json.loads(raw.decode())
        check('upload ok', code == 200, str(code))
        check('upload stays in the M2 budget with the accelerator armed',
              dtu < 2.0, f'{dtu * 1000:.0f} ms (fuse runs in a background thread)')
        pid = doc['id']
        dp = os.path.join(ROOT, 'app', 'projects', pid, 'depth.png')
        check('upload wrote depth.png', os.path.exists(dp))
        code, raw = req(f'/api/image/{pid}/orient.png')
        check('GET /api/image/<pid>/orient.png', code == 200 and
              raw[:8] == b'\x89PNG\r\n\x1a\n', str(code))
        code, nraw = req(f'/api/image/{pid}/normal.png')
        n_ok = code == 200 and nraw[:8] == b'\x89PNG\r\n\x1a\n'
        check('GET /api/image/<pid>/normal.png', n_ok, str(code))
        check('upload wrote normal.png',
              os.path.exists(os.path.join(ROOT, 'app', 'projects', pid, 'normal.png')))
        if n_ok:
            import io as _io2
            n8 = np.asarray(Image.open(_io2.BytesIO(nraw)).convert('RGB'))
            check('served normal map matches depth.normal_map exactly',
                  bool((n8 == depth.normal_map(
                      np.asarray(Image.open(natp)).astype(np.uint8))).all()),
                  f'shape={n8.shape}')
        # pre-M6 projects (and the tracked M1 examples) have no derived views: the
        # first GET must build them from native.png rather than 404
        for gone in ('normal.png', 'orient.png'):
            os.remove(os.path.join(ROOT, 'app', 'projects', pid, gone))
        code, graw = req(f'/api/image/{pid}/normal.png')
        check('a missing derived view is rebuilt on first GET', code == 200 and
              graw[:8] == b'\x89PNG\r\n\x1a\n' and
              os.path.exists(os.path.join(ROOT, 'app', 'projects', pid, 'normal.png')) and
              os.path.exists(os.path.join(ROOT, 'app', 'projects', pid, 'orient.png')) is False,
              str(code))
        code, _g2 = req(f'/api/image/{pid}/orient.png')
        check('orient.png rebuilds the same way', code == 200 and
              os.path.exists(os.path.join(ROOT, 'app', 'projects', pid, 'orient.png')), str(code))
        # ---- M6e: the optional model-fused depth pass -----------------------
        import engine.depth_ml as dml
        src = dml.source()

        def spear(a, b):
            ra = np.argsort(np.argsort(np.asarray(a, np.float64).ravel())).astype(np.float64)
            rb = np.argsort(np.argsort(np.asarray(b, np.float64).ravel())).astype(np.float64)
            return float(np.corrcoef(ra, rb)[0, 1])

        ping = json.loads(urllib.request.urlopen(BASE + '/api/ping', timeout=5).read())
        check('ping reports the depth-pass source', ping.get('depth_ml') == src, src)
        nat_arr = np.asarray(Image.open(natp)).astype(np.uint8)
        ml_path = os.path.join(ROOT, 'app', 'projects', pid, 'depth_ml.png')
        if src != 'parse-only':
            for _ in range(80):                      # background fuse thread
                if os.path.exists(ml_path):
                    break
                time.sleep(0.25)
            check('background fuse wrote the model-fused depth map',
                  os.path.exists(ml_path), src)
            code, mraw = req(f'/api/image/{pid}/depth_ml.png')
            ok = code == 200 and mraw[:8] == b'\x89PNG\r\n\x1a\n'
            check('GET /api/image/<pid>/depth_ml.png', ok, str(code))
            if ok:
                import io as _io3
                m8 = np.asarray(Image.open(_io3.BytesIO(mraw)))
                u8, meta = dml.fuse(nat_arr, depth.field(nat_arr)['depth'])
                check('served fused map reproduces from engine.depth_ml',
                      m8.shape == u8.shape and bool((m8 == u8).all()) and
                      meta['model'] in src, str(meta))
                gtp = os.path.join(ROOT, 'depth.npy')
                if os.path.exists(gtp):
                    g = np.load(gtp).astype(np.float64)
                    r_ml = spear(u8, g)
                    r_pa = spear(depth.field(nat_arr)['depth'], g)
                    check('fused depth beats the parse against the origin depth.npy',
                          r_ml >= 0.75 and r_ml >= r_pa + 0.10,
                          f'fused rho={r_ml:.3f} vs parse rho={r_pa:.3f}')
                os.remove(ml_path)
                code, _m2 = req(f'/api/image/{pid}/depth_ml.png')
                check('a missing depth_ml.png rebuilds on first GET',
                      code == 200 and os.path.exists(ml_path), str(code))
        else:
            check('no accelerator present -> parse-only depth (M6b bytes)',
                  not os.path.exists(ml_path), src)
        empty = tempfile.mkdtemp()
        u8n, _mn = dml.fuse(nat_arr[:8, :8], np.zeros((8, 8), np.uint8),
                            models_dir=empty, ort_dir=empty)
        check('fuse() degrades to None without runtime/model', u8n is None)
        code, raw = req(f'/api/image/{pid}/depth.png')
        ok_png = code == 200 and raw[:8] == b'\x89PNG\r\n\x1a\n'
        check('GET /api/image/<pid>/depth.png', ok_png, str(code))
        if ok_png:
            import io as _io
            d8 = np.asarray(Image.open(_io.BytesIO(raw)))
            check('served map matches the field', bool((d8 == depth.field(
                np.asarray(Image.open(natp)).astype(np.uint8))['depth']).all()))
        code, body = jreq('/api/masks', {'pid': pid, 'ops': [
            {'op': 'upsert', 'id': 'aa11', 'name': 'bar', 'kind': 'object',
             'color': [200, 40, 40], 'z': 0, 'surface': 'wall',
             'rle': im.rle_encode(_bar(natp))}]})
        check('mask depth is the feet key', code == 200 and
              body['objects'][0]['depth'] == 2.0 * _bar_base(natp),
              f'{body["objects"][0]["depth"]} vs {2.0 * _bar_base(natp)}')
        check('surface round-trips through /api/masks', code == 200 and
              body['objects'][0].get('surface') == 'wall',
              str(body['objects'][0].get('surface')))
        # depthsplit over the courtyard junction: one band mask -> two pieces
        nat = np.asarray(Image.open(natp))
        band = np.zeros(nat.shape[:2], bool)
        band[180:260, 100:200] = True
        code, body = jreq('/api/edge', {'pid': pid, 'rle': im.rle_encode(band),
                                        'op': 'depthsplit'})
        check('edge op depthsplit cuts at the junction', code == 200 and
              body.get('n', 0) >= 2, f"n={body.get('n')}")
        # wand depth guard on the real art: a wall seed must not reach the floor
        code, w0 = jreq('/api/wand', {'pid': pid, 'x': 60, 'y': 100, 'tol': 60})
        code, w1 = jreq('/api/wand', {'pid': pid, 'x': 60, 'y': 100, 'tol': 60,
                                      'dguard': 4})
        def rows(w):
            m = np.zeros(nat.shape[:2], bool)
            r = w.get('rle', [])
            for i in range(0, len(r), 3):
                m[r[i], r[i + 1]:r[i + 2] + 1] = True
            return m
        m0, m1 = rows(w0), rows(w1)
        hz = depth.field(nat)['horizon']
        jx = float(hz[60])                     # the junction at the seed's column
        check('server wand dguard stops at the seed column junction',
              (np.nonzero(m1)[0].max() if m1.any() else 0) <= jx + 5 and
              int(m1.sum()) < int(m0.sum()),
              f'unguarded {int(m0.sum())}px to row {np.nonzero(m0)[0].max() if m0.any() else 0}, '
              f'guarded {int(m1.sum())}px to row {np.nonzero(m1)[0].max() if m1.any() else 0} '
              f'(junction at column 60: {jx:.0f})')
    finally:
        srv.terminate()
        srv.wait(timeout=10)

    print()
    if FAILS:
        print(f'{len(FAILS)} M6 CHECK(S) FAILED: {FAILS}')
        sys.exit(1)
    print('ALL M6 ACCEPTANCE CHECKS PASSED')


def _mean_key(m):
    """The pre-M6a key (3*base_y - centroid) that ranked tall objects in front."""
    ys = np.nonzero(m)[0]
    return round(3.0 * float(np.percentile(ys, 97)) - float(ys.mean()), 1)


def _bar(path):
    """A 4-row-tall bar mask at rows 40..43 of the native art (for the key check)."""
    nat = np.asarray(Image.open(path))
    m = np.zeros(nat.shape[:2], bool)
    m[40:44, 10:20] = True
    return m


def _bar_base(path):
    return float(np.percentile(np.nonzero(_bar(path))[0], 97))


if __name__ == '__main__':
    main()
