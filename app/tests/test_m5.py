"""M5 acceptance test — hardening: perf pass, texture fidelity, concurrency guard,
docs page, atlas packing (plan §6 M5 / build_log entry 12).

Sections:
  blit parity    _Blit == reference slice blit for every offset in range (clip incl.)
  texture        static sprites/frames carry the object's OWN pixels (M3/M4 regression:
                 the old _put_runs sampled nat at cell-relative x — left-edge garbage)
  leaf units     origin multi-source BFS split: <150px stays whole, combs partition
                 into >=2 units, tips land on their own arm's unit
  shadow hole    a mover's claimed ring is inpainted OUT of the plate (sync visible)
  perf           masks.npy worst case 48f @ 384x512 < 10 s, audit 0, deterministic
  server         ping M5 + engine version; render cache flag; concurrency guard
                 (same-pid serialises, cross-pid bounded, 429 shape, no 5xx)
  docs           GET /docs serves the guide (atlas/shadow/429/shortcuts sections)
  atlas          bundle carries atlas pages + scene.json rects; crops == sheets;
                 atlas recomposite == render frames exactly; atlas=0 stays M4-shaped
  pack unit      pack_atlas: in-page, non-overlapping, verbatim, oversized -> None

Run:  python3 app/tests/test_m5.py
"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'app'))
sys.path.insert(0, os.path.join(ROOT, 'app', 'tests'))
from engine import animate, export, imaging as im              # noqa: E402
import test_m3 as T3                                            # noqa: E402 (helpers)

PORT = int(os.environ.get('M5_TEST_PORT', 8147))
T3.PORT = PORT
T3.BASE = f'http://127.0.0.1:{PORT}'
BASE = T3.BASE
check, req, jreq = T3.check, T3.req, T3.jreq
zone_mask, _rle = T3.zone_mask, T3._rle
frames_from_strip = T3.frames_from_strip
FAILS = T3.FAILS
PIDS = []                                                       # cleanup list


# ------------------------------------------------------------------ engine bits

def ref_blit(art, runs, dx, dy, cell_shape, ox, oy):
    """Ground truth: absolute-source slice blit with cell clipping."""
    H, W = art.shape[:2]
    ch, cw = cell_shape
    ref = np.zeros((ch, cw, 4), np.uint8)
    for (y, a, b) in runs:
        if y < 0 or y >= H:
            continue
        yy = y + dy - oy
        if yy < 0 or yy >= ch:
            continue
        da = a - ox + dx
        c0, c1 = max(da, 0), min(da + (b - a + 1), cw)
        if c0 >= c1:
            continue
        s0 = a + (c0 - da)
        n = c1 - c0
        if s0 < 0 or s0 + n > W:
            continue
        ref[yy, c0:c1, :3] = art[y, s0:s0 + n]
        ref[yy, c0:c1, 3] = 255
    return ref


def upload(art):
    buf = io.BytesIO()
    Image.fromarray(art).save(buf, 'PNG')
    r = urllib.request.Request(BASE + '/api/upload', data=buf.getvalue(),
                               headers={'Content-Type': 'image/png'})
    with urllib.request.urlopen(r, timeout=60) as resp:
        meta = json.loads(resp.read().decode())
    PIDS.append(meta['id'])
    return meta


def main():
    rng = np.random.default_rng(11)

    # ------------------------------------------------------- _Blit parity
    print('\nM5 blit — cached flat-index blit == reference slice blit')
    art = rng.integers(0, 255, (40, 50, 3), dtype=np.uint8)
    blob = np.zeros((40, 50), bool)
    blob[8:30, 12:40] = True
    blob[rng.integers(0, 40, 60), rng.integers(0, 50, 60)] = False   # ragged holes
    runs = animate._runs_of(*np.nonzero(blob))
    ox, oy, cw, ch = 10, 6, 34, 28
    nat3 = np.ascontiguousarray(art).reshape(-1, 3)
    bad = 0
    for dx in range(-3, 4):
        for dy in range(-2, 3):
            cell = np.zeros((ch, cw, 4), np.uint8)
            animate._Blit(runs, ox, oy, cw, ch, 50).put(cell.reshape(-1), nat3, dx, dy)
            if not np.array_equal(cell, ref_blit(art, runs, dx, dy, (ch, cw), ox, oy)):
                bad += 1
    check('all 35 offsets pixel-identical to reference (clipping included)', bad == 0,
          f'{bad} mismatches')

    # ------------------------------------------------- texture fidelity (static)
    print('\nM5 texture — sprites carry the object’s own pixels')
    sart, zones = T3.synth_art()
    H, W = sart.shape[:2]
    m = zone_mask((H, W), zones['static'])
    o = {'id': 'st', 'name': 'st', 'kind': 'object', 'visible': True, 'z': 0,
         'anim': 'static', 'params': {}, 'mask': m}
    rr = animate.render(sart, [o], N=4)
    plan = rr['plans'][0]
    x0, y0 = plan.cell[0], plan.cell[1]
    cell = rr['sprites'][0][0]
    ys, xs = np.nonzero(m)
    own = (cell[ys - y0, xs - x0, :3] == sart[ys, xs]).all(axis=1)
    check('static cell == source art at the same coordinates', bool(own.all()),
          f'{int(own.sum())}/{len(ys)} px')
    fr0 = rr['frames'][0]
    check('static frame == art inside the mask', bool((fr0[ys, xs] == sart[ys, xs]).all()))

    # --------------------------------------------------------- leaf-unit split
    print('\nM5 leaf units — origin multi-source BFS split restored')
    shape = (120, 90)
    tiny = np.zeros(shape, bool)
    tiny[10:20, 10:20] = True                                   # 100 px < 150
    units = animate._split_leaf_units(*np.nonzero(tiny), shape)
    check('blob <150px stays one unit', len(units) == 1 and len(units[0][0]) == 100,
          f'{len(units)} units')
    comb = np.zeros(shape, bool)
    comb[10:97, 20:29] = True                                   # arm 1
    comb[10:97, 56:65] = True                                   # arm 2
    comb[95:97, 29:56] = True                                   # 2px isthmus:
    cys, cxs = np.nonzero(comb)                                 # erode cuts it, so
    #                                                           # each arm keeps its
    #                                                           # own core (origin rule)
    units = animate._split_leaf_units(cys, cxs, shape)
    lab = -np.ones(shape, np.int32)
    cover = 0
    for i, (uys, uxs) in enumerate(units):
        lab[uys, uxs] = i
        cover += len(uys)
    check('comb partitions the blob exactly', cover == len(cys)
          and bool((lab[cys, cxs] >= 0).all()), f'{len(units)} units, {cover}/{len(cys)} px')
    check('comb splits into >=2 units', len(units) >= 2, f'{len(units)}')
    t1, t2 = lab[11, 24], lab[11, 60]
    check('each arm tip rides its own arm’s unit', t1 >= 0 and t2 >= 0 and t1 != t2,
          f'tips {t1} vs {t2}')

    # --------------------------------------------------- shadow ring plate hole
    print('\nM5 shadow — claimed ring leaves the plate (sync is visible)')
    gart = np.full((120, 120, 3), (180, 160, 120), np.uint8)     # flat ground
    pad = np.zeros((120, 120), bool)
    pad[30:45, 30:45] = True
    gart[pad] = (90, 200, 90)
    band = im.dilate4(pad.copy(), 2) & ~pad
    gart[band] = (30, 34, 48)                                    # cool dark shadow
    oo = {'id': 'pad', 'name': 'pad', 'kind': 'object', 'visible': True, 'z': 0,
          'anim': 'bob', 'params': {'speed': 1, 'amp': 1}, 'mask': pad.copy(),
          'shadow': True}
    rr = animate.render(gart, [oo], N=8)
    plate = rr['plate']
    moved = (plate[band] != gart[band]).any(axis=1)
    check('ring pixels are inpainted out of the plate', float(moved.mean()) >= 0.8,
          f'{moved.mean():.2f} of {int(band.sum())} ring px')
    check('plate exact outside mask+ring',
          bool((plate[~(band | pad)] == gart[~(band | pad)]).all()))
    d0 = (np.abs(rr['frames'][0].astype(int) - plate.astype(int)).sum(2) > 0) & band
    check('the moving ring repaints the hole in frames', int(d0.sum()) > 0,
          f'{int(d0.sum())} px differ from plate')
    check('shadow render audit 0', rr['audit']['stray_max'] == 0, str(rr['audit']))

    # -------------------------------------------------------------- perf budget
    print('\nM5 perf — worst case 48f @ 384×512 (plan budget <10 s)')
    nat = np.asarray(Image.open(os.path.join(ROOT, 'native.png')), np.uint8)
    M = np.load(os.path.join(ROOT, 'masks.npy'))
    objs = [{'id': i, 'name': f'b{i}', 'kind': 'object', 'visible': True, 'z': i,
             'anim': 'sway', 'params': {}, 'mask': m, 'shadow': True}
            for i, m in enumerate(M)]
    t0 = time.time()
    r1 = animate.render(nat, objs, N=48)
    dt = time.time() - t0
    check('worst-case render < 10 s', dt < 10.0, f'{dt:.2f} s, '
          f'{sum(len(p.units) for p in r1["plans"])} units')
    check('worst-case audit 0', r1['audit']['stray_max'] == 0, str(r1['audit']))
    h1 = hashlib.md5(r1['frames'][7].tobytes()).hexdigest()
    del r1
    r2 = animate.render(nat, objs, N=48)
    h2 = hashlib.md5(r2['frames'][7].tobytes()).hexdigest()
    check('worst-case render deterministic', h1 == h2)
    del r2

    # ------------------------------------------------------------- server suite
    srv = subprocess.Popen([sys.executable, os.path.join(ROOT, 'app', 'server.py'),
                            str(PORT)], cwd=os.path.join(ROOT, 'app'),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        for _ in range(60):
            try:
                code, _ = jreq('/api/ping')
                if code == 200:
                    break
            except OSError:
                time.sleep(0.25)
        code, ping = jreq('/api/ping')
        check('ping: milestone >= M5 + engine version',
              ping.get('ok') and ping.get('milestone', 'M1') >= 'M5'
              and len(ping.get('engine', '')) == 12, str(ping))

        # small project with 3 objects for the server-side checks
        art, zones = T3.synth_art()
        shape = art.shape[:2]
        meta = upload(art)
        pid = meta['id']
        ops = []
        for z_i, typ in enumerate(('sway', 'bob', 'static')):
            m = zone_mask(shape, zones[typ])
            ops.append({'op': 'upsert', 'id': typ, 'name': typ, 'kind': 'object',
                        'color': [200, 200, 200], 'z': z_i, 'anim': typ,
                        'rle': [int(v) for run in _rle(m) for v in run]})
        code, _ = jreq('/api/masks', {'pid': pid, 'ops': ops})
        check('masks upsert 200', code == 200, f'{code}')

        # render cache
        code, r_a = jreq('/api/render', {'pid': pid, 'N': 24, 'fps': 12})
        code2, r_b = jreq('/api/render', {'pid': pid, 'N': 24, 'fps': 12})
        check('render cache: first cold, second cached',
              code == 200 and code2 == 200 and r_a.get('cached') is False
              and r_b.get('cached') is True, f"{r_a.get('cached')} / {r_b.get('cached')}")

        # concurrency: same pid serialises, results identical
        out = []

        def hit(body):
            try:
                c, r = jreq('/api/render', body)
                out.append((c, r))
            except Exception as e:                              # noqa: BLE001
                out.append((0, {'error': repr(e)}))

        ths = [threading.Thread(target=hit,
                                args=({'pid': pid, 'N': 24, 'fps': 12},))
               for _ in range(3)]
        for t in ths:
            t.start()
        for t in ths:
            t.join(timeout=120)
        codes = [c for c, _ in out]
        check('3 concurrent renders on one pid: all 200 (serialised)',
              codes == [200, 200, 200], str(codes))
        check('concurrent renders agree on audit',
              all(r.get('audit', {}).get('stray_max') == 0 for _, r in out))
        n_cached = sum(1 for _, r in out if r.get('cached'))
        check('late arrivals ride the cache', n_cached >= 1, f'{n_cached}/3 cached')

        # concurrency: cross-pid mix, bounded — no 5xx, 429s well-formed
        pids = [pid] + [upload(art)['id'] for _ in range(2)]
        jobs = [('/api/render', {'pid': pids[0], 'N': 24}),
                ('/api/gif', {'pid': pids[1], 'N': 24, 'scale': 1}),
                ('/api/bundle', {'pid': pids[2], 'N': 24}),
                ('/api/render', {'pid': pids[1], 'N': 24}),
                ('/api/gif', {'pid': pids[0], 'N': 24, 'scale': 1}),
                ('/api/bundle', {'pid': pids[1], 'N': 24})]
        out2 = []

        def hit2(path, body):
            try:
                c, r = jreq(path, body)
                out2.append((c, r))
            except Exception as e:                              # noqa: BLE001
                out2.append((0, {'error': repr(e)}))

        ths = [threading.Thread(target=hit2, args=j) for j in jobs]
        for t in ths:
            t.start()
        for t in ths:
            t.join(timeout=180)
        ok = all(c in (200, 429) for c, _ in out2)
        shape_ok = all(('error' in r and r.get('retry') is True)
                       for c, r in out2 if c == 429)
        check('cross-pid burst: no 5xx, 429s well-formed', ok and shape_ok,
              str([c for c, _ in out2]))
        code, ping = jreq('/api/ping')
        check('server healthy after the burst', code == 200 and ping.get('ok'))

        # docs page
        code, raw = req('/docs')
        body = raw.decode('utf-8', 'replace')
        check('GET /docs 200 html', code == 200 and '<html' in body, f'{code}')
        need = ('atlas', 'Shadow sync', '429', 'walkmask', 'Keyboard shortcuts',
                'multi-source BFS')
        check('docs cover the M5 topics', all(s in body for s in need),
              str([s for s in need if s not in body]))

        # ------------------------------------------------------------ atlas bundle
        print('\nM5 atlas — bundle packs every sheet into pages')
        code, bz = jreq('/api/bundle', {'pid': pid, 'N': 24, 'fps': 12, 'atlas': 1})
        check('bundle(atlas=1) 200', code == 200,
              f"{bz.get('bytes')} B, {bz.get('sheets')} sheets")
        check('bundle info reports atlas pages', bz.get('atlas_pages', 0) >= 1,
              f"{bz.get('atlas_pages')} pages")
        code, zraw = req(bz['url'])
        zf = zipfile.ZipFile(io.BytesIO(zraw))
        names = set(zf.namelist())
        scene = json.loads(zf.read('scene.json'))
        pages = scene.get('atlas', {}).get('pages', [])
        check('atlas pages exist in zip + scene.json',
              pages and all(p['file'] in names for p in pages),
              f'{len(pages)} pages, {[p["file"] for p in pages]}')
        atl = {p['file']: np.asarray(Image.open(io.BytesIO(zf.read(p['file']))))
               for p in pages}
        exact = True
        have_rect = 0
        for o in scene['objects']:
            a = o.get('atlas')
            if not a:
                continue
            have_rect += 1
            page = atl[pages[a['page']]['file']]
            crop = page[a['y']:a['y'] + a['h'], a['x']:a['x'] + a['w']]
            sheet = np.asarray(Image.open(io.BytesIO(zf.read(o['sheet']['file']))))
            if crop.shape != sheet.shape or not np.array_equal(crop, sheet):
                exact = False
        check('every atlas rect crops to its sheet, pixel-exact',
              exact and have_rect == len(scene['objects']),
              f'{have_rect}/{len(scene["objects"])} objects in atlas')

        # atlas recomposite == render frames (engine imports from the atlas alone)
        code, rr = jreq('/api/render', {'pid': pid, 'N': 24, 'fps': 12})
        code, sraw = req(rr['strip'])
        frames = frames_from_strip(sraw, rr['cols'], rr['fw'], rr['fh'], 24)
        plate = np.asarray(Image.open(io.BytesIO(zf.read('plate.png'))).convert('RGB'))

        def recomposite_atlas(t):
            fr = plate.copy()
            for o in sorted(scene['objects'], key=lambda o: o['z']):
                a = o.get('atlas')
                sh = o['sheet']
                if a:
                    img = atl[pages[a['page']]['file']]
                    bx, by = a['x'], a['y']
                else:
                    img = np.asarray(Image.open(io.BytesIO(zf.read(sh['file']))))
                    bx, by = 0, 0
                cols, cw, chh = sh['cols'], sh['cell_w'], sh['cell_h']
                ft = t % sh['frames']
                r, c = divmod(ft, cols)
                cell = img[by + r * chh:by + r * chh + chh,
                           bx + c * cw:bx + c * cw + cw]
                ax, ay = o['anchor']
                Hh, Ww = fr.shape[:2]
                h, w = cell.shape[:2]
                x1, y1 = min(Ww, ax + w), min(Hh, ay + h)
                sx, sy = max(0, -ax), max(0, -ay)
                ax, ay = max(0, ax), max(0, ay)
                sub = cell[sy:sy + (y1 - ay), sx:sx + (x1 - ax)]
                al = sub[..., 3] > 0
                fr[ay:y1, ax:x1][al] = sub[..., :3][al]
            return fr

        eq = all(np.array_equal(recomposite_atlas(t), frames[t]) for t in (0, 11, 23))
        check('atlas-only recomposite == render frames (t=0,11,23)', eq)

        # atlas off -> M4-shaped bundle
        code, bz0 = jreq('/api/bundle', {'pid': pid, 'N': 24, 'fps': 12, 'atlas': 0})
        code, zraw0 = req(bz0['url'])
        zf0 = zipfile.ZipFile(io.BytesIO(zraw0))
        names0 = set(zf0.namelist())
        scene0 = json.loads(zf0.read('scene.json'))
        check('atlas=0 keeps the M4 layout',
              not any(n.startswith('atlas_') for n in names0)
              and 'atlas' not in scene0
              and {'plate.png', 'walkmask.png', 'palette.png', 'scene.json'} <= names0,
              f'{len(names0)} files')

        # pack_atlas unit checks
        sh_a = np.full((30, 20, 4), 7, np.uint8)
        sh_b = np.full((50, 40, 4), 9, np.uint8)
        sh_c = np.full((10, 10, 4), 3, np.uint8)
        pgs, rects = export.pack_atlas([('a', sh_a), ('b', sh_b), ('c', sh_c)],
                                       max_page=128)
        okp = True
        for name, arr in (('a', sh_a), ('b', sh_b), ('c', sh_c)):
            pi, x, y, w, h = rects[name]
            pg = pgs[pi]
            if (w, h) != (arr.shape[1], arr.shape[0]) or x < 0 or y < 0 \
                    or x + w > pg.shape[1] or y + h > pg.shape[0] \
                    or not np.array_equal(pg[y:y + h, x:x + w], arr):
                okp = False
        boxes = [(rects[n][1], rects[n][2], rects[n][1] + rects[n][3],
                  rects[n][2] + rects[n][4]) for n in 'abc']
        disjoint = all(b1[2] <= b2[0] or b2[2] <= b1[0] or b1[3] <= b2[1] or b2[3] <= b1[1]
                       for i, b1 in enumerate(boxes) for b2 in boxes[i + 1:])
        check('pack_atlas: verbatim, in-page, non-overlapping', okp and disjoint,
              f'{len(pgs)} pages')
        pgs2, rects2 = export.pack_atlas([('big', sh_b)], max_page=32)
        check('oversized sheet -> standalone fallback (rect None)', rects2['big'] is None)

    finally:
        srv.terminate()
        try:
            srv.wait(timeout=5)
        except subprocess.TimeoutExpired:
            srv.kill()
        for p in PIDS:
            shutil.rmtree(os.path.join(ROOT, 'app', 'projects', p), ignore_errors=True)

    print()
    if FAILS:
        print(f'{len(FAILS)} M5 CHECK(S) FAILED: {FAILS}')
        return 1
    print('ALL M5 ACCEPTANCE CHECKS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
