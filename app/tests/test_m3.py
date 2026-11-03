"""M3 acceptance test — engine + export, end to end over HTTP (plan §6).

Starts the real server on a spare port and checks:
  M3 catalog   all 7 anim types move (or hold still, for static) on synthetic art;
               frame diffs stay inside allowed zones (audit = 0 stray px, wrap included)
  M3 plate     art outside every object mask is untouched by the inpaint
  M3 determinism  two renders of the same doc are byte-identical (preview == export)
  M3 params    bogus/out-of-range/hostile anim+params are coerced, never crash
  M3 courtyard upload -> auto-seg import -> 48f loop: audit 0 stray, GIF <= 2 MB,
               n_frames == N, loops forever
  M3 bundle    zip carries plate/walkmask/palette/scene.json + >=3 RGBA sheets;
               sheets have binary alpha and zeroed RGB under alpha 0 (no halos);
               re-importing the bundle recomposites to EXACTLY the rendered frames

Run:  python3 app/tests/test_m3.py
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
from engine import animate                                        # noqa: E402

PORT = int(os.environ.get('M3_TEST_PORT', 8139))
BASE = f'http://127.0.0.1:{PORT}'
FAILS = []


def check(name, ok, detail=''):
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'  — {detail}' if detail else ''))
    if not ok:
        FAILS.append(name)


def req(path, body=None, raw=None, method=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    r = urllib.request.Request(BASE + path, data=data,
                               method=method or ('POST' if data is not None else 'GET'),
                               headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(r, timeout=300) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def jreq(path, body=None):
    code, raw = req(path, body)
    return code, json.loads(raw.decode())


# ------------------------------------------------------------------ synthetic art


def synth_art():
    """64x64 pixel art with one blob per catalog type (dither-free on purpose: the
    catalog checks are about motion, the dither-proofing is M2's wand story)."""
    W = H = 64
    a = np.zeros((H, W, 3), np.uint8)
    a[:] = (150, 140, 120)                       # floor
    a[:20] = (90, 90, 110)                       # wall band
    zones = {}
    a[30:46, 6:30] = (60, 130, 200)              # water sheet
    zones['ripple'] = (30, 46, 6, 30)
    a[34:40, 12:20] = (140, 190, 90)             # lily pad on the water
    zones['bob'] = (34, 40, 12, 20)
    a[24:46, 44:52] = (70, 170, 80)              # fern column
    a[24:46, 44:52][::2] = (60, 150, 70)        # interior shading rows
    zones['sway'] = (24, 46, 44, 52)
    a[20:24, 45:50] = (230, 120, 180)            # blossoms above the fern
    zones['flutter'] = (20, 24, 45, 50)
    a[50:56, 34:42] = (240, 225, 150)            # tassel fringe
    zones['gustflick'] = (50, 56, 34, 42)
    a[52:60, 4:12] = (140, 80, 40)               # terracotta pot
    zones['static'] = (52, 60, 4, 12)
    a[8:16, 40:60] = (60, 130, 200)              # wall water glint band (shimmer)
    zones['shimmer'] = (8, 16, 40, 60)
    return a, zones


def zone_mask(shape, z):
    m = np.zeros(shape[:2], bool)
    m[z[0]:z[1], z[2]:z[3]] = True
    return m


def frames_from_strip(png_bytes, cols, fw, fh, N):
    sheet = np.asarray(Image.open(io.BytesIO(png_bytes)).convert('RGB'))
    out = []
    for t in range(N):
        r, c = divmod(t, cols)
        out.append(sheet[r * fh:(r + 1) * fh, c * fw:(c + 1) * fw])
    return np.stack(out)


def recomposite_from_bundle(zf, t):
    """Simulate the target engine: plate + sheets pasted at anchors in z order."""
    scene = json.loads(zf.read('scene.json'))
    plate = np.asarray(Image.open(io.BytesIO(zf.read('plate.png'))).convert('RGB')).copy()
    fr = plate.copy()
    for o in sorted(scene['objects'], key=lambda o: o['z']):
        sh = o['sheet']
        a = np.asarray(Image.open(io.BytesIO(zf.read(sh['file']))))
        cols, cw, ch = sh['cols'], sh['cell_w'], sh['cell_h']
        ft = t % sh['frames']
        r, c = divmod(ft, cols)
        cell = a[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
        ax, ay = o['anchor']
        H, W = fr.shape[:2]
        h, w = cell.shape[:2]
        x1, y1 = min(W, ax + w), min(H, ay + h)
        sx, sy = max(0, -ax), max(0, -ay)
        ax, ay = max(0, ax), max(0, ay)
        sub = cell[sy:sy + (y1 - ay), sx:sx + (x1 - ax)]
        al = sub[..., 3] > 0
        fr[ay:y1, ax:x1][al] = sub[..., :3][al]
    return fr, scene


def main():
    srv = subprocess.Popen([sys.executable, os.path.join(ROOT, 'app', 'server.py'), str(PORT)],
                           cwd=os.path.join(ROOT, 'app'),
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
        check('ping', ping.get('ok') and ping.get('milestone', 'M1') >= 'M3', str(ping))

        # ------------------------------------------------------------ 7-type catalog
        print('\nM3 catalog — all 7 anim types on synthetic art')
        art, zones = synth_art()
        buf = io.BytesIO()
        Image.fromarray(art).save(buf, 'PNG')
        r = urllib.request.Request(BASE + '/api/upload', data=buf.getvalue(),
                                   headers={'Content-Type': 'image/png'})
        with urllib.request.urlopen(r, timeout=60) as resp:
            meta = json.loads(resp.read().decode())
        pid = meta['id']
        shape = (meta['native_h'], meta['native_w'])
        from engine import imaging as im
        z = 0
        ops = []
        want = {'sway': 'sway', 'bob': 'bob', 'ripple': 'ripple', 'shimmer': 'shimmer',
                'flutter': 'flutter', 'gustflick': 'gustflick', 'static': 'static'}
        for typ in want:
            m = zone_mask(shape, zones[typ])
            ops.append({'op': 'upsert', 'id': typ, 'name': typ, 'kind': 'object',
                        'color': [200, 200, 200], 'z': z, 'anim': typ,
                        'rle': [int(v) for run in _rle(m) for v in run]})
            z += 1
        code, r = jreq('/api/masks', {'pid': pid, 'ops': ops})
        check('masks upsert 200', code == 200, f'{code}')
        N = 24
        code, rr = jreq('/api/render', {'pid': pid, 'N': N, 'fps': 12})
        check('render 200', code == 200, f'{code} {rr.get("error")}')
        check('audit: 0 stray px (wrap included)', rr['audit']['stray_max'] == 0,
              str(rr['audit']))
        code, raw = req(rr['strip'])
        frames = frames_from_strip(raw, rr['cols'], rr['fw'], rr['fh'], N)

        changed = (np.abs(frames.astype(int)[1:] - frames.astype(int)[:-1]).sum(3) > 0)
        wrap = (np.abs(frames[0].astype(int) - frames[-1].astype(int)).sum(2) > 0)
        allowed_all = np.zeros(shape, bool)
        for typ in want:
            if typ != 'static':
                allowed_all |= im.dilate4(zone_mask(shape, zones[typ]).copy(), 3)
        for typ in want:
            m = zone_mask(shape, zones[typ])
            if typ == 'static':
                st = frames[:, m]
                check(f'{typ}: pixels constant across the loop',
                      bool((st == st[0]).all()), f'{int(m.sum())}px held')
                continue
            zone = im.dilate4(m.copy(), 3)
            moved = int(changed[:, zone].sum())
            check(f'{typ}: motion inside its zone', moved > 0,
                  f'{moved} changed px·frames')
        outside = int((changed & ~allowed_all[None]).sum())
        check('no motion outside any animated zone', outside == 0, f'{outside}px')
        moved_union = int(changed.any(axis=0).sum())
        check('loop seam carries motion (no frozen wrap)', int(wrap.sum()) > 0
              and int(wrap.sum()) <= moved_union, f'wrap {int(wrap.sum())} vs union {moved_union}')

        # determinism: same doc -> same strip bytes
        code, rr2 = jreq('/api/render', {'pid': pid, 'N': N, 'fps': 12})
        code2, raw2 = req(rr2['strip'])
        check('render is deterministic', raw == raw2, f'{len(raw)}B')

        # params coercion
        code, r = jreq('/api/masks', {'pid': pid, 'ops': [
            {'op': 'upsert', 'id': 'sway', 'name': 'sway', 'kind': 'object',
             'z': 0, 'anim': 'sway (articulated fronds)',
             'params': {'amp': 99, 'speed': -5, 'gust': 'loud', 'phase': 3},
             'rle': [int(v) for run in _rle(zone_mask(shape, zones['sway'])) for v in run]}]})
        code, doc = jreq(f'/api/project/{pid}')
        o = [x for x in doc['objects'] if x['id'] == 'sway'][0]
        check('hint text coerced to catalog type', o['anim'] == 'sway', o['anim'])
        check('params clamped/whitelisted', o['params']['amp'] == 2.0
              and o['params']['speed'] == 1 and o['params']['gust'] == 0.15
              and o['params']['phase'] == 0.99, str(o['params']))
        code, r = jreq('/api/masks', {'pid': pid, 'ops': [
            {'op': 'upsert', 'id': 'sway', 'name': 'sway', 'kind': 'object', 'z': 0,
             'anim': 'nonsense hint', 'params': {'amp': [1, 2], 'speed': {'a': 1}},
             'rle': [int(v) for run in _rle(zone_mask(shape, zones['sway'])) for v in run]}]})
        code, doc = jreq(f'/api/project/{pid}')
        o = [x for x in doc['objects'] if x['id'] == 'sway'][0]
        check('hostile params fall back to defaults', o['anim'] == 'static'
              and o['params'] == {}, str(o['params']))

        # out-dir whitelist
        check('out dir is whitelisted', req(f'/api/out/{pid}/../../server.py')[0] == 404
              and req(f'/api/out/{pid}/nope.png')[0] == 404)

        # ------------------------------------------------------------ courtyard E2E
        print('\nM3 acceptance — courtyard masks -> seamless GIF <= 2 MB, audit 0')
        with open(os.path.join(ROOT, 'uploads', 'iso_00080_.png'), 'rb') as f:
            art_bytes = f.read()
        r = urllib.request.Request(BASE + '/api/upload', data=art_bytes,
                                   headers={'Content-Type': 'image/png'})
        with urllib.request.urlopen(r, timeout=60) as resp:
            meta = json.loads(resp.read().decode())
        cpid = meta['id']
        t0 = time.time()
        code, seg = jreq('/api/autoseg', {'pid': cpid, 'max_suggestions': 60,
                                          'cfg': {'recognise': True}})
        check('autoseg 200', code == 200, f"{seg.get('ms')} ms")
        sugs = [s for s in seg['suggestions'] if s['kind'] == 'object']
        ops = []
        for i, s in enumerate(sugs):
            ops.append({'op': 'upsert', 'id': s['id'], 'name': s['name'], 'kind': 'object',
                        'color': s['color'], 'z': i, 'label': s.get('label', s['name']),
                        'conf': s.get('conf', 0), 'anim': s.get('anim', ''),
                        'anim_hint': s.get('anim', ''), 'nfrag': s.get('nfrag', 1),
                        'rle': s['rle']})
        code, r = jreq('/api/masks', {'pid': cpid, 'ops': ops})
        check(f'imported {len(ops)} courtyard objects', code == 200 and len(ops) >= 20,
              f'{len(ops)}')
        code, r = jreq(f'/api/project/{cpid}', {'N': 48, 'fps': 12})
        t0 = time.time()
        code, rr = jreq('/api/render', {'pid': cpid, 'N': 48, 'fps': 12})
        check('courtyard render', code == 200, f"{rr.get('ms')} ms, "
              f"{rr.get('animated')}/{rr.get('objects')} animated")
        check('courtyard audit = 0 stray px', rr['audit']['stray_max'] == 0,
              str(rr['audit']))
        check('render fast enough for a button', rr['total_ms'] < 60000,
              f"{rr['total_ms']} ms")

        t0 = time.time()
        code, gif = jreq('/api/gif', {'pid': cpid, 'N': 48, 'fps': 12, 'scale': 2,
                                      'colors': 128})
        check('gif 200', code == 200, f"{gif.get('bytes')} B in {gif.get('total_ms')} ms")
        check('GIF <= 2 MB', gif['bytes'] <= 2 * 1024 * 1024,
              f"{gif['bytes'] / 1048576:.2f} MB")
        code, graw = req(gif['url'])
        g = Image.open(io.BytesIO(graw))
        check('GIF loops, right size/frame count', g.n_frames == 48
              and (g.width, g.height) == (768, 1024) and g.info.get('loop') == 0,
              f'{g.n_frames}f {g.width}x{g.height} loop={g.info.get("loop")}')
        check('GIF audit also 0', gif['audit']['stray_max'] == 0)

        # ------------------------------------------------------------ bundle
        print('\nM3 acceptance — scene bundle re-imports with clean RGBA edges')
        t0 = time.time()
        code, bz = jreq('/api/bundle', {'pid': cpid, 'N': 48, 'fps': 12})
        check('bundle 200', code == 200, f"{bz.get('bytes')} B, {bz.get('sheets')} sheets,"
              f" {bz.get('total_ms')} ms")
        check('bundle carries >= 3 RGBA sheets', bz['sheets'] >= 3, f"{bz['sheets']}")
        code, zraw = req(bz['url'])
        zf = zipfile.ZipFile(io.BytesIO(zraw))
        names = set(zf.namelist())
        check('bundle layout (plate/walkmask/palette/scene.json)',
              {'plate.png', 'walkmask.png', 'palette.png', 'scene.json'} <= names,
              f'{len(names)} files')
        scene = json.loads(zf.read('scene.json'))
        check('scene.json anchors/depths/bboxes/clips', all(
            'anchor' in o and 'bbox' in o and 'depth' in o and o.get('clips')
            for o in scene['objects']), f"{len(scene['objects'])} objects")
        wm = np.asarray(Image.open(io.BytesIO(zf.read('walkmask.png'))).convert('L'))
        check('walkmask is a real mask (part walkable)', 0 < (wm > 0).mean() < 1,
              f"{(wm > 0).mean():.2f} walkable")
        clean = True
        for o in scene['objects']:
            a = np.asarray(Image.open(io.BytesIO(zf.read(o['sheet']['file']))))
            if a.ndim != 3 or a.shape[2] != 4:
                clean = False
                break
            al = a[..., 3]
            if not np.isin(al, (0, 255)).all():
                clean = False
                break
            if (a[..., :3][al == 0] != 0).any():     # zeroed RGB under alpha 0
                clean = False
                break
        check('sheets: binary alpha + zeroed RGB under alpha (no halos)', clean)

        # parity: bundle recomposite == rendered frames (strip), several frames
        code, sraw = req(rr['strip'])
        frames = frames_from_strip(sraw, rr['cols'], rr['fw'], rr['fh'], 48)
        ok_par = True
        for t in (0, 17, 31, 47):
            fr, _ = recomposite_from_bundle(zf, t)
            if not (fr == frames[t]).all():
                d = int((fr != frames[t]).any(axis=2).sum())
                check(f'bundle frame {t} == render frame {t}', False, f'{d}px differ')
                ok_par = False
                break
        if ok_par:
            check('bundle recomposites to EXACT render frames (0,17,31,47)', True)

        # plate invariant on the synthetic project
        code, doc = jreq(f'/api/project/{pid}')
        print('\nM3 plate invariant')
        plate = None
        zf2 = None
        code, bz2 = jreq('/api/bundle', {'pid': pid, 'N': 24, 'fps': 12})
        code2, zraw2 = req(bz2['url'])
        zf2 = zipfile.ZipFile(io.BytesIO(zraw2))
        plate = np.asarray(Image.open(io.BytesIO(zf2.read('plate.png'))).convert('RGB'))
        allm = np.zeros(shape, bool)
        for o in doc['objects']:
            m = zone_mask(shape, zones[o['id']]) if o['id'] in zones else None
            if m is not None:
                allm |= m
        check('plate == art outside every object mask',
              bool((plate[~allm] == art[~allm]).all()))

        print()
        if FAILS:
            print(f'{len(FAILS)} M3 CHECK(S) FAILED: {FAILS}')
            return 1
        print('ALL M3 ACCEPTANCE CHECKS PASSED')
        return 0
    finally:
        srv.terminate()
        srv.wait(timeout=10)


def _rle(m):
    out = []
    for y in range(m.shape[0]):
        row = m[y]
        x = 0
        W = m.shape[1]
        while x < W:
            if row[x]:
                x0 = x
                while x + 1 < W and row[x + 1]:
                    x += 1
                out.append((y, x0, x))
            x += 1
    return out


if __name__ == '__main__':
    sys.exit(main())
