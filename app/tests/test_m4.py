"""M4 acceptance test — polish: shadow sync, vertical-blend inpaint, project
management, palette option (plan §6 / build_log entry 11).

Starts the real server on a spare port and checks:
  M4 inpaint   vertical-blend pass: no streak regression, plate untouched outside masks
  M4 shadow    sync on -> the dark hug ring moves with its object (more than the
               overhang alone), audit stays 0; pots/statics keep their shadow still;
               bundle still recomposites exactly with shadows on; renders deterministic
  M4 projects  rename / duplicate / delete over HTTP; hostile pids 404
  M4 palette   bundle honours the palette-size option (64 / 256)

Run:  python3 app/tests/test_m4.py
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
sys.path.insert(0, os.path.join(ROOT, 'app', 'tests'))
from engine import animate, imaging as im                    # noqa: E402
import test_m3 as T3                                         # noqa: E402  (helpers)

PORT = int(os.environ.get('M4_TEST_PORT', 8143))
T3.PORT = PORT
T3.BASE = f'http://127.0.0.1:{PORT}'
BASE = T3.BASE
check, req, jreq = T3.check, T3.req, T3.jreq
zone_mask, _rle = T3.zone_mask, T3._rle
frames_from_strip, recomposite_from_bundle = T3.frames_from_strip, T3.recomposite_from_bundle
FAILS = T3.FAILS


def synth_shadow_art():
    """synth art + a dark cast-shadow ring around the pad and around the pot."""
    art, zones = T3.synth_art()
    ring = np.zeros(art.shape[:2], bool)
    z = zones['bob']
    ring[z[0] - 2:z[1] + 2, z[2] - 2:z[3] + 2] = True
    ring &= ~zone_mask(art.shape, z)
    art[ring] = (28, 32, 46)                       # cool dark shadow (val < .32, hue >= 150)
    pring = np.zeros(art.shape[:2], bool)
    pz = zones['static']
    pring[pz[0] - 2:pz[1] + 2, pz[2] - 2:pz[3] + 2] = True
    pring &= ~zone_mask(art.shape, pz)
    art[pring] = (30, 33, 45)
    return art, zones, ring, pring


def upload(art):
    buf = io.BytesIO()
    Image.fromarray(art).save(buf, 'PNG')
    r = urllib.request.Request(BASE + '/api/upload', data=buf.getvalue(),
                               headers={'Content-Type': 'image/png'})
    with urllib.request.urlopen(r, timeout=60) as resp:
        return json.loads(resp.read().decode())


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
        check('ping', ping.get('ok') and ping.get('milestone', 'M1') >= 'M4', str(ping))

        # ------------------------------------------------ vertical-blend inpaint
        print('\nM4 inpaint — vertical-blend pass under big masks')
        nat = np.asarray(Image.open(os.path.join(ROOT, 'native.png')), np.uint8)
        M = np.load(os.path.join(ROOT, 'masks.npy'))
        big = M[0] | M[1]                            # plants + pads: long horizontal runs
        p_on = animate.inpaint(nat, big, vblend=True)
        p_off = animate.inpaint(nat, big, vblend=False)

        def streak(p):
            d = np.abs(p[1:].astype(int) - p[:-1].astype(int)).sum(2)
            return float(d[big[1:]].mean())
        s_on, s_off = streak(p_on), streak(p_off)
        check('vertical blend does not streak more than h-only', s_on <= s_off + 1e-9,
              f'vblend {s_on:.3f} vs h-only {s_off:.3f}')
        check('plate still exact outside masks', bool((p_on[~big] == nat[~big]).all()))
        # streaks = hard row-to-row steps INSIDE the hole (the rim is real art edge)
        inner = im.erode4(big.copy(), 3)
        d_on = np.abs(p_on[1:].astype(int) - p_on[:-1].astype(int)).sum(2)
        d_off = np.abs(p_off[1:].astype(int) - p_off[:-1].astype(int)).sum(2)
        n_on = int(((d_on > 48) & inner[1:]).sum())
        n_off = int(((d_off > 48) & inner[1:]).sum())
        check('streak lines inside the hole are reduced', n_on < n_off,
              f'{n_off} -> {n_on} hard row steps')

        # ------------------------------------------------ shadow sync
        print('\nM4 shadow sync — hug ring rides the mover, pots keep theirs')
        art, zones, ring, pring = synth_shadow_art()
        shape = art.shape[:2]
        meta = upload(art)
        pid = meta['id']

        def objs(shadow):
            return [
                {'op': 'upsert', 'id': 'pad', 'name': 'pad', 'kind': 'object', 'z': 0,
                 'anim': 'bob', 'params': {'speed': 1, 'amp': 1}, 'shadow': shadow,
                 'rle': [v for r in _rle(zone_mask(shape, zones['bob'])) for v in r]},
                {'op': 'upsert', 'id': 'plant', 'name': 'plant', 'kind': 'object', 'z': 1,
                 'anim': 'sway', 'params': {'speed': 2, 'amp': 1}, 'shadow': shadow,
                 'rle': [v for r in _rle(zone_mask(shape, zones['sway'])) for v in r]},
                {'op': 'upsert', 'id': 'pot', 'name': 'pot', 'kind': 'object', 'z': 2,
                 'anim': 'static', 'shadow': shadow,
                 'rle': [v for r in _rle(zone_mask(shape, zones['static'])) for v in r]},
            ]
        renders = {}
        for shadow in (False, True):
            code, r = jreq('/api/masks', {'pid': pid, 'ops': objs(shadow)})
            assert code == 200, r
            code, rr = jreq('/api/render', {'pid': pid, 'N': 24, 'fps': 12})
            check(f'render 200 (shadow={shadow})', code == 200, str(rr.get('audit')))
            check(f'audit 0 (shadow={shadow})', rr['audit']['stray_max'] == 0,
                  str(rr['audit']))
            code, raw = req(rr['strip'])
            renders[shadow] = (frames_from_strip(raw, rr['cols'], rr['fw'], rr['fh'], 24),
                               rr)
        f_off, f_on = renders[False][0], renders[True][0]
        ch = lambda f: (np.abs(f.astype(int)[1:] - f.astype(int)[:-1]).sum(3) > 0)
        mv_off, mv_on = int(ch(f_off)[:, ring].sum()), int(ch(f_on)[:, ring].sum())
        check('shadow ring moves with the pad when sync is on', mv_on > mv_off,
              f'ring changed px·frames: off {mv_off} -> on {mv_on}')
        check('pot keeps its own shadow still', bool(
            (f_on[:, pring] == f_on[0][pring]).all()),
            f'{int(pring.sum())}px constant')
        check('shadow off leaves the ring alone (only overhang changes)',
              mv_off < mv_on)
        # determinism with shadows
        code, rr2 = jreq('/api/render', {'pid': pid, 'N': 24, 'fps': 12})
        code, raw2 = req(rr2['strip'])
        _, rr1 = renders[True]
        code1, raw1 = req(rr1['strip'])
        check('render deterministic with shadows', raw1 == raw2)
        # bundle parity with shadows on
        code, bz = jreq('/api/bundle', {'pid': pid, 'N': 24, 'fps': 12})
        check('bundle 200 with shadows', code == 200, f"{bz.get('sheets')} sheets")
        code, zraw = req(bz['url'])
        zf = zipfile.ZipFile(io.BytesIO(zraw))
        ok_par = all((recomposite_from_bundle(zf, t)[0] == f_on[t]).all()
                     for t in (0, 11, 23))
        check('bundle recomposites exactly with shadows on', ok_par)
        scene = json.loads(zf.read('scene.json'))
        pad = [o for o in scene['objects'] if o['id'] == 'pad'][0]
        check('pad sheet covers the ring (bbox grew)', pad['bbox'][0] <=
              int(np.nonzero(zone_mask(shape, zones['bob']) | ring)[1].min()),
              str(pad['bbox']))

        # ------------------------------------------------ project management
        print('\nM4 projects — rename / duplicate / delete')
        code, r = jreq(f'/api/project/{pid}/meta', {'name': 'shadow lab'})
        check('rename 200', code == 200 and r.get('name') == 'shadow lab', str(r.get('name')))
        code, lst = jreq('/api/projects')
        check('list shows the name', any(p['id'] == pid and p['name'] == 'shadow lab'
                                          for p in lst))
        code, dup = jreq(f'/api/project/{pid}/duplicate', {})
        check('duplicate 200 with new id + copy name', code == 200 and dup['id'] != pid
              and dup['name'].endswith('copy'), f"{dup['id']} {dup.get('name')}")
        code, ddoc = jreq(f'/api/project/{dup["id"]}')
        check('duplicate carries the objects', len(ddoc['objects']) == 3,
              f"{len(ddoc['objects'])}")
        code, r = jreq(f'/api/project/{dup["id"]}/delete', {})
        check('delete 200', code == 200 and r.get('ok'))
        check('deleted project is gone', jreq(f'/api/project/{dup["id"]}')[0] == 404)
        code, lst = jreq('/api/projects')
        check('list no longer has it', not any(p['id'] == dup['id'] for p in lst))
        check('hostile pid refused', jreq('/api/project/..%2F..%2Fetc/delete', {})[0] == 404
              and jreq('/api/project/zzzzzzzz/delete', {})[0] == 404)

        # ------------------------------------------------ palette option
        print('\nM4 palette size option')
        # flat synthetic art has ~10 distinct colours, so the option is checked on the
        # colour-rich courtyard art (bundle/gif work fine with zero objects: plate only)
        with open(os.path.join(ROOT, 'uploads', 'iso_00080_.png'), 'rb') as f:
            r = urllib.request.Request(BASE + '/api/upload', data=f.read(),
                                       headers={'Content-Type': 'image/png'})
            cpid = json.loads(urllib.request.urlopen(r, timeout=60).read())['id']
        for colors in (64, 256):
            code, bz = jreq('/api/bundle', {'pid': cpid, 'N': 8, 'fps': 12,
                                            'colors': colors})
            code, zraw = req(bz['url'])
            zf = zipfile.ZipFile(io.BytesIO(zraw))
            pal = np.asarray(Image.open(io.BytesIO(zf.read('palette.png'))))
            check(f'bundle palette honours colours={colors}', pal.shape[1] == colors,
                  f'{pal.shape[1]} entries')
        code, gif = jreq('/api/gif', {'pid': cpid, 'N': 8, 'fps': 12, 'colors': 64})
        check('gif honours colours=64', gif['colors'] == 64, str(gif['colors']))

        print()
        if FAILS:
            print(f'{len(FAILS)} M4 CHECK(S) FAILED: {FAILS}')
            return 1
        print('ALL M4 ACCEPTANCE CHECKS PASSED')
        return 0
    finally:
        srv.terminate()
        srv.wait(timeout=10)


if __name__ == '__main__':
    sys.exit(main())
