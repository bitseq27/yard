"""M2 acceptance test — end to end over HTTP (plan §6).

Starts the real server on a spare port and checks:
  M1 regression  1536x2048 upload -> grid 4 -> native 384x512 in < 2 s; routes 200/404
  M2 mask        wand grabs a fern in ONE click at tolerance 30
  M2 autoseg     >= 80% of the courtyard's objects returned as <= 60 suggestions
                 (ground truth = origin session's seg.npy / seg_legend.json)
  M2 persist     mask upsert -> PNG on disk -> reload round-trips exactly
  M2 edge        grow/shrink/outline/hole-fill behave (px monotonic, rim only, no holes)

Run:  python3 app/tests/test_m2.py
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gt_stats                                             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PORT = int(os.environ.get('M2_TEST_PORT', 8137))
BASE = f'http://127.0.0.1:{PORT}'
FAILS = []


def _hsv(a):
    arr = np.asarray(a, np.float32) / 255.0
    cmax, cmin = arr.max(-1), arr.min(-1)
    df = cmax - cmin
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    hue = np.zeros(arr.shape[:-1], np.float32)
    m = (cmax == r) & (df > 0)
    hue[m] = ((g - b)[m] / df[m]) % 6
    m = (cmax == g) & (df > 0)
    hue[m] = ((b - r)[m] / df[m]) + 2
    m = (cmax == b) & (df > 0)
    hue[m] = ((r - g)[m] / df[m]) + 4
    return hue * 60, np.where(cmax > 0, df / np.maximum(cmax, 1e-6), 0), cmax


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
        with urllib.request.urlopen(r, timeout=120) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def jreq(path, body=None):
    code, raw = req(path, body)
    return code, json.loads(raw.decode())


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
        else:
            raise RuntimeError('server did not come up')
        code, ping = jreq('/api/ping')
        check('ping', ping.get('ok') and ping.get('milestone', 'M1') >= 'M2', str(ping))

        # ---- M1 regression: upload + grid detect ---------------------------
        print('\nM1 regression')
        with open(os.path.join(ROOT, 'uploads', 'iso_00080_.png'), 'rb') as f:
            raw = f.read()
        t0 = time.time()
        code, rawmeta = req('/api/upload', raw=raw)
        meta = json.loads(rawmeta.decode())
        dt = time.time() - t0
        check('upload 200', code == 200, f'{code}')
        check('grid detected = 4', meta.get('grid') == 4, str(meta.get('grid')))
        check('native 384x512', (meta.get('native_w'), meta.get('native_h')) == (384, 512),
              f"{meta.get('native_w')}x{meta.get('native_h')}")
        check('round-trip < 2 s', dt < 2.0, f'{dt * 1000:.0f} ms')
        pid = meta['id']
        check('native.png served', req(f'/api/image/{pid}/native.png')[0] == 200)
        check('unknown route 404', req('/api/nope')[0] == 404)
        check('unknown project 404', req('/api/project/deadbeef')[0] == 404)

        # ---- M2: magic wand grabs a fern in one click at tol 30 -------------
        print('\nM2 wand')
        seg, gt = gt_stats.objects(25)
        nat = np.asarray(Image.open(os.path.join(ROOT, 'native.png'))).astype(np.int16)
        hue, sat, val = _hsv(nat)

        def frond_click(name):
            """Where a human clicks: the pixel closest to the object's mid-tone colour."""
            g = seg == next(o for o in gt if o['name'] == name)['cid']
            P = nat[g]
            ss, vv = sat[g], val[g]
            mid = (ss >= 0.35) & (vv >= 0.25) & (vv <= 0.85)
            med = np.median(P[mid], 0) if mid.sum() else np.median(P, 0)
            d = np.abs(P - med.astype(np.int16)).sum(1)
            yy, xx = np.nonzero(g)
            k = int(d.argmin())
            return g, int(xx[k]), int(yy[k])

        def wand_at(cx, cy, **kw):
            body = {'pid': pid, 'x': cx, 'y': cy, 'tol': 30, 'conn': 8, 'smooth': True}
            body.update(kw)
            code, w = jreq('/api/wand', body)
            wm = np.zeros((512, 384), bool)
            rle = w.get('rle', [])
            for i in range(0, len(rle), 3):
                wm[rle[i], rle[i + 1]:rle[i + 2] + 1] = True
            return code, w, wm

        fern = next(o for o in gt if o['name'] == 'plant_9')       # the courtyard fern
        gtm, fx, fy = frond_click('plant_9')
        code, w, wm = wand_at(fx, fy)
        cover = (wm & gtm).sum() / fern['px']
        check('wand 200', code == 200, f'{code}')
        check('one click @ tol 30 grabs the fern', w.get('px', 0) >= 200 and cover >= 0.25,
              f"click ({fx},{fy}) -> {w.get('px')}px, {cover * 100:.0f}% of the fern object")
        check('wand is not the whole frame', w.get('px', 0) < 0.12 * 384 * 512,
              f"{w.get('px')}px = {w.get('px', 0) / (384 * 512) * 100:.1f}% of frame")
        # leak guard: a shared hue family (terracotta pots vs wood doors) must stay local
        pg, px_, py_ = frond_click('pot_3')
        pot = next(o for o in gt if o['name'] == 'pot_3')
        code, w2, wm2 = wand_at(px_, py_)
        check('leak guard keeps shared materials local',
              wm2.sum() < 0.03 * 384 * 512 and (wm2 & pg).sum() / pot['px'] >= 0.5,
              f"{w2.get('px')}px = {w2.get('px', 0) / (384 * 512) * 100:.1f}% of frame, "
              f"{(wm2 & pg).sum() / pot['px'] * 100:.0f}% of the pot")
        code, w3, wm3 = wand_at(fx, fy, mode='seed', snap=48)
        check('seed metric still available', code == 200 and wm3.sum() >= 1, f"{w3.get('px')}px")

        # ---- M2: auto-seg >=80% of objects as <=60 suggestions --------------
        print('\nM2 auto-seg')
        t0 = time.time()
        code, sg = jreq('/api/autoseg', {'pid': pid, 'max_suggestions': 60})
        dt = time.time() - t0
        check('autoseg 200', code == 200, f'{code}')
        sugs = sg.get('suggestions', [])
        check('<= 60 suggestions', len(sugs) <= 60, f'{len(sugs)}')
        masks = []
        for s in sugs:
            m = np.zeros((512, 384), bool)
            rle = s['rle']
            for i in range(0, len(rle), 3):
                m[rle[i], rle[i + 1]:rle[i + 2] + 1] = True
            masks.append(m)
        hit = 0
        for o in gt:
            g = seg == o['cid']
            for m in masks:
                if (m & g).sum() / o['px'] >= 0.5 or (m & g).sum() / max(1, (m | g).sum()) >= 0.3:
                    hit += 1
                    break
        rec = hit / len(gt)
        check('>= 80% of courtyard objects returned', rec >= 0.8,
              f'{hit}/{len(gt)} = {rec * 100:.1f}%')
        check('autoseg fast enough for a button', dt < 15, f'{dt:.1f} s')
        check('suggestions carry name/kind/colour/bbox', all(
            s.get('name') and s.get('kind') in ('object', 'region') and len(s.get('color', [])) == 3
            and s.get('bbox') for s in sugs))

        # ---- M2: auto-seg settings reach the engine ------------------------
        print('\nM2 auto-seg settings')
        code, r1 = jreq('/api/autoseg', {'pid': pid, 'max_suggestions': 60,
                                         'cfg': {'include_regions': False}})
        check('setting: include_regions=false -> objects only',
              all(s['kind'] == 'object' for s in r1['suggestions']),
              f"{len(r1['suggestions'])} suggestions")
        code, r2 = jreq('/api/autoseg', {'pid': pid, 'max_suggestions': 60,
                                         'cfg': {'min_object_px': 400}})
        plants2 = [s['px'] for s in r2['suggestions'] if s.get('cls') == 'plant']
        check('setting: min_object_px filters small objects',
              plants2 and min(plants2) >= 400,
              f"smallest plant {min(plants2)}px (other kinds keep their own floors)")
        code, r3 = jreq('/api/autoseg', {'pid': pid, 'max_suggestions': 20})
        check('setting: cap is honoured', len(r3['suggestions']) <= 20, f"{len(r3['suggestions'])}")
        code, r4 = jreq('/api/autoseg', {'pid': pid, 'max_suggestions': 9999,
                                         'cfg': {'min_object_px': -5, 'merge_k': 'xx',
                                                 'bogus_key': 1, 'absorb_gap': 1e9}})
        check('setting: bogus/hostile cfg is coerced, not crashed',
              code == 200 and 0 < len(r4['suggestions']) <= 200, f'{code}, {len(r4["suggestions"])}')
        code, r5 = jreq('/api/autoseg', {'pid': pid, 'max_suggestions': 60, 'conn': 4})
        check('setting: connectivity accepted', code == 200 and r5['ok'])

        # ---- M2: object recognition ----------------------------------------
        print('\nM2 recognition')
        check('suggestions carry label/conf/anim', all(
            s.get('label') and isinstance(s.get('conf'), (int, float)) and 'anim' in s
            for s in sugs))
        fix = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          'recog_fixture.json')))
        gtmask = {o['name']: (seg == o['cid']) for o in gt}
        pool_cid = next(int(k) for k, v in gt_stats.load()[1].items() if v['name'] == 'pool_water')
        gtmask['pool_water'] = seg == pool_cid
        good, seen = 0, 0
        detail = []
        for gname, want in fix.items():
            g = gtmask[gname]
            px = int(g.sum())
            hit = [(sg, mk) for sg, mk in zip(sugs, masks) if (mk & g).sum() / px >= 0.5]
            if not hit:
                detail.append(f'{gname}: not returned')
                continue
            seen += 1
            lab = max(hit, key=lambda t: (t[1] & g).sum())[0]['label']
            good += lab == want
            detail.append(f'{gname}->{lab}{"!" if lab != want else ""}')
        check('recognised labels match the hand-labelled fixture', seen >= 8 and good / seen >= 0.8,
              f'{good}/{seen}: ' + ', '.join(detail))
        census = {}
        for sg in sugs:
            census[sg['label']] = census.get(sg['label'], 0) + 1
        check('scene inventory is plausible', census.get('potted plant', 0) >= 5
              and census.get('rug', 0) + census.get('runner rug', 0) >= 3
              and census.get('palm', 0) >= 1 and census.get('pool', 0) >= 1, str(census))
        code, r6 = jreq('/api/autoseg', {'pid': pid, 'max_suggestions': 60,
                                         'cfg': {'recognise': False}})
        check('setting: recognise=false keeps class names', all(
            s['label'] == s['cls'] and s['conf'] == 0 for s in r6['suggestions']))

        # ---- M2: import + persist round-trip ------------------------------
        print('\nM2 persist')
        import random
        rng = random.Random(7)
        test_mask = np.zeros((512, 384), bool)
        test_mask[100:140, 60:110] = True
        test_mask[120:160, 90:130] = True
        rle = []
        for y in range(512):
            x = 0
            while x < 384:
                if test_mask[y, x]:
                    x0 = x
                    while x + 1 < 384 and test_mask[y, x + 1]:
                        x += 1
                    rle += [y, x0, x]
                x += 1
        code, r = jreq('/api/masks', {'pid': pid, 'ops': [
            {'op': 'upsert', 'id': 'aa11', 'name': 'fern test', 'kind': 'object',
             'color': [200, 60, 60], 'rle': rle}]})
        check('mask upsert 200', code == 200, f'{code}')
        check('mask PNG on disk', os.path.exists(os.path.join(
            ROOT, 'app', 'projects', pid, 'masks', 'aa11.png')))
        code, doc = jreq(f'/api/project/{pid}')
        check('project doc lists the mask', any(o['id'] == 'aa11' and o['name'] == 'fern test'
                                                 and o['px'] == int(test_mask.sum())
                                                 for o in doc['objects']),
              str([(o['id'], o['px']) for o in doc['objects']]))
        code, png = req(f'/api/mask/{pid}/aa11.png')
        import io as _io
        back = np.asarray(Image.open(_io.BytesIO(png)).convert('L')) > 127
        check('mask round-trips exactly', bool((back == test_mask).all()),
              f'{int(back.sum())} vs {int(test_mask.sum())} px')
        code, r = jreq('/api/masks', {'pid': pid, 'ops': [{'op': 'delete', 'id': 'aa11'}]})
        code, doc = jreq(f'/api/project/{pid}')
        check('mask delete', not any(o['id'] == 'aa11' for o in doc['objects']))

        # ---- M2: edge helpers ---------------------------------------------
        print('\nM2 edge ops')
        sq = []
        m0 = np.zeros((512, 384), bool)
        m0[50:90, 50:90] = True
        m0[60, 60] = False                      # a hole to fill
        for y in range(512):
            x = 0
            while x < 384:
                if m0[y, x]:
                    x0 = x
                    while x + 1 < 384 and m0[y, x + 1]:
                        x += 1
                    sq += [y, x0, x]
                x += 1
        base_px = int(m0.sum())
        dil = m0.copy()
        dil[1:, :] |= m0[:-1, :]
        dil[:-1, :] |= m0[1:, :]
        dil[:, 1:] |= m0[:, :-1]
        dil[:, :-1] |= m0[:, 1:]
        code, g = jreq('/api/edge', {'pid': pid, 'rle': sq, 'op': 'grow'})
        check('grow adds exactly a 1px rim', g['px'] == int(dil.sum()),
              f"{base_px} -> {g['px']} (expect {int(dil.sum())})")
        ero = m0.copy()
        ero[1:, :] &= m0[:-1, :]
        ero[:-1, :] &= m0[1:, :]
        ero[:, 1:] &= m0[:, :-1]
        ero[:, :-1] &= m0[:, 1:]
        code, s = jreq('/api/edge', {'pid': pid, 'rle': sq, 'op': 'shrink'})
        check('shrink removes exactly the rim', s['px'] == int(ero.sum()),
              f"{base_px} -> {s['px']} (expect {int(ero.sum())})")
        code, o = jreq('/api/edge', {'pid': pid, 'rle': sq, 'op': 'outline'})
        check('outline keeps only the rim', o['px'] < base_px and o['px'] > 0, f"{o['px']}px rim")
        code, f = jreq('/api/edge', {'pid': pid, 'rle': sq, 'op': 'fill'})
        check('hole-fill closes the hole', f['px'] == base_px + 1, f"{base_px} -> {f['px']}")
        # solid: two blocks 2px apart + a hole -> one component, no holes
        lac = np.zeros((512, 384), bool)
        lac[50:60, 50:60] = True
        lac[50:60, 62:72] = True            # 1px gap column at x=61
        lac[80:90, 50:60] = True
        lac[84, 54] = False                 # enclosed hole
        lr = []
        for y in range(512):
            x = 0
            while x < 384:
                if lac[y, x]:
                    x0 = x
                    while x + 1 < 384 and lac[y, x + 1]:
                        x += 1
                    lr += [y, x0, x]
                x += 1
        code, so = jreq('/api/edge', {'pid': pid, 'rle': lr, 'op': 'solid'})
        sm = np.zeros((512, 384), bool)
        for i in range(0, len(so['rle']), 3):
            sm[so['rle'][i], so['rle'][i + 1]:so['rle'][i + 2] + 1] = True
        sys.path.insert(0, os.path.join(ROOT, 'app'))
        from engine import imaging as _im
        comps = _im.components(sm, 1, 8)
        inner = ~_im.scanline_flood(~sm, [(0, 0)], 8)      # everything inside the outer rim
        # the 1px gap and the enclosed hole are gone; the far-away block stays its own
        # component (solid must not glue disconnected parts together)
        check('solid bridges gaps and fills holes',
              len(comps) == 2 and bool(sm[55, 61]) and bool(sm[84, 54])
              and not (inner & ~sm).any() and so['px'] > int(lac.sum()),
              f"{int(lac.sum())}px/3 comps -> {so['px']}px/{len(comps)} comps, "
              f"gap={bool(sm[55, 61])}, hole={bool(sm[84, 54])}")
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=5)
        except subprocess.TimeoutExpired:
            srv.kill()

    print('\n' + ('ALL M2 ACCEPTANCE CHECKS PASSED' if not FAILS
                  else f'{len(FAILS)} FAILURES: {FAILS}'))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
