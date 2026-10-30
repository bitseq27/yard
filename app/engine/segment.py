"""Pixel Sway engine — auto-segmentation (plan §4 "one-click auto-segment suggestions").

Ported from the origin session's courtyard pipeline (`seg_depth.py`) but made *generic*:
the old script hardcoded courtyard polygons (pool quad, wall poly) and sampled floor
colours from fixed pixel patches. None of that can ship in an app that accepts arbitrary
pixel art, so the geometry is replaced by portable rules:

  1. Colour classes are decided per *palette entry* after a palette snap, not per pixel —
     dither noise cannot shred an object into a hundred classes.
  2. A core class covers only ~50% of an object's real pixels (measured against the origin
     ground truth: the rest is interior shading/outline that landed in other classes), so
     every core is **grown** into its own silhouette (dilate -> hole-fill -> erode,
     windowed, ratio-capped).
  3. One object is usually *several* colour-class fragments (a rug = red field + border
     motifs, a palm = a few frond clusters), so same-class fragments within `merge_k` px
     are merged before growing. Without this the candidate count is ~2x the object count
     and the suggestion cap throws real objects away.
  4. A "background sheet" test (big + solid) demotes floor/wall-scale blobs out of the
     object list — that is what the hardcoded floor-patch sampling used to achieve.
  5. Water touching the frame border is floor/wall *material*; only interior, compact
     water is a pool, and only pools can hold lily pads.

Suggestions are never authoritative (plan §7): the editor imports them as editable masks.
"""
import time

import numpy as np

from . import imaging as im
from . import recognise as rec

# kind: 'object' = animatable / game object (plan §8), 'region' = plate material
# (floor/wall/water sheet), unchecked by default in the import dialog.
CLASS_INFO = {
    'plant':     ('object', (90, 210, 90)),
    'lily_pad':  ('object', (70, 200, 160)),
    'pot':       ('object', (210, 130, 60)),
    'carpet':    ('object', (225, 80, 90)),
    'fringe':    ('object', (245, 225, 150)),
    'water':     ('object', (60, 130, 220)),
    'stone':     ('region', (200, 200, 210)),
    'wood':      ('region', (150, 100, 55)),
    'shadow':    ('region', (70, 70, 90)),
    'light':     ('region', (240, 240, 235)),
    'ground':    ('region', (110, 170, 120)),
    'misc':      ('region', (130, 130, 140)),
}

DEFAULTS = dict(
    snap=96,               # palette colours used for class decisions
    min_object_px=25,      # origin used 25px for plants; below that is dither speckle
    min_core_px=12,        # smaller *cores* are allowed: growth usually lifts them over
                           # min_object_px, and dropping them here lost real small plants
    min_carpet_px=60,
    min_pot_px=25,
    min_pad_px=16,
    min_region_px_frac=0.004,
    sheet_area_frac=0.02,  # >=2% of frame ...
    sheet_fill=0.72,       # ... and this solid -> background material, not an object
    huge_area_frac=0.06,   # >=6% of frame -> never an object, whatever the shape
    close_plants=1,        # morphological close iterations before component split
    close_carpets=2,
    pad_water_frac=0.50,   # share of a green core inside a pool -> lily pad
    pad_max_px=2600,
    grow_r=2,              # core growth: dilate radius
    grow_erode=1,          # core growth: erode back (net +1px halo, plan §4 halo control)
    grow_max_ratio=3.5,    # grown area cap vs core area
    merge_k=2,             # same-class fragments within 2*k px are one object
    # per-kind ceiling on a merged object, as a fraction of the frame. Without it,
    # chained merges produce 12%-of-frame "plants" that are not editable objects.
    max_px_frac={'plant': 0.05, 'carpet': 0.09, 'pot': 0.012, 'lily_pad': 0.012,
                 'fringe': 0.01, 'water': 0.20},
    pot_max_h=26, pot_max_w=28,
    pot_dir='any',         # foliage near a brown comp ('any') or only above it ('above')
    pot_near=4,
    max_per_kind=45,       # safety valve: no single class may flood the suggestion list
    absorb_gap=14,         # over-cap candidates merge into a same-kind neighbour this close
                           # (sweep: 14 -> 85.9% recall vs 8 -> 81.2%; see tests/sweep.py)
    kind_floor=3,          # every detected kind keeps at least this many suggestion slots
    region_reserve=4,      # suggestion slots kept for plate material (walkmask/regions)
    include_regions=True,  # UI toggle: offer plate material alongside recognised objects
    recognise=True,        # UI toggle: name suggestions by recognised object type
    min_suggestions_cap=200,
)

# claiming priority: small/under-foot objects first so big growers cannot eat them
ORDER = ('pot', 'lily_pad', 'plant', 'carpet', 'fringe', 'water')
MIN_PX_KEY = {'pot': 'min_pot_px', 'lily_pad': 'min_pad_px', 'carpet': 'min_carpet_px',
              'plant': 'min_object_px', 'fringe': None, 'water': 'min_region_px'}


# ---------------------------------------------------------------- colour classes


def classify(a, cfg=None):
    """Palette-snap RGB -> (class map int16, class names, cfg).

    Rule order matters (a palette entry takes the first class it matches): shadow, water,
    carpet, blossom, brown, light, stone, fringe, plant, ground, misc.
    """
    d = dict(DEFAULTS, **(cfg or {}))
    idx, pal = im.quantize(a, d['snap'])
    hue, sat, val = im.hsv_maps(pal)
    r, g, b = (pal[:, i].astype(np.int16) for i in range(3))

    names = []

    def cid(name):
        if name not in names:
            names.append(name)
        return names.index(name) + 1          # 0 = unassigned

    cls_entry = np.zeros(len(pal), np.int16)
    for i in range(len(pal)):
        h, s, v = float(hue[i]), float(sat[i]), float(val[i])
        green = 55 <= h <= 180 and s > 0.22 and v > 0.10 and g[i] > b[i]
        if v < 0.16:
            n = 'shadow'
        elif green:
            n = 'plant'                       # green wins over teal: pads are yellow-green
        elif 160 <= h <= 270 and s > 0.20 and v > 0.12:
            n = 'water'
        elif (h < 15 or h > 345) and s > 0.35 and v > 0.18:
            n = 'carpet'
        elif 285 <= h <= 345 and s > 0.22 and v > 0.30:
            n = 'blossom'                     # pink flowers: foliage, not carpet
        elif 8 <= h <= 45 and s > 0.28 and 0.15 <= v <= 0.88 and r[i] >= g[i] >= b[i]:
            n = 'brown'
        elif s < 0.22 and v > 0.72:
            n = 'light'
        elif s < 0.45 and v > 0.42 and 10 <= h <= 85:
            n = 'stone'
        elif s < 0.35 and v > 0.55 and (h < 70 or h > 300):
            n = 'fringe'
        elif 15 <= h <= 50 and s > 0.40 and v > 0.45:
            n = 'blossom'                     # orange blossoms
        elif 70 <= h <= 165 and s > 0.18 and 0.15 <= v <= 0.70:
            n = 'ground'                      # olive/desaturated green: floor, not foliage
        else:
            n = 'misc'
        cls_entry[i] = cid(n)

    return cls_entry[idx], names, d


def _mask(cls, names, name):
    if name not in names:
        return np.zeros(cls.shape, bool)
    return cls == (names.index(name) + 1)


def _is_sheet(px, bbox, total, d):
    """Big + solid = background material (floor/wall), not an object."""
    if px >= d['huge_area_frac'] * total:
        return True
    x0, y0, x1, y1 = bbox
    area = max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
    return px >= d['sheet_area_frac'] * total and (px / area) >= d['sheet_fill']


def _touches_border(bbox, shape):
    x0, y0, x1, y1 = bbox
    return x0 <= 0 or y0 <= 0 or x1 >= shape[1] - 1 or y1 >= shape[0] - 1


# ---------------------------------------------------------------- merge + grow


def _agglomerate(masks, shape, k, max_group_px, conn=8):
    """Group near-touching fragments, never letting a group exceed `max_group_px`.

    Plain "dilate the union then take components" chains: A~B~C~... can swallow a whole
    hedge into one 12%-of-frame "object" (measured: 23700px on the courtyard). Real objects
    have a size ceiling per kind, so agglomeration is greedy + size-capped: biggest fragment
    seeds a group, neighbours join the largest group they touch while it stays under budget.
    """
    n = len(masks)
    if n == 1:
        return [[masks[0]]]
    lab = np.zeros(shape, np.int16)
    for i, m in enumerate(masks):
        lab[m] = i + 1
    pxs = [int(m.sum()) for m in masks]
    bboxes = [im.bbox_of(m) for m in masks]
    adj = [set() for _ in range(n)]
    for i, m in enumerate(masks):
        y0, y1, x0, x1 = im.window(bboxes[i], shape, k + 1)
        dsub = im.dilate4(m[y0:y1, x0:x1].copy(), k)
        for v in np.unique(lab[y0:y1, x0:x1][dsub]).tolist():
            j = int(v) - 1
            if 0 <= j < n and j != i:
                adj[i].add(j)
                adj[j].add(i)
    groups = []
    for i in sorted(range(n), key=lambda i: -pxs[i]):
        best = None
        for gi, g in enumerate(groups):
            if max_group_px and g['px'] + pxs[i] > max_group_px:
                continue
            if adj[i] & g['ids'] and (best is None or g['px'] > groups[best]['px']):
                best = gi
        if best is None:
            groups.append({'ids': {i}, 'px': pxs[i]})
        else:
            groups[best]['ids'].add(i)
            groups[best]['px'] += pxs[i]
    return [[masks[i] for i in sorted(g['ids'])] for g in groups]


def _merge_and_grow(cores, shape, d, conn=8, grow_r=None, max_ratio=None, restrict=None,
                    min_px=1, max_group_px=None):
    """Group near-touching same-class fragments, then grow each group into its silhouette."""
    if not cores:
        return []
    grow_r = d['grow_r'] if grow_r is None else grow_r
    max_ratio = d['grow_max_ratio'] if max_ratio is None else max_ratio
    out = []
    for group in _agglomerate(cores, shape, d['merge_k'], max_group_px, conn):
        m = group[0].copy()
        for extra in group[1:]:
            m |= extra
        px = int(m.sum())
        if px < min_px:
            continue
        g = im.grow_core(m, im.bbox_of(m), grow_r, d['grow_erode'], max_ratio)
        if restrict is not None:
            g &= restrict
        if int(g.sum()) < min_px:
            continue
        out.append((g, im.bbox_of(g), int(g.sum()), len(group)))
    return out



def _bbox_gap(a, b):
    """Chebyshev pixel gap between two bboxes (0 = touching/overlapping)."""
    if not a or not b:
        return 10 ** 6
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]) - 1)
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]) - 1)
    return max(dx, dy)


def _fit_to_cap(objects, cap, d, total):
    """Trim candidates to `cap` without losing coverage.

    Two policies, both measured against the courtyard ground truth:
      * A size-only cap ("keep the 60 biggest") threw away every pot and lily pad —
        they are small — and recall fell 80% -> 68%. So slots are allocated **per kind**,
        proportional to how many candidates that kind produced, with a small floor.
      * A kind's overflow is **absorbed** into its nearest kept neighbour (bbox gap <=
        absorb_gap) rather than dropped: coverage is preserved and the result stays
        editable. Overflow with no neighbour that close is genuine noise -> dropped.
    Absorbed suggestions carry nfrag > 1; the UI shows that so the user can split them
    (plan §8: connected objects -> user combines or splits at a cut line).
    """
    if cap <= 0:
        return [], len(objects)
    objs = [[base, m, bbox, px, nfrag] for base, m, bbox, px, nfrag in objects]
    if len(objs) <= cap:
        return objs, 0

    kinds = {}
    for o in objs:
        kinds.setdefault(o[0], []).append(o)
    n_all = len(objs)
    quota = {k: max(min(d['kind_floor'], len(v)), int(round(cap * len(v) / n_all)))
             for k, v in kinds.items()}
    guard = 0
    while sum(quota.values()) > cap and guard < 1000:
        k = max(quota, key=lambda k: (quota[k], len(kinds[k])))
        if quota[k] <= 1:
            break
        quota[k] -= 1
        guard += 1
    guard = 0
    while sum(quota.values()) < cap and guard < 1000:
        cands = [k for k in kinds if quota[k] < len(kinds[k])]
        if not cands:
            break
        quota[max(cands, key=lambda k: len(kinds[k]) - quota[k])] += 1
        guard += 1

    kept, dropped = [], 0
    for k, v in kinds.items():
        v.sort(key=lambda t: -t[3])
        keep = v[:quota[k]]
        for small in sorted(v[quota[k]:], key=lambda t: t[3]):
            best, best_d = None, 10 ** 9
            for o in keep:
                dist = _bbox_gap(small[2], o[2])
                if dist < best_d:
                    best, best_d = o, dist
            if best is None or best_d > d['absorb_gap']:
                dropped += 1
                continue
            best[1] |= small[1]
            best[2] = im.bbox_of(best[1])
            best[3] = int(best[1].sum())
            best[4] += small[4]
        kept.extend(keep)
    return kept, dropped


# ---------------------------------------------------------------- main entry


def auto_segment(a, max_suggestions=60, cfg=None, conn=8):
    """Auto-segment a native-res RGB array into editable object/region suggestions.

    Returns {'suggestions': [{'id','name','kind','color','mask','px','bbox','depth'}],
             'classes', 'n_objects', 'n_regions', 'dropped', 'ms', 'cfg'}
    """
    t0 = time.time()
    d = dict(DEFAULTS, **(cfg or {}))
    a = np.asarray(a, np.uint8)
    H, W = a.shape[:2]
    total = H * W
    cls, names, d = classify(a, d)

    water = _mask(cls, names, 'water')
    plant_raw = _mask(cls, names, 'plant')
    blossom = _mask(cls, names, 'blossom')
    carpet_raw = _mask(cls, names, 'carpet')
    brown_raw = _mask(cls, names, 'brown')
    light = _mask(cls, names, 'light')
    fringe_raw = _mask(cls, names, 'fringe')
    stone = _mask(cls, names, 'stone')
    shadow = _mask(cls, names, 'shadow')
    ground = _mask(cls, names, 'ground')
    misc = _mask(cls, names, 'misc')
    min_region = max(8, int(d['min_region_px_frac'] * total))

    # per-pixel feature maps for the recogniser (on the palette-snapped colours, so
    # dither cannot sway a feature): hue/sat/val + the three palette families the
    # recognition rules look at
    qh, qs, qv = im.hsv_maps(_denoised(a, d))
    qr, qg, qb = (_denoised(a, d)[..., i].astype(np.int16) for i in range(3))
    F = {
        'H': H, 'W': W, 'hue': qh, 'sat': qs, 'val': qv,
        'green': (qh >= 55) & (qh <= 180) & (qs > 0.22) & (qv > 0.10) & (qg > qb),
        'brown': (qh >= 8) & (qh <= 45) & (qs > 0.28) & (qr >= qg) & (qg >= qb) & (qv >= 0.15),
        'blossom': (((qh >= 285) & (qh <= 345) & (qs > 0.22) & (qv > 0.30))
                    | ((qh >= 15) & (qh <= 50) & (qs > 0.40) & (qv > 0.45))
                    | ((qs < 0.22) & (qv > 0.72))),
    }

    # ---- water: interior compact bodies are pools; border-touching sheets are material
    pool = np.zeros((H, W), bool)
    water_sheet = np.zeros((H, W), bool)
    for c in im.components(water, min_region, conn):
        m = im.mask_from_runs(c['runs'], (H, W))
        if not _touches_border(c['bbox'], (H, W)) and c['px'] <= 0.30 * total:
            pool |= m
        else:
            water_sheet |= m
    pool_d = im.dilate4(pool.copy(), 3)

    # ---- blossoms adjacent to greenery join the foliage (origin masks.py rule)
    plant_seed = plant_raw.copy()
    if blossom.any():
        plant_seed |= blossom & im.dilate4(plant_raw.copy(), 2)
    plant_c = im.close4(plant_seed, d['close_plants']) if d['close_plants'] else plant_seed
    ctx = {'plant_d': im.dilate4(plant_seed, 4)}

    raw = {k: [] for k in ORDER}
    wood = np.zeros((H, W), bool)

    # ---- pots: brown comps with foliage directly above, compact, not a sheet
    plant_d3 = im.dilate4(plant_seed, 3)
    for c in im.components(im.close4(brown_raw.copy(), 1), 8, conn):
        m = im.mask_from_runs(c['runs'], (H, W))
        x0, y0, x1, y1 = c['bbox']
        h_, w_ = y1 - y0, x1 - x0
        if d['pot_dir'] == 'above':
            band = plant_d3[max(y0 - 4, 0):y0 + 2, x0:x1 + 1]
        else:
            wy0, wy1, wx0, wx1 = im.window(c['bbox'], (H, W), d['pot_near'])
            band = plant_d3[wy0:wy1, wx0:wx1]
        fill = c['px'] / max(1.0, (h_ + 1) * (w_ + 1))
        if (c['px'] >= d['min_pot_px'] and h_ <= d['pot_max_h'] and w_ <= d['pot_max_w']
                and fill >= 0.30 and bool(band.any())
                and not _is_sheet(c['px'], c['bbox'], total, d)):
            raw['pot'].append(m)
        else:
            wood |= m

    # ---- plants vs lily pads (a pad is a green core sitting in a pool)
    for c in im.components(plant_c, 12, conn):
        m = im.mask_from_runs(c['runs'], (H, W))
        if _is_sheet(c['px'], c['bbox'], total, d):
            ground |= m                        # olive floor sheet, not foliage
            continue
        in_pool = float(pool_d[m].mean()) if pool_d.any() else 0.0
        if in_pool >= d['pad_water_frac'] and c['px'] <= d['pad_max_px']:
            raw['lily_pad'].append(m)
        elif c['px'] >= d['min_core_px']:
            raw['plant'].append(m)

    # ---- carpets: rugs are fragmented red, so close hard before splitting
    carpet_cores = []
    for c in im.components(im.close4(carpet_raw.copy(), d['close_carpets']), 12, conn):
        m = im.mask_from_runs(c['runs'], (H, W)) & ~pool
        if m.sum() >= 12:
            carpet_cores.append(m)
    raw['carpet'] = carpet_cores

    # ---- fringe: bright desaturated tassels hugging a carpet (origin rule)
    carpet_union = np.zeros((H, W), bool)
    for m in carpet_cores:
        carpet_union |= m
    if carpet_union.any():
        fr = fringe_raw & im.dilate4(carpet_union.copy(), 2)
        raw['fringe'] = [im.mask_from_runs(c['runs'], (H, W))
                         for c in im.components(fr, 6, conn)]

    # ---- pools themselves are objects (ripple animates water surfaces)
    raw['water'] = [im.mask_from_runs(c['runs'], (H, W))
                    for c in im.components(pool, min_region, conn)]

    # ---- merge fragments + grow, claiming pixels in priority order
    cores = []
    claimed = np.zeros((H, W), bool)
    for base in ORDER:
        cs = [m & ~claimed for m in raw[base]]
        cs = [m for m in cs if int(m.sum()) >= 4]
        if not cs:
            continue
        key = MIN_PX_KEY.get(base)
        min_px = min_region if key == 'min_region_px' else (d[key] if key else 6)
        budget = d['max_px_frac'].get(base)
        max_group_px = int(budget * total) if budget else None
        if base == 'lily_pad':
            for g, bbox, px, nfrag in _merge_and_grow(cs, (H, W), d, conn, grow_r=2,
                                                      max_ratio=8.0, restrict=pool_d,
                                                      min_px=d['min_pad_px'],
                                                      max_group_px=max_group_px):
                g |= light & pool_d & im.dilate4(g.copy(), 2) & ~claimed   # rim/highlights
                g &= ~claimed
                claimed |= g
                cores.append((base, g, im.bbox_of(g), int(g.sum()), nfrag))
            continue
        for g, bbox, px, nfrag in _merge_and_grow(cs, (H, W), d, conn, min_px=min_px,
                                                  max_group_px=max_group_px):
            g &= ~claimed
            if int(g.sum()) < min_px:
                continue
            claimed |= g
            cores.append((base, g, im.bbox_of(g), int(g.sum()), nfrag))

    # ---- safety valve: cap runaway classes, keep the biggest
    kept, per_kind = [], {}
    for base, m, bbox, px, nfrag in sorted(cores, key=lambda t: -t[3]):
        n = per_kind.get(base, 0)
        if n >= d['max_per_kind']:
            continue
        per_kind[base] = n + 1
        kept.append((base, m, bbox, px, nfrag))
    cores = kept

    # ---- regions: leftover plate material (optional in the UI)
    regions = []
    region_defs = (('water', water_sheet), ('stone', stone), ('wood', wood),
                   ('shadow', shadow), ('ground', ground), ('light', light),
                   ('misc', misc))
    for name, base in (region_defs if d['include_regions'] else ()):
        base = base & ~claimed
        if not base.any():
            continue
        comps = im.components(base, min_region, conn)[:6]
        for i, c in enumerate(comps):
            m = im.mask_from_runs(c['runs'], (H, W)) & ~claimed
            if m.sum() < min_region:
                continue
            regions.append((name if len(comps) == 1 else f'{name}_{i}', m, 'region'))

    # ---- relevance filter: below a kind's min px a candidate is noise, so it is dropped
    # (distinct from the cap policy, where over-cap candidates are absorbed, not dropped)
    kind_min = {'plant': d['min_object_px'], 'pot': d['min_pot_px'],
                'carpet': d['min_carpet_px'], 'lily_pad': d['min_pad_px'],
                'fringe': 6, 'water': min_region}
    cores = [c for c in cores if c[3] >= kind_min.get(c[0], 1)]

    # ---- cap: objects first (the point of auto-seg); over-cap candidates are absorbed
    # into their nearest same-kind neighbour rather than dropped (see _fit_to_cap).
    # A few slots stay reserved for plate material: absorbing costs no coverage, so
    # reserving region slots is free.
    room = max(0, int(max_suggestions))
    reserve = min(d['region_reserve'], len(regions), max(0, room // 4))
    cores, n_absent = _fit_to_cap(cores, max(0, room - reserve), d, total)
    objects = sorted(cores, key=lambda t: -t[3])
    n_drop = n_absent
    out, counts = [], {}
    for base, m, bbox, px, nfrag in objects:
        i = counts.get(base, 0)
        counts[base] = i + 1
        out.append(_suggest(f'{base}_{i}', base, 'object', m, nfrag))
    for name, m, kind in regions:
        if len(out) >= max_suggestions:
            break
        out.append(_suggest(name, name.split('_')[0], kind, m))

    # UI order: objects in reading order (top-to-bottom, left-to-right), regions last
    out.sort(key=lambda s: (0 if s['kind'] == 'object' else 1,
                            round((s['bbox'][1] if s['bbox'] else 0) / 8),
                            round((s['bbox'][0] if s['bbox'] else 0) / 8), -s['px']))
    if d['recognise']:
        counters = {}
        for s in out:
            lab, conf = rec.recognise_with_context(s['mask'], F, s['kind'], s['cls'], ctx)
            s['label'], s['conf'] = lab, round(conf, 2)
            s['anim'] = rec.ANIM_HINT.get(lab, '')
            n = counters.get(lab, 0)
            counters[lab] = n + 1
            s['name'] = f'{lab} {n + 1}'
    else:
        for s in out:
            s['label'], s['conf'], s['anim'] = s['cls'], 0.0, ''
    for i, s in enumerate(out):
        s['id'] = f'sg{i:03d}'
        if s['kind'] == 'object':
            s['color'] = list(im.golden_color(i))

    classes = {n: int((cls == i + 1).sum()) for i, n in enumerate(names)}
    return {'suggestions': out, 'classes': {k: v for k, v in classes.items() if v},
            'total_px': total, 'n_objects': len(cores), 'n_regions': len(regions),
            'dropped': n_drop, 'ms': round((time.time() - t0) * 1000), 'cfg': d}


def _suggest(name, base, kind, mask, nfrag=1):
    st = im.stats_of(mask)
    col = CLASS_INFO.get(base, CLASS_INFO['misc'])[1]
    return {'name': name, 'kind': kind, 'cls': base, 'color': list(col), 'mask': mask,
            'px': st['px'], 'bbox': st['bbox'], 'depth': st['depth'], 'nfrag': nfrag}


def _denoised(a, d):
    """Palette-snapped RGB — the recogniser measures colours on this, never on dither."""
    idx, pal = im.quantize(np.asarray(a, np.uint8), d['snap'])
    return pal[idx].astype(np.uint8)


# ---------------------------------------------------------------- debug aid


def render_debug(res, shape):
    """RGB image of every suggestion in its chip colour (for eyeballing auto-seg)."""
    rgb = np.zeros(tuple(shape) + (3,), np.uint8)
    for s in res['suggestions']:
        rgb[s['mask']] = s['color']
    return rgb
