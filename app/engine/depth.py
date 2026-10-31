"""Pixel Sway engine — depth & surface-orientation field (plan §10, M6a).

The origin session's `seg_depth.py:250-303` produced `depth.npy` / `depth_map.png`
(closeness = 3*base_y - y) from a hand-tuned courtyard rule set: a teal wall class,
a hardcoded `ys.min() < 240` stone band, carpets below row 260, per-column bottoms
for four hardcoded vertical classes. None of that survives contact with arbitrary
pixel art, so this module ports the *idea* — every pixel learns the ground row its
surface stands on — with portable rules:

  1. JUNCTION  the wall/floor junction per column is the top row of the *floor sheet*:
               the horizontal plate band at the wall base (stone steps + ground +
               shadow lines, FLOOR_BAND). Its top boundary is where the standing
               surface begins — geometric, no colour archaeology, no objects needed.
               Uncovered columns are interpolated between their covered neighbours
               and the line is median-smoothed (the origin's stripe killer).
  2. WALL      plate material above the junction stands up: base_y = the junction row
               of its column (the origin's per-column-bottom rule, expressed on the
               junction line instead of per hardcoded class).
  3. GROUND    plate material at or below the junction: base_y = its own row, which is
               exactly what the plan §8 iso formula assumes.
  4. OBJECT    pixels owned by an object mask: base_y = the object's ground contact
               (97th row percentile — the app's own rule, imaging.stats_of), so the
               depth map and the mask list can never disagree.
  5. LABELS    an object/region whose recognised label is in WALL_LABELS /
               GROUND_LABELS forces its surface (user-correctable in M6b).
  6. NONE      whatever is left (highlight/shadow specks, tile outlines) is closed
               onto the surface it sits on by a box-majority pass, then inherits
               base_y from its labelled neighbours.

Consumers today (M6a):
  * `object_key` — the z sort key is the iso closeness formula evaluated at an
    object's *feet* (3*base_y - y at y == base_y == 2*base_y), not averaged over its
    body. The old mean-closeness key ranked a tall palm in front of a short pot
    standing on the same ground row; feet keys rank them equal.
  * `hole_axis` — which surface a hole sits in, measured for M6b/M6c (the fill-axis
    experiment is logged in build_log entry 13: it did not beat M4 on the courtyard,
    so animate.inpaint keeps its proven behaviour for now).

Deliberately conservative: when no floor sheet is found (flat backgrounds, side-view
art) the field degrades to base_y = y everywhere, i.e. today's behaviour.
"""
import time
import warnings

import numpy as np

from . import imaging as im

NONE, GROUND, WALL, OBJECT = 0, 1, 2, 3
ORIENT_NAMES = ('none', 'ground', 'wall', 'object')

# surface material: floor OR wall candidates (region-ish classes of classify()).
PLATE = ('stone', 'brown', 'ground', 'misc', 'water')
# the horizontal plate band at the wall base; its top row per column is the junction
FLOOR_BAND = ('stone', 'ground', 'misc')
# recogniser labels that force a surface (see recognise.ANIM_HINT); M6b lets the user
# set these per region in the inspector.
WALL_LABELS = ('wall tiles', 'wooden door', 'woodwork', 'wall')
GROUND_LABELS = ('floor tiles', 'floor', 'pool')

CFG = dict(
    snap=96,             # palette colours for class decisions (same as auto-seg)
    wall_margin=1,       # rows at the junction itself stay GROUND
    median_w=9,          # horizontal median window along the junction line
    fill_passes=2,       # box-majority passes that close lacy surfaces
    min_wall_votes=8,    # rim votes a hole needs before its fill axis may flip
)


# ------------------------------------------------------------------ helpers


def _box_count(m, r):
    """Integral-image count of `m` in a (2r+1)x(2r+1) window, same shape as m."""
    H, W = m.shape
    pad = np.zeros((H + 2 * r + 1, W + 2 * r + 1), np.int32)
    pad[r + 1:r + 1 + H, r + 1:r + 1 + W] = m.astype(np.int32)
    c = pad.cumsum(0).cumsum(1)
    k = 2 * r + 1
    return (c[k:k + H, k:k + W] - c[:H, k:k + W] - c[k:k + H, :W] + c[:H, :W])


def _majority_fill(orient, passes=2, r=2):
    """Close lacy surfaces: NONE pixels take the majority orientation of their box.

    Tilework outlines and cast-shadow specks are not plate classes, so a wall comes
    out of the class step as lace although the surface itself is solid.
    """
    for _ in range(int(passes)):
        none = orient == NONE
        if not none.any():
            break
        cnt = {WALL: _box_count(orient == WALL, r),
               GROUND: _box_count(orient == GROUND, r),
               OBJECT: _box_count(orient == OBJECT, r)}
        best = np.zeros(orient.shape, np.uint8)
        bestn = np.zeros(orient.shape, np.int32)
        for lab in (WALL, GROUND, OBJECT):
            take = none & (cnt[lab] > bestn)
            best[take] = lab
            bestn[take] = cnt[lab][take]
        assign = none & (bestn > 0)
        orient[assign] = best[assign]
    return orient


def _median_smooth_row(base, sel, w):
    """Horizontal sliding median of `base` over `sel` pixels (stripe killer).

    The origin did this with a python double loop and a per-pixel np.median; a
    strided NaN-median over the (H, W, w) window is the same rule at ~30 ms.
    """
    if not sel.any() or w < 3:
        return base
    H, W = base.shape
    half = w // 2
    pad = np.full((H, W + 2 * half), np.nan, np.float32)
    pad[:, half:half + W] = np.where(sel, base, np.nan)
    win = np.lib.stride_tricks.sliding_window_view(pad, w, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)   # all-NaN window slices
        med = np.nanmedian(win, axis=2)
    out = base.copy()
    ok = sel & ~np.isnan(med)
    out[ok] = med[ok]
    return out


def _propagate(base, labelled, passes):
    """Unlabelled pixels inherit base_y from their nearest labelled 4-neighbour."""
    for _ in range(int(passes)):
        unlab = ~labelled
        if not unlab.any():
            break
        acc = np.zeros(base.shape, np.float32)
        cnt = np.zeros(base.shape, np.int32)
        for sy, sx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            src = np.roll(np.roll(base, sy, 0), sx, 1)
            slab = np.roll(np.roll(labelled, sy, 0), sx, 1)
            acc += np.where(slab, src, 0.0)
            cnt += slab.astype(np.int32)
        take = unlab & (cnt > 0)
        base[take] = acc[take] / cnt[take]
        labelled = labelled | take
    return base


# ------------------------------------------------------------------ the junction


def floor_sheet_top(cls, names, shape, min_px=None):
    """Per-column top row of the floor sheet (NaN where a column lacks coverage).

    The floor sheet is the horizontal plate band at the wall base: stone steps +
    ground + shadow lines. Its top boundary is the wall/floor junction per column.
    Objects standing in the band cut it into pieces, so the minimum size scales with
    the frame (0.2%) instead of a fixed pixel count.
    """
    H, W = shape
    from . import segment as seg          # lazy: segment imports this module (M6b)
    min_px = max(48, int(0.002 * H * W)) if min_px is None else min_px
    band = np.zeros((H, W), bool)
    for n in FLOOR_BAND:
        band |= seg._mask(cls, names, n)
    comps = im.components(band, min_px, 8)
    if not comps:
        return None
    c = max(comps, key=lambda c: c['px'])
    m = im.mask_from_runs(c['runs'], (H, W))
    ys = np.where(m, np.arange(H)[:, None], H).min(0).astype(np.float32)
    ys[ys >= H] = np.nan
    return ys if np.isfinite(ys).any() else None


def horizon(sheet_top, shape, d):
    """Per-column junction row: interpolate uncovered columns linearly between their
    covered neighbours, then median-smooth. None when there is no coverage at all."""
    if sheet_top is None:
        return None
    _H, W = shape
    ys = np.asarray(sheet_top, np.float32).copy()
    known = np.isfinite(ys)
    if not known.any():
        return None
    if not known.all():
        idx = np.arange(W, dtype=np.float32)
        good, vals = idx[known], ys[known]
        pos = np.searchsorted(good, idx)
        lp = np.clip(pos - 1, 0, good.size - 1)
        rp = np.clip(pos, 0, good.size - 1)
        left, right = good[lp], good[rp]
        span = np.where(right > left, right - left, 1.0)
        t = np.clip((idx - left) / span, 0.0, 1.0)
        ys = np.where(known, ys, vals[lp] + (vals[rp] - vals[lp]) * t)
    return _median_smooth_row(ys[None, :], np.ones((1, W), bool), d['median_w'])[0]


# ------------------------------------------------------------------ the field


def field(nat, objects=None, cfg=None):
    """{'base_y', 'orient', 'depth', 'horizon', 'walls_px', 'ms'}.

    nat: native-res HxWx3 uint8. objects: optional iterable of bool masks, or dicts
    carrying 'mask' (+ optional 'label'), whose pixels are OBJECT. `depth` is the
    origin's uint8 closeness map (3*base_y - y, min-max normalised) so depth.npy
    comparisons stay direct.
    """
    from . import segment as seg          # lazy: segment imports this module (M6b)
    t0 = time.time()
    nat = np.ascontiguousarray(nat, np.uint8)
    H, W = nat.shape[:2]
    d = dict(CFG, **(cfg or {}))

    cls, names, _d = seg.classify(nat, {'snap': d['snap']})
    plate = np.zeros((H, W), bool)
    for n in PLATE:
        plate |= seg._mask(cls, names, n)
    orient = np.where(plate, GROUND, NONE).astype(np.uint8)
    base = np.repeat(np.arange(H, dtype=np.float32)[:, None], W, axis=1)

    # source 1: a surface the user set in the inspector beats the recognised label
    forced = np.zeros((H, W), bool)
    for o in (objects or []):
        if not isinstance(o, dict):
            continue
        surf = str(o.get('surface', '') or '').strip().lower()
        lab = str(o.get('label', '')).strip().lower()
        if surf in ('wall', 'ground'):
            want = WALL if surf == 'wall' else GROUND
        elif lab in WALL_LABELS:
            want = WALL
        elif lab in GROUND_LABELS:
            want = GROUND
        else:
            continue
        m = np.asarray(o.get('mask'), bool) if o.get('mask') is not None else None
        if m is None or not m.any():
            continue
        orient[m] = want
        forced |= m

    # source 2: plate material above the floor-sheet junction stands up
    hz = horizon(floor_sheet_top(cls, names, (H, W)), (H, W), d)
    if hz is not None:
        rows = np.arange(H, dtype=np.float32)[:, None]
        standing = plate & (rows < hz[None, :] - d['wall_margin']) & ~forced
        orient[standing] = WALL

    # source 3: objects own their pixels; ground contact = the app's 97pct rule.
    # Regions whose surface/label forced an orientation in source 1 keep it.
    for o in (objects or []):
        if isinstance(o, dict):
            surf = str(o.get('surface', '') or '').strip().lower()
            lab = str(o.get('label', '')).strip().lower()
            if surf in ('wall', 'ground') or lab in WALL_LABELS + GROUND_LABELS:
                continue
        m = o['mask'] if isinstance(o, dict) else o
        m = np.asarray(m, bool)
        if not m.any():
            continue
        ys = np.nonzero(m)[0]
        orient[m] = OBJECT
        base[m] = float(np.percentile(ys, 97))

    # source 4: close lacy surfaces, then pin every wall pixel to its column's
    # junction row and smooth along the face
    orient = _majority_fill(orient, d['fill_passes'])
    wallm = orient == WALL
    if hz is not None and wallm.any():
        ys, xs = np.nonzero(wallm)
        base[ys, xs] = hz[xs]
        base = _median_smooth_row(base, wallm, int(d['median_w']))

    # whatever is still unlabelled inherits base_y from its neighbourhood
    base = _propagate(base, orient != NONE, 2)

    close = 3.0 * base - np.arange(H, dtype=np.float32)[:, None]
    lo, hi = float(close.min()), float(close.max())
    depth8 = ((close - lo) / max(hi - lo, 1e-6) * 255.0).astype(np.uint8)
    return {'base_y': base.astype(np.float32), 'orient': orient, 'depth': depth8,
            'horizon': None if hz is None else hz.astype(np.float32),
            'walls_px': int(wallm.sum()), 'ms': round((time.time() - t0) * 1000)}


def object_key(mask, base_y=None):
    """z sort key of one object: iso closeness at its feet, 3*base_y - y at y = base_y.

    base_y defaults to the mask's own ground contact (97th row percentile), i.e. the
    same value imaging.stats_of reports. Two objects on the same ground row tie
    whatever their heights — the mean-closeness key (M5 and earlier) put tall objects
    in front of short ones on the same row.
    """
    m = np.asarray(mask, bool)
    ys = np.nonzero(m)[0]
    if not len(ys):
        return 0.0
    by = float(base_y) if base_y is not None else float(np.percentile(ys, 97))
    return round(2.0 * by, 1)


def hole_axis(orient, hole, cfg=None):
    """WALL or GROUND: the surface a hole sits in, from the labelled rim around it.

    Majority vote of the 2px ring outside the hole; weak evidence (fewer than
    min_wall_votes votes for either label) answers GROUND, i.e. "keep today's fill".
    """
    d = dict(CFG, **(cfg or {}))
    m = np.asarray(hole, bool)
    if not m.any():
        return GROUND
    rim = im.dilate4(m.copy(), 2) & ~m
    votes = orient[rim]
    nw = int((votes == WALL).sum())
    if nw >= int((votes == GROUND).sum()) and nw >= d['min_wall_votes']:
        return WALL
    return GROUND


# ------------------------------------------------------------------ depth breaks


def break_lines(base_y, thresh):
    """1px lines where the ground row jumps by > `thresh` between 4-neighbours.

    Both sides of a jump are marked so an 8-connected flood cannot slip diagonally
    across a break. On a wall base_y is constant along the face (no lines); between
    a wall and the floor, or between two objects at different depths, every border
    pixel lands on a line.
    """
    b = np.asarray(base_y, np.float32)
    cuts = np.zeros(b.shape, bool)
    dy = np.abs(b[1:, :] - b[:-1, :]) > thresh
    dx = np.abs(b[:, 1:] - b[:, :-1]) > thresh
    cuts[1:, :] |= dy
    cuts[:-1, :] |= dy
    cuts[:, 1:] |= dx
    cuts[:, :-1] |= dx
    return cuts


def surface_edges(orient):
    """1px lines where the orientation label changes between 4-neighbours (both
    sides marked). base_y is continuous across the wall/floor junction by
    construction (a wall pixel's ground row *is* the junction row), so the discrete
    surface boundary is the orientation label, not a base_y jump."""
    o = np.asarray(orient)
    cuts = np.zeros(o.shape, bool)
    dy = o[1:, :] != o[:-1, :]
    dx = o[:, 1:] != o[:, :-1]
    cuts[1:, :] |= dy
    cuts[:-1, :] |= dy
    cuts[:, 1:] |= dx
    cuts[:, :-1] |= dx
    return cuts


def split_pieces(mask, base_y, thresh=3, min_px=25, orient=None):
    """Connected pieces of `mask` separated by surface boundaries.

    Cuts run where the orientation label changes (wall meets floor) or where the
    ground row jumps by > `thresh` between neighbours (an object's feet against the
    surface behind it, a step between wall faces). The segregation help M6 exists
    for: same-colour material on two surfaces fuses into one colour-class component
    (plan §8's "user chooses combine-or-split at a cut line") — the surface boundary
    *is* that cut line, proposed automatically. Honest limit: two objects on the
    *same* surface at different depths share one ground plane and need the Phase-2
    recomposition tools, not this. Returns [mask] when there is nothing to split.
    """
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    cuts = break_lines(base_y, thresh)
    if orient is not None:
        cuts |= surface_edges(orient)
    comps = im.components(m & ~cuts, min_px, 8)
    if len(comps) < 2:
        return [m]
    return [im.mask_from_runs(c['runs'], m.shape) for c in comps]


def _box_mean(a, r):
    """(2r+1)x(2r+1) box mean of a float map, integral-image style.

    Zero-padded by r+1 on every side so edge windows shrink toward 0 rather than
    off the map; the returned value is the true mean of the window that exists
    inside the padding, so a constant map stays constant (verified in test_m6).
    """
    H, W = a.shape
    k = 2 * r + 1
    pad = np.pad(a.astype(np.float32), r, mode='edge')      # edge, not zero: a
    c = np.zeros((H + 2 * r + 1, W + 2 * r + 1), np.float32)   # constant map must
    c[1:, 1:] = pad.cumsum(0).cumsum(1)                     # stay constant
    s = (c[k:k + H, k:k + W] - c[:H, k:k + W]
         - c[k:k + H, :W] + c[:H, :W])
    return s / (k * k)


def normal_map(nat, bump=3.2, snap=96):
    """Tangent-space normal map (M6c tab), derived, not solved.

    Pixel art carries no geometry to photogrammetry-solve, so this is a bump map:
    height = the palette-snapped luminance (dither-safe, see segment._denoised),
    box-meaned once, then central differences. Encoding is the usual
    (nx, ny, nz)*0.5+0.5 with G up (OpenGL convention): flat shading reads
    (128, 128, 255). M6c's shading-aware inpaint and the optional `relight` clip
    consume this; the editor shows it as the normals tab.
    """
    from . import segment as seg          # lazy: segment imports this module (M6b)
    a = seg._denoised(np.ascontiguousarray(nat, np.uint8), {'snap': snap})
    lum = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float32)
    lum = _box_mean(lum, 1)
    hx = np.gradient(lum, axis=1)
    hy = np.gradient(lum, axis=0)
    nx = -bump * hx / 255.0
    ny = bump * hy / 255.0               # +Y up: negate the down-image derivative
    nz = np.ones_like(lum)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    rgb = np.stack((nx / ln, ny / ln, nz / ln), -1)
    # round, don't truncate: flat art must land on exactly (128, 128, 255)
    return np.clip(np.rint(rgb * 127.5 + 127.5), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ debug view


def render_orient(f):
    """Orientation tint only (wall red, ground green, object blue, none grey) —
    served as <pid>/orient.png for the editor's surface overlay (M6b)."""
    o = f['orient']
    rgb = np.zeros(o.shape + (3,), np.uint8)
    rgb[o == WALL] = (215, 60, 60)
    rgb[o == GROUND] = (70, 170, 90)
    rgb[o == OBJECT] = (120, 130, 220)
    rgb[o == NONE] = (128, 128, 128)
    return rgb


def render_debug(f, shape=None):
    """RGB overlay: closeness as luminance, orientation as tint (wall red, ground
    green, object blue, none grey). app/tests/depth_probe.py writes it to a PNG."""
    d8 = f['depth'].astype(np.float32) / 255.0
    o = f['orient']
    rgb = np.stack((d8 * 0.75 + 0.25 * (o == WALL),
                    d8 * 0.75 + 0.25 * (o == GROUND),
                    d8 * 0.75 + 0.25 * (o == OBJECT)), -1)
    rgb[o == NONE] = 0.5
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
