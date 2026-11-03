"""Pixel Sway engine — animation render (plan §2/§3, M3).

Generalised port of the proven origin pipeline (`animate3.py` → courtyard GIF). The
origin script animated four hardcoded masks of one artwork; this module animates an
arbitrary object list from the project doc, driven by the plan §3 catalog:

    sway · bob · ripple · shimmer · flutter · gustflick · static

Architecture — **parity by construction**. Every object is rendered as per-frame RGBA
*sprites* (a fixed bbox cell + a native-res anchor). Composite frames are built ONLY by
pasting those sprites onto the inpainted plate in z order, and the scene bundle exports
the very same sprite cells as sheets. Preview GIF and game bundle therefore cannot
disagree, and "bundle re-imports with clean RGBA edges" is an invariant, not a hope.

Origin lessons baked in (plan §2, build_log entry 2 bug ledger):
  1. plate = art inpainted under every visible object mask (horizontal run fill, the
     origin `hfill`) — vacated pixels reveal plausible background, not object ghosts.
  2. every object is redrawn EVERY frame, including offset 0 (the "disappearing lily").
     With sprite pasting this is structural: the full sprite is pasted per frame.
  3. trailing-edge holes: mover sprites leave vacated pixels *transparent*, so the
     composite shows the pre-object frame (`fr_bg`) — origin rule 4, for free.
  4. all motion frequencies are integer cycles per loop → seamless by construction;
     gust windows are placed so frame N-1 and frame 0 are both at rest.
  5. articulated chunks (base/mid/top) instead of per-row shear ("screen tearing");
     big foliage units erode-split into leaf units ("palm not moving" fix); units <8px
     ride along static ("stray specks swaying on walls" fix).
  6. an object's sprite source excludes pixels covered by higher-z objects, so ripple
     never moves a leaf-coloured pixel (the "plant over water dissolved" fix): the
     overhanging object repaints them from its own sprite every frame.
  7. deterministic: per-object RNG is seeded from a hash of id+anim+params, so two
     renders of the same doc are byte-identical (preview == export).

Audit hook (rule 8): frame-to-frame diffs (including the N-1 → 0 wrap) must be zero
outside the allowed zones (animated objects' masks dilated by their max shift).
"""
import json
import time
import zlib

import numpy as np

from . import imaging as im

TAU = 2.0 * np.pi

ANIM_TYPES = ('sway', 'bob', 'ripple', 'shimmer', 'flutter', 'gustflick', 'static')
MOVERS = ('sway', 'bob', 'flutter', 'gustflick')   # shadow-sync applies to these only

# param -> (min, max, default, step). `phase` default None = auto (derived per object).
PARAM_DEFS = {
    'sway':      {'speed': (1, 3, 1, 1), 'amp': (0.5, 2.0, 1.0, 0.1),
                  'gust': (0.0, 0.3, 0.15, 0.01), 'phase': (0.0, 0.99, None, 0.01)},
    'bob':       {'speed': (1, 3, 1, 1), 'amp': (0.0, 2.0, 1.0, 0.1),
                  'circ': (0.0, 1.0, 0.6, 0.05), 'phase': (0.0, 0.99, None, 0.01)},
    'ripple':    {'speed': (1, 3, 3, 1), 'wavelength': (6, 40, 14, 1),
                  'amp': (0.5, 2.0, 1.0, 0.1)},
    'shimmer':   {'speed': (2, 4, 3, 1), 'coverage': (0.0, 1.0, 0.5, 0.05)},
    'flutter':   {'speed': (2, 4, 3, 1), 'amp': (1.0, 2.0, 1.0, 0.5)},
    'gustflick': {'gusts': (1, 4, 3, 1), 'strength': (1, 2, 1, 1), 'travel': (8, 40, 20, 1)},
    'static':    {},
}

# recognition hints (engine/recognise.py ANIM_HINT) -> catalog type. First match wins;
# combined hints keep the primary motion ('ripple + shimmer' -> ripple; the user can
# duplicate the mask and pick shimmer for the second object — z-composited, they stack).
HINT_MAP = (
    ('ripple', 'ripple'), ('sway', 'sway'), ('bob', 'bob'),
    ('gustflick', 'gustflick'), ('flutter', 'flutter'), ('shimmer', 'shimmer'),
    ('static', 'static'),
)


def coerce_anim(a):
    """Any stored anim string (catalog type, recognition hint, junk) -> catalog type."""
    a = (a or '').strip().lower()
    if a in ANIM_TYPES:
        return a
    for needle, typ in HINT_MAP:
        if needle in a:
            return typ
    return 'static'


def clean_params(anim, raw):
    """Whitelist + clamp params against PARAM_DEFS (same policy as segment cfg)."""
    anim = coerce_anim(anim)
    out = {}
    raw = raw if isinstance(raw, dict) else {}
    for k, (lo, hi, dflt, _step) in PARAM_DEFS[anim].items():
        v = raw.get(k, dflt)
        if v is None:
            out[k] = None                      # phase: auto
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = dflt if dflt is not None else lo
        out[k] = int(round(min(hi, max(lo, v)))) if float(lo) == int(lo) and float(hi) == int(hi) \
            and _step >= 1 else round(min(hi, max(lo, v)), 3)
    return out


def _rng_for(obj):
    """Deterministic per-object RNG: same id+anim+params -> same motion, every render."""
    key = json.dumps([obj.get('id'), coerce_anim(obj.get('anim')),
                      clean_params(obj.get('anim'), obj.get('params'))], sort_keys=True)
    return np.random.default_rng(zlib.crc32(key.encode()) & 0xFFFFFFFF)


# ---------------------------------------------------------------- plate / inpaint


def _hfill(plate, mask):
    """Per-row horizontal run fill, lerped between boundary colours (origin `hfill`).
    Reads its left colour from the *output*, so adjacent runs continue smoothly."""
    H, W = mask.shape[:2]
    lens = np.zeros((H, W), np.int32)          # run length of each masked pixel
    for y in range(H):
        row = mask[y]
        if not row.any():
            continue
        d = np.diff(np.concatenate((np.int8([0]), row.view(np.int8), np.int8([0]))))
        for a, b in zip(np.nonzero(d == 1)[0].tolist(), (np.nonzero(d == -1)[0] - 1).tolist()):
            has_l, has_r = a > 0, b + 1 < W
            if not has_l and not has_r:
                continue                       # fully masked row: nothing to fill from
            lc = plate[y, a - 1] if has_l else plate[y, b + 1]
            rc = plate[y, b + 1] if has_r else plate[y, a - 1]
            n = b - a + 1
            t = (np.arange(1, n + 1) / (n + 1.0))[:, None]
            plate[y, a:b + 1] = (lc[None, :].astype(np.float32) * (1 - t)
                                 + rc[None, :].astype(np.float32) * t).astype(np.uint8)
            lens[y, a:b + 1] = n
    return lens


def _vfill(plate, mask):
    """Same lerp fill top-to-bottom per column (the multi-direction pass of plan §7)."""
    H, W = mask.shape[:2]
    for x in range(W):
        col = mask[:, x]
        if not col.any():
            continue
        d = np.diff(np.concatenate((np.int8([0]), col.view(np.int8), np.int8([0]))))
        for a, b in zip(np.nonzero(d == 1)[0].tolist(), (np.nonzero(d == -1)[0] - 1).tolist()):
            has_t, has_b = a > 0, b + 1 < H
            if not has_t and not has_b:
                continue
            lc = plate[a - 1, x] if has_t else plate[b + 1, x]
            rc = plate[b + 1, x] if has_b else plate[a - 1, x]
            n = b - a + 1
            t = (np.arange(1, n + 1) / (n + 1.0))[:, None]
            plate[a:b + 1, x] = (lc[None, :].astype(np.float32) * (1 - t)
                                 + rc[None, :].astype(np.float32) * t).astype(np.uint8)
    return plate


def _bifill(nat, mask):
    """Inverse-distance blend of the four run-boundary colours per masked pixel.

    Averaging a horizontal and a vertical run fill just trades row banding for column
    banding (measured, M4); weighting each of the four boundary colours by 1/distance
    to it is smooth in both axes — this is the multi-direction pass of plan §7 risk 1,
    applied only where horizontal runs are long enough to streak.
    """
    H, W = mask.shape[:2]
    wsum = np.zeros((H, W), np.float32)
    acc = np.zeros((H, W, 3), np.float32)

    def add(w, col):
        sl = w[:, None] * col[None, :].astype(np.float32)
        return sl

    for y in range(H):                             # left/right boundaries per row run
        row = mask[y]
        if not row.any():
            continue
        d = np.diff(np.concatenate((np.int8([0]), row.view(np.int8), np.int8([0]))))
        for a, b in zip(np.nonzero(d == 1)[0].tolist(), (np.nonzero(d == -1)[0] - 1).tolist()):
            xs = np.arange(a, b + 1)
            if a > 0:
                w = 1.0 / (xs - a + 1.0)
                wsum[y, a:b + 1] += w
                acc[y, a:b + 1] += add(w, nat[y, a - 1])
            if b + 1 < W:
                w = 1.0 / (b - xs + 1.0)
                wsum[y, a:b + 1] += w
                acc[y, a:b + 1] += add(w, nat[y, b + 1])
    for x in range(W):                             # top/bottom boundaries per column run
        col = mask[:, x]
        if not col.any():
            continue
        d = np.diff(np.concatenate((np.int8([0]), col.view(np.int8), np.int8([0]))))
        for a, b in zip(np.nonzero(d == 1)[0].tolist(), (np.nonzero(d == -1)[0] - 1).tolist()):
            ys = np.arange(a, b + 1)
            if a > 0:
                w = 1.0 / (ys - a + 1.0)
                wsum[a:b + 1, x] += w
                acc[a:b + 1, x] += w[:, None] * nat[a - 1, x][None, :].astype(np.float32)
            if b + 1 < H:
                w = 1.0 / (b - ys + 1.0)
                wsum[a:b + 1, x] += w
                acc[a:b + 1, x] += w[:, None] * nat[b + 1, x][None, :].astype(np.float32)
    ok = wsum > 0
    out = np.zeros((H, W, 3), np.uint8)
    out[ok] = np.clip(acc[ok] / wsum[ok][:, None], 0, 255).astype(np.uint8)
    return out, ok


VBLEND_LONG_RUN = 10        # horizontal runs longer than this get the bi-directional pass


def inpaint(nat, mask, vblend=True):
    """Remove `mask` from the art (plan §2.2). Horizontal run fill is the proven origin
    behaviour; long horizontal runs are where it streaks (plan §7 risk 1), so those pixels
    are refilled with the inverse-distance four-boundary blend instead — M4's
    vertical-blend pass."""
    plate = np.array(nat, np.uint8)
    if mask is None or not mask.any():
        return plate
    lens = _hfill(plate, mask)
    if vblend:
        long = mask & (lens > VBLEND_LONG_RUN)
        if long.any():
            bi, ok = _bifill(nat, long)
            sel = long & ok
            plate[sel] = bi[sel]
    return plate


# ---------------------------------------------------------------- shadow sync


def shadow_pixels(nat):
    """Dark, cool-or-neutral pixels = cast shadow material (measured on the courtyard:
    the origin shadow class sits at val median 0.10; hue>=150 keeps brown door crevices
    and dark foliage out, sat<0.35 keeps shadows on neutral stone in)."""
    hue, sat, val = im.hsv_maps(nat)
    return (val < 0.32) & ((hue >= 150) | (sat < 0.35))


def shadow_ring(nat, dark, mask, exclude, dil=2):
    """Dark pixels *hugging* `mask` (outside it, within `dil`) that nobody else owns.
    plan §3: pots/statics keep theirs — the caller puts their hug zone in `exclude`."""
    if not mask.any():
        return np.zeros(mask.shape, bool)
    ring = dark & im.dilate4(mask.copy(), dil) & ~mask & ~exclude
    return ring


# ---------------------------------------------------------------- motion plans


def _components_of(src):
    """Connected components of src -> list of (ys, xs) coordinate arrays (desc by px)."""
    out = []
    for c in im.components(src, 8, 8):
        m = im.mask_from_runs(c['runs'], src.shape)
        ys, xs = np.nonzero(m)
        out.append((ys, xs))
    return out


def _split_leaf_units(ys, xs, shape):
    """Big foliage blob -> leaf units (origin: erode -> cores >=6px -> multi-source
    BFS reassignment; blob <150px or <2 cores stays one unit). Without this a 7.5k-px
    palm crown sways as one rigid +-1px block and reads as "not moving" (entry 2).

    M5 perf+parity: the M3 port approximated the BFS with synchronous max-label
    propagation over the FULL canvas, truncated at 4096 rounds — both slow (~9 s of a
    10 s worst-case render on fractal fern masks) and unfaithful (label far from the
    nearest core, arbitrary cutoff). This restores the origin rule: nearest core wins
    (ties -> lower core index), computed as synchronous BFS rounds confined to the
    blob's bounding box."""
    comp = np.zeros(shape, bool)
    comp[ys, xs] = True
    if len(ys) < 150:                           # origin: small blob = single unit
        return [(ys, xs)]
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    H, W = shape
    # crop to the blob bbox, padded by a virtual ring that reproduces full-canvas
    # erode4 semantics: erode4 treats off-canvas as True, so the pad is False where
    # the canvas continues (bbox is minimal -> no blob pixels out there) and True
    # where the blob touches the canvas border.
    c = np.zeros((y1 - y0 + 3, x1 - x0 + 3), bool)
    c[1:-1, 1:-1] = comp[y0:y1 + 1, x0:x1 + 1]
    if y0 == 0:
        c[0, 1:-1] = True
    if y1 == H - 1:
        c[-1, 1:-1] = True
    if x0 == 0:
        c[1:-1, 0] = True
    if x1 == W - 1:
        c[1:-1, -1] = True
    er = im.erode4(c.copy(), 1)[1:-1, 1:-1]
    c = c[1:-1, 1:-1].copy()
    cores = im.components(er, 6, 8)
    if len(cores) < 2:
        return [(ys, xs)]
    h, w = c.shape
    lab = np.zeros((h, w), np.int16)
    for i, cc in enumerate(cores):
        lab[im.mask_from_runs(cc['runs'], (h, w))] = np.int16(i + 1)
    INF = np.int16(32767)
    un = c & (lab == 0)
    guard = 0
    while un.any() and guard <= len(ys):        # synchronous BFS rounds (8-neighbour)
        p = np.where(lab > 0, lab, INF).astype(np.int16)
        g = p.copy()
        g[1:, :] = np.minimum(g[1:, :], p[:-1, :])
        g[:-1, :] = np.minimum(g[:-1, :], p[1:, :])
        g[:, 1:] = np.minimum(g[:, 1:], p[:, :-1])
        g[:, :-1] = np.minimum(g[:, :-1], p[:, 1:])
        g[1:, 1:] = np.minimum(g[1:, 1:], p[:-1, :-1])
        g[1:, :-1] = np.minimum(g[1:, :-1], p[:-1, 1:])
        g[:-1, 1:] = np.minimum(g[:-1, 1:], p[1:, :-1])
        g[:-1, :-1] = np.minimum(g[:-1, :-1], p[1:, 1:])
        take = un & (g < INF)
        lab[take] = g[take]
        un = c & (lab == 0)
        guard += 1
    units = []
    for i in range(1, len(cores) + 1):
        uys, uxs = np.nonzero(lab == i)
        if len(uys):
            units.append((uys + y0, uxs + x0))
    if un.any() and units:                      # unreachable remainder rides unit 0
        uys, uxs = np.nonzero(un)
        units[0] = (np.concatenate((units[0][0], uys + y0)),
                    np.concatenate((units[0][1], uxs + x0)))
    return units


def _runs_of(ys, xs):
    """Coordinate arrays -> per-row inclusive runs [(y, a, b)] (sprite blitting)."""
    runs = []
    if not len(ys):
        return runs
    order = np.lexsort((xs, ys))
    ys, xs = ys[order], xs[order]
    y = int(ys[0])
    a = prev = int(xs[0])
    for yy, xx in zip(ys[1:].tolist(), xs[1:].tolist()):
        if yy == y and xx == prev + 1:
            prev = xx
            continue
        runs.append((y, a, prev))
        y, a, prev = yy, xx, xx
    runs.append((y, a, prev))
    return runs


class _Unit:
    """One rigid-ish piece of foliage: 3 articulated chunks of row-runs + a waveform."""
    __slots__ = ('y1', 'hgt', 'sz', 'k', 'ph', 'w2', 'ps2', 'ampT', 'j1', 'j2', 'f1',
                 'envA', 'envP', 'bias', 'maxdx', 'chunked', 'chunks', 'static',
                 'cy', 'cx')

    def __init__(self, ys, xs, rng, p, shape):
        self.sz = int(len(ys))
        self.cy, self.cx = float(ys.mean()), float(xs.mean())
        y0, self.y1 = int(ys.min()), int(ys.max())
        hgt = max(self.y1 - y0, 1)
        self.hgt = hgt
        speed = int(p['speed'])
        if p['phase'] is None:
            self.ph = float(rng.random())
            self.k = speed
        else:
            self.ph = float(p['phase'])
            self.k = speed
        self.w2 = float(rng.uniform(0.10, 0.38))
        self.ps2 = float(rng.uniform(0, TAU))
        self.ampT = float(p['amp']) * (rng.uniform(0.90, 1.25) if self.sz >= 250
                                       else rng.uniform(0.85, 1.10))
        self.j1 = float(rng.uniform(0.30, 0.45))
        self.j2 = float(rng.uniform(0.62, 0.80))
        self.f1 = float(rng.uniform(0.45, 0.65))
        self.envA = float(p['gust']) * float(rng.uniform(0.7, 1.3))
        self.envP = float(rng.uniform(0, TAU))
        self.bias = float(rng.uniform(-0.12, 0.12))
        self.maxdx = int(min(3, max(1, round(p['amp']) + (1 if self.sz >= 180 else 0))))
        self.chunked = hgt >= 14
        self.static = self.sz < 8              # specks ride along, never sway on their own
        # chunk runs: fy measured from the unit's feet (origin convention)
        fy = (self.y1 - ys) / float(hgt)
        ch_of = np.where(fy < self.j1, 0, np.where(fy < self.j2, 1, 2))
        self.chunks = []
        for ch in range(3):
            if not self.chunked:
                self.chunks.append(_runs_of(ys, xs) if ch == 2 else [])
                continue
            sel = ch_of == ch
            self.chunks.append(_runs_of(ys[sel], xs[sel]) if sel.any() else [])

    def offsets(self, t, N):
        """Integer dx per chunk at frame t (d0 base, d1 mid, d2 top)."""
        if self.static:
            return (0, 0, 0)
        th = TAU * (self.k * t / N + self.ph)
        sv = (1 - self.w2) * np.sin(th) + self.w2 * np.sin(2 * th + self.ps2)
        env = 1.0 + self.envA * np.sin(TAU * t / N + self.envP)
        s = self.ampT * env * sv + self.bias
        d2 = int(np.clip(round(s), -self.maxdx, self.maxdx))
        if not self.chunked:
            return (d2, d2, d2)
        d1 = int(np.clip(round(s * self.f1), -self.maxdx, self.maxdx))
        return (0, d1, d2)


class Plan:
    """Per-object render plan: sprite cell geometry + everything needed per frame."""

    def __init__(self, obj, src, anim, p, rng, N, shape, ring=None):
        self.id = obj.get('id')
        self.name = obj.get('name', self.id)
        self.label = obj.get('label', '')
        self.anim = anim
        self.params = dict(p)
        self.shadow = bool(obj.get('shadow')) and anim in MOVERS
        self.src = src                         # pixels this object actually owns
        self.z = obj.get('z', 0)
        self.depth = obj.get('depth', 0.0)
        bbox = im.bbox_of(src)
        self.bbox = bbox
        self.maxsh = 0
        self.units = []
        self.comps = []
        self.runs = []                         # ripple: runs of erode4(src)
        self.flick = []                        # gustflick: (ys,xs,start,dur,dx)
        self.shimmer_px = None
        self.ringm = ring.copy() if ring is not None else None
        body = src & ~self.ringm if self.ringm is not None else src
        H, W = shape
        if bbox is None:
            self.cell = (0, 0, 0, 0)
            self.ring = []
            return

        if anim == 'sway':
            for ys, xs in _components_of(body):
                if len(ys) >= 600:
                    for u in _split_leaf_units(ys, xs, shape):
                        self.units.append(_Unit(u[0], u[1], rng, p, shape))
                else:
                    self.units.append(_Unit(ys, xs, rng, p, shape))
            self.maxsh = max((u.maxdx for u in self.units), default=0)
        elif anim in ('bob', 'flutter'):
            k = int(p['speed'])
            for ys, xs in _components_of(body):
                cx, cy = float(xs.mean()), float(ys.mean())
                if p.get('phase') is not None:
                    ph = float(p['phase']) * TAU
                    ph2 = ph + np.pi / 2
                else:                          # origin: phase from position (staggered)
                    ph = TAU * ((int(cx) * 11 + int(cy) * 5) % 32) / 32.0
                    ph2 = TAU * ((int(cx) * 7 + int(cy) * 13) % 32) / 32.0
                self.comps.append({'ys': ys, 'xs': xs, 'ph': ph, 'ph2': ph2, 'k': k})
            amp = float(p['amp'])
            self.maxsh = int(np.ceil(amp)) + (int(np.ceil(amp * 0.7)) if anim == 'flutter' else 0)
        elif anim == 'ripple':
            self.runs = _runs_of(*np.nonzero(im.erode4(src.copy(), 1)))
            amp = float(p['amp'])
            self.maxsh = int(np.ceil(1.15 * amp))
        elif anim == 'shimmer':
            ys, xs = np.nonzero(src)
            self.shimmer_px = (ys, xs)
            self.maxsh = 0
        elif anim == 'gustflick':
            G = int(p['gusts'])
            strength = int(p['strength'])
            travel = max(1, int(p['travel']))
            hi = max(4, N - 8)
            gust_at = sorted(rng.choice(np.arange(2, hi), size=min(G, hi - 2),
                                        replace=False).tolist()) if hi > 3 else [2]
            for ys, xs in _components_of(body):
                if len(ys) < 4:
                    continue
                cx = float(xs.mean())
                g = int(gust_at[int(rng.integers(0, len(gust_at)))])
                delay = int(cx // travel) % 10 + int(rng.integers(0, 3))
                dur = int(rng.integers(3, 6))
                start = g + delay
                if start + dur <= N - 1:       # frame N-1 and 0 both at rest -> seam safe
                    self.flick.append((ys, xs, start, dur, strength))
            self.maxsh = strength if self.flick else 0
        # shadow-sync (plan §3): the hug ring rides along with the piece it belongs to
        self.ring = []
        if self.ringm is not None and self.ringm.any():
            for rys, rxs in _components_of(self.ringm):
                self.ring.append({'runs': _runs_of(rys, rxs),
                                  'owner': self._ring_owner(rys, rxs)})
        x0, y0, x1, y1 = bbox
        m = self.maxsh + 1
        self.cell = (max(0, x0 - m), max(0, y0 - m), min(W - 1, x1 + m), min(H - 1, y1 + m))

    def _ring_owner(self, ys, xs):
        """Nearest moving piece (unit/comp/flick comp) — its motion carries the shadow."""
        cy, cx = float(ys.mean()), float(xs.mean())
        if self.anim == 'sway':
            cand = [(abs(u.cy - cy) + abs(u.cx - cx), i) for i, u in enumerate(self.units)]
        elif self.anim in ('bob', 'flutter'):
            cand = [(abs(float(c['ys'].mean()) - cy) + abs(float(c['xs'].mean()) - cx), i)
                    for i, c in enumerate(self.comps)]
        elif self.anim == 'gustflick':
            cand = [(abs(float(f[0].mean()) - cy) + abs(float(f[1].mean()) - cx), i)
                    for i, f in enumerate(self.flick)]
        else:
            return -1
        return min(cand)[1] if cand else -1

    # -- per-frame sprite ------------------------------------------------------
    def frame_offsets(self, t, N):
        """bob/flutter component offsets at frame t (also useful for tests/docs)."""
        p = self.params
        out = []
        for c in self.comps:
            th = TAU * (c['k'] * t / N)
            if self.anim == 'bob':
                dx = round(float(p['amp']) * np.sin(th + c['ph']))
                dy = round(float(p['amp']) * float(p['circ']) * np.cos(th + c['ph']))
            else:                              # flutter: fast small Lissajous jitter
                dx = round(float(p['amp']) * np.sin(th + c['ph']))
                dy = round(float(p['amp']) * 0.7 * np.sin(TAU * ((c['k'] + 1) * t / N) + c['ph2']))
            out.append((int(dx), int(dy)))
        return out

    def ripple_dx(self, t, N):
        """Integer dx per run at frame t: two integer-cycle terms (origin waveform)."""
        p = self.params
        k = int(p['speed'])
        k2 = max(1, k - 1)
        wy = TAU / float(p['wavelength'])
        A1, A2 = 0.7 * float(p['amp']), 0.45 * float(p['amp'])
        th = TAU * t / N
        out = []
        for (y, a, b) in self.runs:
            c = (a + b) // 2
            out.append(int(round(A1 * np.sin(k * th + wy * y) + A2 * np.sin(k2 * th + 0.30 * c))))
        return out

    def shimmer_sel(self, t, N):
        """(bright_idx, dark_idx) into self.shimmer_px for frame t (origin waveform)."""
        ys, xs = self.shimmer_px
        p = self.params
        k = int(p['speed'])
        thr = float(np.clip(1.0 - 0.30 * float(p['coverage']), 0.40, 0.97))
        wv = np.sin(TAU * (k * t / N) + 0.50 * ys + 0.35 * xs + 1.3)
        return np.nonzero(wv > thr)[0], np.nonzero(wv < -thr - 0.06)[0]


# ---------------------------------------------------------------- sprite build


def _empty_cell(plan):
    x0, y0, x1, y1 = plan.cell
    return np.zeros((y1 - y0 + 1, x1 - x0 + 1, 4), np.uint8)


class _Blit:
    """Cached flat-index blitter (M5 perf pass).

    For a fixed set of row-runs and cell geometry, the (src, dst) pixel indices for a
    given integer offset (dx, dy) are computed once and reused on every frame needing
    the same offset — turning the per-run python slice loop into two fancy-index
    assignments per piece per frame. Offsets repeat heavily across the loop (small
    amplitudes, many units), so the cache covers nearly all blits after frame ~2."""

    __slots__ = ('runs', 'ox', 'oy', 'cw', 'ch', 'W', 'cache')

    def __init__(self, runs, ox, oy, cw, ch, W):
        self.runs = runs
        self.ox, self.oy, self.cw, self.ch, self.W = ox, oy, cw, ch, W
        self.cache = {}

    def pairs(self, dx, dy=0):
        key = (dx, dy)
        if key in self.cache:
            return self.cache[key]
        src, dst = [], []
        for (y, a, b) in self.runs:
            yy = y + dy - self.oy
            if yy < 0 or yy >= self.ch:
                continue
            da = a - self.ox + dx
            c0, c1 = max(da, 0), min(da + (b - a + 1), self.cw)
            if c0 >= c1:
                continue
            s0 = c0 + self.ox - dx            # absolute source x of the clipped run
            n = c1 - c0
            if s0 < 0 or s0 + n > self.W:
                continue
            src.append(y * self.W + s0 + np.arange(n))
            dst.append(yy * self.cw + c0 + np.arange(n))
        if not src:
            self.cache[key] = None
            return None
        s = np.concatenate(src)
        d = np.concatenate(dst) * 4
        pr = (s, d[:, None] + np.arange(3)[None, :], d + 3)
        self.cache[key] = pr
        return pr

    def put(self, rgba4, nat3, dx, dy=0):
        """rgba4: cell viewed as (-1,4); nat3: source art viewed as (-1,3)."""
        pr = self.pairs(dx, dy)
        if pr is None:
            return
        s, d3, da = pr
        rgba4[d3] = nat3[s]
        rgba4[da] = 255


def build_sprites(nat, plan, N):
    """Per-frame RGBA cells for one object (static -> a single reused cell).

    M5 perf pass: every piece (unit chunk / component / ring / ripple run) is blitted
    through a cached _Blit, so a frame costs a handful of fancy-index assignments
    instead of a python loop over runs. Identical offset patterns repeat across the
    loop, and finished frames keyed by their offset tuple are memoised as well."""
    srcm = plan.src
    x0, y0, x1, y1 = plan.cell
    cw, ch = x1 - x0 + 1, y1 - y0 + 1
    W = nat.shape[1]
    nat3 = np.ascontiguousarray(nat).reshape(-1, 3)
    anim = plan.anim
    ys, xs = np.nonzero(srcm)

    def blit(runs):
        return _Blit(runs, x0, y0, cw, ch, W)   # empty runs -> pairs() None -> put no-op

    if anim == 'static' or (anim != 'shimmer' and not len(ys)):
        rgba = _empty_cell(plan)
        if len(ys):
            blit(_runs_of(ys, xs)).put(rgba.reshape(-1), nat3, 0, 0)
        return [rgba]

    if anim == 'shimmer':
        base = _empty_cell(plan)
        blit(_runs_of(ys, xs)).put(base.reshape(-1), nat3, 0, 0)
        pys, pxs = plan.shimmer_px
        pyf = (pys - y0) * cw + (pxs - x0)          # flat cell idx of shimmer pixels
        cells = []
        for t in range(N):
            rgba = base.copy()
            v = rgba.reshape(-1)
            br, dk = plan.shimmer_sel(t, N)
            if len(br):
                idx = (pyf[br] * 4)[:, None] + np.arange(3)[None, :]
                v[idx] = np.clip(v[idx].astype(np.int16)
                                 + np.array([14, 18, 16], np.int16), 0, 255).astype(np.uint8)
            if len(dk):
                idx = (pyf[dk] * 4)[:, None] + np.arange(3)[None, :]
                v[idx] = np.clip(v[idx].astype(np.int16)
                                 - np.array([8, 10, 10], np.int16), 0, 255).astype(np.uint8)
            cells.append(rgba)
        return cells

    if anim == 'ripple':
        base = _empty_cell(plan)
        blit(_runs_of(ys, xs)).put(base.reshape(-1), nat3, 0, 0)
        # footprint stays exactly `src`: shifted run pixels that would land outside the
        # water silhouette are clipped (origin bug ledger: "ripple edge writes spilled").
        keep = np.zeros((ch, cw), bool)
        keep[srcm[y0:y1 + 1, x0:x1 + 1]] = True
        run_blits = [blit([(y, a, b)]) for (y, a, b) in plan.runs]
        # frame memo keyed by the per-run dx tuple (offsets repeat across the loop)
        memo = {}
        cells = []
        for t in range(N):
            dxs = plan.ripple_dx(t, N)
            key = tuple(dxs)
            hit = memo.get(key)
            if hit is None:
                rgba = base.copy()
                v = rgba.reshape(-1)
                for rb, dx in zip(run_blits, dxs):
                    if dx:
                        rb.put(v, nat3, dx, 0)
                rgba[..., 3] = np.where(keep & (rgba[..., 3] > 0), 255, 0).astype(np.uint8)
                memo[key] = rgba
                hit = rgba
            cells.append(hit)
        return cells

    if anim in ('bob', 'flutter'):
        comp_blits = [blit(_runs_of(c['ys'], c['xs'])) for c in plan.comps]
        ring_blits = [blit(rc['runs']) for rc in plan.ring]
        memo = {}
        cells = []
        for t in range(N):
            offs = tuple(plan.frame_offsets(t, N))
            hit = memo.get(offs)
            if hit is None:
                rgba = _empty_cell(plan)
                v = rgba.reshape(-1)
                for cb, (dx, dy) in zip(comp_blits, offs):
                    if cb:
                        cb.put(v, nat3, dx, dy)
                for rb, rc in zip(ring_blits, plan.ring):
                    if rb and rc['owner'] >= 0:
                        dx, dy = offs[rc['owner']]
                        rb.put(v, nat3, dx, dy)
                memo[offs] = rgba
                hit = rgba
            cells.append(hit)
        return cells

    if anim == 'gustflick':
        body = srcm & ~plan.ringm if plan.ringm is not None else srcm
        bys, bxs = np.nonzero(body)
        base_blit = blit(_runs_of(bys, bxs))
        flick_cells = [np.zeros(srcm.shape, bool) for _ in plan.flick]
        for (fys, fxs, _s, _d, _dx), fm in zip(plan.flick, flick_cells):
            fm[fys, fxs] = True
        flick_blits = [blit(_runs_of(*np.nonzero(fm))) for fm in flick_cells]
        ring_blits = [blit(rc['runs']) for rc in plan.ring]
        rest_memo = {}                          # active-set -> blit of body minus flicks
        cells = []
        for t in range(N):
            active = tuple(i for i, f in enumerate(plan.flick) if f[2] <= t < f[2] + f[3])
            rgba = _empty_cell(plan)
            v = rgba.reshape(-1)
            if not active:
                base_blit.put(v, nat3, 0, 0)
            else:
                rb = rest_memo.get(active)
                if rb is None:
                    rest = body.copy()
                    for i in active:
                        rest &= ~flick_cells[i]
                    rb = blit(_runs_of(*np.nonzero(rest)))
                    rest_memo[active] = rb
                rb.put(v, nat3, 0, 0)
                for i in active:                # flicked tassels jump by their dx
                    flick_blits[i].put(v, nat3, plan.flick[i][4], 0)
            for rb, rc in zip(ring_blits, plan.ring):   # shadow flicks with its tassel
                if not rb:
                    continue
                fdx = 0
                if rc['owner'] >= 0:
                    _s, _d = plan.flick[rc['owner']][2], plan.flick[rc['owner']][3]
                    if _s <= t < _s + _d:
                        fdx = plan.flick[rc['owner']][4]
                rb.put(v, nat3, fdx, 0)
            cells.append(rgba)
        return cells

    # sway: articulated chunks per unit, every unit every frame (rule 3)
    unit_blits = [[blit(u.chunks[c]) for c in range(3)] for u in plan.units]
    ring_blits = [blit(rc['runs']) for rc in plan.ring]
    memo = {}
    cells = []
    for t in range(N):
        du = tuple(tuple(u.offsets(t, N)) for u in plan.units)
        hit = memo.get(du)
        if hit is None:
            rgba = _empty_cell(plan)
            v = rgba.reshape(-1)
            for ub, (d0, d1, d2) in zip(unit_blits, du):
                for cb, dx in zip(ub, (d0, d1, d2)):
                    if cb:
                        cb.put(v, nat3, dx, 0)
            for rb, rc in zip(ring_blits, plan.ring):   # cast shadow rides top chunk
                if rb:
                    dx = du[rc['owner']][2] if rc['owner'] >= 0 else 0
                    rb.put(v, nat3, dx, 0)
            memo[du] = rgba
            hit = rgba
        cells.append(hit)
    return cells


# ---------------------------------------------------------------- composite


def paste(fr, rgba, ax, ay):
    """Alpha-over paste of an RGBA cell at native anchor (ax,ay), bounds-clipped."""
    H, W = fr.shape[:2]
    h, w = rgba.shape[:2]
    x1, y1 = ax + w, ay + h
    sx, sy = max(0, -ax), max(0, -ay)
    ax, ay = max(0, ax), max(0, ay)
    x1, y1 = min(W, x1), min(H, y1)
    if ax >= x1 or ay >= y1:
        return
    sub = rgba[sy:sy + (y1 - ay), sx:sx + (x1 - ax)]
    a = sub[..., 3] > 0
    if not a.any():
        return
    reg = fr[ay:y1, ax:x1]
    reg[a] = sub[..., :3][a]


def render(nat, objects, N=48, fps=12):
    """Render the scene: frames + plate + per-object sprites + audit.

    objects: doc-order list of {'id','name','label','kind','visible','z','depth',
    'anim','params','mask'(bool HxW)}. Only visible kind=='object' entries animate;
    regions and hidden masks stay untouched plate material (eye-off = "not an object").
    """
    t0 = time.time()
    nat = np.ascontiguousarray(nat, np.uint8)
    H, W = nat.shape[:2]
    N = int(max(2, min(96, N)))
    objs = [o for o in objects
            if o.get('visible', True) and o.get('kind', 'object') == 'object'
            and o.get('mask') is not None and o['mask'].any()]
    objs.sort(key=lambda o: o.get('z', 0))

    # rule 6: an object never owns pixels a higher object covers; shadow-sync rings
    # (plan §3) extend a mover's footprint with the dark pixels hugging it — first
    # claimer in z order wins, and pots/statics keep theirs (static_keep).
    dark = shadow_pixels(nat)
    static_keep = np.zeros((H, W), bool)
    for o in objs:
        if coerce_anim(o.get('anim')) == 'static':
            static_keep |= dark & im.dilate4(o['mask'].copy(), 2)
    above = np.zeros((H, W), bool)
    claimed = np.zeros((H, W), bool)
    srcs, rings = [], []
    for o in objs:
        m = o['mask']
        anim = coerce_anim(o.get('anim'))
        ring = np.zeros((H, W), bool)
        if o.get('shadow') and anim in MOVERS:
            ring = shadow_ring(nat, dark, m, above | claimed | static_keep)
            claimed |= ring
        srcs.append((m | ring) & ~above)
        rings.append(ring)
        above |= m

    # M5 fix: a mover's claimed shadow ring must leave the plate too — otherwise the
    # static original shadow stays painted under the moving ring and sync is invisible
    # (the ring would ride pixel-identical copies of itself).
    holes = above.copy()
    for r in rings:
        holes |= r
    plate = inpaint(nat, holes)

    plans, sprites = [], []
    for o, src, ring in zip(objs, srcs, rings):
        anim = coerce_anim(o.get('anim'))
        p = clean_params(anim, o.get('params'))
        plan = Plan(o, src, anim, p, _rng_for(o), N, (H, W), ring=ring)
        plans.append(plan)
        sprites.append(build_sprites(nat, plan, N))

    frames = np.repeat(plate[None], N, axis=0)
    for plan, cells in zip(plans, sprites):
        ax, ay = plan.cell[0], plan.cell[1]
        for t in range(N):
            paste(frames[t], cells[t % len(cells)], ax, ay)

    # audit (rule 8): diffs only inside allowed zones, wrap included
    allowed = np.zeros((H, W), bool)
    for plan in plans:
        if plan.anim != 'static' and plan.bbox is not None:
            zm = np.zeros((H, W), bool)
            zm[plan.src] = True
            # maxsh==0 (shimmer) still writes inside its own mask -> radius 1
            allowed |= im.dilate4(zm, plan.maxsh + 1)
    aud = audit(frames, allowed)
    aud['allowed_px'] = int(allowed.sum())
    return {'frames': frames, 'plate': plate, 'plans': plans, 'sprites': sprites,
            'objects': objs, 'N': N, 'fps': int(fps), 'audit': aud,
            'ms': round((time.time() - t0) * 1000)}


def audit(frames, allowed):
    """Max changed-px outside `allowed` over consecutive frames incl. the loop wrap."""
    N = len(frames)
    worst, worst_t, stray_frames = 0, -1, 0
    f = frames.astype(np.int16)
    for t in range(N):
        d = (np.abs(f[t] - f[t - 1]).sum(2) > 0) & ~allowed   # t=0 wraps to N-1 ✓
        n = int(d.sum())
        if n:
            stray_frames += 1
        if n > worst:
            worst, worst_t = n, t
    return {'stray_max': worst, 'stray_frame': worst_t, 'stray_frames': stray_frames}
