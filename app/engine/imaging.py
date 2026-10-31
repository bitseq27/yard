"""Pixel Sway engine — integer-pixel image operators.

Everything here works on the art's *native* grid (after grid detection + downsample).
Hard rule (plan §2): integer-pixel operators only — no interpolation, no float resampling,
so masks and sprites stay pixel-crisp and palette-safe.

Contents
    detect_grid      pixel-art grid (largest uniform p x p block)
    quantize         palette snap (denoises dither; plan §2.7)
    hsv_maps         hue/sat/val from RGB
    dilate4/erode4/close4/holefill   4-neighbour morphology
    scanline_flood   run-based flood fill (fast on pixel art)
    wand             magic-wand colour-tolerance fill
    components       run-based connected components (union-find, no scipy)
    rle_encode/decode  compact wire format for masks (JSON friendly)
    mask png helpers
"""
import io

import numpy as np
from PIL import Image

# ---------------------------------------------------------------- grid


def detect_grid(a):
    """Largest p in 1..16 whose p x p blocks are uniformly coloured (>99% of blocks).

    M1 bug ledger: returning the *smallest* such p is useless (p=1 is trivially
    uniform), so we keep the largest.
    """
    H, W = a.shape[:2]
    best = 1
    for p in range(1, 17):
        if H % p or W % p:
            continue
        b = a[:H // p * p, :W // p * p].reshape(H // p, p, W // p, p, a.shape[2])
        if (b == b[:, :1, :, :1, :]).all(axis=(1, 3)).mean() > 0.99:
            best = p
    return best


def native_from(a, p):
    """Downsample by grid size p (take the top-left pixel of each block)."""
    return a[::p, ::p]


# ---------------------------------------------------------------- colour


def quantize(a, colors=64):
    """Palette-snap an HxWx3 uint8 image -> (index map, palette Nx3 int16).

    Origin session lesson: the courtyard art carries ~19.7k near-duplicate colours
    (dither noise). Snapping to a small palette both denoises it and shrinks GIFs.
    """
    im = Image.fromarray(np.ascontiguousarray(a.astype(np.uint8))).convert('RGB')
    q = im.quantize(colors=int(colors), method=Image.MEDIANCUT, dither=Image.Dither.NONE)
    idx = np.asarray(q).astype(np.int32)
    pal = np.array(q.getpalette(), np.int16).reshape(-1, 3)
    n = int(idx.max()) + 1 if idx.size else 0
    return idx, pal[:max(n, 1)]


def hsv_maps(a):
    """RGB (0..255, any int/float dtype) -> (hue deg 0..360, sat 0..1, val 0..1).

    Accepts both HxWx3 images and Nx3 palettes (channel axis is always last).
    """
    arr = np.asarray(a, np.float32) / 255.0
    cmax = arr.max(-1)
    cmin = arr.min(-1)
    df = cmax - cmin
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    hue = np.zeros(arr.shape[:-1], np.float32)
    m = (cmax == r) & (df > 0)
    hue[m] = ((g - b)[m] / df[m]) % 6
    m = (cmax == g) & (df > 0)
    hue[m] = ((b - r)[m] / df[m]) + 2
    m = (cmax == b) & (df > 0)
    hue[m] = ((r - g)[m] / df[m]) + 4
    hue *= 60.0
    sat = np.where(cmax > 0, df / np.maximum(cmax, 1e-6), 0.0).astype(np.float32)
    return hue, sat, cmax


def golden_color(i):
    """Golden-angle HSV colour for suggestion/mask chips (distinct, no two alike)."""
    import colorsys
    h = (i * 0.61803398875) % 1.0
    s, v = [(0.85, 0.95), (0.55, 0.9), (0.9, 0.55), (0.7, 0.75)][i % 4]
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


# ---------------------------------------------------------------- morphology


def dilate4(m, it=1):
    for _ in range(int(it)):
        d = m.copy()
        m[1:, :] |= d[:-1, :]
        m[:-1, :] |= d[1:, :]
        m[:, 1:] |= d[:, :-1]
        m[:, :-1] |= d[:, 1:]
    return m


def erode4(m, it=1):
    for _ in range(int(it)):
        d = m.copy()
        m[1:, :] &= d[:-1, :]
        m[:-1, :] &= d[1:, :]
        m[:, 1:] &= d[:, :-1]
        m[:, :-1] &= d[:, 1:]
    return m


def close4(m, it=1):
    """Dilate then erode — joins fragmented parts, fills 1px gaps (lily-pad lesson)."""
    return erode4(dilate4(m.copy(), it), it)


def _border_seeds(m):
    H, W = m.shape
    seeds = []
    for y in (0, H - 1):
        for x in range(W):
            if m[y, x]:
                seeds.append((y, x))
    for x in (0, W - 1):
        for y in range(H):
            if m[y, x]:
                seeds.append((y, x))
    return seeds


def scanline_flood(ok, seeds, conn=8):
    """Run-based flood fill over boolean predicate `ok` -> visited bool mask.

    Scanline (not per-pixel BFS): pixel-art regions are run-heavy, so this is
    ~100x faster than the origin session's deque BFS and cannot hang on it.
    """
    H, W = ok.shape
    vis = np.zeros((H, W), bool)
    stack = [(y, x) for (y, x) in seeds
             if 0 <= y < H and 0 <= x < W and ok[y, x]]
    pad = 1 if conn == 8 else 0
    while stack:
        y, x = stack.pop()
        if vis[y, x] or not ok[y, x]:
            continue
        x0 = x
        while x0 - 1 >= 0 and ok[y, x0 - 1] and not vis[y, x0 - 1]:
            x0 -= 1
        x1 = x
        while x1 + 1 < W and ok[y, x1 + 1] and not vis[y, x1 + 1]:
            x1 += 1
        vis[y, x0:x1 + 1] = True
        lo = max(x0 - pad, 0)
        hi = min(x1 + pad, W - 1)
        for ny in (y - 1, y + 1):
            if ny < 0 or ny >= H:
                continue
            row = ok[ny]
            vrow = vis[ny]
            xx = lo
            while xx <= hi:
                if row[xx] and not vrow[xx]:
                    stack.append((ny, xx))
                    while xx <= hi and row[xx]:
                        xx += 1
                else:
                    xx += 1
    return vis


def holefill(m):
    """Fill enclosed holes: flood the complement from the image border, keep the rest."""
    inv = ~m
    reach = scanline_flood(inv, _border_seeds(inv), conn=8)
    return m | ~reach


def window(bbox, shape, margin=4):
    """Clipped (y0, y1, x0, x1) slice around a bbox, expanded by `margin`."""
    H, W = shape[:2]
    x0, y0, x1, y1 = bbox
    return (max(0, y0 - margin), min(H, y1 + margin + 1),
            max(0, x0 - margin), min(W, x1 + margin + 1))


def grow_core(m, bbox, r=2, erode_back=1, max_ratio=3.0):
    """Dilate -> hole-fill -> erode, but only inside a bbox window.

    Why: an object's *core* colour class covers only ~50% of its real pixels (the rest is
    interior shading/outline in other classes), so suggestions must grow into their own
    silhouette. Windowing keeps this O(bbox) instead of O(frame) — that is the difference
    between 0.5 s and 20 s for ~70 objects. `max_ratio` stops a lacy core from swallowing
    the frame when hole-fill runs away.
    """
    base = int(m.sum())
    y0, y1, x0, x1 = window(bbox, m.shape, r + 2)
    sub = m[y0:y1, x0:x1].copy()
    sub = dilate4(sub, r)
    sub = holefill(sub)
    if erode_back:
        sub = erode4(sub, erode_back)
    if base and int(sub.sum()) > max_ratio * base:
        sub = m[y0:y1, x0:x1].copy()
        sub = dilate4(sub, max(1, r // 2))
    out = np.zeros(m.shape, bool)
    out[y0:y1, x0:x1] = sub
    return out



# ---------------------------------------------------------------- wand


def wand(nat, y, x, tol=30, conn=8, snap=0, smooth=True, mode='hue',
         base_y=None, dguard=0):
    """Magic wand from native pixel (y, x). `tol` is the tolerance slider value (0..128).

    mode='hue' (default) — hue-family flood: a pixel joins if its circular hue distance to
      the clicked pixel is <= tol degrees, it is chromatic (sat >= 0.30) and not deep shadow
      (val >= 0.14). Pixel-art shading swings value wildly while hue stays put, so this is
      the only metric under which "one click at tol 30 grabs a fern" is true on dithered art
      (measured on the courtyard ground truth: RGB Chebyshev at tol 30 grabs 4% of the fern,
      and at tol 45 already swallows 12% of the whole frame; hue-family grabs 57% of the fern
      and 1.2% of the frame).
      Leak guard: if that family covers >12% of the frame you clicked a *shared* material
      (terracotta pots and wood doors are one hue family), so the wand tightens to the seed's
      own shade family (Chebyshev <= k*tol) until the grab is local again.
    mode='seed' — classic flat flood: Chebyshev RGB distance to the seed colour on the
      palette-snapped image (`snap` colours, 0 = raw). Precise on clean palettes, useless on
      dither; kept because it is what people expect from other tools.
    smooth  morphological close + hole-fill, so the grab is not swiss cheese.
    base_y, dguard (M6b)  depth guard: with a ground-row field from engine/depth and
      dguard > 0, the flood may not leave the seed's ground-row band (|base_y -
      base_y[seed]| <= dguard rows). Colour floods leak along shared materials; a
      ground-row jump is a surface boundary, so this stops the wand at the wall/floor
      junction and at object feet without touching tolerance. dguard=0 keeps M2's
      measured behaviour exactly.
    """
    H, W = nat.shape[:2]
    if not (0 <= y < H and 0 <= x < W):
        raise ValueError(f'seed ({x},{y}) outside {W}x{H}')
    a = np.asarray(nat, np.int16)
    if mode == 'seed':
        if snap:
            idx, pal = quantize(a, snap)
            a = pal[idx].astype(np.int16)
        seed = a[y, x]
        ok = (np.abs(a - seed[None, None, :]).max(axis=2)) <= max(0, int(tol))
        ok = _depth_guard(ok, base_y, y, x, dguard)
        m = scanline_flood(ok, [(y, x)], conn)
        if smooth:
            m = holefill(close4(m, 1))
    else:
        m = _hue_wand(a, y, x, max(0, int(tol)), conn, smooth, base_y, dguard)
    return m


def _depth_guard(ok, base_y, y, x, dguard):
    """Intersect a flood predicate with the seed's ground-row band (M6b)."""
    if not dguard or base_y is None:
        return ok
    b = np.asarray(base_y, np.float32)
    return ok & (np.abs(b - b[y, x]) <= float(dguard))


_HUE_SAT_MIN = 0.30
_HUE_VAL_MIN = 0.14
_LEAK_FRAC = 0.12          # >12% of the frame means "shared material" -> tighten
_LEAK_STEPS = (2.0, 1.5, 1.0)


def _hue_wand(a, y, x, tol, conn=8, smooth=True, base_y=None, dguard=0):
    hue, sat, val = hsv_maps(a)
    if sat[y, x] < _HUE_SAT_MIN or val[y, x] < _HUE_VAL_MIN:
        # clicked a shadow or a highlight: step to the nearest chromatic pixel first
        cand = (sat >= _HUE_SAT_MIN) & (val >= _HUE_VAL_MIN)
        if cand.any():
            yy, xx = np.nonzero(cand)
            k = int(((yy - y) ** 2 + (xx - x) ** 2).argmin())
            y, x = int(yy[k]), int(xx[k])
    dh = np.abs(hue - hue[y, x])
    dh = np.minimum(dh, 360.0 - dh)
    fam = (dh <= tol) & (sat >= _HUE_SAT_MIN) & (val >= _HUE_VAL_MIN)
    fam = _depth_guard(fam, base_y, y, x, dguard)

    def finish(mask):
        return holefill(close4(mask, 1)) if smooth else mask

    limit = _LEAK_FRAC * hue.size
    m = finish(scanline_flood(fam, [(y, x)], conn))
    if m.sum() <= limit:
        return m
    # leak guard: the hue family is a shared material (terracotta pots and wood doors are
    # one family) -> tighten to the seed's own shade family until the grab is local again.
    seed = a[y, x]
    rgb = np.abs(a - seed[None, None, :]).max(axis=2)
    for k in _LEAK_STEPS + (1.0,):
        t = finish(scanline_flood(fam & (rgb <= k * tol), [(y, x)], conn))
        if t.sum() <= limit:
            return t
    return finish(scanline_flood(fam & (rgb <= max(1, tol // 2)), [(y, x)], conn))




# ---------------------------------------------------------------- components


class _DSU:
    __slots__ = ('p',)

    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def row_runs(m):
    """Per-row inclusive runs of a bool mask -> (runs list of (y,x0,x1), row index)."""
    H, W = m.shape
    runs = []
    rows = []
    for y in range(H):
        row = m[y]
        start = len(runs)
        if row.any():
            p = np.concatenate((np.int8([0]), row.view(np.int8), np.int8([0])))
            d = np.diff(p)
            s = np.nonzero(d == 1)[0]
            e = np.nonzero(d == -1)[0] - 1
            for a, b in zip(s.tolist(), e.tolist()):
                runs.append((y, a, b))
        rows.append((start, len(runs)))
    return runs, rows


def components(m, min_px=1, conn=8):
    """Connected components via run-based union-find. Returns dicts sorted by px desc:
    {'px', 'bbox'=(x0,y0,x1,y1), 'runs'=[(y,x0,x1), ...]}
    """
    if not m.any():
        return []
    H, W = m.shape
    runs, rows = row_runs(m)
    n = len(runs)
    dsu = _DSU(n)
    pad = 1 if conn == 8 else 0
    for y in range(H - 1):
        a0, a1 = rows[y]
        b0, b1 = rows[y + 1]
        if a0 == a1 or b0 == b1:
            continue
        i, j = a0, b0
        while i < a1 and j < b1:
            _, s1, e1 = runs[i]
            _, s2, e2 = runs[j]
            if e2 < s1 - pad:
                j += 1
            elif e1 < s2 - pad:
                i += 1
            else:
                dsu.union(i, j)
                if e1 < e2:
                    i += 1
                else:
                    j += 1
    groups = {}
    for i in range(n):
        groups.setdefault(dsu.find(i), []).append(i)
    out = []
    for g in groups.values():
        px = 0
        x0, y0, x1, y1 = W, H, -1, -1
        gr = []
        for i in g:
            y, a, b = runs[i]
            gr.append((y, a, b))
            px += b - a + 1
            if a < x0:
                x0 = a
            if b > x1:
                x1 = b
            if y < y0:
                y0 = y
            if y > y1:
                y1 = y
        if px < min_px:
            continue
        out.append({'px': px, 'bbox': (x0, y0, x1, y1), 'runs': gr})
    out.sort(key=lambda c: -c['px'])
    return out


def mask_from_runs(runs, shape):
    m = np.zeros(shape, bool)
    for y, a, b in runs:
        m[y, a:b + 1] = True
    return m


def biggest(comps):
    """Union of component pixel sets -> bool mask."""
    return comps


# ---------------------------------------------------------------- rle / png


def rle_encode(m):
    """Row-run RLE as a flat JSON list [y, x0, x1, y, x0, x1, ...] (inclusive)."""
    out = []
    for y, a, b in row_runs(np.ascontiguousarray(m, bool))[0]:
        out.append(y)
        out.append(a)
        out.append(b)
    return out


def rle_decode(rle, shape):
    m = np.zeros(shape, bool)
    it = iter(rle)
    for y, a, b in zip(it, it, it):
        m[int(y), int(a):int(b) + 1] = True
    return m


def mask_png_bytes(m):
    """Mask -> RGBA PNG (white where set, transparent elsewhere)."""
    rgba = np.zeros(m.shape + (4,), np.uint8)
    rgba[..., 0:3] = 255
    rgba[..., 3] = np.where(m, 255, 0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba, 'RGBA').save(buf, 'PNG', optimize=True)
    return buf.getvalue()


def mask_from_png_bytes(data, shape=None):
    im = Image.open(io.BytesIO(data))
    if im.mode != 'RGBA':
        im = im.convert('RGBA')
    a = np.asarray(im)
    m = a[..., 3] > 127
    if shape is not None and m.shape != tuple(shape):
        out = np.zeros(shape, bool)
        h = min(shape[0], m.shape[0])
        w = min(shape[1], m.shape[1])
        out[:h, :w] = m[:h, :w]
        return out
    return m


def bbox_of(m):
    ys, xs = np.nonzero(m)
    if not len(ys):
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def stats_of(m):
    """{'px', 'bbox', 'depth'} — depth = iso sort key at the object's feet (plan §8).

    M6a: the key is the closeness formula 3*base_y - y evaluated where the object
    *stands* (y == base_y), i.e. 2*base_y — see engine/depth.object_key. The old
    mean-closeness key (3*base_y - centroid) ranked a tall palm in front of a short
    pot on the same ground row; the client mirror in index.html maskStats matches.
    """
    ys, xs = np.nonzero(m)
    px = int(len(ys))
    if not px:
        return {'px': 0, 'bbox': None, 'depth': 0.0}
    base_y = float(np.percentile(ys, 97))
    return {'px': px,
            'bbox': (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            'depth': round(2.0 * base_y, 1)}
