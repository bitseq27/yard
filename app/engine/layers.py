"""Pixel Sway engine — depth layers for game-scene prep (M7, plan §8b).

The depth pass (M6a parse, or the M6e/M7 model-fused map — DA3 by default) answers
"what is in front of what" per pixel. This module turns that field into the thing
a 2.5D adventure game actually consumes: a stack of depth-banded plates with
sprites slotted between them, so a character walking around the scene is
occluded by (and occludes) the right art at every step.

  bands      1-D k-means over the depth histogram → k-1 cut points. Bands are
             ordered FAR (0) → NEAR (k-1); quantile-seeded, deterministic, so the
             same field + k always gives the same cut points.
  plates     per band i, an RGBA plate:
               member   (band==i, not object-owned)   original art, opaque
               nearer   (band>i) ∪ object footprints   INPAINTED, opaque —
                        disocclusion backing: when a sprite sways, a character
                        walks past, or the camera parallaxes, the revealed hole
                        shows plausible continuation instead of transparency
               farther  (band<i)                       alpha 0 — the plates
                        behind show through
             Consequence (pinned in test_m7): compositing plates far→near and
             drawing each object's sprite after its own band's plate reproduces
             the native art pixel-exactly — layers are a lossless re-packaging
             of the scene, not a filter.
  objects    every visible kind=='object' mask is assigned to the band of its
             FEET (97th-percentile ground row, the app's own contact rule —
             imaging.stats_of), with the raw foot depth kept as a continuous
             0..1 value so an engine can y-sort smoothly inside a band.
  regions    band_map()/region_masks() expose the bands as plain masks, which the
             editor offers as one-click region suggestions ("add bands as
             regions") — depth-driven segmentation that lands in the same
             editable object list as auto-seg.

Engine rule holds: numpy + PIL only. Inpainting reuses animate.inpaint (the
proven plate chain: horizontal run fill + boundary blend on long runs).
"""
import time

import numpy as np

from . import animate
from . import imaging as im

# band tints for the overlay preview: far = cool, near = warm
RAMP = ((59, 91, 219), (64, 156, 255), (55, 178, 122), (240, 177, 41),
        (247, 103, 7), (214, 48, 49))

CFG = dict(
    k=3,               # default band count
    kmeans_iters=12,   # 1-D k-means over a 256-bin histogram: cheap and stable
    min_k=2,
    max_k=6,
)


# ------------------------------------------------------------------ banding


def bands(d, k=CFG['k'], iters=CFG['kmeans_iters']):
    """(cuts, centers) for a u8 closeness map (higher = nearer).

    cuts: k-1 ascending thresholds; band i is cuts[i-1] <= d < cuts[i] with
    band 0 the farthest. centers: the per-band mean depth. 1-D k-means on the
    histogram, seeded at pixel quantiles — deterministic, no RNG.
    """
    d = np.asarray(d)
    k = int(max(CFG['min_k'], min(CFG['max_k'], k)))
    hist = np.bincount(np.clip(d, 0, 255).astype(np.uint8).ravel(),
                       minlength=256).astype(np.float64)
    cdf = np.cumsum(hist) / max(hist.sum(), 1.0)
    centers = np.array([float(np.searchsorted(cdf, (j + 0.5) / k))
                        for j in range(k)])
    bins = np.arange(256, dtype=np.float64)
    for _ in range(iters):
        idx = np.abs(bins[:, None] - centers[None, :]).argmin(1)
        new = centers.copy()
        for j in range(k):
            m = idx == j
            if hist[m].sum() > 0:
                new[j] = float((bins[m] * hist[m]).sum() / hist[m].sum())
        if np.allclose(new, centers):
            centers = new
            break
        centers = new
    centers = np.sort(centers)
    cuts = np.array([int(round((centers[j] + centers[j + 1]) / 2.0))
                     for j in range(k - 1)])
    # strictly increasing cuts even on degenerate (near-constant) fields
    for j in range(1, len(cuts)):
        if cuts[j] <= cuts[j - 1]:
            cuts[j] = cuts[j - 1] + 1
    return cuts, centers


def band_map(d, cuts):
    """Per-pixel band index (0 = farthest … k-1 = nearest)."""
    return np.searchsorted(np.asarray(cuts), np.asarray(d), side='right')


def region_masks(bm, k):
    """[bool mask per band] — the bands as plain segmentation masks."""
    return [bm == i for i in range(k)]


# ------------------------------------------------------------------ objects


def object_bands(objects, d, cuts):
    """Per-object band assignment at the FEET, the app's ground-contact rule
    (97th percentile row — imaging.stats_of). Returns a list parallel to the
    input objects: {'band', 'foot_y', 'foot_x', 'foot_depth' (u8 median at the
    contact row), 'depth_n' (0..1 continuous, for smooth y-sort inside a band)}.
    """
    out = []
    d = np.asarray(d)
    H, W = d.shape[:2]
    rows = np.mgrid[0:H, 0:W][0]
    k = k_from_cuts(cuts)
    for o in objects:
        m = np.asarray(o['mask'], bool)
        ys, xs = np.nonzero(m)
        if not len(ys):
            out.append({'band': k - 1, 'foot_y': 0, 'foot_x': 0,
                        'foot_depth': 0, 'depth_n': 0.0})
            continue
        foot_y = float(np.percentile(ys, 97))
        foot = m & (rows >= int(foot_y) - 1)
        if not foot.any():
            foot = m & (rows == int(ys.max()))
        fys, fxs = np.nonzero(foot)
        fd = int(np.median(d[foot]))
        out.append({'band': int(np.searchsorted(cuts, fd, side='right')),
                    'foot_y': int(round(foot_y)),
                    'foot_x': int(np.median(fxs)),
                    'foot_depth': fd,
                    'depth_n': round(fd / 255.0, 4)})
    return out


def k_from_cuts(cuts):
    return len(cuts) + 1


# ------------------------------------------------------------------ separation


def separate(nat, d, objects=None, k=CFG['k'], fill=True):
    """Depth-band the scene into game-ready layers.

    nat: HxWx3 uint8 native art. d: u8 closeness map (higher = nearer) — the
    model-fused field when available, the parse otherwise. objects: visible
    kind=='object' entries with 'mask' attached (regions are plate material and
    stay in the plates). fill=False gives pure cut-out bands (no inpainting).

    Returns {'k', 'cuts', 'centers', 'bands': [{'i','lo','hi','px','plate'}],
    'objects': object_bands(...) + the object's 'id'/'name', 'obj_any', 'ms'}.
    """
    t0 = time.time()
    nat = np.ascontiguousarray(nat, np.uint8)
    d = np.asarray(d)
    H, W = nat.shape[:2]
    cuts, centers = bands(d, k)
    k = k_from_cuts(cuts)
    bm = band_map(d, cuts)

    objects = [o for o in (objects or [])
               if o.get('mask') is not None and np.asarray(o['mask'], bool).any()]
    obj_any = np.zeros((H, W), bool)
    for o in objects:
        obj_any |= np.asarray(o['mask'], bool)

    lob = [0] + [int(c) for c in cuts]
    hib = [int(c) - 1 for c in cuts] + [255]
    out_bands = []
    for i in range(k):
        member = (bm == i) & ~obj_any
        removed = obj_any | (bm > i)          # everything drawn after plate i
        if fill and removed.any():
            base = animate.inpaint(nat, removed)
        else:
            base = nat
        alpha = member | (removed if fill else False)
        rgba = np.zeros((H, W, 4), np.uint8)
        rgba[..., :3][alpha] = base[alpha]
        rgba[..., 3][alpha] = 255
        out_bands.append({'i': i, 'lo': lob[i], 'hi': hib[i],
                          'px': int(member.sum()), 'plate': rgba})

    ob = object_bands(objects, d, cuts)
    for o, e in zip(objects, ob):
        e['id'] = o.get('id')
        e['name'] = o.get('name')
    return {'k': k, 'cuts': [int(c) for c in cuts],
            'centers': [round(float(c), 1) for c in centers],
            'bands': out_bands, 'objects': ob, 'obj_any': obj_any,
            'ms': round((time.time() - t0) * 1000)}


# ------------------------------------------------------------------ previews


def overlay(nat, bm, k):
    """Band-tinted preview (far cool → near warm) over the art, for the depth tab."""
    nat = np.asarray(nat, np.float32)
    tint = np.zeros(nat.shape, np.float32)
    for i in range(k):
        c = RAMP[int(round(i * (len(RAMP) - 1) / max(k - 1, 1)))]
        tint[bm == i] = c
    out = nat * 0.55 + tint * 0.45
    return np.clip(out, 0, 255).astype(np.uint8)


def strip(res, bg=(24, 24, 28), gap=2):
    """Plates side by side (far → near) on a dark ground, as one RGB array."""
    plates = [b['plate'] for b in res['bands']]
    H = plates[0].shape[0]
    ws = [p.shape[1] for p in plates]
    out = np.zeros((H, sum(ws) + gap * (len(plates) - 1), 3), np.uint8)
    out[:] = bg
    x = 0
    for p, w in zip(plates, ws):
        rgb = np.asarray(p[..., :3]).copy()
        a = (p[..., 3:4] > 0).astype(np.float32)
        comp = rgb * a + np.array(bg, np.float32) * (1 - a)
        out[:, x:x + w] = comp.astype(np.uint8)
        x += w + gap
    return out


def composite(res):
    """Plates far→near — the reference recomposition test_m7 pins: on every
    non-object pixel this equals the native art, and object pixels carry the
    inpainted backing a sprite is drawn on (sprite cut-outs restore the art)."""
    p0 = res['bands'][0]['plate']
    out = np.zeros(p0.shape[:2] + (3,), np.uint8)
    for b in res['bands']:
        p = b['plate']
        a = p[..., 3] > 0
        out[a] = p[..., :3][a]
    return out
