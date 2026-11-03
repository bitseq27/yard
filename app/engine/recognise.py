"""Pixel Sway engine — object recognition on top of auto-segmentation.

Auto-seg answers "where are the objects and what *material* are they" (plant / pot /
carpet / water). This module answers the next question a user actually has: "what IS it?"
— palm vs fern vs flowering plant vs potted plant vs rug vs lily pad vs pool.

No ML: pixel art gives away its subjects through geometry + palette, and every feature
below is measured on the object's own mask and pixels (plus a little context: is there
foliage sitting on this pot? is this pad inside a pool?). Each rule returns a label and a
confidence; unmatched objects fall back to their material class at low confidence, and the
UI shows the confidence so a wrong guess is obvious and renameable (plan §7: suggestions
are never authoritative).

Features are cheap on purpose: recognition runs per suggestion inside the auto-seg request,
so the whole pass must stay in the millisecond range (measured ~1 ms/object).
"""
import numpy as np

from . import imaging as im

# label -> (anim hint for M3, purely advisory, shown in the UI)
ANIM_HINT = {
    'palm': 'sway (articulated fronds)',
    'fern': 'sway',
    'flowering plant': 'sway + flutter on blossoms',
    'grass / reeds': 'gustflick',
    'shrub': 'sway (slow)',
    'potted plant': 'static pot + sway top',
    'pot / planter': 'static',
    'lily pad': 'bob',
    'flowering lily pad': 'bob + flutter',
    'pool': 'ripple + shimmer',
    'rug': 'static',
    'runner rug': 'static',
    'tassel fringe': 'gustflick',
    'wooden door': 'static',
    'wall tiles': 'static',
    'floor': 'static',
    'stone trim': 'static',
    'shadow': 'static',
}


def _stats(mask, F):
    """Geometric + colour features of one object mask.

    Everything runs inside the object's bbox window: hole-fill and connected components
    are O(frame) per call, and at 60 suggestions that turned a 1.3 s auto-seg into a 6 s
    one (measured). Windowed, recognition costs ~1 ms per object again.
    """
    ys, xs = np.nonzero(mask)
    px = len(ys)
    if not px:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    w, h = x1 - x0 + 1, y1 - y0 + 1
    wy0, wy1, wx0, wx1 = im.window((x0, y0, x1, y1), (F['H'], F['W']), 2)
    sub = mask[wy0:wy1, wx0:wx1]
    hue_w, sat_w, val_w = (F[k][wy0:wy1, wx0:wx1] for k in ('hue', 'sat', 'val'))
    brown_w, green_w, blossom_w = (F[k][wy0:wy1, wx0:wx1] for k in ('brown', 'green', 'blossom'))

    filled = im.holefill(sub.copy())
    st = {
        'px': px, 'w': w, 'h': h, 'aspect': w / h, 'fill': px / (w * h),
        'solidity': px / max(1, int(filled.sum())),
        'cy': ys.mean() / F['H'],                     # 0 = top of frame, 1 = bottom
        'top_frac': float((ys < y0 + h / 3).mean()),  # mass in the top third
        'bot_frac': float((ys > y1 - h / 3).mean()),
    }
    hue, sat, val = hue_w[sub], sat_w[sub], val_w[sub]
    st['green'] = float(((hue >= 55) & (hue <= 180) & (sat > 0.22)).mean())
    st['blossom'] = float(blossom_w[sub].mean())
    st['brown'] = float(brown_w[sub].mean())
    st['dark'] = float((val < 0.16).mean())
    st['bright'] = float((val > 0.7).mean())

    # brown just below the feet: trunk of a palm, or the rim of a planter
    by0, by1 = min(y1 + 1, F['H'] - 1), min(y1 + 13, F['H'])
    cx0, cx1 = x0 + w // 5, max(x0 + w // 5 + 1, x1 - w // 5)
    st['brown_below'] = float(F['brown'][by0:by1, cx0:cx1].mean()) if by1 > by0 else 0.0

    # trunk: a narrow brown column reaching the object's feet (palms, trees)
    st['trunk'] = False
    bm = sub & brown_w
    if bm.sum() >= 8:
        for c in im.components(bm, 6, 8):
            cw = c['bbox'][2] - c['bbox'][0] + 1
            ch = c['bbox'][3] - c['bbox'][1] + 1
            if ch >= 8 and cw <= max(4, w * 0.3) and ch >= cw * 1.6 \
                    and c['bbox'][3] >= (y1 - wy0) - 3:
                st['trunk'] = True
                break

    # frond units: green survives erosion as separate lobes -> lacy/radiating foliage
    gm = im.erode4((sub & green_w).copy(), 1)
    st['units'] = len(im.components(gm, 5, 8))
    st['lace'] = st['fill'] < 0.45 and st['solidity'] < 0.8

    # holes / fragments
    st['holes'] = int(filled.sum() - px)
    st['frags'] = len(im.components(sub, 8, 8))
    return st


def recognise(mask, F, kind, cls):
    """(label, confidence) for one suggestion mask. F = feature maps from segment."""
    st = _stats(mask, F)
    if st is None:
        return cls, 0.0

    # ---- water ------------------------------------------------------------
    if cls == 'water':
        if kind == 'object':
            return 'pool', 0.8          # interior water bodies only, by construction
        return ('wall tiles', 0.55) if st['cy'] < 0.45 else ('floor tiles', 0.55)

    # ---- lily pads ---------------------------------------------------------
    if cls == 'lily_pad':
        if st['blossom'] > 0.10 or st['bright'] > 0.25:
            return 'flowering lily pad', 0.75
        return 'lily pad', 0.85

    # ---- pots (context rule in recognise_with_context upgrades to potted plant)
    if cls == 'pot':
        return 'pot / planter', 0.6

    # ---- carpets -----------------------------------------------------------
    if cls == 'carpet':
        return ('runner rug', 0.7) if st['aspect'] >= 1.7 else ('rug', 0.75)

    if cls == 'fringe':
        return 'tassel fringe', 0.6

    # ---- foliage -----------------------------------------------------------
    if cls == 'plant':
        if st['blossom'] >= 0.06 and st['px'] <= 2500:
            return 'flowering plant', 0.75
        if (st['trunk'] or st['brown_below'] >= 0.12) and st['h'] >= 90 and st['top_frac'] >= 0.28:
            return 'palm', 0.75
        if st['brown_below'] >= 0.25 and st['h'] <= 70:
            return 'potted plant', 0.7       # the foliage half of a potted plant
        if st['units'] >= 18 and st['aspect'] >= 1.3 and st['h'] <= 110:
            return 'fern', 0.7
        if st['units'] >= 10 and st['fill'] < 0.5 and st['h'] <= 70:
            return 'fern', 0.6
        if st['aspect'] < 0.7 and st['h'] >= 22 and st['fill'] < 0.5:
            return 'grass / reeds', 0.5
        if st['fill'] >= 0.5 and st['px'] <= 900:
            return 'shrub', 0.5
        return 'plant', 0.4

    # ---- regions -----------------------------------------------------------
    if cls == 'wood':
        if st['aspect'] < 1.3 and st['h'] >= 36 and st['brown'] > 0.4:
            return 'wooden door', 0.7
        return 'woodwork', 0.5
    if cls == 'water':
        return 'wall tiles', 0.5 if st['cy'] < 0.5 else 0.5
    if cls == 'stone':
        return 'stone trim', 0.55
    if cls == 'shadow':
        return 'shadow', 0.6
    if cls == 'ground':
        return 'floor', 0.5
    if cls == 'light':
        return 'plaster / highlight', 0.4
    return cls, 0.3


def recognise_with_context(mask, F, kind, cls, ctx):
    """recognise() plus the two context rules that need neighbours:
    a pot with foliage on top is a *potted plant*; a pad is only a pad inside a pool."""
    if cls == 'pot':
        x0, y0, x1, y1 = im.bbox_of(mask)
        band = ctx['plant_d'][max(y0 - 5, 0):y0 + 2, x0:x1 + 1]
        if band.any():
            return 'potted plant', 0.8
        return 'pot / planter', 0.7
    return recognise(mask, F, kind, cls)
