"""Pixel Sway engine — exports (plan §2.7, §8; M3).

Three outputs over one render (parity by construction — see animate.py):
  strip_png    N frames as one grid image ("strip") the browser plays frame-by-frame
  gif_bytes    shared-palette GIF: quantize once from a frame montage, per-frame
               dither=NONE, ×2 NEAREST upscale, gifsicle -O3 (origin chain, 10.8MB -> 0.8MB)
  bundle_zip   the game bundle (plan §8): plate.png, walkmask.png, palette.png,
               scene.json, objects/<id>.png sheets + clips metadata.

Sheets are the render's sprite cells verbatim, laid out in the same grid as the strip,
with RGB zeroed under alpha==0 — engines that blend premultiplied or bilinear can never
pick up a halo colour from a transparent pixel ("clean RGBA edges" acceptance).
"""
import io
import json
import os
import subprocess
import tempfile
import zipfile

import numpy as np
from PIL import Image

from . import imaging as im

STRIP_COLS = 8
WALKABLE_LABELS = ('rug', 'runner rug', 'carpet', 'floor')      # objects you walk ON
NONWALK_REGIONS = ('water', 'shadow')                           # plate material you don't


# ---------------------------------------------------------------- strip


def strip_png(frames, cols=STRIP_COLS):
    """Frames -> one grid PNG (cols per row). Returns (png_bytes, cols, fw, fh)."""
    N, H, W = frames.shape[:3]
    cols = int(min(max(1, cols), N))
    rows = (N + cols - 1) // cols
    sheet = np.zeros((rows * H, cols * W, 3), np.uint8)
    for t in range(N):
        r, c = divmod(t, cols)
        sheet[r * H:(r + 1) * H, c * W:(c + 1) * W] = frames[t]
    buf = io.BytesIO()
    Image.fromarray(sheet).save(buf, 'PNG', optimize=True)
    return buf.getvalue(), cols, W, H


def cells_grid(cells, cols=STRIP_COLS):
    """Sprite cells -> one RGBA grid array. Returns (sheet, cols, cw, ch)."""
    n = len(cells)
    ch, cw = cells[0].shape[:2]
    cols = int(min(max(1, cols), n))
    rows = (n + cols - 1) // cols
    sheet = np.zeros((rows * ch, cols * cw, 4), np.uint8)
    for t, cell in enumerate(cells):
        r, c = divmod(t, cols)
        sheet[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = cell
    return sheet, cols, cw, ch


def cells_png(cells, cols=STRIP_COLS):
    """Sprite cells -> one RGBA grid PNG. Returns (png_bytes, cols, cw, ch)."""
    sheet, cols, cw, ch = cells_grid(cells, cols)
    buf = io.BytesIO()
    Image.fromarray(sheet, 'RGBA').save(buf, 'PNG', optimize=True)
    return buf.getvalue(), cols, cw, ch


def pack_atlas(sheets, pad=1, max_page=4096):
    """M5 atlas packing: shelf-pack named RGBA sheet arrays into square pages.

    sheets: [(name, rgba HxWx4)]. Returns (pages, rects) where pages are RGBA arrays
    and rects[name] = (page_idx, x, y, w, h) or None when a sheet is too big for a
    page (callers keep its standalone file as the fallback). Deterministic: sheets are
    packed tallest-first, ties by name."""
    items = sorted(sheets, key=lambda t: (-t[1].shape[0], -t[1].shape[1], t[0]))
    if not items:
        return [], {}
    area = sum(int(a.shape[0] + 2 * pad) * int(a.shape[1] + 2 * pad) for _, a in items)
    biggest = max(max(a.shape[:2]) for _, a in items) + 2 * pad
    p = 256
    while p < max_page and (p * p * 3 < area * 4 or p < biggest):
        p *= 2
    p = int(min(max(p, 64), max_page))
    pages = []
    rects = {}

    def newpage():
        pages.append({'arr': np.zeros((p, p, 4), np.uint8), 'x': pad, 'y': pad, 'sh': 0})

    newpage()
    for name, a in items:
        h, w = int(a.shape[0]), int(a.shape[1])
        if w + 2 * pad > p or h + 2 * pad > p:
            rects[name] = None                # oversized: standalone sheet only
            continue
        pg = pages[-1]
        if pg['x'] + w + pad > p:             # next shelf
            pg['y'] += pg['sh'] + pad
            pg['x'] = pad
            pg['sh'] = 0
        if pg['y'] + h + pad > p:             # next page
            newpage()
            pg = pages[-1]
        x, y = pg['x'], pg['y']
        pg['arr'][y:y + h, x:x + w] = a
        pg['x'] = x + w + pad
        pg['sh'] = max(pg['sh'], h)
        rects[name] = (len(pages) - 1, x, y, w, h)
    return [pg['arr'] for pg in pages], rects


# ---------------------------------------------------------------- palette + gif


def shared_palette(frames, colors=128, montage=8):
    """Quantize once from an evenly spaced frame montage -> (palette Nx3 uint8).

    Origin rule: a shared palette + dither=NONE means no inter-frame flicker and tiny
    LZW output; per-frame palettes would re-map dither every frame (plan §2.7).
    """
    N = len(frames)
    idxs = sorted(set(int(round(i * (N - 1) / max(1, montage - 1))) for i in range(montage)))
    mon = np.concatenate([frames[i] for i in idxs], axis=0)
    _idx, pal = im.quantize(mon, colors)
    return pal.astype(np.uint8)


def gif_bytes(frames, fps=12, scale=2, colors=128, pal=None):
    """Render frames -> optimized animated GIF bytes. Returns (gif_bytes, info)."""
    N = len(frames)
    fps = int(max(2, min(30, fps)))
    scale = int(max(1, min(4, scale)))
    pal = shared_palette(frames, colors) if pal is None else np.asarray(pal, np.uint8)
    pal_img = Image.new('P', (1, 1))
    pal_img.putpalette(np.asarray(pal, np.uint8).reshape(-1).tolist())
    imgs = []
    for t in range(N):
        fr = Image.fromarray(frames[t])
        q = fr.quantize(palette=pal_img, dither=Image.Dither.NONE)
        if scale > 1:
            q = q.resize((q.width * scale, q.height * scale), Image.NEAREST)
        imgs.append(q)
    dur = int(round(1000.0 / fps))
    raw = io.BytesIO()
    imgs[0].save(raw, 'GIF', save_all=True, append_images=imgs[1:], duration=dur,
                 loop=0, optimize=False, disposal=1)
    raw = raw.getvalue()
    out = raw
    try:                                       # gifsicle -O3: lossless, ~10x smaller
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, 'in.gif')
            dst = os.path.join(td, 'out.gif')
            with open(src, 'wb') as f:
                f.write(raw)
            r = subprocess.run(['gifsicle', '-O3', src, '-o', dst],
                               capture_output=True, timeout=120)
            if r.returncode == 0 and os.path.getsize(dst):
                with open(dst, 'rb') as f:
                    out = f.read()
    except (OSError, subprocess.SubprocessError):
        pass                                   # gifsicle missing: raw PIL GIF still valid
    info = {'bytes': len(out), 'raw_bytes': len(raw), 'frames': N, 'fps': fps,
            'scale': scale, 'colors': len(pal), 'duration_ms': dur}
    return out, info


# ---------------------------------------------------------------- scene bundle


def walkmask(shape, plans_or_objs, regions):
    """Default walkmask (white = walkable): everything except object footprints you
    cannot walk on and water/shadow regions. Rugs/carpets stay walkable (adventure-game
    convention). M4 adds region tools to edit this by hand (plan §8)."""
    H, W = shape[:2]
    wm = np.full((H, W), 255, np.uint8)
    for o in plans_or_objs:
        label = str(o.get('label') or o.get('name') or '').lower()
        if any(label.startswith(w) for w in WALKABLE_LABELS):
            continue
        src = o['mask'] if isinstance(o.get('mask'), np.ndarray) else o.src
        wm[src] = 0
    for r in regions or []:
        name = str(r.get('name') or '').lower()
        if any(n in name for n in NONWALK_REGIONS):
            wm[r['mask']] = 0
    return wm


def bundle_zip(nat, res, doc, fps=None, scale=1, colors=128, pal=None, atlas=True):
    """Scene bundle per plan §8. `res` = animate.render() output.

    Returns (zip_bytes, info). Layout:
        plate.png            art with every visible object removed + inpainted
        walkmask.png         white = walkable (derived default, M4 makes it editable)
        palette.png          shared palette strip (the GIF palette when provided)
        scene.json           anchors, depths, bboxes, clips, layout metadata
        objects/<id>.png     RGBA sprite sheets (grid, cols=8) — render sprites verbatim
        atlas_<i>.png        M5: all sheets shelf-packed (scene.json objects[].atlas;
                             oversized sheets keep only their standalone file)
    """
    fps = int(fps or res.get('fps') or doc.get('fps') or 12)
    N = int(res['N'])
    plate = res['plate']
    if scale and scale > 1:
        plate = np.asarray(Image.fromarray(plate).resize(
            (plate.shape[1] * scale, plate.shape[0] * scale), Image.NEAREST))
    objs = res['objects']
    regions = [o for o in doc.get('objects', []) if o.get('kind') == 'region']
    region_masks = []
    for r in regions:                        # regions carry masks by id -> load lazily
        m = r.get('_mask')
        if m is not None:
            region_masks.append({'name': r.get('name'), 'mask': m})
    wm = walkmask(nat.shape[:2], objs, region_masks)

    pal = shared_palette(res['frames'], colors) if pal is None else np.asarray(pal, np.uint8)

    scene = {
        'format': 'pixel-sway-bundle', 'version': 1,
        'native_w': int(nat.shape[1]), 'native_h': int(nat.shape[0]),
        'grid': doc.get('grid', 1), 'scale': int(scale or 1),
        'fps': fps, 'loop_frames': N,
        'palette': {'file': 'palette.png', 'colors': len(pal)},
        'plate': {'file': 'plate.png'},
        'walkmask': {'file': 'walkmask.png', 'note': 'white=walkable; derived default'},
        'regions': {'hotspots': doc.get('regions', {}).get('hotspots', []),
                    'spawnpoints': doc.get('regions', {}).get('spawnpoints', [])},
        'objects': [],
    }
    files = {}
    files['plate.png'] = _png_bytes(plate, 'RGB')
    files['walkmask.png'] = _png_bytes(wm, 'L')
    files['palette.png'] = _png_bytes(pal.reshape(1, -1, 3), 'RGB')

    n_sheets = 0
    atlas_src = []
    for plan, cells in zip(res['plans'], res['sprites']):
        if plan.bbox is None:
            continue
        # clean RGBA edges: zero RGB wherever alpha is 0 (no halo colour to bleed)
        clean = []
        for c in cells:
            c2 = np.array(c)
            c2[..., :3][c2[..., 3] == 0] = 0
            clean.append(c2)
        grid, cols, cw, chh = cells_grid(clean)
        png = _png_bytes(grid, 'RGBA')
        oid = str(plan.id)
        fname = f'objects/{oid}.png'
        files[fname] = png
        atlas_src.append((oid, grid))
        n_frames = len(clean)
        n_sheets += 1
        scene['objects'].append({
            'id': oid, 'name': plan.name, 'label': plan.label,
            'z': int(plan.z), 'depth': plan.depth,
            'anchor': [int(plan.cell[0]), int(plan.cell[1])],
            'bbox': [int(v) for v in plan.bbox],
            'sheet': {'file': fname, 'frames': n_frames, 'cols': cols,
                      'cell_w': int(cw), 'cell_h': int(chh), 'loop': True},
            'clips': [{'name': 'idle', 'anim': plan.anim, 'params': plan.params,
                       'frames': N if plan.anim != 'static' else 1,
                       'fps': fps, 'loop': True}],
        })
    if atlas and atlas_src:
        pages, rects = pack_atlas(atlas_src)
        if pages:
            scene['atlas'] = {
                'pad': 1,
                'pages': [{'file': f'atlas_{i}.png', 'w': int(pg.shape[1]),
                           'h': int(pg.shape[0])} for i, pg in enumerate(pages)],
            }
            for i, pg in enumerate(pages):
                files[f'atlas_{i}.png'] = _png_bytes(pg, 'RGBA')
            for o in scene['objects']:
                r = rects.get(o['id'])
                if r is not None:
                    o['atlas'] = {'page': r[0], 'x': r[1], 'y': r[2], 'w': r[3], 'h': r[4]}
    scene['objects'].sort(key=lambda o: o['z'])
    files['scene.json'] = json.dumps(scene, indent=1).encode()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):
            z.writestr(name, files[name])
    data = buf.getvalue()
    info = {'bytes': len(data), 'sheets': n_sheets, 'objects': len(scene['objects']),
            'walkable_px': int((wm > 0).sum()), 'N': N, 'fps': fps,
            'atlas_pages': len(scene.get('atlas', {}).get('pages', []))}
    return data, info


def _png_bytes(arr, mode):
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr), mode).save(buf, 'PNG', optimize=True)
    return buf.getvalue()
