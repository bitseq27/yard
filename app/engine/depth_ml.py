"""M6e — optional monocular-depth accelerator for the depth pass.

The M6a field is a *parse*: ground-row closeness per surface. Measured against the
origin session's depth.npy (app/tests/depth_alts.py, entry 16) it scores spearman
0.556 and reads flat — smooth plates, cut-out silhouettes, no facade. Monocular
depth models score far better on layout (Depth Anything V2 small int8: 0.804;
MiDaS v2.1 small: 0.781) but blur object interiors; blending model layout with the
parse's crisp structure beats both (0.777 / 0.772 at the weights below).

This module is an OPTIONAL accelerator, not engine core: the engine rule stays
"numpy + PIL only" for everything the render/sprite/audit path touches, and this
file imports onnxruntime lazily, inside `fuse()`, only when a runtime AND a model
file are actually present. Without them `fuse()` returns None and the server keeps
shipping the pure parse field — same bytes as M6b.

Fusion weights were picked by a documented rule on the bake-off, not by eye:
maximise rho + conc + wallrho subject to within-object rho_in >= 0.40 (keep a
majority of the parse's inside-object ordering), scored against the ART-ONLY
parse because that is what upload-time fusion sees. That gives w=0.8 for DA-v2
(rho 0.829, conc 0.853, rho_in 0.454) and w=0.6 for MiDaS (rho 0.785, rho_in
0.406 — its facade term never turns positive, so the constraint caps the weight
while interiors stay parse-driven). test_m6 pins fused rho >= 0.75 and at least
0.10 above the parse it fuses.
"""
import os

import numpy as np
from PIL import Image

VERSION = 'm6e-da0.8-mi0.6'

IMNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# name, onnx file, input edge, normalisation, blend weight (model share)
MODELS = (
    ('da-v2-small-int8', 'da_v2_small_quant.onnx', 518, IMNET, 0.8),
    # fp32 twin of the int8 model, same weight: kept as a fallback for installs
    # where only the bigger file survived (workspace snapshots cap ~128 MB and
    # dropped the 27 MB quant file); bake-off rho .794 vs .804 for the int8
    ('da-v2-small', 'da_v2_small.onnx', 518, IMNET, 0.8),
    ('midas-v21-small', 'midas_v21_small.onnx', 256, IMNET, 0.6),
)


def _search_dirs(extra=None):
    here = os.path.dirname(os.path.abspath(__file__))
    app = os.path.dirname(here)
    if extra:                      # explicit override = search ONLY there
        return [extra]
    dirs = []
    if os.environ.get('DEPTH_MODEL_DIR'):
        dirs.append(os.environ['DEPTH_MODEL_DIR'])
    dirs += [os.path.join(app, 'models'), os.path.join(app, 'tests', 'models')]
    return dirs


def _ort_dirs(extra=None):
    here = os.path.dirname(os.path.abspath(__file__))
    app = os.path.dirname(here)
    if extra:                      # explicit override = search ONLY there
        return [extra]
    dirs = []
    if os.environ.get('ORT_PATH'):
        dirs.append(os.environ['ORT_PATH'])
    dirs += [os.path.join(app, 'tests', '_ort'), os.path.join(app, '_ort')]
    return dirs


def available(models_dir=None):
    """(model_name, weight, onnx_path) for the first model we could actually run,
    else None. Cheap: file checks only, no import.

    DEPTH_MODEL env picks the path: 'off' disables the accelerator, 'da' / 'midas'
    force one family (midas = the 0.4 s path for slow machines), unset = prefer DA.
    """
    want = os.environ.get('DEPTH_MODEL', '').lower()
    if want == 'off':
        return None
    for name, fn, _edge, _norm, w in MODELS:
        if want and not name.startswith({'da': 'da', 'midas': 'midas'}.get(want, want)):
            continue
        for d in _search_dirs(models_dir):
            p = os.path.join(d, fn)
            if os.path.exists(p) and os.path.getsize(p) > 1e6:
                return name, w, p
    return None


def source(models_dir=None):
    a = available(models_dir)
    return f'{a[0]}@w{a[1]:g}' if a else 'parse-only'


def _session(path, ort_dir=None):
    import sys
    for d in _ort_dirs(ort_dir):
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.inter_op_num_threads = 4
    so.intra_op_num_threads = 4
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])


def _run(sess, nat, edge, norm):
    img = Image.fromarray(np.ascontiguousarray(nat, np.uint8)).convert('RGB')
    a = img.resize((edge, edge), Image.BILINEAR)
    a = np.asarray(a, np.float32) / 255.0
    mean, std = norm
    a = (a - np.array(mean, np.float32)) / np.array(std, np.float32)
    x = np.transpose(a, (2, 0, 1))[None].astype(np.float32)
    inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
    y = sess.run([out.name], {inp.name: x})[0]
    d = np.asarray(y[0], np.float64)
    while d.ndim > 2:
        d = d[0]
    return np.asarray(Image.fromarray(d).resize((nat.shape[1], nat.shape[0]),
                                                Image.BILINEAR), np.float64)


def _znorm(a):
    a = np.asarray(a, np.float64)
    return (a - a.mean()) / (a.std() + 1e-9)


def fuse(nat, parse_depth, models_dir=None, ort_dir=None):
    """Blended depth (uint8, higher = nearer) or None when no model is runnable.

    blend = w*znorm(model) + (1-w)*znorm(parse), encoded with a 1-99 percentile
    window so one outlier pixel cannot flatten the contrast (min-max did, in the
    bake-off montages). Orientation of the model output is fixed without ground
    truth: in iso art depth grows toward the bottom row, so a model map that
    correlates negatively with the row axis is sign-flipped first.
    """
    a = available(models_dir)
    if not a:
        return None, None
    name, w, path = a
    for mname, fn, edge, norm, mw in MODELS:
        if mname != name:
            continue
        w = mw
        try:
            sess = _session(path, ort_dir)
            d = _run(sess, nat, edge, norm)
        except Exception as e:                      # accelerator stays optional
            return None, {'model': name, 'error': f'{type(e).__name__}: {e}'}
        ys = np.mgrid[0:nat.shape[0], 0:nat.shape[1]][0]
        if np.corrcoef(d.ravel(), ys.ravel())[0, 1] < 0:
            d = -d
        blend = w * _znorm(d) + (1 - w) * _znorm(parse_depth)
        lo, hi = np.percentile(blend, (1, 99))
        u8 = np.clip((blend - lo) / (hi - lo + 1e-9) * 255.0, 0, 255).astype(np.uint8)
        return u8, {'model': name, 'w': w, 'version': VERSION}
    return None, None
