"""M6e/M7 — optional monocular-depth accelerator for the depth pass.

The M6a field is a *parse*: ground-row closeness per surface. Measured against the
origin session's depth.npy (app/tests/depth_alts.py) it scores spearman 0.556 and
reads flat — smooth plates, cut-out silhouettes, no facade. Monocular depth models
score far better on layout, and Depth Anything 3 small (fp32 onnx) is the best of
the family on this art:

  model              rho     conc    rho_in  wallrho  wflat   (bake-off, courtyard)
  m6a parse          0.556   0.728   0.703   -0.166   0.42
  midas-v21-small    0.822   0.849   0.364   -0.230   0.20
  da-v2-small-int8   0.848   0.866   0.431   +0.209   0.22
  da3-small (fp32)   0.887   0.887   0.403   +0.240   0.22
  fuse da3 w=0.9     0.885   0.886   0.455   +0.246   0.23   (M7 shipped default)
  da-v2-ViT-L int8   0.832*  0.855*  0.412*  +0.112*  0.23*  (entry 20 default, w=1.0)

* measured through the quantised export at edge 336: this sandbox's 1 GiB cgroup
cannot hold the 1.34 GB fp32 weights (nor the int8 attention buffer at 518²), so
the fp32 map the user's machine runs is unmeasurable here — the proxy understates
it. The user asked for exactly this model (their ComfyUI node runs the same
weights, fp32), and the documented fusion rule picks w=1.0 for the large family:
raw model beats every parse blend on rho+conc+wallrho at rho_in >= 0.40.

Fusion weights come from the documented rule, not the eye: maximise
rho + conc + wallrho subject to within-object rho_in >= 0.40 (keep a majority of
the parse's inside-object ordering), scored against the ART-ONLY parse because
that is what upload-time fusion sees. For DA3 the rule picks w=0.9 (2.017, just
over raw DA3's 2.014 — the parse share buys interior ordering, rho_in 0.403 ->
0.455, at no layout cost). wflat 0.42 -> 0.23 vs the gt's 0.20: the flatness
complaint is the thing that moved most. DA-v2 (w=0.8) and MiDaS (w=0.6) stay as
fallbacks for installs without the 101 MB DA3 file; MiDaS remains the fast path
(0.3 s vs 4.8 s for DA3 on CPU).

DA3 licence/output notes: DA3-SMALL is Apache-2.0 (the any-view Large/Giant
checkpoints are CC-BY-NC and deliberately NOT used here). Its onnx export is
depth-only, and its raw output is *far = large* — the opposite sign of DA-v2's
disparity-style output — so each model entry carries an explicit orientation and
the row-correlation heuristic stays only as a backstop.

This module is an OPTIONAL accelerator, not engine core: the engine rule stays
"numpy + PIL only" for everything the render/sprite/audit path touches, and this
file imports onnxruntime (or torch + the official depth-anything-3 package)
lazily, inside `fuse()`, only when a runtime AND a model are actually present.
Without them `fuse()` returns None and the server keeps shipping the pure parse
field — same bytes as M6b.

Backends, in preference order:
  1. torch — the official `depth-anything-3` package when importable (the path
     for machines that already run DA3 locally; GPU when present; checkpoint via
     DA3_TORCH_MODEL, default depth-anything/DA3-SMALL, Apache-2.0). No onnx
     install needed at all on those machines.
  2. onnx  — Depth Anything V2 ViT-L first (fp32 `da_v2_large.onnx`, the weights
     the user's ComfyUI node runs; quantised twin at edge 336 for low-RAM
     installs), then DA3 small, then DA-v2 small / MiDaS as fallbacks; fetched
     by app/install_depth_models.py (~1.9 GB with the ViT-L pair).
"""
import contextlib
import os
import sys

import numpy as np
from PIL import Image

VERSION = 'm7-da2l-1.0'

IMNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# name, onnx file, family, input edge, normalisation, blend weight (model share),
# orientation ('auto' = row-correlation heuristic; 'far-large' = flip always).
MODELS = (
    # Depth Anything V2 ViT-L fp32 — the same weights the user's ComfyUI node
    # (DownloadAndLoadDepthAnythingV2Model, depth_anything_v2_vitl_fp32.safetensors)
    # runs; default since entry 20 at user request. w=1.0: the documented rule
    # (max rho+conc+wallrho s.t. rho_in >= 0.40) picks the RAW model map for the
    # large family — blending with the parse only costs (measured, entry 20).
    ('da-v2-large', 'da_v2_large.onnx', 'da', 518, IMNET, 1.0, 'auto'),
    # quantised twin at a reduced edge for low-RAM machines (1 GiB cgroups: the
    # fp32 weights alone are 1.34 GB and even the int8 attention buffer at 518²
    # is 120 MB — edge 256 fits inside the server process and scores rho .840
    # on the courtyard, above the 336-edge run's .832).
    ('da-v2-large-int8', 'da_v2_large_quant.onnx', 'da', 256, IMNET, 1.0, 'auto'),
    # DA3 small, fp32, folded-preprocess export: uint8 NHWC in, normalisation is
    # inside the graph; `edge` is the fixed square side the trace was cut for.
    ('da3-small', 'da3_small.onnx', 'da3', 518, IMNET, 0.9, 'far-large'),
    ('da-v2-small-int8', 'da_v2_small_quant.onnx', 'da', 518, IMNET, 0.8, 'auto'),
    # fp32 twin of the int8 model, same weight: kept as a fallback for installs
    # where only the bigger file survived (workspace snapshots cap ~128 MB and
    # dropped the 27 MB quant file); bake-off rho .794 vs .804 for the int8
    ('da-v2-small', 'da_v2_small.onnx', 'da', 518, IMNET, 0.8, 'auto'),
    ('midas-v21-small', 'midas_v21_small.onnx', 'midas', 256, IMNET, 0.6, 'auto'),
)

# DEPTH_MODEL env spellings -> family
_FAMILIES = {'da3': 'da3', 'da': 'da', 'da2': 'da', 'dav2': 'da', 'da-v2': 'da',
             'midas': 'midas', 'torch': 'torch'}


def _torch_ok():
    """True when the OFFICIAL depth-anything-3 package + torch are importable —
    the path for machines that already run DA3 locally (GPU when present, any
    checkpoint via DA3_TORCH_MODEL). Checked lazily; never a hard dependency."""
    import importlib.util as iu
    return (iu.find_spec('torch') is not None
            and iu.find_spec('depth_anything_3') is not None)


_TORCH = {}


def _torch_session():
    if 'm' not in _TORCH:
        import torch
        from depth_anything_3.api import DepthAnything3
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        m = DepthAnything3.from_pretrained(
            os.environ.get('DA3_TORCH_MODEL', 'depth-anything/DA3-SMALL'))
        _TORCH['m'] = (m.to(device=dev), dev)
    return _TORCH['m']


def _run_torch(nat):
    """Depth via the official package. Its output is distance (far = large), so
    the sign is flipped here to this module's closer = larger convention."""
    try:
        m, _dev = _torch_session()
        pred = m.inference([np.ascontiguousarray(nat, np.uint8)])
        d = np.asarray(pred.depth[0], np.float64)
    except Exception as e:                      # accelerator stays optional
        return None, {'model': 'da3-torch', 'error': f'{type(e).__name__}: {e}'}
    return -d, {'model': 'da3-torch'}


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
    """(model_name, weight, onnx_path|'torch') for the first backend we could
    actually run, else None. Cheap: import/file checks only, no model load.

    Preference: the official depth-anything-3 torch package when DEPTH_MODEL
    asks for it explicitly (machines that run DA3 locally — GPU if present,
    checkpoint via DA3_TORCH_MODEL, default DA3-SMALL), then the onnx files in
    MODELS order (ViT-L first since entry 20, then DA3, DA-v2 small, MiDaS).

    DEPTH_MODEL env picks the family: 'off' disables the accelerator, 'da3' /
    'da' ('da2', 'dav2') / 'midas' / 'torch' force one, unset = MODELS order.
    The preference is only a first try: model_depth() falls through this list
    at RUNTIME too, because a model can be present and still fail to allocate
    (ViT-L in a 1 GiB cgroup, entry 20) — failing to the next model beats
    silently shipping the flat parse.
    """
    c = candidates(models_dir)
    return c[0] if c else None


def candidates(models_dir=None):
    """Ordered (name, weight, path|'torch') run-preference list, best first."""
    want = os.environ.get('DEPTH_MODEL', '').lower().strip()
    if want == 'off':
        return []
    out = []
    if want in ('da3', 'torch') and _torch_ok():
        out.append(('da3-torch', 0.9, 'torch'))
    ort_ok = None                                  # probe lazily, only if needed
    fam = _FAMILIES.get(want, want or None)
    for entry in MODELS:
        name, fn = entry[0], entry[1]
        if fam and entry[2] != fam and not name.startswith(fam):
            continue
        for d in _search_dirs(models_dir):
            p = os.path.join(d, fn)
            if os.path.exists(p) and os.path.getsize(p) > 1e6:
                if ort_ok is None:
                    ort_ok = runtime_ok()
                if not ort_ok:
                    break                          # files yes, runtime no: parse
                out.append((name, entry[5], p))
                break
    return out


def source(models_dir=None):
    a = available(models_dir)
    return f'{a[0]}@w{a[1]:g}' if a else 'parse-only'


_SESS = {}                                  # path -> live InferenceSession (entry 21)
_SESS_FLOOR = 2 * 1024 ** 3                # cache sessions only above this headroom
_ORT_OK = None                             # entry 22: cached runtime probe


def runtime_ok(ort_dir=None):
    """Entry 22: can onnxruntime actually be imported here? available() used to
    promise a model on file presence alone, so a box with the 1.3 GB weights but
    no runtime reported 'da-v2-large@w1' while every fuse died in milliseconds.
    Imported once per process; the import (not find_spec) also catches broken
    DLL/CPU-signal loads."""
    global _ORT_OK
    if _ORT_OK is None:
        import sys
        for d in _ort_dirs(ort_dir):
            if os.path.isdir(d) and d not in sys.path:
                sys.path.insert(0, d)
        try:
            import onnxruntime                     # noqa: F401
            _ORT_OK = True
        except Exception:
            _ORT_OK = False
    return _ORT_OK


def _mem_headroom():
    """Bytes of memory this process may still allocate. cgroup-aware (the
    sandbox caps the whole box at 1 GiB while /proc/meminfo advertises the
    host), falling back to MemAvailable, then to 'plenty'."""
    try:
        with open('/sys/fs/cgroup/memory.max') as f:
            mx = f.read().strip()
        if mx != 'max':
            with open('/sys/fs/cgroup/memory.current') as f:
                return int(mx) - int(f.read().strip())
    except Exception:
        pass
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 1 << 62


def _session(path, ort_dir=None):
    """One InferenceSession per model file per process WHEN THE BOX CAN AFFORD
    to keep it. Re-creating a session for every fuse leaves resident RAM behind
    each cycle (five fuses killed a 1 GiB server, entry 21), but caching every
    session pins ~300 MB per process — fatal when two processes fuse at once
    (test client + server). So: cache above 2 GB headroom, per-call below."""
    s = _SESS.get(path)
    if s is not None:
        return s
    import sys
    for d in _ort_dirs(ort_dir):
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.inter_op_num_threads = 4
    so.intra_op_num_threads = 4
    so.log_severity_level = 3
    sess = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
    if _mem_headroom() > _SESS_FLOOR:
        _SESS[path] = sess
    return sess


def _run(sess, nat, edge, norm):
    """Run one depth model. Both onnx input contracts are auto-detected from the
    session: float32 NCHW ImageNet-normalised (DA-v2, MiDaS), or the 'folded
    preprocess' uint8 NHWC 0..255 with normalisation inside the graph (DA3)."""
    img = Image.fromarray(np.ascontiguousarray(nat, np.uint8)).convert('RGB')
    img = img.resize((edge, edge), Image.BILINEAR)
    inp = sess.get_inputs()[0]
    shp = [d if isinstance(d, int) else edge for d in inp.shape]
    if inp.type == 'tensor(uint8)' or (len(shp) == 4 and shp[-1] == 3):
        x = np.asarray(img, np.uint8)[None]                # NHWC uint8
    else:
        a = np.asarray(img, np.float32) / 255.0
        mean, std = norm
        a = (a - np.array(mean, np.float32)) / np.array(std, np.float32)
        x = np.transpose(a, (2, 0, 1))[None].astype(np.float32)
        if inp.type == 'tensor(float16)':
            x = x.astype(np.float16)
    out = sess.get_outputs()[0]
    y = sess.run([out.name], {inp.name: x})[0]
    d = np.asarray(y[0], np.float32)
    while d.ndim > 2:
        d = d[0]
    return np.asarray(Image.fromarray(d).resize((nat.shape[1], nat.shape[0]),
                                                Image.BILINEAR), np.float64)


def _znorm(a):
    a = np.asarray(a, np.float64)
    return (a - a.mean()) / (a.std() + 1e-9)


def _worker_main():
    """Entry 21: run one model in a throwaway process. argv: onnx, edge, png.
    Writes the raw float32 map to stdout. Small boxes (<= 2 GB headroom) fuse
    out-of-process because each in-process session cycle ratchets resident RAM
    and never fully returns it — a subprocess gives it all back on exit."""
    import sys
    path, edge, png = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    nat = np.asarray(Image.open(png).convert('RGB'))
    sess = _session(path)
    d = _run(sess, nat, edge, IMNET)
    sys.stdout.buffer.write(d.astype(np.float32).tobytes())


def _run_subprocess(path, edge, nat):
    import subprocess
    import tempfile
    app = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tf:
        tmp = tf.name
        Image.fromarray(np.ascontiguousarray(nat, np.uint8)).save(tf)
    try:
        code = ('import sys; sys.path.insert(0, %r); '
                'from engine import depth_ml as m; m._worker_main()' % app)
        r = subprocess.run([sys.executable, '-c', code, path, str(edge), tmp],
                           capture_output=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode()[-300:] or 'worker exit %d'
                               % r.returncode)
        d = np.frombuffer(r.stdout, np.float32).astype(np.float64)
        return d.reshape(nat.shape[0], nat.shape[1])
    finally:
        with contextlib.suppress(Exception):
            os.remove(tmp)


def model_depth(nat, models_dir=None, ort_dir=None):
    """Raw model closeness map (float64, higher = nearer, native resolution) or
    (None, meta). Orientation is made explicit per model (DA3's raw output is
    far=large, so it is always flipped); 'auto' keeps the M6e heuristic: in iso
    art depth grows toward the bottom row, so a map that correlates negatively
    with the row axis is sign-flipped. Separated from fuse() so the layers pass
    can reuse the model map without re-running the blend."""
    a = candidates(models_dir)
    if not a:
        return None, None
    errs = []
    for name, _w, path in a:
        if path == 'torch':
            d, meta = _run_torch(nat)
            if d is not None:
                return d, meta
            errs.append(f'{name}: {meta.get("error")}')
            continue
        orient = 'auto'
        edge, norm = 518, IMNET
        for m in MODELS:
            if m[0] == name:
                edge, norm, orient = m[3], m[4], m[6]
        sess = None
        try:
            if _mem_headroom() > _SESS_FLOOR:
                sess = _session(path, ort_dir)
                d = _run(sess, nat, edge, norm)
                if _SESS.get(path) is not sess:
                    del sess        # not cached (small box): free it now
            else:
                d = _run_subprocess(path, edge, nat)   # RAM returns on exit
        except Exception as e:                  # accelerator stays optional
            errs.append(f'{name}: {type(e).__name__}: {e}')
            if sess is not None and _SESS.get(path) is sess:
                # a session that failed mid-run keeps its arena: evict it so the
                # fallback model gets the RAM (entry 20's starvation case)
                del _SESS[path]
                del sess
                import gc
                gc.collect()
            continue                            # next model in preference
        if orient == 'far-large':
            d = -d
        elif orient == 'auto':
            ys = np.mgrid[0:nat.shape[0], 0:nat.shape[1]][0]
            if np.corrcoef(d.ravel(), ys.ravel())[0, 1] < 0:
                d = -d
        meta = {'model': name}
        if errs:
            meta['fallback_from'] = '; '.join(errs)
        return d, meta
    return None, {'model': a[0][0], 'error': '; '.join(errs)}


def fuse(nat, parse_depth, models_dir=None, ort_dir=None):
    """Blended depth (uint8, higher = nearer) or None when no model is runnable.

    blend = w*znorm(model) + (1-w)*znorm(parse), encoded with a 1-99 percentile
    window so one outlier pixel cannot flatten the contrast (min-max did, in the
    bake-off montages).
    """
    d, meta = model_depth(nat, models_dir, ort_dir)
    if d is None:
        return None, meta
    name = meta['model']
    w = next((m[5] for m in MODELS if m[0] == name), 0.9)   # torch path: DA3's w
    blend = w * _znorm(d) + (1 - w) * _znorm(parse_depth)
    lo, hi = np.percentile(blend, (1, 99))
    u8 = np.clip((blend - lo) / (hi - lo + 1e-9) * 255.0, 0, 255).astype(np.uint8)
    return u8, {'model': name, 'w': w, 'version': VERSION}
