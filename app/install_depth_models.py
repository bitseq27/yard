"""One-click installer for the optional M6e depth accelerator.

A fresh clone has NO depth models and NO onnxruntime (both gitignored, ~140 MB
combined), so the depth pass boots parse-only — which reads flat. This module is
what the editor's "install models" button and `server.py --fetch-depth-models`
call; it downloads

  onnxruntime wheel   ~24 MB  unpacked into app/_ort
  Depth Anything V2 q ~27 MB  app/models/da_v2_small_quant.onnx
  MiDaS v2.1 small    ~66 MB  app/models/midas_v21_small.onnx

with progress in a module-level PROGRESS dict that GET /api/depth_ml/status
surfaces. Stdlib only; the engine never imports this (rule 11 keeps engine core
numpy+PIL — this is installer tooling, like app/tests/dl_models.py).
"""
import json
import os
import threading
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))          # app/
ORT_DIR = os.path.join(HERE, '_ort')
MODEL_DIR = os.path.join(HERE, 'models')

LOCK = threading.Lock()
PROGRESS = {'running': False, 'stage': '', 'got': 0, 'total': 0,
            'done': False, 'error': None, 'log': []}

MODELS = (
    ('da_v2_small_quant.onnx',
     'https://huggingface.co/onnx-community/depth-anything-v2-small/'
     'resolve/main/onnx/model_quantized.onnx'),
    ('midas_v21_small.onnx',
     'https://huggingface.co/Heliosoph/midas-small-onnx/resolve/main/'
     'midas_v21_small_256.onnx'),
)
TOTAL_MB = 24 + 27 + 66


def _note(msg):
    PROGRESS['log'] = (PROGRESS['log'] + [msg])[-8:]


def _fetch(url, dest):
    req = urllib.request.Request(url, headers={'User-Agent': 'pixel-sway-installer'})
    with urllib.request.urlopen(req, timeout=300) as r:
        total = int(r.headers.get('Content-Length') or 0)
        PROGRESS['total'] = total
        tmp = dest + '.part'
        got = 0
        with open(tmp, 'wb') as fh:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                got += len(chunk)
                PROGRESS['got'] = got
        os.replace(tmp, dest)
    return os.path.getsize(dest)


def _ort_wheel():
    if os.path.isdir(ORT_DIR) and os.listdir(ORT_DIR):
        _note('onnxruntime already unpacked')
        return
    with urllib.request.urlopen('https://pypi.org/pypi/onnxruntime/json',
                                timeout=60) as r:
        meta = json.load(r)
    url = None
    for f in meta['urls']:
        if f['packagetype'] == 'bdist_wheel' and 'cp311' in f['filename'] \
                and 'manylinux' in f['filename'] and 'x86_64' in f['filename']:
            url = f['url']
    if not url:
        raise RuntimeError('no cp311 manylinux x86_64 onnxruntime wheel on pypi')
    PROGRESS['stage'] = 'onnxruntime wheel'
    PROGRESS['got'] = 0
    whl = os.path.join(HERE, 'projects', '_ort.whl')
    os.makedirs(os.path.dirname(whl), exist_ok=True)
    _fetch(url, whl)
    os.makedirs(ORT_DIR, exist_ok=True)
    with zipfile.ZipFile(whl) as z:
        z.extractall(ORT_DIR)
    os.remove(whl)
    _note(f'onnxruntime {meta["info"]["version"]} unpacked')


def install():
    """Run the whole install synchronously, updating PROGRESS. Returns True on
    success. Callers that want it backgrounded wrap this in a thread."""
    if not LOCK.acquire(blocking=False):
        return False
    try:
        PROGRESS.update(running=True, done=False, error=None, got=0, total=0, log=[])
        os.makedirs(MODEL_DIR, exist_ok=True)
        _ort_wheel()
        for fn, url in MODELS:
            dest = os.path.join(MODEL_DIR, fn)
            if os.path.exists(dest) and os.path.getsize(dest) > 1e6:
                _note(f'{fn} already present')
                continue
            PROGRESS['stage'] = fn
            PROGRESS['got'] = 0
            n = _fetch(url, dest)
            _note(f'{fn} {n / 1e6:.0f} MB')
        PROGRESS['done'] = True
        PROGRESS['stage'] = 'done'
        return True
    except Exception as e:
        PROGRESS['error'] = f'{type(e).__name__}: {e}'
        _note(PROGRESS['error'])
        return False
    finally:
        PROGRESS['running'] = False
        LOCK.release()


def install_async():
    """True if a new install thread started, False if one is already running."""
    if PROGRESS['running']:
        return False
    t = threading.Thread(target=install, daemon=True)
    t.start()
    return True


def status():
    p = dict(PROGRESS)
    p['pct'] = round(100 * p['got'] / p['total'], 1) if p['total'] else 0.0
    p['total_mb'] = TOTAL_MB
    return p
