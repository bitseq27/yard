"""Fetch an ONNX runtime + monocular depth models for the depth bake-off (M6e).

Kept OUT of the engine (plan rule: numpy+PIL only) — this is evaluation tooling.
Downloads into app/tests/_ort/ (unzipped wheel) and app/tests/models/; both are
gitignored. Re-run after a sandbox reset.
"""
import io
import json
import os
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ORT = os.path.join(HERE, '_ort')
MODELS = os.path.join(HERE, 'models')


def get(url, dest, tries=2):
    for i in range(tries):
        try:
            print('  GET', url, flush=True)
            with urllib.request.urlopen(url, timeout=180) as r, open(dest, 'wb') as fh:
                fh.write(r.read())
            print('  -> %s (%.1f MB)' % (dest, os.path.getsize(dest) / 1e6), flush=True)
            return True
        except Exception as e:
            print('  retry %d: %s' % (i, e), flush=True)
    return False


def ort_wheel():
    if os.path.isdir(ORT) and os.listdir(ORT):
        print('onnxruntime already unpacked'); return True
    with urllib.request.urlopen('https://pypi.org/pypi/onnxruntime/json', timeout=60) as r:
        meta = json.load(r)
    url = None
    for f in meta['urls']:
        if f['packagetype'] == 'bdist_wheel' and 'cp311' in f['filename'] \
                and 'manylinux' in f['filename'] and 'x86_64' in f['filename']:
            url = f['url']
    if not url:
        print('no cp311 manylinux x86_64 wheel found'); return False
    whl = '/tmp/ort.whl'
    if not get(url, whl):
        return False
    os.makedirs(ORT, exist_ok=True)
    with zipfile.ZipFile(whl) as z:
        z.extractall(ORT)
    print('unpacked onnxruntime', meta['info']['version'])
    return True


def models():
    os.makedirs(MODELS, exist_ok=True)
    want = [
        # the isl-org GitHub release assets 404; the HF mirror works (M6e, entry 16)
        ('midas_v21_small.onnx',
         'https://huggingface.co/Heliosoph/midas-small-onnx/resolve/main/'
         'midas_v21_small_256.onnx'),
        ('da_v2_small_quant.onnx',
         'https://huggingface.co/onnx-community/depth-anything-v2-small/'
         'resolve/main/onnx/model_quantized.onnx'),
        ('da_v2_small.onnx',
         'https://huggingface.co/onnx-community/depth-anything-v2-small/'
         'resolve/main/onnx/model.onnx'),
    ]
    ok = []
    for name, url in want:
        dest = os.path.join(MODELS, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 1e6:
            ok.append(name); continue
        if get(url, dest):
            ok.append(name)
    print('models available:', ok)
    return ok


if __name__ == '__main__':
    a = ort_wheel()
    b = models()
    print('DONE ort=%s models=%s' % (a, b))
    sys.exit(0 if b else 1)
