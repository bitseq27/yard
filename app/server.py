#!/usr/bin/env python3
"""Pixel Sway — M3 server: masking API + animation render + exports.

Thin JSON API over `app/engine` (plan §1). All heavy work is server-side numpy; masks
travel as row-run RLE (tiny JSON) or PNG. Stdlib ThreadingHTTPServer only — no framework,
so a sandbox reset never breaks the app beyond `apt-get install python3-numpy python3-pil
gifsicle` (see build_log.md Setup).

Routes
  GET  /                                  editor (single file, no CDN deps)
  GET  /api/ping                          {ok, milestone}
  POST /api/upload                        raw image bytes -> project meta
  GET  /api/image/<pid>/{native,original,depth,orient,normal}.png
                                  (the last three are derived; a project that
                                   predates them rebuilds the file on first GET)
  GET  /api/project/<pid>                 project doc (objects/regions/N/fps/palette)
  POST /api/project/<pid>                 save the doc (names, colours, z, visibility)
  GET  /api/mask/<pid>/<mid>.png          one mask, grayscale PNG
  POST /api/masks                         {pid, ops:[upsert|delete]} -> persist masks
  POST /api/wand                          {pid,x,y,tol,conn,snap,smooth} -> {rle,px,bbox}
  POST /api/autoseg                       {pid,max_suggestions} -> editable suggestions
  POST /api/edge                          {pid,rle,op} -> grown/shrunk/outlined/filled rle
  POST /api/render                        {pid,N,fps} -> frame-strip preview + audit
  POST /api/gif                           {pid,N,fps,scale,colors} -> optimized GIF
  POST /api/bundle                        {pid,N,fps} -> game bundle zip (plan §8)
  GET  /api/out/<pid>/{strip.png,preview.gif,bundle.zip}
"""
import hashlib
import io
import json
import os
import shutil
import contextlib
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
from PIL import Image

try:
    import install_depth_models as dim
except ImportError:                       # imported as a module, not run from app/
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import install_depth_models as dim
from engine import depth_ml, imaging as im
from engine import segment, animate, export, depth

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.join(ROOT, 'projects')
MILESTONE = 'M6'


def _engine_version():
    """Content hash of the engine modules: any code change invalidates render caches
    (M5 — stale caches must never serve frames from an older engine)."""
    h = hashlib.md5()
    base = os.path.dirname(os.path.abspath(__file__))
    for fn in ('engine/animate.py', 'engine/export.py', 'engine/imaging.py',
               'engine/segment.py', 'engine/recognise.py', 'engine/depth.py',
               'server.py'):
        try:
            with open(os.path.join(base, fn), 'rb') as f:
                h.update(f.read())
        except OSError:
            h.update(fn.encode())
    return h.hexdigest()[:12]


ENGINE_V = _engine_version()

# M5 concurrency guard: heavy endpoints take (1) their project's lock — concurrent
# requests for the same pid serialise and the second one hits the render cache — and
# (2) a slot in a small global semaphore, so a handful of browsers can't pin every
# core. When no slot frees up in time the request answers 429 instead of queueing
# forever behind the UI.
_BUSY_TIMEOUT = 90.0
_WORK_SEM = threading.BoundedSemaphore(2)
_PID_LOCKS = {}
_PID_LOCKS_GUARD = threading.Lock()


class BusyError(Exception):
    pass


def _pid_lock(pid):
    with _PID_LOCKS_GUARD:
        lk = _PID_LOCKS.get(pid)
        if lk is None:
            lk = _PID_LOCKS[pid] = threading.Lock()
        return lk


@contextlib.contextmanager
def _work_slot(pid, sem=True):
    lk = _pid_lock(pid)
    if not lk.acquire(timeout=_BUSY_TIMEOUT):
        raise BusyError('this project is busy with another render/export — retry shortly')
    try:
        if sem and not _WORK_SEM.acquire(timeout=_BUSY_TIMEOUT):
            raise BusyError('server busy: too many concurrent renders — retry shortly')
        try:
            yield
        finally:
            if sem:
                _WORK_SEM.release()
    finally:
        lk.release()
os.makedirs(PROJ, exist_ok=True)
_LOCK = threading.Lock()
_CACHE = {}                                    # pid -> (key, render result)
_FIELD = {}                                    # pid -> (key, depth field) M6b
_OUT_FILES = ('strip.png', 'preview.gif', 'bundle.zip')


# ------------------------------------------------------------------ project files


def _pdir(pid):
    return os.path.join(PROJ, pid)


def _load_doc(pid):
    fp = os.path.join(_pdir(pid), 'project.json')
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        return json.load(f)


def _save_doc(pid, doc):
    with open(os.path.join(_pdir(pid), 'project.json'), 'w') as f:
        json.dump(doc, f, indent=1)


def _native(pid):
    a = np.asarray(Image.open(os.path.join(_pdir(pid), 'native.png')))
    return a


def _mask_path(pid, mid):
    return os.path.join(_pdir(pid), 'masks', f'{mid}.png')


def _new_mask_entry(mid, name, kind, color, shape, rle=None):
    st = im.stats_of(im.rle_decode(rle, shape)) if rle else {'px': 0, 'bbox': None, 'depth': 0.0}
    return {'id': mid, 'name': name, 'kind': kind, 'color': list(color), 'z': 0,
            'visible': True, 'px': st['px'], 'bbox': st['bbox'], 'depth': st['depth'],
            'nfrag': 1, 'mask': f'masks/{mid}.png', 'offset': [0, 0], 'clips': [],
            'label': name, 'conf': 0.0, 'anim': '', 'params': {}, 'shadow': False}


def _safe_pid(pid):
    """Project ids are our own hex slugs; anything else never reaches a path."""
    return isinstance(pid, str) and len(pid) == 8 and all(c in '0123456789abcdef' for c in pid)


# ------------------------------------------------------------------ M3 render glue


def _mask_array(pid, mid, shape):
    fp = _mask_path(pid, mid)
    if not os.path.exists(fp):
        return np.zeros(shape, bool)
    a = np.asarray(Image.open(fp).convert('L'))
    return a > 127


def _load_objects(pid, doc, with_regions=False):
    """Doc objects in z order with their mask arrays attached (engine input)."""
    shape = (doc['native_h'], doc['native_w'])
    out = []
    for o in doc.get('objects', []):
        e = dict(o)
        e['mask'] = _mask_array(pid, o['id'], shape)
        if e.get('kind') == 'region' and not with_regions:
            continue
        out.append(e)
    out.sort(key=lambda o: o.get('z', 0))
    return out


def _render_key(pid, doc, N, fps):
    """Cache key: anything that can change a render invalidates it."""
    h = hashlib.md5()
    for o in doc.get('objects', []):
        h.update(json.dumps([o.get('id'), o.get('kind'), o.get('visible'), o.get('z'),
                             o.get('shadow', False),
                             animate.coerce_anim(o.get('anim')),
                             animate.clean_params(o.get('anim'), o.get('params'))],
                            sort_keys=True).encode())
        fp = _mask_path(pid, o['id'])
        h.update(str(int(os.path.getmtime(fp) * 1000) if os.path.exists(fp) else 0).encode())
    h.update(f'{N}|{fps}|{ENGINE_V}'.encode())
    return h.hexdigest()


_ML_PENDING = set()          # pids whose background depth-ml fuse is still running


def _bg_fuse(pid, nat, parse):
    try:
        u8, _mm = depth_ml.fuse(nat, parse)
        if u8 is not None:
            Image.fromarray(u8).save(os.path.join(_pdir(pid), 'depth_ml.png'))
    except Exception as e:
        print(f'depth_ml fuse failed for {pid}: {type(e).__name__}: {e}')
    finally:
        _ML_PENDING.discard(pid)


def _write_art_views(pid, nat=None, only=None):
    """The art-only derived views: depth.png (M6a), orient.png (M6b), normal.png (M6c).

    Written at upload, and rebuilt on demand when a project predates them — the four
    tracked M1 examples and anything uploaded before M6 — so a tab never shows a
    placeholder for a view the engine can derive from native.png alone.
    """
    d = _pdir(pid)
    if nat is None:
        nat = _native(pid)
    f = {}

    def fld():
        if 'f' not in f:
            f['f'] = depth.field(nat)
        return f['f']

    writers = (('depth.png', lambda: fld()['depth']),
               ('orient.png', lambda: depth.render_orient(fld())),
               ('normal.png', lambda: depth.normal_map(nat)))
    for name, fn in writers:
        if only and name != only:
            continue
        Image.fromarray(fn()).save(os.path.join(d, name))
    return fld()


def _field_key(pid, doc):
    """Anything that can change the depth field invalidates it: mask pixels, the
    recognised labels and the surfaces the user set in the inspector (M6b)."""
    h = hashlib.md5()
    for o in doc.get('objects', []):
        h.update(json.dumps([o.get('id'), o.get('kind'), o.get('label'),
                             o.get('surface', '')], sort_keys=True).encode())
        fp = _mask_path(pid, o['id'])
        h.update(str(int(os.path.getmtime(fp) * 1000) if os.path.exists(fp) else 0).encode())
    h.update(ENGINE_V.encode())
    return h.hexdigest()


def _field(pid, doc):
    """Depth/orientation field including the project's masks (wand guard, split).
    The art-only field is written at upload; this one adds OBJECT rows and the
    surfaces/labels the user corrected, cached like the render cache (M5)."""
    key = _field_key(pid, doc)
    hit = _FIELD.get(pid)
    if hit and hit[0] == key:
        return hit[1]
    f = depth.field(_native(pid), objects=_load_objects(pid, doc, with_regions=True))
    _FIELD[pid] = (key, f)
    if len(_FIELD) > 2:
        for k in list(_FIELD):
            if k != pid:
                del _FIELD[k]
    return f


def _get_render(pid, doc, N, fps):
    """Returns (result, cached). Guarded by the caller's _work_slot(pid)."""
    key = _render_key(pid, doc, N, fps)
    hit = _CACHE.get(pid)
    if hit and hit[0] == key:
        return hit[1], True
    nat = _native(pid)
    objs = _load_objects(pid, doc)
    res = animate.render(nat, objs, N=N, fps=fps)
    res['_key'] = key
    _CACHE[pid] = (key, res)
    if len(_CACHE) > 2:                        # keep memory sane: evict others
        for k in list(_CACHE):
            if k != pid:
                del _CACHE[k]
    return res, False


def _out_dir(pid):
    d = os.path.join(_pdir(pid), 'out')
    os.makedirs(d, exist_ok=True)
    return d


def _clean_cfg(raw):
    """Whitelist + type-coerce auto-seg settings from the UI against engine DEFAULTS.

    The settings panel posts arbitrary JSON; only known keys with sane types reach the
    engine, so a bad value can neither crash the server nor smuggle a huge workload in.
    """
    out = {}
    for k, v in (raw or {}).items():
        if k not in segment.DEFAULTS:
            continue
        dv = segment.DEFAULTS[k]
        try:
            if isinstance(dv, bool):
                out[k] = bool(v)
            elif isinstance(dv, int):
                out[k] = int(v)
            elif isinstance(dv, float):
                out[k] = float(v)
            elif isinstance(dv, dict) and isinstance(v, dict):
                out[k] = {kk: float(vv) for kk, vv in v.items()}
        except (TypeError, ValueError):
            continue
    return out


# ------------------------------------------------------------------ handler


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    # -- plumbing -----------------------------------------------------------
    def _send(self, code, body, ctype='application/json'):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path, ctype):
        if not os.path.exists(path):
            self._send(404, {'error': 'not found'})
            return
        with open(path, 'rb') as f:
            self._send(200, f.read(), ctype)

    def _json_body(self):
        raw = getattr(self, '_raw', b'')
        return json.loads(raw.decode('utf-8')) if raw else {}

    def _err(self, e, code=400):
        self._send(code, {'error': f'{type(e).__name__}: {e}'})

    # -- GET ----------------------------------------------------------------
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == '/api/ping':
            self._send(200, {'ok': True, 'milestone': MILESTONE, 'engine': ENGINE_V,
                              'depth_ml': depth_ml.source()})
            return
        if u.path == '/api/depth_ml/status':
            self._send(200, {'source': depth_ml.source(), 'progress': dim.status()})
            return
        if u.path in ('/', '/index.html'):
            with open(os.path.join(ROOT, 'static', 'index.html'), 'rb') as f:
                self._send(200, f.read(), 'text/html; charset=utf-8')
            return
        if u.path in ('/docs', '/docs/'):
            with open(os.path.join(ROOT, 'static', 'docs.html'), 'rb') as f:
                self._send(200, f.read(), 'text/html; charset=utf-8')
            return
        parts = [p for p in u.path.split('/') if p]
        if len(parts) == 4 and parts[0] == 'api' and parts[1] == 'image':
            pid, which = parts[2], parts[3]
            if which not in ('native.png', 'original.png', 'depth.png', 'orient.png',
                              'normal.png', 'depth_ml.png'):
                self._send(404, {'error': 'not found'})
                return
            fp = os.path.join(_pdir(pid), which)
            if not os.path.exists(fp) and which in ('depth.png', 'orient.png', 'normal.png'):
                with contextlib.suppress(Exception):
                    _write_art_views(pid, only=which)     # pre-M6 project: derive it now
            if not os.path.exists(fp) and which == 'depth_ml.png':
                if pid in _ML_PENDING:               # still fusing: don't double-run
                    self._send(404, {'error': 'fusing'})
                    return
                with contextlib.suppress(Exception):
                    nat = _native(pid)
                    u8, _mm = depth_ml.fuse(nat, depth.field(nat)['depth'])
                    if u8 is not None:
                        Image.fromarray(u8).save(fp)
            self._file(fp, 'image/png')
            return
        if len(parts) == 4 and parts[0] == 'api' and parts[1] == 'mask':
            self._file(_mask_path(parts[2], parts[3].replace('.png', '')), 'image/png')
            return
        if len(parts) == 4 and parts[0] == 'api' and parts[1] == 'out':
            if parts[3] not in _OUT_FILES:
                self._send(404, {'error': 'not found'})
                return
            ctype = 'image/png' if parts[3].endswith('.png') else (
                'image/gif' if parts[3].endswith('.gif') else 'application/zip')
            self._file(os.path.join(_pdir(parts[2]), 'out', parts[3]), ctype)
            return
        if len(parts) == 3 and parts[0] == 'api' and parts[1] == 'project':
            doc = _load_doc(parts[2])
            if doc is None:
                self._send(404, {'error': 'no such project'})
                return
            self._send(200, doc)
            return
        if len(parts) == 2 and parts[0] == 'api' and parts[1] == 'projects':
            out = []
            if os.path.isdir(PROJ):
                for pid in sorted(os.listdir(PROJ)):
                    d = _load_doc(pid)
                    if d:
                        out.append({'id': pid, 'name': d.get('name', ''),
                                    'w': d.get('w'), 'h': d.get('h'),
                                    'objects': len(d.get('objects', [])),
                                    'mtime': int(os.path.getmtime(_pdir(pid)))})
            out.sort(key=lambda p: p['mtime'])          # recents = recent, not alphabetical
            self._send(200, out)
            return
        self._send(404, {'error': 'not found'})

    # -- POST ---------------------------------------------------------------
    def do_POST(self):
        u = urlparse(self.path)
        if u.path == '/api/depth_ml/install':
            # M6e one-click accelerator: download onnxruntime + the two onnx models
            if os.environ.get('DEPTH_MODEL', '').lower() == 'off':
                self._send(200, {'ok': False, 'disabled': True})
            elif depth_ml.available():
                self._send(200, {'ok': True, 'already': True,
                                  'source': depth_ml.source()})
            elif dim.install_async():
                self._send(200, {'ok': True, 'started': True})
            else:
                self._send(200, {'ok': True, 'busy': True})
            return
        # always drain the body first: an unread body desyncs the next request on a
        # keep-alive connection (M4 ledger: 404-on-unknown-route left bytes behind and
        # a later response parsed as HTML)
        n = int(self.headers.get('Content-Length', 0))
        self._raw = self.rfile.read(n) if n else b''
        try:
            if u.path == '/api/upload':
                self._upload()
            elif u.path.startswith('/api/project/'):
                rest = u.path[len('/api/project/'):]
                if rest.endswith('/meta'):
                    self._project_meta(rest[:-len('/meta')])
                elif rest.endswith('/duplicate'):
                    self._project_duplicate(rest[:-len('/duplicate')])
                elif rest.endswith('/delete'):
                    self._project_delete(rest[:-len('/delete')])
                else:
                    self._save_project(rest)
            elif u.path == '/api/masks':
                self._masks()
            elif u.path == '/api/wand':
                self._wand()
            elif u.path == '/api/autoseg':
                self._autoseg()
            elif u.path == '/api/edge':
                self._edge()
            elif u.path == '/api/render':
                self._render()
            elif u.path == '/api/gif':
                self._gif()
            elif u.path == '/api/bundle':
                self._bundle()
            else:
                self._send(404, {'error': 'not found'})
        except Exception as e:            # noqa: BLE001 — report, never hang the editor
            self._err(e)

    def _upload(self):
        raw = self._raw
        t0 = time.time()
        try:
            img = Image.open(io.BytesIO(raw)).convert('RGB')
        except Exception as e:
            self._send(400, {'error': f'bad image: {e}'})
            return
        a = np.asarray(img)
        p = im.detect_grid(a)
        nat = im.native_from(a, p)
        pid = uuid.uuid4().hex[:8]
        d = _pdir(pid)
        os.makedirs(os.path.join(d, 'masks'), exist_ok=True)
        Image.fromarray(a).save(os.path.join(d, 'original.png'))
        Image.fromarray(nat).save(os.path.join(d, 'native.png'))
        # M6a/M6c: the derived views depend on the art alone, so they are computed
        # once at upload and served for the editor tabs + M6b surface tools.
        fld = _write_art_views(pid, nat)
        # M6e: the optional model-fused depth runs in the BACKGROUND — the M2
        # upload budget (<2 s) protects the parse pipeline, and the depth tab
        # lazy-fetches depth_ml.png when the user first opens it. Without the
        # accelerator runtime/model this is a no-op and the bytes stay M6b's.
        if depth_ml.available():
            _ML_PENDING.add(pid)
            threading.Thread(target=_bg_fuse, args=(pid, nat, fld['depth']),
                             daemon=True).start()
        doc = {'id': pid, 'w': img.width, 'h': img.height, 'grid': p,
               'native_w': int(nat.shape[1]), 'native_h': int(nat.shape[0]),
               'process_ms': round((time.time() - t0) * 1000),
               'objects': [], 'regions': {}, 'N': 48, 'fps': 12, 'palette': None}
        _save_doc(pid, doc)
        self._send(200, doc)

    def _save_project(self, pid):
        if not _safe_pid(pid):
            self._send(404, {'error': 'no such project'})
            return
        doc = _load_doc(pid)
        if doc is None:
            self._send(404, {'error': 'no such project'})
            return
        body = self._json_body()
        with _LOCK:
            for k in ('objects', 'regions', 'N', 'fps', 'palette'):
                if k in body:
                    doc[k] = body[k]
            _save_doc(pid, doc)
        self._send(200, doc)

    # -- M4: project management ------------------------------------------------
    def _project_meta(self, pid):
        if not _safe_pid(pid):
            self._send(404, {'error': 'no such project'})
            return
        doc = _load_doc(pid)
        if doc is None:
            self._send(404, {'error': 'no such project'})
            return
        body = self._json_body()
        name = str(body.get('name', '')).strip()[:60]
        if name:
            doc['name'] = name
            _save_doc(pid, doc)
        self._send(200, doc)

    def _project_duplicate(self, pid):
        if not _safe_pid(pid):
            self._send(404, {'error': 'no such project'})
            return
        doc = _load_doc(pid)
        if doc is None:
            self._send(404, {'error': 'no such project'})
            return
        with _LOCK:
            new = uuid.uuid4().hex[:8]
            shutil.copytree(_pdir(pid), _pdir(new))
            doc2 = _load_doc(new)
            doc2['id'] = new
            doc2['name'] = ((doc.get('name') or pid) + ' copy')[:60]
            _save_doc(new, doc2)
        self._send(200, doc2)

    def _project_delete(self, pid):
        if not _safe_pid(pid):
            self._send(404, {'error': 'no such project'})
            return
        with _LOCK:
            _CACHE.pop(pid, None)
            d = _pdir(pid)
            if not os.path.isdir(d):
                self._send(404, {'error': 'no such project'})
                return
            shutil.rmtree(d)
        _FIELD.pop(pid, None)
        self._send(200, {'ok': True})

    def _masks(self):
        body = self._json_body()
        pid = body.get('pid')
        doc = _load_doc(pid)
        if doc is None:
            self._send(404, {'error': 'no such project'})
            return
        shape = (doc['native_h'], doc['native_w'])
        mdir = os.path.join(_pdir(pid), 'masks')
        os.makedirs(mdir, exist_ok=True)
        with _LOCK:
            for op in body.get('ops', []):
                if op.get('op') == 'delete':
                    doc['objects'] = [o for o in doc['objects'] if o['id'] != op['id']]
                    fp = _mask_path(pid, op['id'])
                    if os.path.exists(fp):
                        os.remove(fp)
                elif op.get('op') == 'upsert':
                    mid = op.get('id') or uuid.uuid4().hex[:8]
                    m = im.rle_decode(op.get('rle', []), shape)
                    Image.fromarray(np.where(m, 255, 0).astype(np.uint8), 'L').save(
                        _mask_path(pid, mid), optimize=True)
                    entry = _new_mask_entry(mid, op.get('name', f'mask {mid}'),
                                            op.get('kind', 'object'),
                                            op.get('color', [200, 200, 200]), shape,
                                            op.get('rle'))
                    for k in ('z', 'visible', 'nfrag', 'depth', 'label', 'conf'):
                        if k in op:
                            entry[k] = op[k]
                    if op.get('surface') in ('', 'wall', 'ground', None):
                        entry['surface'] = op.get('surface') or ''
                    if 'shadow' in op:
                        entry['shadow'] = bool(op['shadow'])
                    if 'anim' in op or 'params' in op:
                        entry['anim'] = animate.coerce_anim(op.get('anim'))
                        entry['params'] = animate.clean_params(entry['anim'],
                                                               op.get('params'))
                    if 'anim_hint' in op:
                        entry['anim_hint'] = str(op.get('anim_hint') or '')[:80]
                    for i, o in enumerate(doc['objects']):
                        if o['id'] == mid:
                            entry['z'] = op.get('z', o.get('z', i))
                            entry['visible'] = op.get('visible', o.get('visible', True))
                            doc['objects'][i] = entry
                            break
                    else:
                        entry['z'] = op.get('z', len(doc['objects']))
                        doc['objects'].append(entry)
                else:
                    raise ValueError(f'unknown op {op.get("op")}')
            doc['objects'].sort(key=lambda o: o.get('z', 0))
            for i, o in enumerate(doc['objects']):
                o['z'] = i
            _save_doc(pid, doc)
        self._send(200, {'ok': True, 'objects': doc['objects']})

    def _wand(self):
        b = self._json_body()
        pid = b['pid']
        nat = _native(pid)
        dguard = min(max(0, int(b.get('dguard', 0) or 0)), 64)
        base_y = None
        if dguard:                       # M6b depth guard: stop at ground-row jumps
            base_y = _field(pid, _load_doc(pid))['base_y']
        m = im.wand(nat, int(b['y']), int(b['x']), tol=int(b.get('tol', 30)),
                    conn=int(b.get('conn', 8)), snap=int(b.get('snap', 0)),
                    smooth=bool(b.get('smooth', True)), mode=b.get('mode', 'hue'),
                    base_y=base_y, dguard=dguard)
        st = im.stats_of(m)
        self._send(200, {'ok': True, 'rle': im.rle_encode(m), 'px': st['px'],
                         'bbox': st['bbox'], 'mode': b.get('mode', 'hue')})

    def _autoseg(self):
        b = self._json_body()
        nat = _native(b['pid'])
        cfg = _clean_cfg(b.get('cfg'))
        cap = min(max(1, int(b.get('max_suggestions', 60))), 200)
        conn = 4 if int(b.get('conn', 8)) == 4 else 8
        try:
            with _work_slot(b['pid'], sem=False):   # one segmentation per project
                res = segment.auto_segment(nat, max_suggestions=cap, cfg=cfg, conn=conn)
        except BusyError as e:
            self._send(429, {'error': str(e), 'retry': True})
            return
        out = []
        for s in res['suggestions']:
            out.append({'id': s['id'], 'name': s['name'], 'kind': s['kind'],
                        'cls': s.get('cls', s['name']), 'label': s.get('label', s['name']),
                        'conf': s.get('conf', 0.0), 'anim': s.get('anim', ''),
                        'color': s['color'], 'rle': im.rle_encode(s['mask']),
                        'px': s['px'], 'bbox': s['bbox'], 'depth': s['depth'],
                        'nfrag': s.get('nfrag', 1)})
        self._send(200, {'ok': True, 'suggestions': out, 'classes': res['classes'],
                         'n_objects': res['n_objects'], 'n_regions': res['n_regions'],
                         'dropped': res['dropped'], 'ms': res['ms'], 'cfg': res['cfg']})

    def _edge(self):
        b = self._json_body()
        doc = _load_doc(b['pid'])
        shape = (doc['native_h'], doc['native_w'])
        m = im.rle_decode(b.get('rle', []), shape)
        op = b.get('op', 'grow')
        n = min(max(1, int(b.get('n', 1))), 8)
        if op == 'grow':
            m = im.dilate4(m, n)
        elif op == 'shrink':
            m = im.erode4(m, n)
        elif op == 'outline':                      # keep only the 1px rim
            m = m & ~im.erode4(m.copy(), 1)
        elif op == 'solid':                        # the inverse of outline: a filled
            m = im.holefill(im.close4(m.copy(), n))  # silhouette (bridges <=2n px gaps)
        elif op == 'fill':
            m = im.holefill(m)
        elif op == 'close':
            m = im.close4(m, n)
        elif op == 'depthsplit':
            # M6b: cut the mask at ground-row jumps (plan §8's combine-or-split cut
            # line, proposed automatically) and hand the pieces back
            thresh = min(max(1, int(b.get('thresh', 3))), 16)
            min_px = min(max(4, int(b.get('min_px', 25))), 1000)
            fld = _field(b['pid'], doc)
            pieces = depth.split_pieces(m, fld['base_y'], thresh, min_px,
                                        orient=fld['orient'])
            out = []
            for pm in pieces:
                st = im.stats_of(pm)
                out.append({'rle': im.rle_encode(pm), 'px': st['px'], 'bbox': st['bbox']})
            self._send(200, {'ok': True, 'pieces': out, 'n': len(out)})
            return
        else:
            raise ValueError(f'unknown edge op {op}')
        st = im.stats_of(m)
        self._send(200, {'ok': True, 'rle': im.rle_encode(m), 'px': st['px'],
                         'bbox': st['bbox']})

    # -- M3: render + exports --------------------------------------------------
    def _render_args(self, b, doc):
        N = int(b.get('N', doc.get('N', 48)) or 48)
        fps = int(b.get('fps', doc.get('fps', 12)) or 12)
        return min(96, max(4, N)), min(30, max(2, fps))

    def _render(self):
        b = self._json_body()
        doc = _load_doc(b.get('pid'))
        if doc is None:
            self._send(404, {'error': 'no such project'})
            return
        N, fps = self._render_args(b, doc)
        t0 = time.time()
        try:
            with _work_slot(b['pid']):
                res, cached = _get_render(b['pid'], doc, N, fps)
                png, cols, fw, fh = export.strip_png(res['frames'])
                with open(os.path.join(_out_dir(b['pid']), 'strip.png'), 'wb') as f:
                    f.write(png)
        except BusyError as e:
            self._send(429, {'error': str(e), 'retry': True})
            return
        animated = sum(1 for p in res['plans'] if p.anim != 'static')
        self._send(200, {'ok': True, 'N': N, 'fps': fps, 'cols': cols, 'fw': fw, 'fh': fh,
                         'strip': f'/api/out/{b["pid"]}/strip.png',
                         'audit': res['audit'], 'ms': res['ms'], 'cached': cached,
                         'total_ms': round((time.time() - t0) * 1000),
                         'objects': len(res['objects']), 'animated': animated})

    def _gif(self):
        b = self._json_body()
        doc = _load_doc(b.get('pid'))
        if doc is None:
            self._send(404, {'error': 'no such project'})
            return
        N, fps = self._render_args(b, doc)
        scale = min(4, max(1, int(b.get('scale', 2) or 2)))
        colors = min(256, max(8, int(b.get('colors', 128) or 128)))
        t0 = time.time()
        try:
            with _work_slot(b['pid']):
                res, _cached = _get_render(b['pid'], doc, N, fps)
                pal = export.shared_palette(res['frames'], colors)
                gif, info = export.gif_bytes(res['frames'], fps=fps, scale=scale,
                                             colors=colors, pal=pal)
                with open(os.path.join(_out_dir(b['pid']), 'preview.gif'), 'wb') as f:
                    f.write(gif)
        except BusyError as e:
            self._send(429, {'error': str(e), 'retry': True})
            return
        info.update({'ok': True, 'url': f'/api/out/{b["pid"]}/preview.gif',
                     'audit': res['audit'], 'total_ms': round((time.time() - t0) * 1000)})
        if info['bytes'] > 8 * 1024 * 1024:              # plan §7 risk: warn, don't block
            info['warn'] = ('GIF over 8 MB — try fewer frames (N), scale 1, '
                            'or a smaller palette')
        self._send(200, info)

    def _bundle(self):
        b = self._json_body()
        doc = _load_doc(b.get('pid'))
        if doc is None:
            self._send(404, {'error': 'no such project'})
            return
        N, fps = self._render_args(b, doc)
        colors = min(256, max(8, int(b.get('colors', 128) or 128)))
        atlas = bool(int(b.get('atlas', 1)))
        t0 = time.time()
        try:
            with _work_slot(b['pid']):
                res, _cached = _get_render(b['pid'], doc, N, fps)
                nat = _native(b['pid'])
                doc2 = json.loads(json.dumps(doc))
                for o in doc2.get('objects', []):            # regions feed the walkmask
                    if o.get('kind') == 'region':
                        o['_mask'] = _mask_array(b['pid'], o['id'],
                                                 (doc['native_h'], doc['native_w']))
                pal = export.shared_palette(res['frames'], colors)
                data, info = export.bundle_zip(nat, res, doc2, fps=fps, scale=1,
                                               colors=colors, pal=pal, atlas=atlas)
                with open(os.path.join(_out_dir(b['pid']), 'bundle.zip'), 'wb') as f:
                    f.write(data)
        except BusyError as e:
            self._send(429, {'error': str(e), 'retry': True})
            return
        info.update({'ok': True, 'url': f'/api/out/{b["pid"]}/bundle.zip',
                     'audit': res['audit'], 'total_ms': round((time.time() - t0) * 1000)})
        self._send(200, info)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    args = [a for a in sys.argv[1:]]
    if '--fetch-depth-models' in args:
        args.remove('--fetch-depth-models')
        ok = dim.install()
        print('depth models:', 'installed' if ok else dim.status()['error'], flush=True)
    port = int(args[0]) if args else 8000
    srv = ThreadingHTTPServer(('0.0.0.0', port), Handler)
    src = depth_ml.source()
    print(f'Pixel Sway {MILESTONE} listening on {port}', flush=True)
    print('depth pass: ' + src + ('' if src != 'parse-only' else
          ' — the depth tab offers a one-click model install (~120 MB), or run '
          '`python3 server.py --fetch-depth-models`'), flush=True)
    srv.serve_forever()
