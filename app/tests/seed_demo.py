"""Seed a demo courtyard project (runtime data, not committed): upload the origin art,
auto-seg it, import the recognised objects with their advisory anim types.
Run:  python3 app/tests/seed_demo.py [port]
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PORT = sys.argv[1] if len(sys.argv) > 1 else '8126'
BASE = f'http://127.0.0.1:{PORT}'


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())


def main():
    srv = subprocess.Popen([sys.executable, 'server.py', PORT],
                           cwd=os.path.join(ROOT, 'app'),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(BASE + '/api/ping', timeout=2)
                break
            except OSError:
                time.sleep(0.25)
        with open(os.path.join(ROOT, 'uploads', 'iso_00080_.png'), 'rb') as f:
            req = urllib.request.Request(BASE + '/api/upload', data=f.read(),
                                         headers={'Content-Type': 'image/png'})
            meta = json.loads(urllib.request.urlopen(req, timeout=60).read())
        pid = meta['id']
        seg = post('/api/autoseg', {'pid': pid, 'max_suggestions': 60,
                                    'cfg': {'recognise': True}})
        ops = []
        for i, s in enumerate([s for s in seg['suggestions'] if s['kind'] == 'object']):
            ops.append({'op': 'upsert', 'id': s['id'], 'name': s['name'],
                        'kind': 'object', 'color': s['color'], 'z': i,
                        'label': s.get('label', s['name']), 'conf': s.get('conf', 0),
                        'anim': s.get('anim', ''), 'anim_hint': s.get('anim', ''),
                        'nfrag': s.get('nfrag', 1), 'rle': s['rle']})
        r = post('/api/masks', {'pid': pid, 'ops': ops})
        print(f'DEMO PROJECT {pid} objects {len(r["objects"])}')
        return pid
    finally:
        srv.terminate()
        srv.wait(timeout=10)


if __name__ == '__main__':
    main()
