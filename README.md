# Pixel Sway — pixel-art animation editor

Single-page editor + stdlib Python server that turns pixel-art scenes into seamless
animated loops, mask-driven sprites and game bundles. Milestones M1–M6 complete and
machine-verified (`app/tests/`); the living ledger is `build_log.md`, the plan is
`plan.md`, in-app docs are served at `/docs`.

## Quick start

```bash
# 1. dependencies (debian/ubuntu) — the engine needs only numpy + PIL + gifsicle
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  python3-pil python3-numpy gifsicle

# 2. launch
cd app && python3 server.py 8000        # http://localhost:8000

# 3. (optional, recommended) the monocular depth accelerator, ~120 MB, NOT in git:
python3 server.py --fetch-depth-models  # one-shot fetch, then launch as usual
```

Without step 3 the server boots **parse-only**: the depth tab shows the M6a
ground-row field (piecewise plates — useful for the wand guard / split / sort tools,
but visually flat). With the accelerator installed the depth tab shows the
model-fused map (Depth Anything V2 / MiDaS blended with the parse; spearman vs the
reference depth 0.83 vs 0.56). You can also install it **from the editor**: the
right-hand end of the view-tab bar shows the depth-pass source and, on a fresh
clone, an `install models (~120 MB)` button with live download progress
(`GET /api/depth_ml/status`, `POST /api/depth_ml/install`). `DEPTH_MODEL=off|da|midas`
overrides the choice; `off` forces parse-only.

Drop a pixel-art PNG on the editor to start a project, or open one of the seeded
examples from the projects dropdown.

## Tests

```bash
npm install                 # jsdom, only for the UI smoke test
python3 app/tests/test_m2.py   # … test_m3 … test_m6
node    app/tests/ui_smoke.js
python3 app/tests/depth_alts.py --models   # the M6e depth bake-off (needs the models)
```

`app/tests/dl_models.py` re-fetches onnxruntime + the onnx models after a sandbox
reset; both live in gitignored `app/tests/{_ort,models}` (the installer above uses
`app/{_ort,models}`).

## Layout

```
app/server.py            stdlib ThreadingHTTPServer (routes in its docstring)
app/static/index.html    the whole editor, no CDN dependencies
app/static/docs.html     API + engine rules, served at /docs
app/engine/              imaging · segment · animate · recognise · export ·
                         depth (M6a parse) · depth_ml (M6e optional accelerator)
app/tests/               acceptance suites + bake-off harness
plan.md · build_log.md   the ledger: decisions, measurements, negative results
```
