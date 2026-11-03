# Pixel Sway — pixel-art animation editor

Single-page editor + stdlib Python server that turns pixel-art scenes into seamless
animated loops, mask-driven sprites and game bundles. Milestones M1–M7 complete and
machine-verified (`app/tests/`); the living ledger is `build_log.md`, the plan is
`plan.md`, in-app docs are served at `/docs`.

## Quick start

```bash
# 1. dependencies (debian/ubuntu) — the engine needs only numpy + PIL + gifsicle
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  python3-pil python3-numpy gifsicle

# 2. launch
cd app && python3 server.py 8000        # http://localhost:8000

# 3. (optional, recommended) the monocular depth accelerator, ~220 MB, NOT in git:
python3 server.py --fetch-depth-models  # one-shot fetch, then launch as usual
```

Without step 3 the server boots **parse-only**: the depth tab shows the M6a
ground-row field (piecewise plates — useful for the wand guard / split / sort tools,
but visually flat). With the accelerator installed the depth tab shows the model
map — **Depth Anything V2 ViT-L fp32** (the same weights as ComfyUI's
`DownloadAndLoadDepthAnythingV2Model` / `depth_anything_v2_vitl_fp32.safetensors`),
used raw: the documented fusion rule picks w=1.0 for the large family, so what
you see is exactly what that node produces. A quantised twin at a reduced input
edge covers low-RAM machines; Depth Anything 3 small (blended at w=0.9, spearman
0.885 on the courtyard reference), DA-v2 small and MiDaS stay in as fallbacks,
and `DEPTH_MODEL=off|da3|da|midas|torch` overrides the choice (`off` forces
parse-only, `da3`/`torch` prefer the DA3 paths). You can also install it **from
the editor**: the right-hand end of the view-tab bar shows the depth-pass source
and, on a fresh clone, an `install models (~1.9 GB)` button with live download
progress (`GET /api/depth_ml/status`, `POST /api/depth_ml/install`). If you
already run **the official `depth-anything-3` torch package** locally,
`DEPTH_MODEL=torch` uses it (GPU when available; checkpoint via `DA3_TORCH_MODEL`,
default `depth-anything/DA3-SMALL`). If your depth output looks flat, the status
chip reads `parse` — the accelerator is missing or failed to load, and the map
you see is the parse-only fallback. If a model is present but cannot allocate
on your machine, the server falls through to the next model in preference
instead of dropping to parse (the status chip names the one that ran).

**Auto-seg labels (M7 follow-up).** Palette alone cannot tell a red tiled roof
from a red rug, so the recogniser double-gates ground sheets: what sits directly
above the sheet's top edge (floor material ⇒ rug/pool; sky/foliage/another tier
⇒ roof) *and* where the sheet sits in the frame (roofs ride high). This also
matters for game bundles — rugs are walkable in the bundle walkmask, roofs must
not be. See build_log entry 19 for the measurements behind the rule.

**Game-scene layers (M7).** The same depth field slices the scene into depth-band
plates for a game where characters walk around: `POST /api/layers {pid, k}` builds
k RGBA plates (far→near; every nearer footprint inpainted as disocclusion backing,
so composing plates + object sheets recomposes the art pixel-exactly), objects are
assigned to the band at their feet with a continuous foot depth for y-sorting, and
the game bundle ships `layers/L*.png` + `depthmap.png` (u8 closeness — sample at a
character's foot pixel) + the `scene.json.layers` contract. The bands double as
segmentation: `POST /api/layers_regions` offers them as editable region masks, and
the editor's Layers panel (build / k / "as regions") plus the `layers` tab wrap all
of it. See plan.md §8b.

Drop a pixel-art PNG on the editor to start a project, or open one of the seeded
examples from the projects dropdown.

## Tests

```bash
npm install                 # jsdom, only for the UI smoke test
python3 app/tests/test_m2.py   # … test_m3 … test_m6 … test_m7 (layers + DA3)
node    app/tests/ui_smoke.js
python3 app/tests/depth_alts.py --models   # the depth bake-off (needs the models)
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
                         depth (M6a parse) · depth_ml (M6e/M7 optional accelerator,
                         DA3 first) · layers (M7 depth-band game plates)
app/tests/               acceptance suites + bake-off harness
plan.md · build_log.md   the ledger: decisions, measurements, negative results
```
