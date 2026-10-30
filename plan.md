# Pixel Sway — plan.md

**Goal:** A web app (hosted in this workspace with live preview) where the user loads a pixel-art
image, masks objects, picks an animation type per mask from a catalog, and exports a seamlessly
looping animated GIF.

**Locked scope decisions (2026-10-30):**
- Platform: web app served from this sandbox (server renders, browser edits).
- Art: pixel art only (grid detection, integer-pixel operators, palette snapping).
- Masking: manual brush/eraser + magic wand + one-click auto-segment suggestions (user-correctable).
- Export: **game bundle first-class** (per-object RGBA sprite sheets + clip metadata + scene
  plate/walkmask/palette JSON); GIF kept as preview/social output. MP4/APNG later.
- Direction (2026-10-30): assets are destined for a pixel-art adventure game → masks are
  modeled as *objects* (anchor, depth sort key, clips) from day one; see §8.
- Target engine: **custom web JS** → engine-neutral JSON + PNG sheets is the final format.
- Editor recomposition: **phase 2**; transforms reserved in the data model now.
- Docs first: this plan + `build_log.md` precede any code.

---

## 1. Architecture

```
browser (vanilla JS, single page, no CDN deps)
  │  relative-URL fetch/XHR only
  ▼
Python backend (stdlib http.server or lightweight framework via apt)
  │  /api/upload   /api/wand   /api/autoseg   /api/render   /api/gif   /api/project
  ▼
Engine module (numpy) — ported from the proven animate3.py pipeline
  │  frames → shared-palette quantize → GIF → gifsicle -O3
  ▼
workspace files (project dir per session: img/, masks/, out/)
```

- **Frontend:** one HTML file with inline CSS/JS (preview iframe is sandboxed; zero external
  deps). Canvas editor with zoom/pan, mask painting, mask list panel, per-mask animation
  inspector, timeline scrubber + play, export button.
- **Backend:** thin JSON API over the engine. All heavy work server-side (numpy), results
  streamed back as PNG (masks/frames preview) and GIF (final).
- **State:** a project = JSON doc `{image, grid, N, fps, masks:[{id, name, z, png, anim, params, shadow}]}`.
  Masks stored as indexed PNGs (1 file per mask or one multi-index PNG + sidecar JSON).

## 2. Engine rules (hard-won lessons — non-negotiable)

1. Detect pixel grid (uniform p×p block test), work at native res, upscale ×2 NEAREST on export.
2. Inpaint background **under every mask** before animating (horizontal run fill; optional
   vertical blend pass) so vacated pixels reveal plausible background.
3. **Redraw every mask every frame**, including when its offset is 0 (the "disappearing lily" bug).
4. Hole-fill trailing edges from the pre-mask background frame (`fr_bg`), 1–2 px max.
5. All motion frequencies = integer cycles per loop → seamless loop by construction.
6. Per-object individuality: own phase, speed (1–3 cycles/loop), amplitude, waveform mix
   (2nd harmonic), gust envelope; articulated chunks (base/mid/top) instead of per-row shear
   (per-row shear = "screen tearing" look).
7. Shared global palette (quantize once from a frame montage), `dither=NONE` → no flicker,
   small LZW size; finish with `gifsicle -O3` (lossless, ~10× smaller).
8. Audit hook: frame-diff outside allowed zones must be 0 (catches stray animated pixels).

## 3. Animation catalog (MVP)

| type      | applies to            | params (sliders)                              | notes |
|-----------|-----------------------|-----------------------------------------------|-------|
| sway      | plants, fronds, flags | speed 1–3, amp 1–2 px, gust 0–0.3, phase auto | articulated chunks, joints auto from mask height |
| bob       | pads, floaters, clouds| speed 1–3, amp 0–2, circularity, phase        | 2-D drift, whole-mask rigid |
| ripple    | water surfaces        | speed 1–3, wavelength, amp                    | per-run row shift, run-based |
| shimmer   | water, glints, metal  | speed 2–4, coverage                           | brightness wave, palette-snapped |
| flutter   | blossoms, tiny leaves | speed 2–4, amp 1                              | fast small jitter |
| gustflick | fringes, tassels, grass| gusts/loop 1–4, strength, travel             | event windows with x-travel delay |
| static    | occluders, background | —                                             | mask participates in z-order/inpaint only |

Per-mask optional toggle: **shadow sync** (dark pixels hugging the mask move with it; pots/static
owners keep theirs) — implemented, currently rolled back in the courtyard GIF, offered as opt-in.

## 4. Masking tools

- Brush / eraser (square nib, sizes 1–16 native px), rect + lasso fill.
- Magic wand: server-side flood fill (tolerance slider), 4/8-connectivity. Metric is a
  **hue-family flood** (circular hue ≤ tol, sat/val gates) with a leak guard that tightens
  to the seed's shade family above 12% of frame — RGB tolerance cannot work on dithered
  art (measured: the fern spans Chebyshev 155 inside itself). Classic seed-tolerance
  flood kept as an option, with optional palette snap. See build_log entry 7.
- Auto-segment: server runs the courtyard segmentation pipeline (color rules + components +
  speckle cleanup) and returns top-K instance masks as editable suggestions.
- Recognition (entry 9): each suggestion is also *named* by a rule-based recogniser
  (`app/engine/recognise.py`) — palm / fern / flowering plant / potted plant / rug /
  runner / lily pad / pool / wall+floor tiles — with a confidence and an advisory M3 anim
  hint; unmatched objects fall back to their material class at low confidence.
- Mask list: rename, recolor chip, visibility, z-order up/down, duplicate, delete.
- Edge helper: "grow/shrink 1 px" and "include outline" per mask (halo control).

## 5. UI layout (single screen)

- Left: tool rail (brush/eraser/wand/lasso/pan), brush size, tolerance.
- Center: zoomable canvas (checker under transparency), onion-skin of mask colors overlay.
- Right: mask list + inspector (anim type dropdown, param sliders, shadow toggle, z-order).
- Bottom: timeline (frame scrub, play/pause at target fps, loop), render + export buttons,
  status/progress line.

## 6. Milestones & acceptance

**Status 2026-10-30: M1 ✔, M2 ✔, M3 ✔ and M4 ✔ — all machine-verified.**
M2 numbers, measured against the origin session's `seg.npy` ground truth by
`app/tests/test_m2.py`: wand grabs the fern in one click at tol 30 (57% of the object,
1.2% of frame; leak-guarded pot 94%/0.9%); auto-seg returns 85.9% of the 85 courtyard
objects ≥25px as exactly 60 suggestions in ~1.3 s. See build_log entries 5–7.
M3 numbers, measured by `app/tests/test_m3.py` (see build_log entry 10): courtyard
auto-seg import (56 objects) → 48-frame loop renders in ~1.1 s with **audit = 0 stray px
(wrap included)**, GIF = **0.72 MB** ≤ 2 MB at 768×1024/48f/loop-forever; the scene bundle
(56 RGBA sheets) **recomposites to exactly the rendered frames** — binary alpha, zeroed RGB
under alpha 0, plate identical to the art outside every mask.

- **M1 skeleton:** ✔ done (entry 5) — server up; upload → grid detect → native downsample;
  canvas view with zoom/pan. *Accept:* 1536×2048 upload shows crisp 384×512 native at ×2
  zoom < 2 s. (re-verified in M2: 930 ms)
- **M2 masking:** ✔ done (entry 7) — brush/eraser/rect/lasso/wand; mask list with z-order;
  auto-seg import. *Accept:* wand grabs a fern in one click at tol 30; auto-seg returns
  ≥80 % of courtyard objects as ≤60 suggestions.
- **M3 engine + export:** ✔ done (entry 10) — all 7 anim types; render preview (server
  returns N frames as one strip + plays in browser); GIF export via palette+gifsicle;
  **scene bundle export** (plate, walkmask, 56 object RGBA sheets, scene.json).
  *Accept:* courtyard masks → loop-seamless GIF ≤2 MB (measured 0.72 MB), audit = 0 stray
  px; bundle re-imports with clean RGBA edges — recomposite equals render frames exactly.
- **M4 polish:** ✔ done (entry 11) — project rename/duplicate/delete + named recents,
  param presets (built-in + user), shadow-sync toggle (plan §3), palette-size option for
  GIF *and* bundle, error toasts, keyboard shortcuts + help overlay, and the plan §7
  vertical-blend inpaint pass (hard streak lines inside big holes 1003 → 619).
  *Accept:* `app/tests/test_m4.py` 23 checks + ui_smoke 75 checks, all pass.
- **M5 hardening:** ✔ done (entry 12) — perf pass (worst-case 48f @ 384×512:
  10.6 s → **1.1 s**), concurrency guard (per-project lock + 2 global slots, 429 when
  busy), /docs page, atlas packing in the bundle (scene.json rects, pixel-exact crops),
  engine-version cache key, and two correctness fixes found by the new parity harness:
  sprites sampled the source at cell-relative x (texture bug, M3/M4), and the leaf-unit
  split had drifted from the origin's multi-source BFS.
  *Accept:* `app/tests/test_m5.py` 34 checks + ui_smoke (now 78 checks), all pass.
- **Phase 2 (post-M5):** object recomposition editor (drag objects, live re-inpaint of source
  and destination holes), optional runtime snippet export for the custom web engine.

## 7. Risks & mitigations

- **Inpaint streaks under large masks** → multi-direction fill + amplitude caps; later: user
  clean-plate layer.
- **Z-order/occlusion confusion** → explicit z list + "static occluder" mask type.
- **Sandbox resets lose apt packages** → `build_log.md` keeps a one-line reinstall command;
  engine depends only on numpy+PIL+gifsicle(+ffmpeg later).
- **Wand/auto-seg over-grab** → suggestions are always editable masks, never authoritative.
- **GIF size** → shared palette + gifsicle; warn >8 MB with frame-count/palette hints.

## 8. Game pipeline (direction change, 2026-10-30)

Data model (engine-neutral JSON; adapters later):
- Scene project = `{plate.png (native, objects removed+inpainted), palette (locked, project-wide),
  grid, objects[], regions[]}`
- object = `{id, name, mask.png, anchor(x,y), offset(x,y) default (0,0) — reserved for
  phase-2 recomposition, depth (auto from iso formula, editable), z,
  label/conf/anim (object recognition, entry 9 — label is the recognised type such as
  "fern"/"potted plant"/"rug", conf its confidence, anim the advisory clip type),
  clips:[{name, anim, params, N, fps, loop|oneshot}]}`
- regions = `{walkmask.png, hotspots:[{poly,id}], spawnpoints:[]}` — edited with the same mask tools

Exports:
1. preview GIF (as today)
2. object sheets: RGBA PNG strip per object per clip + `clips.json`
3. scene bundle: `plate.png`, `walkmask.png`, `palette.png`, `scene.json` (anchors, depths, bboxes)
4. M5: atlas packing (target engine is a custom web JS engine → plain JSON + PNG sheets
   is the final format; no engine adapters needed)

Clip conventions:
- loops seamless (integer cycles); per-instance variety = engine-side start-frame offset (free)
- one-shots carry lead-in/rest frames; default budget 8 frames/clip @ 8–12 fps (configurable)
- extraction: sprite = mask ∩ source as RGBA; plate hole inpainted; overlapping connected
  objects → user chooses combine-or-split at a cut line
- shadows: optional separate shadow sprite (shadow-sync code from origin session), not baked

## 9. Out of scope (for now)

MP4/APNG export, atlas packing (M5), engine importers (M5), non-pixel art, multi-user/auth, cloud storage, mobile layout,
rotation/scale operators (pixel-unsafe), audio.
