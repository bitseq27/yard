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
9. Every pixel knows its ground row (M6a, `engine/depth.py`): the wall/floor junction per
   column is the top row of the floor sheet (the horizontal plate band at the wall base);
   standing surfaces inherit their column's junction row, ground pixels their own row,
   objects their ground contact (97th row percentile). z sort keys are the iso closeness
   formula evaluated **at the feet** (3·base_y − y at y = base_y), never averaged over the
   body — the mean-closeness key ranked a tall palm in front of a short pot on the same
   ground row (measured counterexample in test_m6).
10. The normal map is a bump, not a solve (M6c, `engine/depth.normal_map`): height = the
    palette-snapped luminance (dither-safe), box-meaned 3×3, central-differenced into a
    tangent-space RGB map, G up (OpenGL). Flat shading encodes exactly (128, 128, 255);
    every vector is unit and faces the viewer. It ships as `<pid>/normal.png` and shows
    in the **normals** view tab — a derived view of the art, not reconstructed geometry.
11. The depth accelerator is OPTIONAL (M6e, `engine/depth_ml.py`): engine core — every
    byte the render/sprite/audit path touches — stays numpy+PIL. The monocular-depth
    blend (onnxruntime, imported lazily inside `fuse()` only when a runtime and a model
    file exist) feeds the depth TAB as `depth_ml.png` and nothing else; without it the
    server ships the pure parse field. `DEPTH_MODEL=off|da|midas` picks the path, and
    the M2 upload budget (<2 s) is kept by fusing in a background thread.

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

**Status 2026-10-31: M1 ✔, M2 ✔, M3 ✔, M4 ✔, M5 ✔, M6a ✔, M6b ✔, M6c ✔, M6e ✔ — all machine-verified.**
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
- **M6 depth & surfaces (started 2026-10-31, see §10):**
  - **M6a ✔ done (entry 13)** — `engine/depth.py`: per-pixel ground-row field (junction
    line from the floor sheet's top row; NONE/GROUND/WALL/OBJECT orientation; closeness
    map in the origin's uint8 format), computed at upload and served as
    `<pid>/depth.png`; z sort key corrected to closeness at the feet (§2.9). *Accept:*
    `app/tests/test_m6.py` — synthetic iso room answers exact (junction row, surface
    labels, per-surface base_y, feet-key ties, hole axis); courtyard field inside the
    perf share (245 ms) with spearman ≥ 0.5 vs the origin `depth.npy` and render audit
    still 0; m2–m5 + ui_smoke unchanged.
  - **M6b ✔ done (entry 14)** — editor exposure: `D` cycles the canvas overlay
    art → depth closeness → surface orientation (`<pid>/depth.png`, `<pid>/orient.png`,
    loaded with the project); the inspector gains a *surface* select (auto / standing /
    ground plane) that overrides the recogniser in the field; a wand **depth guard**
    (rail button, cycles off/2/4/8 rows) stops the colour flood at ground-row jumps;
    `split @depth` (edge op + auto-seg setting, default on) cuts a fused suggestion or
    mask at surface boundaries and makes the pieces their own masks. *Accept:*
    `test_m6.py` 43 checks (guarded wand stays on its surface on synthetic + courtyard,
    fused vine+bush splits into two single-surface suggestions only when asked, surface
    round-trips through /api/masks) and `ui_smoke.js` 89 checks.
  - **M6c ✔ done (entry 15)** — editor ergonomics + view tabs + the derived normal map,
    all user-driven: header **undo/redo buttons** (↶ ↷) that mirror the ctrl+Z history
    (disabled state + "N steps" in the tooltip); the rail's seven tools become inline
    **SVG icons** and gain a **select** tool (`V`: topmost visible mask under the cursor,
    ⇧ adds a 3-px ring search, hidden masks never pick, empty art keeps the selection);
    the overlay-cycle button is replaced by **view tabs** above the stage — masks /
    depth / normals / anim (`D` cycles; the anim tab renders on first open and plays the
    loop in place; non-mask tabs are pan-only so painting can never land on a derived
    view); and `engine/depth.normal_map()` derives the **tangent-space bump map**
    (palette-snapped luminance → 3×3 box mean → central differences → unit encode, G up),
    shipped per project as `normal.png` (projects predating M6 rebuild a missing derived
    view on first GET). *Accept:* `test_m6.py` 54 checks (7 normal-map checks: flat =
    exactly (128, 128, 255), 1-px dither interior deviates 0 levels vs step tilt sd 11.3,
    ramp signs, unit length |n|−1 ≤ 0.006, z ≥ 0.75, determinism; + 2 lazy-rebuild checks)
    and `ui_smoke.js` 104 checks (16 new: buttons, tabs, icons, select); m2–m5 unchanged,
    M2 recall still 85.9 %.
  - **M6e ✔ done (entry 16)** — the depth pass bake-off + optional model fusion. User
    reported the depth pass reads flat; `app/tests/depth_alts.py` scored 24 candidates
    against the origin `depth.npy` (classical: iso ramp, shape-from-shading via
    Poisson-integrated normals, luminance; models: Depth Anything V2 small fp32/int8,
    MiDaS v2.1 small; plus blend grids). Models win layout (DA rho .804 vs parse .556
    with objects / .688 art-only) but blur interiors; the parse wins within-object
    ordering; the blend wins both (DA w=0.8: rho .829, rho_in .454). Shipped as an
    OPTIONAL accelerator per §2.11: background fuse at upload, lazy `depth_ml.png` in
    the depth tab, parse-only fallback, weights chosen by a documented rule. Measured
    negatives: no candidate recovers the reference's facade relief from one pixel-art
    image; iso-physical interior ramps contradict the reference (rho .556 -> .02).
    *Accept:* test_m6 58 checks (fused rho >= .75 and >= parse + .10; upload < 2 s with
    the accelerator armed: 673 ms; graceful None without runtime) + ui_smoke 106.
  - **M6d (open)** — shading-aware inpaint from the orientation field + a light direction
    inferred from the art's own highlight/shadow placement; optional `relight` clip type
    consuming the M6c normal map.
- **Phase 2 (post-M5):** object recomposition editor (drag objects, live re-inpaint of source
  and destination holes), optional runtime snippet export for the custom web engine.

## 7. Risks & mitigations

- **Inpaint streaks under large masks** → multi-direction fill + amplitude caps; later: user
  clean-plate layer. M6a measured the obvious next step — choosing the fill axis from the
  orientation field — and it does NOT beat M4 on patterned tilework (entry 13); M6c's
  shading-aware inpaint is the open mitigation.
- **Z-order/occlusion confusion** → explicit z list + "static occluder" mask type, plus
  feet-based sort keys (M6a, §2.9). M6b's "split at depth break" will propose occlusion
  cut lines automatically instead of asking the user (§8).
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

## 10. Depth & surface orientation (M6, 2026-10-31)

The origin session's depth pass (`seg_depth.py:250-303` → `depth.npy`, `depth_map.png`)
was courtyard-specific: a teal wall class, a hardcoded `ys.min() < 240` stone band,
carpets below row 260, per-column bottoms for four hardcoded vertical classes. M6 ports
the *idea* — every pixel learns the ground row its surface stands on — with portable
rules, and uses it where geometry, not colour, is the missing signal.

- **Field** (`engine/depth.field`) → `base_y` (ground row per pixel), `orient`
  (NONE/GROUND/WALL/OBJECT), `depth` (uint8 closeness in the origin's format),
  `horizon` (the junction line). Junction per column = top row of the *floor sheet*
  (largest component of stone+ground+misc — the horizontal band at the wall base),
  interpolated across columns that objects stand in, median-smoothed along the face.
  WALL = plate material above the junction; OBJECT = an object's ground contact;
  lacy surfaces (tile outlines are shadow class) closed by box-majority passes;
  whatever remains inherits base_y from its labelled neighbours. No floor sheet
  (flat or side-view art) → base_y = y everywhere, i.e. today's behaviour.
- **Sort key** = closeness at the feet (§2.9). `imaging.stats_of`, `depth.object_key`
  and the index.html `maskStats` mirror agree; test_m6 pins all three together.
- **Served**: upload writes `<pid>/depth.png`; `GET /api/image/<pid>/depth.png`.
- **M6b shipped**: overlay (`D`), per-mask surface override (inspector select, stored on
  the doc, honoured ahead of the recogniser label in the field), wand depth guard
  (`dguard` rows: the flood may not leave the seed's ground-row band), and split at
  surface boundaries (`/api/edge` op `depthsplit`, auto-seg cfg `depth_split`, default on
  in the UI, off in the engine so M2's measured recall stays the baseline).
  Honest limit, logged in entry 14: `base_y` is *continuous* across the wall/floor
  junction by construction (a wall pixel's ground row is the junction row), so the
  discrete cut signal is the orientation label (`surface_edges`), not a base_y jump; and
  two objects on the *same* surface share one ground plane — splitting those is Phase-2
  recomposition work, not this.
- **Measured and rejected for now**: choosing the inpaint fill axis from `orient`
  (build_log entry 13). On the courtyard's wall holes, vertical fill does not beat M4's
  horizontal+bifill (24.58 vs 24.34 mean abs error against the art underneath) because
  tilework is patterned in both axes — a lerp cannot reconstruct pattern on either axis.
  `animate.inpaint` keeps its proven behaviour; M6c's shading-aware inpaint is the
  version of this idea worth building.
- **M6b** exposes the field in the editor (overlay toggle, per-region surface type,
  split-at-depth-break, depth-aware wand) — the segregation/masking help this milestone
  exists for.
- **M6c shipped (entry 15)**: the editor's view layer. Tabs (masks / depth / normals /
  anim, `D` cycles) replace the single overlay button; undo/redo gain header buttons;
  the rail is icon-only plus a select tool; and the **normals** tab promotes a derived
  normal map to a first-class view + export artifact (`normal.png`). The M6b position
  that a normal *map* stays out of scope was reversed by the user's tab request, with
  this scope guard: the map is **derived** (a bump from the art's own shading — see §2.10
  and the measured dither/flatness numbers above), never solved geometry; the
  shading-aware inpaint + `relight` ideas that would *consume* it are M6d.
- **M6e shipped (entry 16)**: the depth bake-off and the optional model-fused depth
  pass. The flatness complaint was structural, not cosmetic: the parse is a piecewise
  ground-row plate while the reference carries facade relief and a curved floor. The
  bake-off's honest conclusion — monocular models fix LAYOUT (rho .556/.688 -> .80-.83)
  but not DETAIL (no candidate recovers the facade; percentile-clipped montages show the
  models' wall signal is the row ramp), and the parse still owns within-object ordering
  — is why the shipped artifact is a *blend*, and why it is optional: the engine rule
  (§2.11) keeps core numpy+PIL, with onnxruntime as a lazy accelerator that degrades to
  M6b bytes. Weights by rule, not by eye: maximise rho+conc+wallrho s.t. rho_in >= .40
  against the art-only parse (DA-v2-int8 w=0.8, MiDaS-v21 w=0.6).
- **M6d (open)**: shading-aware inpaint + optional `relight` clip type.
