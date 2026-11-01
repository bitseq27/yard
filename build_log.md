# Pixel Sway — build log.md

Living log of the app build. Newest entries on top. Keep entries short: what changed,
decisions taken, problems hit + fixes, and anything a future session must know
(the sandbox resets installed packages between sessions — see Setup).

---

## Setup (run after any sandbox reset)

```bash
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  python3-pil python3-numpy gifsicle   # add ffmpeg when MP4 export lands
# optional M6e depth accelerator (onnxruntime + Depth Anything V2 / MiDaS onnx,
# ~190 MB into gitignored app/tests/{_ort,models}; server works without it):
python3 app/tests/dl_models.py
```

Workspace root holds: `plan.md`, `build_log.md`, app code (`app/`), and the reference
artifacts from the origin session (`courtyard_animated.gif/.mp4`, `segmentation_mask.png`,
`depth_map.png`, `seg_legend.json`, `contact_sheet.png`, `uploads/iso_00080_.png`).

---

## 2026-10-31 — Entry 17: M6e follow-up — the accelerator installs itself

**Why**: user cloned the repo fresh, launched, and the depth pass was still flat —
correctly so (models + onnxruntime are gitignored, ~120 MB), but nothing in the app
said so. Silence is a bug: the parse-only fallback now announces itself.

**Shipped**: `app/install_depth_models.py` (stdlib installer with byte progress);
`GET /api/depth_ml/status` + `POST /api/depth_ml/install` (background thread, single
flight, `already|started|busy|disabled` answers, refuses while `DEPTH_MODEL=off`);
the tab bar's right end shows the depth-pass source and, when parse-only, an
`install models (~120 MB)` button with live percent that swaps the depth tab to the
fused map by itself when the download lands; `server.py --fetch-depth-models` for
headless setups and a boot log line naming the source plus the hint; new `README.md`
quick-start (the repo had none — part of why the fresh-clone path was unclear);
docs §8 rows for the two endpoints.

**Verified**: test_m6 adds status/install no-op checks plus a second server booted
with `DEPTH_MODEL=off` (parse-only ping, status agrees, install refuses); ui_smoke
checks the tab-bar label matches ping in both states. m2–m6 + ui_smoke green.

**Follow-up fix (user: "the install button did not work")**: the button sat inside
`#tabs`, and the tab bar's own wiring (`querySelectorAll('#tabs button').forEach(b =>
b.onclick = … setTab …)`) runs *after* the install handler is attached — it clobbered
it, so a click called `setTab(undefined)` and corrupted the tab state instead of
installing. Fixed by scoping every tab-bar loop to `#tabs button[data-tab]` and
guarding `setTab` against unknown tabs; ui_smoke now clicks the button and asserts the
tab state survives. While in there: the installer picks the onnxruntime wheel for the
*running* platform (it was cp311-manylinux-x86_64 only — a mac/windows user would have
hit a cryptic error), and failures now surface in the status line + button title
instead of a bare "retry". Re-verified in jsdom against a parse-only server: click →
`installing da_v2_small_quant.onnx N%` with the button disabled during the download.

---

## 2026-10-31 — Entry 16: M6e — depth bake-off, model-fused depth pass (optional accelerator)

**Complaint that started it**: "the depth pass is flat". Confirmed against the origin
`depth.npy`: the M6a field is a piecewise ground-row plate (smooth ramps, cut-out
silhouettes, zero facade; wallrho -0.17) while the reference carries dense facade relief
and a curved floor. So `app/tests/depth_alts.py` became a bake-off: 24 candidates scored
against `depth.npy` (global spearman, pairwise concordance, within-object spearman,
|grad| placement, facade-band spearman, within-object flatness ratio) — classical
solutions, two ONNX model families, and fusions.

| solution | rho | conc | rho_in | facade | flat (gt .20) | cost |
|---|---|---|---|---|---|---|
| m6a shipped | .556 | .728 | **+.703** | -.166 | .42 | 0 ms |
| iso3 (iso-physical ramp) | .556 | .730 | +.706 | -.166 | .42 | 0 ms |
| sfs (Poisson-integrated M6c normals) | .398 | .637 | +.193 | +.159 | .60 | 90 ms |
| luminance proxy | .029 | .508 | -.074 | -.194 | .57 | 0 ms |
| Depth Anything V2 small int8 | **.804** | .808 | +.196 | **+.388** | .21 | 4.5 s |
| MiDaS v2.1 small | .781 | **.819** | +.413 | -.399 | .21 | 0.4 s |
| fuse .8 DA + .2 parse | **.831** | .821 | +.299* | +.286* | .27 | 4.5 s |
| fuse .6 MiDaS + .4 parse | .785* | .826* | +.406* | -.442* | — | 0.4 s |

(*rows marked with the weight finally shipped; the table shows the family extremes and
the shipped blends — full grid in `depth_alts.py` output, entry montages in
`app/projects/_scratch/bakeoff*.png`.)

**Negative results, measured**: shape-from-shading cannot carry layout (rho .398) and
only hints at the facade (+.16); luminance alone is noise; an iso-physical height ramp
inside objects/walls *contradicts* the reference (rho collapses .556 -> .02 by gamma=0.6)
— the reference's object interiors really are near-constant, so "flat objects" is partly
correct behaviour; and **no tested solution recovers the reference's facade relief from a
single pixel-art image** — the models' facade score comes from matching the row ramp,
visible in percentile-clipped montages. That detail came from real geometry in the origin
session.

**What shipped** (user chose "both models", weights picked by rule): optional accelerator
`engine/depth_ml.py` — onnxruntime imported lazily *inside* `fuse()`, only when a runtime
AND a model file exist; otherwise `fuse()` returns None and the server ships M6b bytes.
`DEPTH_MODEL=off|da|midas` picks the path. Blend = w*znorm(model) + (1-w)*znorm(parse),
percentile 1-99 encoded (min-max flattened contrast in the montages); model orientation
fixed without ground truth via the row-axis correlation. Weight rule: maximise
rho+conc+wallrho s.t. within-object rho_in >= .40, scored against the ART-ONLY parse
(what upload sees) -> DA-v2-int8 w=0.8 (rho .829, rho_in .454), MiDaS w=0.6 (rho .785,
rho_in .406). Upload runs the fuse in a **background thread** (M2's <2 s upload budget
stays law: measured 673 ms with the accelerator armed); `depth_ml.png` is then fetched
lazily by the editor (open + first depth-tab visit) and rebuilds synchronously on first
GET for older projects; a still-fusing pid answers 404 instead of double-running.
`/api/ping` reports `depth_ml` so UI/tests know the source; engine tools (wand guard,
split, sort keys) keep using the parse field — only the depth TAB shows the fused map.

**Verification**: test_m6 58 checks (M6e: budget pin, bg-write poll, GET, byte-exact
reproduce from `depth_ml.fuse`, fused rho >= .75 and >= parse+.10 -> .829 vs .688,
lazy rebuild, graceful None without runtime); ui_smoke 106 (ping source + fused img on
the depth tab); m2-m5 unchanged (m2's round-trip first REGRESSED to >2 s with a
synchronous fuse — that is what forced the background-thread design). Models + unpacked
onnxruntime live in gitignored `app/tests/{models,_ort}/`; `app/tests/dl_models.py`
refetches them after a sandbox reset (MiDaS's GitHub release assets 404 — the working
mirror is `Heliosoph/midas-small-onnx` on HF; DA-v2 comes from `onnx-community`).

**Engine rule, amended with the user's approval**: core stays numpy+PIL; `depth_ml` is an
optional accelerator outside the render/sprite/audit path (plan section 2 rule 11).

**Pushed**: `origin/main` = `29640c2` (the origin's squashed root commit) + `1e9c49c`
(this full M1-M6e state, rebuilt after two sandbox resets dropped `.git` — the hashes
cited above, 99263d1/5a34cd7/42bbf86/5a2b2e8, lived only in lost sandboxes). Also fixed
here: `dl_models.py` now pulls MiDaS from the Heliosoph HF mirror (the isl-org release
assets 404), and `depth_ml.MODELS` gained the fp32 DA twin as a fallback for installs
where the quant file is missing.

---

## 2026-10-31 — Entry 15: M6c — undo/redo buttons, icon rail + select tool, view tabs, normal map

**What shipped (all user-driven, in four asks).**
1. **Undo/redo buttons** in the header (↶ ↷). They call the same `undo()`/`redo()` as
   ctrl+Z, and a new `syncHist()` keeps them honest: `disabled` when that stack is empty
   and the depth in the tooltip ("Undo 3 steps (Ctrl+Z)"), refreshed from `pushUndo`,
   `undo`, `redo` and both history clears.
2. **Icons on every tool button** (inline SVG, no font/CDN dependency) and a new
   **select** tool on `V`: `pickMask` walks masks top-down and selects the first visible
   one under the cursor; ⇧ adds a 3-px square-ring search so fringes stay grabbable;
   empty art leaves the selection alone and hidden masks are skipped. Pointer handling
   reads `dataset.tool`, so the SVG children of the buttons are safe under the click
   handler.
3. **View tabs** above the stage — masks / depth / normals / anim — replacing M6b's
   overlay-cycle button (and the `S.view` state with `S.tab`; `D` cycles, ids `#tabs`,
   `aria-pressed` kept in sync for screen readers). The depth and normals tabs blit the
   derived PNG full-bleed and are **pan-only** (a brush stroke must never land on a
   derived view); the anim tab renders on first open (`doRender()` when there is no
   strip yet) and plays the loop in place, pausing when you leave. `openProject` loads
   `normal.png` best-effort next to depth/orient, and a project without one shows a
   "no normal map for this project yet" placeholder.
4. **`engine/depth.normal_map()`** — the derived tangent-space bump map behind the
   normals tab: height = palette-snapped luminance (`segment._denoised`), 3×3 box mean
   (edge-padded: a constant map must stay constant), central differences, unit encode
   with G up (OpenGL), rounded not truncated so flat art is exactly (128, 128, 255).
   Server writes `<pid>/normal.png` at upload via a new `_write_art_views()` (also used
   to **rebuild a missing derived view on first GET**, so pre-M6 projects and the four
   tracked M1 examples are never stuck on a placeholder) and whitelists it on
   `GET /api/image/`.

**Measured** (test_m6, 54 checks): flat patch → unique encoded row (128, 128, 255);
1-px checkerboard dither interior deviates **0 levels** from neutral while a hard step
reads tilt sd **11.3** (the box mean does the work; palette snapping alone cannot kill a
2-colour checker); ramp signs per the G-up convention (bright-top → G 119, bright-right →
R 119); |n|−1 ≤ **0.006** on the courtyard with z ≥ **0.75** (B ≥ 223, nothing faces
away); courtyard tilt sd 13.8/128 (mostly flat, real relief on edges); deterministic
byte-for-byte. ui_smoke now runs **104 checks** (16 new: button/history sync, tab cycling +
first-open render, icon coverage, select pick/ring/hidden/empty); m2–m5 unchanged and
M2's auto-seg recall still 85.9 % @ ≤60 suggestions.

**Problems hit.** (a) First `_box_mean` had an off-by-one in the integral offsets —
caught because the constant-map check failed; fixed with edge padding + verified window
sums. (b) The box-mean smoke test looked like a code bug but was my test art: tiling a
1-D ramp with `np.tile(a, (1,60,1))` made a (1, 3600, 3) image. (c) jsdom has no canvas
2-D text, so the tab placeholder crashed the harness on the anim tab before a strip
exists — `tabPlaceholder` now guards `typeof ctx.fillText` (and mirrors the message to
the status bar, which is friendlier anyway). (d) The smoke test's render-wait loop read a
stale frame label left by the anim tab's first-open render; it now clears `tlstat`/`frlbl`
and waits for the audit line. (e) Select-tool fixtures: the smoke project's *seeded*
mask is empty and the wand later covers the stroke, so the pick fixture scans the live
mask data for a pixel whose topmost owner is the stroke and an empty pixel 1–3 px away,
instead of assuming coordinates.

**Decisions.** The normal map is explicitly a *bump from the art's shading*, not solved
geometry — documented in plan §2.10, docs §9 rule 10 and a docs troubleshooting bullet
("almost flat lavender is correct"). M6c's original scope (shading-aware inpaint +
`relight`) moves to **M6d** and will consume this map.

---

## 2026-10-31 — Entry 14: M6b — the field reaches the editor (overlay, guard, split)

**What shipped.** `D` cycles the canvas overlay art → depth closeness (`depth.png`) →
surface orientation (`orient.png`, new: wall red / ground green / object blue); both are
written at upload and loaded with the project. The inspector gains a **surface** select
(auto / standing / ground plane) stored on the doc and honoured *ahead of* the
recogniser label in `depth.field`. The wand gains a **depth guard** (rail button cycles
off/2/4/8 rows, `dguard` on /api/wand): the flood may not leave the seed's ground-row
band. **split @depth** arrives twice: as an edge op (`/api/edge op=depthsplit` → pieces,
the editor keeps piece 1 and spawns the rest as masks) and as an auto-seg setting
(`depth_split`, default on in the UI, off in engine DEFAULTS so M2's measured recall
stays the baseline). Acceptance: test_m6 43 checks, ui_smoke 89 checks; m2–m5 unchanged.

**Measured.** Guard, synthetic iso room (wall and floor are one hue family on purpose,
joined by a teal drainpipe as the leak path): unguarded hue flood at tol 60 puts >200 px
below the junction, `dguard=4` keeps it under 20 and is byte-identical to the old wand
when off. Courtyard seed (60,100) tol 60: unguarded 16 960 px down to row 259; guarded
3 095 px stopping at row 229 — the junction at that column is 225, i.e. the guard stops
exactly where the surface changes. Split, synthetic vine-on-wall + stem + bush-on-floor
as ONE connected green blob: auto-seg without the split returns one suggestion covering
96% of it; with the split, two suggestions (41% / 35%) each confined to its own surface
(rows 19–55 and 58–86, junction 58). Courtyard: recall identical with the split on and
off (85.9%, 60 suggestions) — nothing fuses there, and the field costs +0.4 s.

**The insight that reshaped the op.** `base_y` is *continuous* across the wall/floor
junction by construction: a wall pixel's ground row **is** the junction row, so the first
cut rule (break where neighbour base_y jumps) found nothing at the junction — the first
M6b run split zero masks and the synthetic check failed loudly. The discrete signal is
the orientation *label*: `depth.surface_edges` (wall meets ground) is the cut line, with
base_y jumps kept for the case they do fire (an object's feet against the surface behind
it, a step between wall faces). Documented in plan §10 with the honest limit: two
objects on the *same* surface share one ground plane, so splitting those belongs to
Phase-2 recomposition, not M6.

**Bug of the entry.** The new rail button took `id="view"` — the canvas's id. In a
browser the rail button wins `getElementById` (document order), so `ctx =
canvas.getContext('2d')` threw at boot; jsdom reported it as "canvas.getContext is not a
function" and ui_smoke died with zero checks run, which is exactly why the smoke test
boots the real page. Renamed to `oview`. Lesson kept: single-file UI means one global id
namespace — new ids get a prefix from now on.

**Next.** M6c (shading-aware inpaint from orientation + inferred light, optional
`relight` clip), then Phase 2 recomposition.

---

## 2026-10-31 — Entry 13: M6a — depth & surface-orientation field in the engine

**What shipped.** New `app/engine/depth.py`: a per-pixel ground-row field
(`base_y`, `orient` NONE/GROUND/WALL/OBJECT, `depth` = the origin's uint8 closeness
map, `horizon`), computed at upload (it depends on the art alone) and served as
`<pid>/depth.png` via `GET /api/image/<pid>/depth.png`. The z sort key is corrected
to iso closeness **at the feet** (`2*base_y`): `imaging.stats_of`, `depth.object_key`
and the index.html `maskStats` mirror now agree, and a tall object no longer jumps
the queue in front of a short one standing on the same ground row (old key on the
test's counterexample: 155.2 vs 145.0 — tall wrongly first; new key: 140.0 vs 140.0).
Acceptance: `app/tests/test_m6.py` (27 checks), dev probe `app/tests/depth_probe.py`
(writes `app/tests/depth_debug.png`). m2–m5 + ui_smoke unchanged, all green.

**The junction, not the colour classes.** First attempt ported the origin's vertical-
surface rules as component geometry tests (vertical sides + straight base line). It
found *zero* walls on the courtyard, and the diagnostic showed why: wall and floor
are the **same colour classes** here (both teal tilework; `classify` puts both in
'water'), and each class fragments into lace across the other, so no per-class
component ever forms a wall silhouette. The portable signal is geometric instead:
the top row of the *floor sheet* — the horizontal stone/ground/shadow band at the
wall base — is the wall/floor junction per column (courtyard: median row 213,
p10–p90 192–228). WALL = plate material above it, per-column base_y = the junction
row (the origin's per-column-bottom rule, expressed on a portable line); uncovered
columns are interpolated and the line median-smoothed (strided NaN-median, ~30 ms,
instead of the origin's python double loop).

**Problems hit + fixes.** (1) Tile outlines and cast-shadow specks are not plate
classes, so a wall came out of the class step as lace (erode-5 left 1.7k of 48k px)
→ two 5×5 box-majority passes close surfaces; NONE then inherits base_y from its
neighbours (106 px left unlabelled on the courtyard). (2) `np.where(mask, rows,
nan).min(0)` is all-NaN per column — NaN propagates through min; sentinel row +
convert afterwards. (3) the floor-sheet component min size must scale with the frame
(0.2%): a fixed 500 px found nothing on the 96×128 synthetic room, where objects
standing in the band cut it into ~150 px pieces. (4) `3*base_y − centroid` looked
like the origin formula averaged over the body; it is the feet formula with the
wrong y — documented as §2.9 rather than silently changed.

**Measured and rejected: fill axis from orientation.** The obvious consumer — holes
in a wall fill top-to-bottom, holes in a floor keep M4 — was prototyped and measured
with punch-hole reconstruction error (boxes punched into clean surface, refilled,
compared to the art underneath): wall holes h=30.85 / m4=24.34 / v=24.58, floor holes
h=44.23 / m4=34.51 / v=34.46. Vertical fill does **not** beat M4 on walls: tilework
is patterned in both axes and a lerp reconstructs pattern on neither, so the axis
choice moves the error by noise. `animate.inpaint` is untouched; `depth.hole_axis`
stays for M6b/c, and M6c's shading-aware inpaint (orientation + inferred light
direction) is the version of the idea worth building. Numbers reproducible via
`python3 app/tests/depth_probe.py`.

**Faithful-port check.** Courtyard field vs the origin `depth.npy`: spearman ρ=0.556
(rank agreement; the origin map additionally carries per-component wall structure the
single-junction-line model deliberately drops), field cost 245 ms on 384×512 — inside
the M5 budget share, cached implicitly because upload computes it once.

**Next.** M6b: overlay toggle in the canvas, per-region surface type in the
inspector, split-at-depth-break edge op, depth-aware wand. Then M6c, then Phase 2.

---

## 2026-10-30 — Entry 12: M5 complete (perf, guard, docs, atlas + two real fixes)

**Perf pass** (budget: 48f @ 384×512 < 10 s): two changes.
1. `_Blit` — cached flat-index blitter. Per (runs, dx, dy) the (src, dst) pixel indices
   are computed once; a frame now costs a few fancy-index assignments per piece instead
   of a python loop over runs (`_put_runs` deleted). Frames whose offset pattern repeats
   are memoised by offset tuple. build_sprites on the 226-unit fern scene: ~9 s → 0.3 s.
2. `_split_leaf_units` — the M3 port approximated the origin's multi-source BFS with
   synchronous max-label propagation over the FULL canvas truncated at 4096 rounds:
   8.5 s of a 10.6 s render on fractal fern masks, and unfaithful (far pixels took the
   max label, arbitrary cutoff). Restored the origin rule (blob ≥150 px → erode →
   cores ≥6 px → nearest-core BFS, 8-neighbour, ties → lower core index) as synchronous
   rounds confined to the bbox, padded so erode4's off-canvas=True edge semantics match
   the full canvas (entry-bbox rows would otherwise survive erosion and fuse cores).
   Worst case now **1.12 s**, audit 0, deterministic.

**Critical bug found by the M5 parity harness (old-vs-new frame hashes):** M3/M4's
`_put_runs` computed `s0 = sa + (c0 - da)` from cell-relative `sa = a - ox` — sprites
were painted from the art at `x - cell_x` (left-edge streaks), not from the object's own
pixels. Every prior check was self-consistent (audit 0, recomposite parity, determinism)
so nothing caught it; demo GIFs shipped with wrong textures. `_Blit` samples absolute
source coords and is verified pixel-identical to a reference slice blit for all 35
offsets incl. clipping; test_m5 adds the static-texture regression check.

**Shadow-sync visibility fix:** render() inpainted only masks, so a mover's claimed ring
repainted pixel-identical copies of its own static original in the plate — sync was
"invisible" (the 56→104 motion number in M4 was texture-bug garbage riding the ring).
The plate now inpaints claimed rings too; M4's motion check reads 56→84 with real ring
pixels and still passes (pots keep theirs: static_keep unchanged).

**Server:** MILESTONE M5; `engine` content hash in /api/ping and in the render cache key
(a code upgrade can never serve stale frames); concurrency guard — heavy endpoints take
their pid lock (same-project requests serialise, the second rides the cache) then one of
2 global slots, 429 `{error, retry}` after a 90 s wait; autoseg takes the pid lock only.
UI: epoch guard fixes a real race (delete during an in-flight openProject re-hid the
empty-state hint — the flaky ui_smoke delete check), docs button 📖, bundle status shows
atlas pages.

**Docs page:** `app/static/docs.html` served at `/docs` — quick start, catalog, shadow
sync, bundle + atlas spec, API table, engine rules, shortcuts, troubleshooting.

**Atlas packing (plan §8):** `export.pack_atlas` shelf-packs sheets into square pages
(≤4096, 1 px pad, tallest-first); bundle carries `atlas_<i>.png` + `scene.json.atlas`
(pages) + per-object `atlas {page,x,y,w,h}`; crops are pixel-exact sheets and an
atlas-only recomposite equals render frames exactly (test_m5). Oversized sheets keep
their standalone file (rect None). `atlas=0` reproduces the M4 layout byte-for-byte
layout-wise. Courtyard bundle 1.51 → 2.84 MB with the atlas; GIF unchanged 0.79 MB.

**Verification:** test_m5.py (34 checks: blit parity, texture, leaf units, shadow hole,
perf budget, cache flag, concurrency burst, /docs, atlas) + ui_smoke 78 checks + m2/m3/m4
all green. Demo bb780421 regenerated: render 1172 ms audit 0, GIF 0.79 MB, bundle
2.84 MB / 56 sheets / 1 atlas page; the frame-0 spot check shows plants in their own
textures for the first time.

## 2026-10-30 — Entry 11: M4 complete (polish + shadow sync + inpaint pass)

- **shadow sync** (plan §3, opt-in per object): the hug ring = dark cool/neutral pixels
  (`val<0.32` and (`hue≥150` or `sat<0.35`) — measured against the origin shadow class,
  which keeps brown door crevices and dark foliage out) within dilate-2 of the mask,
  unowned by anyone else. First claimer in z order wins; **pots/statics keep theirs**
  via a static_keep exclusion. Ring components ride their nearest moving piece (sway:
  the unit's top-chunk dx; bob/flutter: that comp; gustflick: that flick window). The
  ring joins the object footprint, so the plate inpaints under it and the sheets carry
  it — parity invariant intact (test recomposites exactly with shadows on). plan §8's
  *separate* shadow sprite stays deferred: with sync on, the ring is baked into the
  object sheet; noted here so nobody re-litigates it.
- **inpaint vertical-blend pass** (plan §7 risk 1): pixels in long horizontal runs
  (>10 px) are refilled with an inverse-distance blend of the four run-boundary
  colours. Lesson: averaging an h-fill and a v-fill just trades row banding for column
  banding (measured both); the four-boundary blend cuts hard interior row steps
  1003 → 619 and mean row diff 74 → 43 on the courtyard hole, plate still exact outside.
- **projects**: rename/duplicate/delete routes behind a hex-pid guard (rmtree only ever
  under `projects/`), names in `/api/projects` — which now sorts by mtime (it was
  alphabetical, so the "recent 12" dropdown showed the wrong twelve once test runs
  accumulated projects); header ✎ ⧉ 🗑 buttons; deleting the open project resets the
  editor to the empty state.
- **presets**: built-ins per anim type (breeze/windy/storm, calm/chop, …) + user presets
  saved to localStorage, applied from the inspector select.
- **palette option** now feeds GIF *and* bundle; scale/colours persist per browser.
- **toasts + shortcuts**: every failed API call toasts (status line keeps the detail);
  `?` help overlay; P play/pause, `,` `.` step, G / ⇧G exports, Delete, Esc.
- **bug ledger**: `do_POST` answered 404 on unknown routes *without draining the body* —
  with HTTP/1.1 keep-alive that desynced the connection and a later response parsed as
  HTML (surfaced the moment the toast test started POSTing to a bogus route). Bodies are
  drained up front now; `_upload` reads the drained bytes. Also: `api()` without a body
  sends GET — the duplicate button needed an explicit `{}`.
- tests: `test_m4.py` 23 checks (inpaint metrics, shadow motion + static-keeps-its-shadow,
  bundle parity with shadows, project CRUD + hostile pids, palette sizes); ui_smoke grew
  to 75 checks (prompt/confirm stubs). Courtyard demo with shadow on 26 objects: audit 0,
  GIF 0.76 MB, bundle 1.51 MB / 56 sheets.
- Next: **M5 hardening** — perf pass (render <10 s for 48f @ 384×512), concurrency guard,
  docs page, atlas packing.

---

## 2026-10-30 — Entry 10: M3 complete (engine + export)

Sandbox had reset again — Setup line first. Then M3 on the M2 editor: new
`app/engine/animate.py` (generalised render engine) + `app/engine/export.py`
(strip/GIF/bundle), server routes `/api/render`, `/api/gif`, `/api/bundle` +
`GET /api/out/<pid>/…`, editor timeline bar + anim inspector, `app/tests/test_m3.py`
and an M3 section in `ui_smoke.js`.

- **Parity by construction** — the big decision: the renderer builds per-object
  per-frame RGBA *sprites* (fixed cell + native anchor) and composites frames ONLY by
  pasting them on the inpainted plate in z order; the bundle exports those same cells
  as sheets. Preview/GIF/bundle cannot disagree, and "bundle re-imports with clean RGBA
  edges" is an invariant: the test recomposites plate+sheets at anchors and compares
  frames 0/17/31/47 to the rendered strip **exactly**. Sheets zero RGB under alpha 0.
- Engine rules, structural now: 3 (full sprite pasted every frame — the disappearing
  lily cannot happen), 4 (mover vacated px = transparent alpha → composite shows
  `fr_bg`), 5 (integer cycles; gust windows placed so frames N−1 and 0 both rest),
  6 (articulated base/mid/top chunks; foliage ≥600 px erode-splits into leaf units;
  <8 px units ride static), 7 (montage palette once, dither NONE, gifsicle −O3),
  8 (audit includes the N−1→0 wrap).
- Sprite source of an object excludes pixels a higher-z object covers (the "plant over
  water dissolved" fix, generalised): pool ripple can never move a leaf pixel.
- ripple keeps a constant footprint (shifted runs clip to its mask — origin's
  "ripple edge writes spilled" fix) and its vacated px show its own static water, so
  the inpainted plate hole under a pool is never visible.
- hidden masks + regions don't participate (no inpaint/draw/sheet): eye-off means "not
  an object". walkmask default = not under a non-walkable-on object (rugs stay
  walkable) and not under water/shadow regions; region tools = M4.
- determinism: per-object RNG = crc32(id|anim|params) → byte-identical renders; server
  caches a render per pid keyed by doc fields + mask mtimes (gif/bundle reuse it).
- params: `PARAM_DEFS` whitelist lives in the engine, mirrored by inspector sliders;
  server clamps on upsert *and* render. Recognition hints coerce to catalog types
  (`HINT_MAP`), raw hint kept as `anim_hint` for display. New uploads default N=48/fps=12
  (loop semantics; M1's N=8 was the one-shot clip budget).
- UI: inspector gained anim dropdown + per-type sliders + phase-auto; bottom timeline
  plays the strip on a preview canvas (rAF at target fps) with scrub, N/fps, GIF
  scale/colours, Render / Export GIF / Export bundle + audit line.
- measured (test_m3.py, 35 checks): 7-type synthetic art moves per zone with audit 0
  and static held; courtyard 56 auto-seg objects → 48f render 1.1 s, GIF 0.72 MB,
  768×1024, loop=0; bundle 56 sheets, recomposite exact. ui_smoke 57 checks incl. the
  render→play→scrub→export path (jsdom gets an Image stub that fires onload, anchor
  clicks stubbed so downloads don't navigate).
- bug ledger: shimmer's `maxsh=0` first left its own mask out of the audit's allowed
  zone (82 "stray" px on-mask); `_Unit` got a raw bool instead of a selector when the
  unit wasn't chunked; GIF smoke wait-regex matched the "encoding GIF…" status itself.
- `app/tests/seed_demo.py` seeds a runtime demo project (upload origin art → import the
  recognised objects); runtime data, not committed. test_m2's ping check now accepts M2+.
- Next: M4 polish — save/load UX, presets, shadow-sync toggle, palette size option,
  toasts, shortcuts; vertical-blend inpaint pass is a candidate here (plan §7 risk 1).

---

## 2026-10-30 — Entry 9: object recognition ("what IS it?")

User ask: can the segmenter also *recognise* the objects, not just find them. Yes — new
`app/engine/recognise.py`, rule-based over features the segmenter already pays for
(geometry + palette + a little context), ~1 ms per object:

- features: bbox aspect / fill / solidity (hole-fill ratio), mass-in-top-third, eroded
  green unit count ("lace"), blossom/brown/dark/bright fractions, holes, fragments,
  trunk-in-mask **and brown-just-below-the-feet**. That last one mattered: trunk and
  planter rims are *brown class*, so they are never inside a green plant mask — the first
  rule version looked for trunks inside the mask and missed every palm.
- rules (first match wins, each with a confidence): pool (interior water objects only, by
  construction) · wall vs floor tiles (height in frame) · pot → **potted plant** when a
  foliage band sits on its rim · carpet aspect → runner rug vs rug · foliage: flowering →
  palm (tall + trunk below + top-heavy crown) → potted foliage → fern (lacy, many eroded
  units) → grass/reeds → shrub → fallback `plant` at conf 0.4.
- naming: suggestions are now named by recognised label + per-label index (`fern 3`,
  `potted plant 7`); the material class stays as `cls` for the machinery; cfg
  `recognise=false` restores class names. Each label carries an advisory M3 anim hint
  (`palm -> sway (articulated fronds)`, `lily pad -> bob`, …) shown in the UI.
- UI: modal rows show the label with a confidence chip and the anim hint as tooltip; the
  inspector shows recognised label/conf/hint; settings gained a *recognise objects* toggle.
- validation: hand-labelled fixture `app/tests/recog_fixture.json` (10 objects read off the
  art: 2 palms, 2 ferns, 2 rugs, 1 runner, 2 lily pads, the pool) — **10/10**, plus a scene
  inventory sanity check. Two GT pots were left out of the fixture on purpose: the origin
  labelling's own carpet hole-fill swallowed them, which is a segmentation disagreement,
  not a recognition one. Low-confidence fallbacks are honest by design (plan §7).
- tests: test_m2.py 37 checks, ui_smoke.js 36 checks, all pass.

---

## 2026-10-30 — Entry 8: M2 follow-up (autoseg UX + solid op)

User feedback on the M2 editor, three changes:

- **Auto-seg parses the loaded image and presents a choose-from list.** A freshly loaded
  image (no masks yet) is now segmented automatically on open and the modal opens with the
  recognised objects; re-opened projects with masks don't re-run (toggle: *auto-run on
  load*). The list is grouped — *recognised objects* (ticked) and *plate regions*
  (unticked) — with per-group tick-all/none, thumbnails, and **click a row to preview that
  suggestion on the canvas** before importing.
- **Auto-seg settings** (modal → `settings`, persisted in localStorage, map 1:1 onto the
  engine cfg): max suggestions, min object px, merge distance, absorb gap, silhouette
  growth, connectivity, offer-regions, auto-run. `re-run` applies them. Server whitelists
  and type-coerces cfg against `segment.DEFAULTS` (`_clean_cfg`), so hostile/garbage values
  can neither crash it nor inflate the workload (tested). Semantics note: `min_object_px`
  is a *relevance* filter (drops tiny candidates per kind), while the cap's overflow is
  *absorbed* — raising the floor therefore changes the mix, not necessarily the count.
- **`solid` edge op, button right next to `outline`** (inspector): the inverse of outline —
  `close(n)` + hole-fill, i.e. a filled silhouette that bridges ≤2n px gaps and fills every
  hole, without gluing disconnected parts together (that is what the acceptance test pins:
  3 comps → 2, gap column and enclosed hole both filled).
- Tests grew with the features: `test_m2.py` now 33 checks (settings reach the engine,
  cap honoured, cfg coercion, solid), `ui_smoke.js` 35 checks (grouped list, defaults ticked,
  settings panel + persistence, row preview, re-run cap, second boot auto-parse). All pass.

---

## 2026-10-30 — Entry 7: M2 complete (masking)

Sandbox had reset (no numpy/PIL/gifsicle) — re-ran the Setup line first. Then built M2 on
the M1 skeleton: `app/engine/` (new package), masking API on the server, and the full
editor UI. Acceptance (plan §6) now automated in `app/tests/test_m2.py` (HTTP end-to-end)
plus `app/tests/ui_smoke.js` (jsdom + live server, drives the editor like a user).

- **Engine** (`app/engine/imaging.py`): grid detect (M1), palette snap, HSV, 4-neighbour
  morphology, run-based union-find components, scanline flood fill, row-run RLE, and the
  wand. All integer-pixel at native res (plan §2).
- **Wand** — the plan's "one click grabs a fern at tol 30" is impossible with an RGB
  tolerance on this art: measured, the fern spans Chebyshev 155 *inside itself* while its
  shadow ring sits 27 from the seed (i.e. background is closer than the fern's own
  highlights), and tol 45 already swallowed 12% of the frame. So the default metric is a
  **hue-family flood** (circular hue ≤ tol, sat ≥ 0.30, val ≥ 0.14 — pixel-art shading
  swings value while hue stays put): fern = 57% of the object, 1.2% of frame, one click.
  A **leak guard** tightens to the seed's shade family when a grab exceeds 12% of the
  frame (terracotta pots and wood doors are one hue family; guarded pot = 94% of the pot,
  0.9% of frame). Classic seed-tolerance wand kept as `mode:'seed'` (+ palette snap).
- **Auto-seg** (`app/engine/segment.py`): generic port of the origin courtyard pipeline —
  its hardcoded pool/wall polygons and floor-patch colour sampling replaced by portable
  rules. Scores **85.9% recall of the 85 courtyard objects ≥25px as exactly 60 suggestions
  in ~1.3 s** (target ≥80% / ≤60). Ground truth = origin `seg.npy`; harnesses
  `tests/{gt_stats,cov,sweep,seg_dev}.py`. Lessons baked in as code + comments:
  1. classes decided per *palette entry*, not per pixel (dither can't shred an object);
  2. a colour class covers only ~50% of an object's pixels → every core is **grown**
     (dilate → hole-fill → erode, bbox-windowed, ratio-capped);
  3. one object = several fragments → same-class fragments within `merge_k` are merged,
     under a per-kind size budget (unchained merges produced 12%-of-frame "plants");
  4. cap policy: slots per kind, and over-cap candidates are **absorbed** into their
     nearest same-kind neighbour, never dropped (a size-only cap threw away every pot and
     pad: recall 80% → 68%; absorption keeps 86%); absorbed masks carry `nfrag`, the UI
     shows "×N" so the user can split (plan §8 combine-or-split);
  5. big+solid = background sheet (replaces the hardcoded floor patches); water touching
     the frame border is material, interior water is a pool, and only pools hold pads.
- **Perf**: windowed morphology took auto-seg from 20.5 s to 0.65 s (hole-filling inside a
  bbox window instead of the whole frame, per candidate).
- **Server**: `/api/wand`, `/api/autoseg`, `/api/masks` (upsert/delete + PNG per mask),
  `/api/edge` (grow/shrink/outline/hole-fill/close), `/api/project` save, `/api/projects`
  list; masks travel as row-run RLE; project doc gained the plan §8 object fields
  (kind, z, visible, depth, offset, clips reserved).
- **Editor** (`app/static/index.html`, still one file, no CDN): brush/eraser/rect/lasso/
  wand/pan, nib 1–16, tolerance + 4/8-conn + metric toggles, Shift inverts add/subtract,
  onion-skin overlay with white rim on the selection, mask list (rename dbl-click, recolour,
  visibility, z up/down, dup, del), inspector with edge helpers, auto-seg modal with
  thumbnails + kind-aware default ticks, undo/redo (Ctrl+Z / Ctrl+Shift+Z), debounced saves,
  `?p=<id>` reopen + projects dropdown. Masks = Uint8Array + tinted native-res canvas each;
  the overlay is composited once per edit, not per frame.
- **Bug ledger**: nonlocal-augmented-assign on numpy masks inside nested helpers (twice);
  run-window index math for the (dropped) chained flood was off by the direction sign;
  leak guard compared the *raw* flood before close+hole-fill inflated it 23k → 30k px.
- Tests: `python3 app/tests/test_m2.py` (27 checks, all pass) and
  `NODE_PATH=<jsdom> node app/tests/ui_smoke.js` (23 checks, all pass; jsdom is dev-only,
  node_modules stays out of the repo). `app/projects/` is now gitignored runtime data
  (M1's four seeded examples stay tracked).
- Next: M3 engine + export (7 anim types, frame-strip preview, GIF + scene bundle).

---

## 2026-10-30 — Entry 6: M1 hotfix (no-server-context guard)

- User hit `TypeError: Failed to parse URL from /api/upload` → page was opened from the
  **static file viewer** (sandboxed iframe, no network, opaque base URL), not the live preview.
- Fix: frontend now checks `location.protocol` (http/https) and pings `GET /api/ping` on load;
  without a server context it shows guidance (“open via live preview, port 8000”) instead of
  cryptic fetch errors; upload() refuses early with the same guidance.
- Server: added `GET /api/ping` → `{ok, milestone}`.
- Re-verified: ping ok, upload 698 ms, grid 4 → 384×512.

---

## 2026-10-30 — Entry 5: M1 complete (skeleton)

- Built `app/server.py` (stdlib ThreadingHTTPServer, no framework deps) + `app/static/index.html`
  (single-file editor shell: zoom/pan canvas, pixel-crisp, tool rail placeholders for M2).
- Endpoints: `POST /api/upload` (grid detect + native downsample + project dir),
  `GET /api/image/<id>/{native,original}.png`, `GET /api/project/<id>`, `GET /` (editor).
- Projects live in `app/projects/<id>/` (original.png, native.png, project.json with
  objects/regions/N/fps/palette fields reserved per plan §8).
- Run: `python3 app/server.py 8000` (binds 0.0.0.0; live preview port 8000).
- Bugs found & fixed during acceptance: (1) grid detector returned p=1 (trivially uniform) —
  now returns the **largest** uniform block size; (2) image route matched 3 path segments
  instead of 4 → 404s.
- Acceptance: 1536×2048 upload → grid 4 → native 384×512, round-trip 696 ms (<2 s) ✔;
  routes 200/404 as specified ✔.
- Next: M2 masking (brush/eraser/rect/lasso/wand + mask list + auto-seg import).

---

## 2026-10-30 — Entry 4: game-bound direction change

- User plans to use scenes/assets in a pixel-art adventure game → primary output becomes the
  **game bundle** (object RGBA sheets + clip metadata + plate/walkmask/palette JSON); GIF is
  preview-only. Masks promoted to *objects* (anchor, iso depth sort key, clips).
- Origin session's segmentation + depth map = ready-made object list + sort keys; shadow-sync
  returns later as optional separate shadow sprites.
- plan.md amended: locked scope, new §8 game pipeline, M3–M5 updated (bundle export, region
  tools, atlas/adapters in M5).
- Open questions resolved: target engine = **custom web JS** (plain JSON + PNG sheets, no
  adapters); recomposition = **phase 2** (offset(x,y) reserved in object model now);
  default clip budget proposed: 8 frames @ 8–12 fps.
- Next: M1 skeleton on user's go-ahead.

---

## 2026-10-30 — Entry 3: docs first (app kick-off)

- Scope locked via user choices: web app in this workspace; pixel art only; masking =
  manual + wand + auto-seg; **GIF-only export for now**.
- Wrote `plan.md` (architecture, engine rules, anim catalog, UI layout, milestones M1–M5,
  risks) and initialized this log. **No app code yet** — docs were requested first.
- Next: M1 skeleton (server + upload + grid detect + zoomable canvas).

## 2026-10-30 — Entry 2: origin session wrap-up (the proof-of-concept)

Fully hand-built animation of `uploads/iso_00080_.png` (Persian courtyard pixel art):
- Pipeline proven end-to-end: grid detect (4× blocks → 384×512 native) → masks → per-object
  motion → 48-frame seamless loop → 128-color shared palette → gifsicle -O3 (0.86 MB GIF)
  + ffmpeg MP4 (0.79 MB).
- Delivered extras: `segmentation_mask.png` (123 classes), `depth_map.png` (iso geometry:
  closeness = 3·base_y − y), `contact_sheet.png`, `seg_legend.json`.
- Shadow-sync animation was implemented (dark pixels hugging movers, pots keep theirs static)
  then **rolled back at user request**; code knowledge retained for the app's opt-in toggle.

### Bug ledger (→ became Engine Rules in plan.md)

| symptom | cause | fix |
|---|---|---|
| "screen tearing wipe" on plants | per-row shear with traveling phase lag | articulated rigid chunks, fixed joints |
| palm "not moving" | 7.5k-px blob as one ±1 px unit | erosion-split into frond units; ±2 px for big units |
| fern "whole object" motion | connected component = one unit | leaf-unit splitting (erode → cores → BFS reassign) |
| lilies "flashing" (1st report) | shimmer hit blossoms/pad rims; ripple edge writes spilled | shimmer/ripple exclusion rings; eroded ripple mask |
| lilies "lines" | pad highlight/rim px outside pad mask rippled as water | morphological close + hole-fill of pad sprites |
| lilies glitch (final) | pad draw skipped when bob offset rounded to (0,0) | redraw every mask every frame, always |
| plant over water dissolved | dx=0 chunks not redrawn over inpainted base | full redraw per chunk incl. offset 0 |
| animated px on walls/floor | misclassified specks swaying | drop units <25 px isolated from foliage; audit metric = 0 |
| BFS hang (shadow pass) | STATIC label −2 passed `<0` "unlabeled" test | label test `== -1` |

## 2026-10-29 — Entry 1: origin session start

- User asked to animate the courtyard image as a pixel-art GIF (sway, ripple, pad bob).
- Built masks by color rules + geometry; discovered art has ~19.7k near-duplicate colors
  (dither noise) → palette-snap to 128 colors both denoises and shrinks GIFs.
- Export chain settled: PIL quantize(montage) → per-frame `quantize(palette=…, dither=NONE)`
  → ×2 NEAREST → gifsicle `-O3` (10.8 MB → 0.8 MB lossless).
