# Pixel Sway — build log.md

Living log of the app build. Newest entries on top. Keep entries short: what changed,
decisions taken, problems hit + fixes, and anything a future session must know
(the sandbox resets installed packages between sessions — see Setup).

---

## Setup (run after any sandbox reset)

```bash
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  python3-pil python3-numpy gifsicle   # add ffmpeg when MP4 export lands
```

Workspace root holds: `plan.md`, `build_log.md`, app code (`app/`), and the reference
artifacts from the origin session (`courtyard_animated.gif/.mp4`, `segmentation_mask.png`,
`depth_map.png`, `seg_legend.json`, `contact_sheet.png`, `uploads/iso_00080_.png`).

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
