/**
 * Pixel Sway — M2 editor smoke test (jsdom + the real server).
 *
 * Boots app/static/index.html in jsdom with a stubbed 2D canvas context and a fetch that
 * proxies to a live server, then drives the editor the way a user would: open a project,
 * paint a stroke, fire the wand, run auto-seg (settings, grouped list, preview, import),
 * save. A second boot checks that a freshly loaded image is parsed automatically and the
 * choose-from list is presented. Any uncaught exception or failed assertion fails the run.
 *
 * Needs:  npm i jsdom   (dev-only; node_modules stays OUT of the repo)
 * Run:    NODE_PATH=<dir with jsdom> node app/tests/ui_smoke.js [port]
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const { JSDOM } = require('jsdom');

const ROOT = path.resolve(__dirname, '..', '..');
const PORT = process.argv[2] || '8141';
const BASE = `http://127.0.0.1:${PORT}`;
let fails = 0;
const errors = [];
function check(name, ok, detail) {
  console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${name}${detail ? '  — ' + detail : ''}`);
  if (!ok) fails++;
}
const wait = ms => new Promise(r => setTimeout(r, ms));

function stubCtx(cv) {
  if (cv.__ctx) return cv.__ctx;
  const ctx = {
    canvas: cv, globalAlpha: 1, fillStyle: '', strokeStyle: '', lineWidth: 1,
    imageSmoothingEnabled: false,
    setTransform() {}, clearRect() {}, fillRect() {}, strokeRect() {},
    drawImage() {}, beginPath() {}, moveTo() {}, lineTo() {}, closePath() {}, stroke() {},
    createImageData(w, h) { return { width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }; },
    putImageData(img) { cv.__img = img; },
    getImageData(x, y, w, h) {
      if (cv.__img && cv.__img.width === w && cv.__img.height === h) return cv.__img;
      return this.createImageData(w, h);
    },
  };
  cv.__ctx = ctx;
  return ctx;
}

function bootDom(url) {
  const html = fs.readFileSync(path.join(ROOT, 'app', 'static', 'index.html'), 'utf8');
  return new JSDOM(html, {
    url,
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    beforeParse(window) {
      window.Element.prototype.setPointerCapture = function () {};
      window.Element.prototype.releasePointerCapture = function () {};
      window.Element.prototype.getBoundingClientRect = function () {
        return { width: 800, height: 600, left: 0, top: 0, right: 800, bottom: 600, x: 0, y: 0 };
      };
      window.HTMLCanvasElement.prototype.getContext = function () { return stubCtx(this); };
      // M3: the preview strip loads through Image(); jsdom has no image loader, so the
      // stub fires onload like a browser would (playback then runs on stubbed canvases).
      window.Image = class {
        constructor() { this.onload = null; this.onerror = null; }
        set src(v) { this._src = v; setTimeout(() => { this.width = 3072; this.height = 3072;
          if (this.onload) this.onload(); }, 10); }
        get src() { return this._src; }
      };
      window.HTMLAnchorElement.prototype.click = function () { this.__clicked = true; };
      window.prompt = (q, d) => (window.__prompt === undefined ? d : window.__prompt);
      window.confirm = () => !!window.__confirm;
      window.createImageBitmap = async blob => {
        const buf = Buffer.from(await blob.arrayBuffer());
        return { width: buf.readUInt32BE(16), height: buf.readUInt32BE(20), close() {} };
      };
      window.fetch = (u, o) => fetch(new URL(u, BASE), o);
      window.addEventListener('error', e => errors.push('window error: ' + e.message));
    },
  });
}

async function main() {
  // ---- live server + a project with one mask -----------------------------
  const srv = spawn('python3', [path.join(ROOT, 'app', 'server.py'), PORT],
    { cwd: path.join(ROOT, 'app') });
  let up = false;
  for (let i = 0; i < 60 && !up; i++) {
    try { up = (await fetch(BASE + '/api/ping')).ok; } catch (e) { await wait(250); }
  }
  if (!up) { console.error('server did not come up'); process.exit(1); }

  const img = fs.readFileSync(path.join(ROOT, 'uploads', 'iso_00080_.png'));
  const meta = await (await fetch(BASE + '/api/upload', { method: 'POST', body: img })).json();
  const rle = [];
  for (let y = 100; y < 140; y++) rle.push(y, 60, 110);
  await fetch(BASE + '/api/masks', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pid: meta.id, ops: [{ op: 'upsert', id: 'seed1', name: 'seed mask', kind: 'object', color: [200, 80, 80], rle }] })
  });
  const fresh = await (await fetch(BASE + '/api/upload', { method: 'POST', body: img })).json();

  // ---- jsdom boot ---------------------------------------------------------
  const dom = bootDom(BASE + '/?p=' + meta.id);
  dom.window.addEventListener('unhandledrejection', e => errors.push('rejection: ' + e.reason));
  process.on('uncaughtException', e => errors.push('uncaught: ' + e.message));

  const win = dom.window, doc = win.document;
  const key = (k, opt = {}) => win.dispatchEvent(new win.KeyboardEvent('keydown',
    Object.assign({ key: k, bubbles: true, cancelable: true }, opt)));
  const click = id => doc.getElementById(id).dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
  const pev = (type, x, y, opt = {}) => doc.getElementById('view').dispatchEvent(
    new win.MouseEvent(type, Object.assign({ clientX: x, clientY: y, button: 0, bubbles: true }, opt)));

  for (let i = 0; i < 80 && !(win.__PS && win.__PS.meta); i++) await wait(100);
  const S = win.__PS;
  const PV = win.__PV || {};
  for (let i = 0; i < 80 && doc.querySelectorAll('#list .row').length < 1; i++) await wait(100);
  check('editor booted and opened the project', !!S && !!S.meta && S.meta.id === meta.id,
    S ? `project ${S.meta && S.meta.id}` : 'no __PS');
  check('native size from server', S && S.W === 384 && S.H === 512, S ? `${S.W}x${S.H}` : '');
  check('existing mask loaded into the list', S && S.masks.length === 1 && S.masks[0].id === 'seed1',
    S ? `masks=${S.masks.length}` : '');
  check('mask list rendered a row', doc.querySelectorAll('#list .row').length === 1);
  check('inspector rendered', doc.getElementById('insp').textContent.length > 0);
  check('no boot errors', errors.length === 0, errors.join('; '));

  // ---- paint a stroke (creates a new mask, since none selected) -----------
  key('1');                                   // native 1:1, origin 0,0
  pev('pointerdown', 60, 60);
  pev('pointermove', 70, 66);
  pev('pointermove', 80, 70);
  pev('pointerup', 80, 70);
  check('brush stroke created + painted a mask', S.masks.length === 2 && S.masks[1].px > 0,
    `masks=${S.masks.length} px=${S.masks[1] && S.masks[1].px}`);
  check('stroke marked dirty for saving', S.dirty.size === 1);

  // ---- eraser subtracts ----------------------------------------------------
  const before = S.masks[1].px;
  key('e');
  pev('pointerdown', 62, 62); pev('pointermove', 66, 64); pev('pointerup', 66, 64);
  check('eraser removed pixels', S.masks[1].px < before, `${before} -> ${S.masks[1].px}`);

  // ---- M6c: icons + the select tool ---------------------------------------
  const toolBtns = [...doc.querySelectorAll('#rail .tool')];
  check('every tool button carries an inline SVG icon',
    toolBtns.length >= 7 && toolBtns.every(b => b.querySelector('svg') && b.dataset.tool),
    `${toolBtns.filter(b => b.querySelector('svg')).length}/${toolBtns.length} with icons`);
  check('the rail offers a select tool', !!doc.querySelector('#rail .tool[data-tool="select"]'));
  key('v');
  check('V switches to the select tool', S.tool === 'select', S.tool);
  const topAt = (x, y) => { for (let i = S.masks.length - 1; i >= 0; i--)
    if (S.masks[i].visible !== false && S.masks[i].data[y * S.W + x]) return i; return -1; };
  const near1 = (x, y) => { for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
    const xx = x + dx, yy = y + dy;
    if (xx >= 0 && yy >= 0 && xx < S.W && yy < S.H && S.masks[1].data[yy * S.W + xx]) return true;
  } return false; };
  // the seeded mask of this project is empty, so the stroke we just drew is the target
  let pick = null, miss = null;
  for (let y = 0; y < S.H && !(pick && miss); y++)
    for (let x = 0; x < S.W; x++) {
      if (!pick && topAt(x, y) === 1) pick = { x, y };
      else if (!miss && topAt(x, y) === -1 && near1(x, y)) miss = { x, y };
      if (pick && miss) break;
    }
  check('found a painted pixel and an empty one 1-3px away', !!pick && !!miss,
    `painted=${pick && pick.x},${pick && pick.y} empty=${miss && miss.x},${miss && miss.y}`);
  const at = p => [Math.round(S.ox + (p.x + 0.5) * S.zoom), Math.round(S.oy + (p.y + 0.5) * S.zoom)];
  S.sel = 0;
  pev('pointerdown', ...at(pick)); pev('pointerup', ...at(pick));
  check('the select tool picks the mask under the cursor', S.sel === 1 &&
    doc.getElementById('iName').value === 'mask 2',
    `sel=${S.sel} name="${doc.getElementById('iName').value}"`);
  S.sel = 0;
  pev('pointerdown', ...at(miss)); pev('pointerup', ...at(miss));
  check('a plain click on empty art changes nothing', S.sel === 0, `sel=${S.sel}`);
  pev('pointerdown', ...at(miss), { shiftKey: true }); pev('pointerup', ...at(miss), { shiftKey: true });
  check('shift-click ring-searches the 3px neighbourhood', S.sel === 1, `sel=${S.sel}`);
  S.masks[1].visible = false; S.sel = 0;
  pev('pointerdown', ...at(pick)); pev('pointerup', ...at(pick));
  check('a hidden mask is not pickable', S.sel === 0, `sel=${S.sel}`);
  S.masks[1].visible = true; S.sel = 1;
  key('b');
  check('B leaves the select tool for the brush', S.tool === 'brush', S.tool);

  // ---- rect + lasso + undo -------------------------------------------------
  key('r');
  pev('pointerdown', 200, 200); pev('pointermove', 220, 215); pev('pointerup', 220, 215);
  const afterRect = S.masks[1].px;
  check('rect fill added pixels', afterRect >= 20 * 15, `px=${afterRect}`);
  key('l');
  pev('pointerdown', 300, 300); pev('pointermove', 320, 305); pev('pointermove', 310, 320);
  pev('pointerup', 310, 320);
  check('lasso fill added pixels', S.masks[1].px > afterRect, `${afterRect} -> ${S.masks[1].px}`);
  const preUndo = S.masks[1].px;
  key('z', { ctrlKey: true });
  check('undo restores the mask', S.masks[1].px === afterRect, `${preUndo} -> ${S.masks[1].px}`);
  // M6c: the same history is reachable from header buttons, not just ctrl+Z
  const hundo = doc.getElementById('hundo'), hredo = doc.getElementById('hredo');
  check('header carries undo + redo buttons', !!hundo && !!hredo &&
    hundo.title.toLowerCase().includes('undo') && hredo.title.toLowerCase().includes('redo'),
    `undo="${hundo && hundo.title}" redo="${hredo && hredo.title}"`);
  hredo.dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
  check('redo button replays the lasso', S.masks[1].px === preUndo,
    `${S.masks[1].px} vs ${preUndo}`);
  hundo.dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
  check('undo button steps back again', S.masks[1].px === afterRect, `${S.masks[1].px}`);
  check('undo/redo buttons mirror the history depth',
    hundo.disabled === !S.undo.length && hredo.disabled === !S.redo.length &&
    hundo.title.includes(`${S.undo.length} step`) && hredo.title.includes('1 step'),
    `undo=${S.undo.length} redo=${S.redo.length} "${hundo.title}" / "${hredo.title}"`);

  // ---- wand ----------------------------------------------------------------
  key('w');
  pev('pointerdown', 219, 181);               // the courtyard fern
  pev('pointerup', 219, 181);
  for (let i = 0; i < 60 && S.masks[1].px === afterRect; i++) await wait(100);
  check('wand grew the selected mask', S.masks[1].px > afterRect + 500,
    `px=${S.masks[1].px} (was ${afterRect})`);

  // ---- edge ops incl. solid -------------------------------------------------
  S.sel = 1;
  win.document.getElementById('insp').querySelectorAll('[data-edge]');
  const pxBeforeSolid = S.masks[1].px;
  const solidBtn = doc.querySelector('#insp [data-edge=solid]');
  check('inspector shows solid next to outline', !!solidBtn &&
    solidBtn.previousElementSibling.getAttribute('data-edge') === 'outline');
  solidBtn.dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
  for (let i = 0; i < 60 && S.masks[1].px === pxBeforeSolid; i++) await wait(100);
  check('solid op ran through the server', S.masks[1].px >= pxBeforeSolid,
    `${pxBeforeSolid} -> ${S.masks[1].px}`);

  // ---- save + server round trip -------------------------------------------
  key('s', { ctrlKey: true });
  for (let i = 0; i < 60 && S.dirty.size; i++) await wait(100);
  check('save flushed to the server', S.dirty.size === 0, doc.getElementById('save').textContent);
  const docj = await (await fetch(BASE + '/api/project/' + meta.id)).json();
  check('server doc has both masks', docj.objects.length === 2,
    docj.objects.map(o => `${o.id}:${o.px}`).join(', '));

  // ---- M6c: view tabs (art / depth / normals / animation) ------------------
  const tabBtn = v => doc.querySelector(`#tabs button[data-tab="${v}"]`);
  check('the derived views load with the project',
    !!S.depthImg && !!S.orientImg && !!S.normalImg,
    `depth=${!!S.depthImg} orient=${!!S.orientImg} normal=${!!S.normalImg}`);
  const pingj = await (await fetch(BASE + '/api/ping')).json();
  check('ping reports the depth-pass source', typeof pingj.depth_ml === 'string',
    pingj.depth_ml);
  const mlb = doc.getElementById('mlbtn');
  check('tab bar shows the depth-pass source (or the install offer)', !!mlb &&
    (pingj.depth_ml === 'parse-only' ? /install/.test(mlb.textContent)
                                     : mlb.textContent.includes(pingj.depth_ml)),
    mlb ? mlb.textContent : 'no #mlbtn');
  const mli = await (await fetch(BASE + '/api/depth_ml/install',
    { method: 'POST' })).json();
  check('install endpoint no-ops when the accelerator is present',
    mli.already === true || pingj.depth_ml === 'parse-only', JSON.stringify(mli));

  check('tab bar replaces the old overlay cycle button', !!doc.getElementById('tabs') &&
    !doc.getElementById('oview') && ['masks', 'depth', 'normals', 'anim'].every(v => !!tabBtn(v)),
    `tabs=${doc.querySelectorAll('#tabs button').length}`);
  check('the active tab is marked for screen readers',
    tabBtn('masks').getAttribute('aria-pressed') === 'true' &&
    tabBtn('depth').getAttribute('aria-pressed') === 'false');
  key('d');
  check('D cycles to the depth tab', S.tab === 'depth' &&
    tabBtn('depth').getAttribute('aria-pressed') === 'true', S.tab);
  if (pingj.depth_ml && pingj.depth_ml !== 'parse-only') {
    for (let i = 0; i < 80 && !S.depthMlImg; i++) await wait(100);
    check('model-fused depth loaded for the depth tab', !!S.depthMlImg,
      `${!!S.depthMlImg} (${pingj.depth_ml})`);
  }
  key('d');
  check('D cycles on to the normals tab', S.tab === 'normals' &&
    tabBtn('normals').getAttribute('aria-pressed') === 'true', S.tab);
  key('d');
  for (let i = 0; i < 80 && !PV.strip; i++) await wait(100);
  check('D cycles to the animation tab (and it renders on first open)',
    S.tab === 'anim' && !!PV.strip && PV.N > 0,
    `tab=${S.tab} strip=${!!PV.strip} frames=${PV.N}`);
  tabBtn('masks').dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
  check('clicking a tab switches back to the masks', S.tab === 'masks' &&
    tabBtn('masks').getAttribute('aria-pressed') === 'true' &&
    tabBtn('masks').classList.contains('on') && !tabBtn('anim').classList.contains('on'), S.tab);

  // ---- M6b: wand guard, surface select, split ------------------------------
  const guardBtn = doc.getElementById('dguard');
  check('rail carries the depth-guard button', !!guardBtn);
  guardBtn.dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
  check('guard cycles to 2 rows', S.dguard === 2 && guardBtn.textContent === 'guard 2',
    `guard=${S.dguard}`);
  for (let i = 0; i < 3; i++) guardBtn.dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
  check('guard cycles back off', S.dguard === 0 && guardBtn.textContent === 'guard off');
  doc.querySelectorAll('#list .row')[0].dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
  const surf = doc.getElementById('iSurf');
  check('inspector offers a surface select', !!surf && surf.options.length === 3,
    surf ? `${surf.options.length} options` : 'missing');
  check('inspector offers split @depth', !!doc.querySelector('#insp [data-edge=depthsplit]'));
  surf.value = 'wall';
  surf.dispatchEvent(new win.Event('change', { bubbles: true }));
  check('surface lands on the mask', S.masks[0].surface === 'wall', S.masks[0].surface);
  key('s', { ctrlKey: true });
  for (let i = 0; i < 60 && S.dirty.size; i++) await wait(100);
  const docjM6 = await (await fetch(BASE + '/api/project/' + meta.id)).json();
  check('surface round-trips to the server doc',
    (docjM6.objects.find(o => o.id === 'seed1') || {}).surface === 'wall',
    JSON.stringify((docjM6.objects.find(o => o.id === 'seed1') || {}).surface));

  // ---- auto-seg: grouped choose-list, settings, preview, import ------------
  key('a');
  for (let i = 0; i < 120 && doc.querySelectorAll('#mrows .srow').length < 10; i++) await wait(150);
  const nrow = doc.querySelectorAll('#mrows .srow').length;
  check('auto-seg presented a choose-from list', nrow >= 40 && nrow <= 60, `${nrow} rows`);
  check('list is grouped into objects + regions',
    doc.querySelectorAll('#mrows .grp').length === 2,
    `${doc.querySelectorAll('#mrows .grp').length} groups`);
  check('suggestions have thumbnails', doc.querySelectorAll('#mrows .thumb canvas').length === nrow);
  check('rows carry recognised names + confidence', (() => {
    const names = [...doc.querySelectorAll('#mrows .nm')].map(e => e.textContent).join(' | ');
    return /potted plant \d/.test(names) && /rug \d/.test(names) &&
      doc.querySelectorAll('#mrows .conf').length > 0;
  })(), [...doc.querySelectorAll('#mrows .nm')].slice(0, 4).map(e => e.textContent).join(', '));
  check('objects ticked by default, regions not', (() => {
    const rows = [...doc.querySelectorAll('#mrows .srow')];
    const isObj = r => r.querySelector('.kind').textContent.startsWith('object');
    const checked = rows.filter(r => r.querySelector('input').checked);
    return checked.length === rows.filter(isObj).length && checked.every(isObj);
  })(), `${doc.querySelectorAll('#mrows .srow input:checked').length} ticked`);

  click('mset');
  check('settings panel opens', doc.getElementById('msettings').classList.contains('on'));
  check('settings expose the engine knobs', ['asCap', 'asMin', 'asMerge', 'asAbsorb', 'asGrow',
    'asConn', 'asRegions', 'asOnload'].every(id => !!doc.getElementById(id)));
  check('settings show the defaults', doc.getElementById('asCap').value === '60' &&
    doc.getElementById('asMin').value === '25' && doc.getElementById('asMerge').value === '2');

  // row click = preview on canvas
  doc.querySelectorAll('#mrows .srow')[3].dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
  check('clicking a row previews it on the canvas', S.sgPreview === 3, `sgPreview=${S.sgPreview}`);
  doc.querySelectorAll('#mrows .srow')[3].dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
  check('clicking again clears the preview', S.sgPreview === null);

  // re-run with a smaller cap through the settings form
  doc.getElementById('asCap').value = '20';
  doc.getElementById('asCap').dispatchEvent(new win.Event('change'));
  click('mrerun');
  for (let i = 0; i < 120 && doc.getElementById('mstat').textContent.includes('parsing'); i++) await wait(150);
  for (let i = 0; i < 60 && doc.querySelectorAll('#mrows .srow').length > 20; i++) await wait(150);
  const nrow2 = doc.querySelectorAll('#mrows .srow').length;
  check('re-run honours the cap setting', nrow2 > 0 && nrow2 <= 20, `${nrow2} rows`);
  check('settings persisted to localStorage',
    JSON.parse(win.localStorage.getItem('ps_asg')).cap === 20);

  click('mobjs');
  doc.getElementById('mimport').dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
  check('import added the ticked objects', S.masks.length > 2 && S.masks.length <= 22,
    `masks=${S.masks.length}`);
  check('modal closed after import', !doc.getElementById('modal').classList.contains('on'));

  // ---- M3: anim inspector, timeline, render + exports ----------------------
  check('badge shows M6', /M6/.test(doc.querySelector('.badge').textContent));
  check('docs button links to /docs', !!doc.getElementById('docsbtn') &&
    doc.getElementById('docsbtn').getAttribute('href') === '/docs');
  {
    const docsRes = await fetch(BASE + '/docs');
    const docsTxt = await docsRes.text();
    check('/docs serves the guide', docsRes.ok && /Keyboard shortcuts/.test(docsTxt) &&
      /atlas/i.test(docsTxt), `${docsRes.status}, ${docsTxt.length} B`);
  }
  check('timeline bar present', ['pvw', 'play', 'scrub', 'tlN', 'tlFps', 'render',
    'exgif', 'exzip'].every(id => !!doc.getElementById(id)));
  check('timeline N/fps seeded from the doc', doc.getElementById('tlN').value === '48' &&
    doc.getElementById('tlFps').value === '12',
    `${doc.getElementById('tlN').value}/${doc.getElementById('tlFps').value}`);
  check('imported masks carry catalog anim types from the hints', (() => {
    const types = new Set(S.masks.map(m => m.anim));
    return S.masks.every(m => ['sway', 'bob', 'ripple', 'shimmer', 'flutter',
      'gustflick', 'static'].includes(m.anim)) && types.has('sway') && types.size >= 3;
  })(), [...new Set(S.masks.map(m => m.anim))].join(','));
  // select a mask row -> inspector shows the anim dropdown + param sliders
  const iSel = S.masks.findIndex(m => m.anim === 'sway');
  doc.querySelectorAll('#list .row')[iSel]
    .dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
  const iAnim = doc.getElementById('iAnim');
  check('inspector has the 7-type anim dropdown', !!iAnim && iAnim.options.length === 7 &&
    iAnim.value === 'sway', iAnim ? iAnim.value : 'missing');
  check('param sliders built for the type', !!doc.getElementById('p_amp') &&
    !!doc.getElementById('p_speed') && !!doc.getElementById('p_gust') &&
    !!doc.getElementById('pAuto'), 'sway params');
  iAnim.value = 'bob';
  iAnim.dispatchEvent(new win.Event('change'));
  check('changing the type rebuilds sliders + defaults',
    S.masks[iSel].anim === 'bob' && S.masks[iSel].params.amp === 1 &&
    S.masks[iSel].params.circ === 0.6 && !!doc.getElementById('p_circ'),
    JSON.stringify(S.masks[iSel].params));
  const amp = doc.getElementById('p_amp');
  amp.value = '2';
  amp.dispatchEvent(new win.Event('input'));
  check('slider writes params and marks dirty', S.masks[iSel].params.amp === 2 &&
    S.dirty.has(S.masks[iSel]), `amp=${S.masks[iSel].params.amp}`);
  iAnim.value = 'sway';
  iAnim.dispatchEvent(new win.Event('change'));
  key('s', { ctrlKey: true });
  for (let i = 0; i < 60 && S.dirty.size; i++) await wait(100);
  const docj2 = await (await fetch(BASE + '/api/project/' + meta.id)).json();
  const saved = docj2.objects.find(o => o.id === S.masks[iSel].id);
  check('anim + params persist to the doc', saved && saved.anim === 'sway' &&
    saved.params && saved.params.amp === 2, JSON.stringify(saved && saved.params));
  // render -> strip plays in the timeline (clear the labels first: the anim tab
  // already rendered once, so a stale label would end the wait early)
  doc.getElementById('tlstat').textContent = '';
  doc.getElementById('frlbl').textContent = '';
  click('render');
  for (let i = 0; i < 120 && !/stray px/.test(doc.getElementById('tlstat').textContent); i++)
    await wait(150);
  const frlbl = doc.getElementById('frlbl').textContent;
  check('render produced a playable strip', /\/48/.test(frlbl), frlbl);
  check('playback started', doc.getElementById('play').textContent !== '▶');
  check('timeline reports the audit', /stray px 0/.test(doc.getElementById('tlstat').textContent),
    doc.getElementById('tlstat').textContent);
  click('play');                                // pause again
  check('pause stops playback', doc.getElementById('play').textContent === '▶');
  doc.getElementById('scrub').value = '7';
  doc.getElementById('scrub').dispatchEvent(new win.Event('input'));
  check('scrubber shows the picked frame', doc.getElementById('frlbl').textContent === '8/48',
    doc.getElementById('frlbl').textContent);
  // exports
  click('exgif');
  for (let i = 0; i < 160 && !/GIF \d+\.\d\d MB/.test(doc.getElementById('tlstat').textContent); i++)
    await wait(200);
  check('GIF export reports size + downloads', /GIF \d+\.\d\d MB/.test(
    doc.getElementById('tlstat').textContent), doc.getElementById('tlstat').textContent);
  const gifRes = await fetch(BASE + '/api/out/' + meta.id + '/preview.gif');
  check('GIF served from /api/out', gifRes.ok && (await gifRes.blob()).size > 10000);
  click('exzip');
  for (let i = 0; i < 160 && !/RGBA sheets/.test(doc.getElementById('tlstat').textContent); i++)
    await wait(200);
  check('bundle export reports sheets', /RGBA sheets/.test(
    doc.getElementById('tlstat').textContent), doc.getElementById('tlstat').textContent);
  check('bundle export reports atlas pages', /\d+ atlas pages/.test(
    doc.getElementById('tlstat').textContent), doc.getElementById('tlstat').textContent);
  const zipRes = await fetch(BASE + '/api/out/' + meta.id + '/bundle.zip');
  check('bundle zip served from /api/out', zipRes.ok && (await zipRes.blob()).size > 10000);

  // ---- M4: shadow toggle, presets, toasts, shortcuts, rename --------------
  const iSh = doc.getElementById('iShadow');
  check('inspector has the shadow-sync toggle', !!iSh);
  check('shadow toggle enabled for movers only', !iSh.disabled === true &&
    (() => { const st = S.masks.findIndex(m => m.anim === 'static');
      if (st < 0) return true;
      doc.querySelectorAll('#list .row')[st]
        .dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
      const d = doc.getElementById('iShadow').disabled;
      doc.querySelectorAll('#list .row')[iSel]
        .dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
      return d; })(), 'static disabled, mover enabled');
  const iSh2 = doc.getElementById('iShadow');
  iSh2.checked = true;
  iSh2.dispatchEvent(new win.Event('change'));
  check('shadow toggle marks dirty', S.masks[iSel].shadow === true && S.dirty.has(S.masks[iSel]));
  const iP = doc.getElementById('iPreset');
  check('preset select lists built-ins for the type',
    [...iP.querySelectorAll('optgroup')].some(g => g.label === 'built-in') &&
    [...iP.options].some(o => o.value === 'windy'), [...iP.options].map(o => o.value).join(','));
  iP.value = 'windy';
  iP.dispatchEvent(new win.Event('change'));
  check('applying a preset writes params', S.masks[iSel].params.speed === 2 &&
    S.masks[iSel].params.amp === 1.4, JSON.stringify(S.masks[iSel].params));
  win.__prompt = 'my gust';
  doc.getElementById('iPresetSave').dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
  check('user preset saved to localStorage + listed',
    JSON.parse(win.localStorage.getItem('ps_presets')).sway['my gust'] &&
    [...doc.getElementById('iPreset').options].some(o => o.value === 'my gust'));
  win.api('/api/definitely-not-a-route').catch(() => {});
  for (let i = 0; i < 40 && !doc.getElementById('toasts').children.length; i++) await wait(100);
  check('failed API calls raise a toast', doc.getElementById('toasts').children.length > 0,
    doc.getElementById('toasts').textContent.slice(0, 60));
  key('s', { ctrlKey: true });
  for (let i = 0; i < 60 && S.dirty.size; i++) await wait(100);
  const docj3 = await (await fetch(BASE + '/api/project/' + meta.id)).json();
  check('shadow flag persists to the doc',
    docj3.objects.find(o => o.id === S.masks[iSel].id).shadow === true);
  // shortcuts: play/pause, step, help
  click('render');
  for (let i = 0; i < 120 && doc.getElementById('play').textContent !== '❚❚'; i++)
    await wait(150);
  check('second render plays the strip', doc.getElementById('play').textContent === '❚❚');
  key('p');
  check('P pauses playback', doc.getElementById('play').textContent === '▶');
  const frAt = +doc.getElementById('frlbl').textContent.split('/')[0];
  key(',');
  check(', steps one frame back',
    +doc.getElementById('frlbl').textContent.split('/')[0] ===
    ((frAt - 2 + 48) % 48) + 1, doc.getElementById('frlbl').textContent);
  key('?');
  check('? opens the shortcuts help', doc.getElementById('help').classList.contains('on'));
  key('Escape');
  check('Escape closes the help', !doc.getElementById('help').classList.contains('on'));
  // project rename through the header button
  win.__prompt = 'smoke rename';
  click('prename');
  for (let i = 0; i < 40; i++) {
    const l = await (await fetch(BASE + '/api/projects')).json();
    if (l.find(x => x.id === meta.id).name === 'smoke rename') break;
    await wait(100);
  }
  const lst2 = await (await fetch(BASE + '/api/projects')).json();
  check('rename button renames the project',
    lst2.find(x => x.id === meta.id).name === 'smoke rename',
    JSON.stringify(lst2.find(x => x.id === meta.id)));
  for (let i = 0; i < 40 &&
    ![...doc.getElementById('recent').options].some(o => /smoke rename/.test(o.textContent)); i++)
    await wait(100);
  check('projects dropdown shows names',
    [...doc.getElementById('recent').options].some(o => /smoke rename/.test(o.textContent)));

  // ---- mask list ops --------------------------------------------------------
  const rows0 = S.masks.length;
  S.sel = 0;
  click('dup');
  check('duplicate adds a mask', S.masks.length === rows0 + 1, `${rows0} -> ${S.masks.length}`);
  click('del');
  check('delete removes a mask', S.masks.length === rows0, `${S.masks.length}`);
  key('s', { ctrlKey: true });
  for (let i = 0; i < 60 && S.dirty.size; i++) await wait(100);
  check('final save clean', S.dirty.size === 0);

  // ---- second boot: a freshly loaded image is parsed automatically ---------
  const dom2 = bootDom(BASE + '/?p=' + fresh.id);
  for (let i = 0; i < 100 && !(dom2.window.__PS && dom2.window.__PS.meta); i++) await wait(100);
  const S2 = dom2.window.__PS, doc2 = dom2.window.document;
  for (let i = 0; i < 120 && !(doc2.getElementById('modal').classList.contains('on') &&
      doc2.querySelectorAll('#mrows .srow').length > 10); i++) await wait(150);
  check('fresh image is parsed on load and the list is presented',
    S2.masks.length === 0 && doc2.getElementById('modal').classList.contains('on') &&
    doc2.querySelectorAll('#mrows .srow').length >= 40,
    `masks=${S2.masks.length}, rows=${doc2.querySelectorAll('#mrows .srow').length}`);
  check('status explains the choice', /recognised/.test(doc2.getElementById('status').textContent),
    doc2.getElementById('status').textContent.slice(0, 60));

  // ---- M4: duplicate then delete resets the editor -------------------------
  win.__confirm = false;
  click('pdel');
  await wait(200);
  check('delete asks for confirmation first', S.meta && S.meta.id === meta.id);
  win.__confirm = true;
  click('pdup');
  for (let i = 0; i < 80 && (!S.meta || S.meta.id === meta.id); i++) await wait(150);
  check('duplicate opens the copy', S.meta && S.meta.id !== meta.id, S.meta && S.meta.id);
  const copyId = S.meta.id;
  click('pdel');
  for (let i = 0; i < 80 && S.meta; i++) await wait(150);
  check('deleting the open project resets the editor', S.meta === null &&
    doc.getElementById('hint').style.display === 'flex');
  const lst3 = await (await fetch(BASE + '/api/projects')).json();
  check('deleted copy is gone from the server', !lst3.find(x => x.id === copyId));

  check('no runtime errors during the whole session', errors.length === 0, errors.join('; '));
  srv.kill();
  console.log(fails ? `\n${fails} UI SMOKE FAILURES` : '\nUI SMOKE TEST PASSED');
  process.exit(fails ? 1 : 0);
}
main().catch(e => { console.error('harness error:', e); process.exit(1); });
