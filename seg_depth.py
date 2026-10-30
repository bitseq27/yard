import numpy as np, json
from PIL import Image
from collections import deque

nat = np.asarray(Image.open('native.png')).astype(np.int32)
H, W, _ = nat.shape
M = np.load('masks.npy')
plant_mask, pad0, water_anim, quad = M[0], M[1], M[2], M[3]
R, G, B = nat[:,:,0], nat[:,:,1], nat[:,:,2]
arr = nat/255.0
cmax = arr.max(2); cmin = arr.min(2); df = cmax-cmin
r,g,b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
hue = np.zeros((H,W))
m=(cmax==r)&(df>0); hue[m]=((g-b)[m]/df[m])%6
m=(cmax==g)&(df>0); hue[m]=((b-r)[m]/df[m])+2
m=(cmax==b)&(df>0); hue[m]=((r-g)[m]/df[m])+4
hue*=60
sat = np.where(cmax>0, df/np.maximum(cmax,1e-6), 0); val = cmax

# pad_full (same as animator)
def dil4(m,it):
    d=m.copy()
    for _ in range(it):
        d2=d.copy()
        d[1:,:]|=d2[:-1,:]; d[:-1,:]|=d2[1:,:]; d[:,1:]|=d2[:,:-1]; d[:,:-1]|=d2[:,1:]
    return d
def ero4(m,it):
    d=m.copy()
    for _ in range(it):
        d2=d.copy()
        d[1:,:]&=d2[:-1,:]; d[:-1,:]&=d2[1:,:]; d[:,1:]&=d2[:,:-1]; d[:,:-1]&=d2[:,1:]
    return d
pad_full = ero4(dil4(pad0,2),2) & quad
reach = np.zeros((H,W),bool); q=deque()
for y0 in (0,H-1):
    for x0 in range(W):
        if not pad_full[y0,x0] and not reach[y0,x0]: reach[y0,x0]=True; q.append((y0,x0))
for x0 in (0,W-1):
    for y0 in range(H):
        if not pad_full[y0,x0] and not reach[y0,x0]: reach[y0,x0]=True; q.append((y0,x0))
while q:
    y,x=q.popleft()
    for dy in(-1,0,1):
        for dx in(-1,0,1):
            ny,nx=y+dy,x+dx
            if 0<=ny<H and 0<=nx<W and not pad_full[ny,nx] and not reach[ny,nx]:
                reach[ny,nx]=True; q.append((ny,nx))
pad_full |= (~pad_full & ~reach)

def label(mask, min_size=1):
    lab = -np.ones((H,W), np.int32); comps=[]
    for y0 in range(H):
        for x0 in range(W):
            if mask[y0,x0] and lab[y0,x0]<0:
                q=deque([(y0,x0)]); lab[y0,x0]=len(comps); pix=[(y0,x0)]
                while q:
                    y,x=q.popleft()
                    for dy in(-1,0,1):
                        for dx in(-1,0,1):
                            ny,nx=y+dy,x+dx
                            if 0<=ny<H and 0<=nx<W and mask[ny,nx] and lab[ny,nx]<0:
                                lab[ny,nx]=len(comps); q.append((ny,nx)); pix.append((ny,nx))
                if len(pix)>=min_size: comps.append(np.array(pix))
                else:
                    for (y,x) in pix: lab[y,x]=-2
    return comps, lab

# ---- class masks ----
brown  = (hue>=8)&(hue<=30)&(sat>0.45)&(R>G)&(G>=B)&(val>0.25)&(val<0.85)
wood   = (hue>=15)&(hue<=40)&(sat>0.25)&(val<=0.55)&(R>G)&(G>=B)
stone  = (sat<0.45)&(val>0.42)&(hue>=10)&(hue<=80)
carpet_raw = ((hue<15)|(hue>340))&(sat>0.35)&(val>0.18)
carpet = np.zeros((H,W), bool)
_cc, _ = label(dil4(carpet_raw,2), 60)
for c in _cc:
    ys, xs = c[:,0], c[:,1]
    if ys.mean() > 260:
        m2 = np.zeros((H,W),bool); m2[ys,xs]=True
        carpet |= (m2 & carpet_raw)
teal   = (hue>=160)&(hue<=200)&(sat>0.2)&(val>0.22)
blue   = (hue>200)&(hue<=270)&(sat>0.25)&(val>0.15)
dark   = val<0.16

water  = quad & ~pad_full
# wall-region polygon prior (everything above the wall base lines)
_wp = [(0,0),(W,0),(W,240),(300,200),(220,262),(204,300),(0,196)]
ysg, xsg = np.mgrid[0:H,0:W]
wallpoly = np.zeros((H,W), bool)
j = len(_wp)-1
for i in range(len(_wp)):
    xi,yi = _wp[i]; xj,yj = _wp[j]
    cond = ((yi>ysg)!=(yj>ysg)) & (xsg < (xj-xi)*(ysg-yi)/(yj-yi+1e-9)+xi)
    wallpoly ^= cond
    j = i
floor_c = teal & ~wallpoly & ~dil4(quad,1)
# olive shadow diamonds on the floor join the floor class
olive = (hue>=80)&(hue<=160)&(sat>0.2)&(val>0.18)&(val<0.65)
floor_c = floor_c | (olive & dil4(floor_c,2) & ~plant_mask & ~pad0)
pool_wall = teal & ~floor_c & dil4(quad,2) & ~quad
wall_teal = teal & ~floor_c & ~pool_wall
# pots vs wood: brown comps touching plants above = pots
bcomps, _ = label(brown, 10)
pot_m = np.zeros((H,W),bool); wood_m = wood.copy()
plant_d = dil4(plant_mask,3)
for c in bcomps:
    ys,xs = c[:,0], c[:,1]
    h_ = ys.max()-ys.min(); w_ = xs.max()-xs.min()
    above = plant_d[max(ys.min()-3,0):ys.min()+2, xs.min():xs.max()+1].any()
    if len(c)>=12 and h_<=20 and w_<=22 and above and ys.mean()>180:
        pot_m[ys,xs] = True
    else:
        wood_m[ys,xs] = True

bigp = np.zeros((H,W),bool)
_pcomps, _ = label(plant_mask, 1)
for c in _pcomps:
    if len(c)>=25: bigp[c[:,0], c[:,1]] = True
bigpd = dil4(bigp,2)
for c in _pcomps:
    if len(c)<25 and not bigpd[c[:,0], c[:,1]].any():
        plant_mask[c[:,0], c[:,1]] = False

# ---- segmentation ids ----
seg = np.zeros((H,W), np.int32)  # 0 = unassigned
CLASSES = {}
def add_class(name):
    if name not in CLASSES: CLASSES[name] = len(CLASSES)+1
    return CLASSES[name]

fixed = {
 'pool_water': (40,120,190), 'floor_tiles': (90,200,190), 'stone_trim': (230,220,190),
 'wall_tiles_blue': (60,80,200), 'wall_tiles_teal': (60,160,220), 'pool_inner_wall': (30,90,160),
 'wood_doors': (150,90,40), 'shadows_dark': (40,40,50), 'fringe': (250,240,200),
}
seg_colors = {}
def set_mask(mask, rgb, name):
    cid = add_class(name); seg_colors[cid] = rgb
    seg[mask] = cid

import colorsys
def golden(i, s=0.75, v=0.95):
    h = (i*0.61803398875) % 1.0
    sv = [(0.85,0.95),(0.55,0.9),(0.9,0.55),(0.7,0.75)][i % 4]
    r_,g_,b_ = colorsys.hsv_to_rgb(h, sv[0], sv[1])
    return (int(r_*255), int(g_*255), int(b_*255))

# instance-colored classes
def add_instances(mask, base_name, min_sz, sat=0.75, v=0.95):
    comps, _ = label(mask, min_sz)
    comps = sorted(comps, key=lambda c: (c[:,0].mean(), c[:,1].mean()))
    for i,c in enumerate(comps):
        cid = add_class(f'{base_name}_{i}')
        seg_colors[cid] = golden(hash(base_name)%7 + i if False else i + (hash(base_name)%50), sat, v)
        seg[c[:,0], c[:,1]] = cid
    return len(comps)

set_mask(water, fixed['pool_water'], 'pool_water')
set_mask(floor_c, fixed['floor_tiles'], 'floor_tiles')
set_mask(stone & ~quad, fixed['stone_trim'], 'stone_trim')
set_mask(wall_teal, fixed['wall_tiles_teal'], 'wall_tiles_teal')
set_mask(pool_wall, fixed['pool_inner_wall'], 'pool_inner_wall')
set_mask(blue & ~plant_mask, fixed['wall_tiles_blue'], 'wall_tiles_blue')
set_mask(wood_m & ~pot_m, fixed['wood_doors'], 'wood_doors')
set_mask(dark & ~plant_mask & ~pad_full, fixed['shadows_dark'], 'shadows_dark')
n_pad = add_instances(pad_full, 'lily_pad', 4)
n_pl  = add_instances(plant_mask & ~pad_full, 'plant', 25)
n_pot = add_instances(pot_m, 'pot', 6)
n_ca  = add_instances(carpet, 'carpet', 40)
# wood vs stone: wood comps barely touching blue tiles are stone bands (keep doors & trunks)
wcomps, _ = label(wood_m & ~pot_m, 10)
for c in wcomps:
    ys, xs = c[:,0], c[:,1]
    m2 = np.zeros((H,W),bool); m2[ys,xs]=True
    touch_blue = (dil4(m2,1) & blue).sum() / max(len(c),1)
    satm = sat[ys,xs].mean()
    if touch_blue < 0.05 and satm < 0.5 and (ys.max()-ys.min()) < 30:
        wood_m[ys,xs] = False
        stone |= m2
        stone2m = m2
# --- refinement pass ---
def holefill(m):
    reach = np.zeros((H,W),bool); qq=deque()
    for y0 in (0,H-1):
        for x0 in range(W):
            if not m[y0,x0] and not reach[y0,x0]: reach[y0,x0]=True; qq.append((y0,x0))
    for x0 in (0,W-1):
        for y0 in range(H):
            if not m[y0,x0] and not reach[y0,x0]: reach[y0,x0]=True; qq.append((y0,x0))
    while qq:
        y,x=qq.popleft()
        for dy in(-1,0,1):
            for dx in(-1,0,1):
                ny,nx=y+dy,x+dx
                if 0<=ny<H and 0<=nx<W and not m[ny,nx] and not reach[ny,nx]:
                    reach[ny,nx]=True; qq.append((ny,nx))
    return m | (~reach)
fringem = (sat<0.35)&(val>0.55)&((hue<70)|(hue>300)) & dil4(carpet,2) & ~stone
floor_cid = CLASSES['floor_tiles']; water_cid = CLASSES['pool_water']; stone_cid = CLASSES['stone_trim']
misc_cid = len(CLASSES)+1  # will be created later; placeholder fix below
# merge rug fragments + fill rug interiors
carpet_g = carpet & ~dil4(plant_mask,3)
gcomps, _ = label(dil4(carpet_g,2), 1)
for gc in gcomps:
    if len(gc) < 80: continue
    gm = np.zeros((H,W),bool); gm[gc[:,0], gc[:,1]] = True
    vals, cnts = np.unique(seg[gm & (seg>0)], return_counts=True)
    if not len(vals): continue
    rid = vals[cnts.argmax()]
    region = holefill(dil4(gm,1)) & dil4(gm,10) & ~plant_mask & ~pot_m & ~pad_full & ~water & ~fringem
    seg[region] = rid
# floor blue diamond motifs
seg[(blue & dil4(floor_c,2) & ~plant_mask & ~pad_full)] = floor_cid
# pad shadows / dark in pool -> water
seg[(dark & quad & ~pad_full)] = water_cid
# shadowed stone -> stone
seg[(dark & dil4(stone & ~quad,1) & ~quad & ~plant_mask & ~pad_full & ~carpet)] = stone_cid
stone2 = (hue>=10)&(hue<=80)&(sat<0.5)&(val>0.26)&~brown&~carpet&~plant_mask&~pad_full&~water
# (applied after misc exists; store mask for later)
STONE2 = stone2
# fringe tassels own class
set_mask(fringem, fixed['fringe'], 'fringe')
# leftovers
left = seg==0
cid = add_class('misc'); seg_colors[cid] = (120,120,120); seg[left] = cid
seg[left & STONE2 & ~dil4(quad,1)] = stone_cid

# speckle cleanup: tiny components of any class (except fringe/plants) -> neighbourhood majority
fringe_cid = CLASSES['fringe']
tiny = np.zeros((H,W), bool)
for cid in np.unique(seg):
    if cid==0 or cid==fringe_cid: continue
    comps, _ = label(seg==cid, 1)
    for c in comps:
        if len(c) < 6: tiny[c[:,0], c[:,1]] = True
for _ in range(3):
    ys, xs = np.nonzero(tiny)
    if len(ys)==0: break
    for y, x in zip(ys, xs):
        y0,y1 = max(y-1,0), min(y+2,H); x0,x1 = max(x-1,0), min(x+2,W)
        vals, cnts = np.unique(seg[y0:y1, x0:x1][~tiny[y0:y1, x0:x1]], return_counts=True)
        if len(vals): seg[y,x] = vals[cnts.argmax()]; tiny[y,x] = False
rgb = np.zeros((H,W,3), np.uint8)
for c_id, col in seg_colors.items():
    rgb[seg==c_id] = col
Image.fromarray(rgb).resize((W*2,H*2), Image.NEAREST).save('segmentation_mask.png')
np.save('seg.npy', seg)
json.dump({str(k): {'name': n, 'px': int((seg==k).sum())} for k,n in [(k, v) for v,k in CLASSES.items()]}, open('seg_legend.json','w'), indent=0)
print('classes', len(CLASSES), 'plants', n_pl, 'pots', n_pot, 'carpets', n_ca, 'pads', n_pad)

# ---- depth map ----
base_y = np.arange(H)[:,None].repeat(W,1).astype(np.float32)  # ground: base = sy
# vertical surfaces: per-column bottom of their component
def set_base_colbottom(mask, min_sz=30):
    comps, _ = label(mask, min_sz)
    for c in comps:
        ys, xs = c[:,0], c[:,1]
        m2 = np.zeros((H,W),bool); m2[ys,xs]=True
        for x in np.unique(xs):
            yy = ys[xs==x].max()
            base_y[m2[:,x], x] = np.maximum(base_y[m2[:,x], x], 0)*0 + yy  # assign
    return
# walls/doors/pool wall/stone-on-wall: vertical
vert = (wall_teal | pool_wall | (blue & ~plant_mask) | (wood_m & ~pot_m)) 
set_base_colbottom(vert, 30)
# stone: mixed; keep ground rule except wall-band stone (upper area): treat stone comps whose bbox top < 250 as vertical
scomps, _ = label(stone & ~quad, 30)
for c in scomps:
    ys, xs = c[:,0], c[:,1]
    if ys.min() < 240 and (ys.max()-ys.min()) > 12:
        m2 = np.zeros((H,W),bool); m2[ys,xs]=True
        for x in np.unique(xs):
            yy = ys[xs==x].max()
            base_y[m2[:,x], x] = yy
# objects: per-comp bottom
def set_base_comp(mask, min_sz):
    comps, _ = label(mask, min_sz)
    for c in comps:
        ys, xs = c[:,0], c[:,1]
        by = np.percentile(ys, 97)
        base_y[ys, xs] = by
set_base_comp(plant_mask, 25)
set_base_comp(pot_m, 6)
set_base_comp(pad_full, 4)
# smooth base_y horizontally within vertical surfaces to kill column striping
vm = vert.copy()
sm = base_y.copy()
for y in range(H):
    row = base_y[y]
    out = row.copy()
    for x in range(W):
        if vm[y,x]:
            x0,x1 = max(x-4,0), min(x+5,W)
            out[x] = np.median(row[x0:x1][vm[y,x0:x1]]) if vm[y,x0:x1].any() else row[x]
    sm[y] = out
base_y = np.where(vm, sm, base_y)
yy = np.arange(H)[:,None].repeat(W,1).astype(np.float32)
closeness = 3*base_y - yy
d = closeness - closeness.min()
d = d / max(d.max(),1e-6)
depth8 = (d*255).astype(np.uint8)
Image.fromarray(depth8).resize((W*2,H*2), Image.NEAREST).save('depth_map.png')
np.save('depth.npy', depth8)
print('depth range', closeness.min(), closeness.max())
