import numpy as np
from PIL import Image
from collections import deque

nat = np.asarray(Image.open('native.png')).astype(np.uint8)
base = np.asarray(Image.open('base_inpainted.png')).astype(np.uint8)
M = np.load('masks.npy')
plant_mask, pad_mask, water_anim, quad = M[0], M[1], M[2], M[3]
H, W, _ = nat.shape
TAU = np.pi*2
N = 48

arr = nat.astype(np.float32)/255.0
cmax = arr.max(2); cmin = arr.min(2); df = cmax-cmin
r,g,b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
hue = np.zeros((H,W))
m=(cmax==r)&(df>0); hue[m]=((g-b)[m]/df[m])%6
m=(cmax==g)&(df>0); hue[m]=((b-r)[m]/df[m])+2
m=(cmax==b)&(df>0); hue[m]=((r-g)[m]/df[m])+4
hue*=60
sat = np.where(cmax>0, df/np.maximum(cmax,1e-6), 0)
val = cmax
green_all = (hue>=60)&(hue<=175)&(sat>0.25)&((nat[:,:,2].astype(int)-nat[:,:,0].astype(int))<40)
brown = (hue>=10)&(hue<=32)&(sat>0.35)&(nat[:,:,0]>nat[:,:,1])&(nat[:,:,1]>=nat[:,:,2])
waterish = water_anim & ~green_all & ~brown

def dilate(m, it):
    d = m.copy()
    for _ in range(it):
        d2 = d.copy()
        d[1:,:] |= d2[:-1,:]; d[:-1,:] |= d2[1:,:]; d[:,1:] |= d2[:,:-1]; d[:,:-1] |= d2[:,1:]
    return d

def label(mask, sources=None):
    lab = -np.ones((H,W), np.int32); comps=[]
    seeds = sources if sources is not None else [(y,x) for y in range(H) for x in range(W) if mask[y,x]]
    for (y0,x0) in seeds:
        if mask[y0,x0] and lab[y0,x0]<0:
            cid=len(comps); q=deque([(y0,x0)]); lab[y0,x0]=cid; pix=[(y0,x0)]
            while q:
                y,x=q.popleft()
                for dy in(-1,0,1):
                    for dx in(-1,0,1):
                        ny,nx=y+dy,x+dx
                        if 0<=ny<H and 0<=nx<W and mask[ny,nx] and lab[ny,nx]<0:
                            lab[ny,nx]=cid; q.append((ny,nx)); pix.append((ny,nx))
            comps.append(np.array(pix))
    return comps, lab

def erode4(m):
    e = m.copy()
    e[1:,:] &= m[:-1,:]; e[:-1,:] &= m[1:,:]; e[:,1:] &= m[:,:-1]; e[:,:-1] &= m[:,1:]
    return e

# ---- build leaf units ----
comps, _ = label(plant_mask)
unit_lab = -np.ones((H,W), np.int32)
units = []   # list of (ys, xs)
for pix in comps:
    cm = np.zeros((H,W), bool); cm[pix[:,0], pix[:,1]] = True
    if len(pix) >= 150:
        er = erode4(cm)
        cores, clab = label(er)
        keep = [i for i,c in enumerate(cores) if len(c)>=6]
        if len(keep) >= 2:
            # multi-source BFS over cm from kept cores
            ul = -np.ones((H,W), np.int32)
            q = deque()
            for ni,i in enumerate(keep):
                for (y,x) in cores[i]:
                    ul[y,x] = ni; q.append((y,x))
            while q:
                y,x = q.popleft()
                for dy in(-1,0,1):
                    for dx in(-1,0,1):
                        ny,nx = y+dy, x+dx
                        if 0<=ny<H and 0<=nx<W and cm[ny,nx] and ul[ny,nx]<0:
                            ul[ny,nx] = ul[y,x]; q.append((ny,nx))
            for ni in range(len(keep)):
                ys, xs = np.nonzero(ul==ni)
                if len(ys)>0:
                    unit_lab[ys,xs] = len(units); units.append((ys,xs))
            continue
    unit_lab[pix[:,0], pix[:,1]] = len(units); units.append((pix[:,0], pix[:,1]))
print('units', len(units))

# drop tiny units isolated from real foliage (stray specks on walls/floor)
big = np.zeros((H,W), bool)
for (ys, xs) in units:
    if len(ys) >= 25: big[ys, xs] = True
bigd = big.copy()
for _ in range(2):
    d2 = bigd.copy()
    bigd[1:,:] |= d2[:-1,:]; bigd[:-1,:] |= d2[1:,:]; bigd[:,1:] |= d2[:,:-1]; bigd[:,:-1] |= d2[:,1:]
keep_units=[]
for (ys, xs) in units:
    if len(ys) < 25 and not bigd[ys, xs].any():
        plant_mask[ys, xs] = False
        unit_lab[ys, xs] = -1
        continue
    keep_units.append((ys, xs))
units = keep_units
for nid, (ys, xs) in enumerate(units):
    unit_lab[ys, xs] = nid
print('units after speck filter', len(units))

rng = np.random.default_rng(4242)
uinfo=[]
for (ys, xs) in units:
    y0, y1 = int(ys.min()), int(ys.max())
    hgt = max(y1-y0, 1); sz = len(ys)
    if sz > 600:   k = 1
    elif sz > 120: k = int(rng.choice([1,1,2]))
    elif sz > 30:  k = int(rng.choice([1,2,2]))
    else:          k = int(rng.choice([2,2,3]))
    ph   = rng.random()
    w2   = rng.uniform(0.10, 0.38)
    ps2  = rng.uniform(0, TAU)
    ampT = rng.uniform(0.90, 1.25) if sz >= 250 else rng.uniform(0.85, 1.10)
    j1   = rng.uniform(0.30, 0.45)
    j2   = rng.uniform(0.62, 0.80)
    f1   = rng.uniform(0.45, 0.65)
    envA = rng.uniform(0.10, 0.32)
    envP = rng.uniform(0, TAU)
    bias = rng.uniform(-0.12, 0.12)
    maxdx = 2 if sz >= 180 else 1
    chunked = hgt >= 14
    uinfo.append((y1, hgt, sz, k, ph, w2, ps2, ampT, j1, j2, f1, envA, envP, bias, maxdx, chunked))

# ---- rug tassels ----
cream = (sat<0.35)&(val>0.55)&((hue<70)|(hue>300))
redm  = ((hue<20)|(hue>340))&(sat>0.4)&(val>0.3)
tealm = (hue>=160)&(hue<=200)&(sat>0.25)&(val>0.3)
def dilate(m, it):
    d = m.copy()
    for _ in range(it):
        d2 = d.copy()
        d[1:,:] |= d2[:-1,:]; d[:-1,:] |= d2[1:,:]; d[:,1:] |= d2[:,:-1]; d[:,:-1] |= d2[:,1:]
    return d
tass = cream & dilate(redm,2) & dilate(tealm,2)
tcomps, _ = label(tass)
print('tassels', len(tcomps), 'px', tass.sum())
gusts = sorted(rng.choice(np.arange(4, N-16), size=3, replace=False).tolist())
tass_info=[]
for pix in tcomps:
    ys, xs = pix[:,0], pix[:,1]
    if rng.random() < 0.85:
        g = gusts[int(rng.integers(0,3))]
        delay = int(xs.mean()//20 + rng.integers(0,3)) % 10
        dur = int(rng.integers(3,6))
        start = g + delay
        if start + dur < N:
            tass_info.append((ys, xs, start, dur))
print('flicking tassels', len(tass_info), 'gusts at', gusts)

# inpaint water under overhanging leaves as well (so ripple never moves leaf-colored px)
over = plant_mask & water_anim
def hfill(baseimg, mask):
    for y in range(H):
        row = mask[y]; x=0
        while x<W:
            if row[x]:
                x2=x
                while x2+1<W and row[x2+1]: x2+=1
                lc = baseimg[y, x-1] if x>0 else baseimg[y, x2+1]
                rc = baseimg[y, x2+1] if x2+1<W else baseimg[y, x-1]
                for xx in range(x, x2+1):
                    t=(xx-x+1)/(x2-x+2)
                    baseimg[y,xx] = (lc*(1-t)+rc*t).astype(np.uint8)
                x=x2+1
            else: x+=1
    return baseimg
green_raw = (hue>=60)&(hue<=175)&(sat>0.25)&(val>0.12)&((nat[:,:,2].astype(int)-nat[:,:,0].astype(int))<40)
for _ in range(2):
    pad_mask = pad_mask | (quad & dilate(pad_mask,1) & green_raw)
# close pad sprites: fill holes (blossom centres) and attach rim/shadow ring
def close8(m, it):
    d = m.copy()
    for _ in range(it):
        d2 = d.copy()
        d[1:,:] |= d2[:-1,:]; d[:-1,:] |= d2[1:,:]; d[:,1:] |= d2[:,:-1]; d[:,:-1] |= d2[:,1:]
    for _ in range(it):
        d2 = d.copy()
        d[1:,:] &= d2[:-1,:]; d[:-1,:] &= d2[1:,:]; d[:,1:] &= d2[:,:-1]; d[:,:-1] &= d2[:,1:]
    return d
pad_full = close8(pad_mask, 2) & quad
# fill interior holes per pad blob
labp = -np.ones((H,W), np.int32); npad=0
from collections import deque as _dq
for y0 in range(H):
    for x0 in range(W):
        if pad_full[y0,x0] and labp[y0,x0]<0:
            q=_dq([(y0,x0)]); labp[y0,x0]=npad
            while q:
                y,x=q.popleft()
                for dy in(-1,0,1):
                    for dx in(-1,0,1):
                        ny,nx=y+dy,x+dx
                        if 0<=ny<H and 0<=nx<W and pad_full[ny,nx] and labp[ny,nx]<0:
                            labp[ny,nx]=npad; q.append((ny,nx))
            npad+=1
outside = ~pad_full
reach = np.zeros((H,W), bool)
q=_dq()
for y0 in (0,H-1):
    for x0 in range(W):
        if outside[y0,x0] and not reach[y0,x0]: reach[y0,x0]=True; q.append((y0,x0))
for x0 in (0,W-1):
    for y0 in range(H):
        if outside[y0,x0] and not reach[y0,x0]: reach[y0,x0]=True; q.append((y0,x0))
while q:
    y,x=q.popleft()
    for dy in(-1,0,1):
        for dx in(-1,0,1):
            ny,nx=y+dy,x+dx
            if 0<=ny<H and 0<=nx<W and outside[ny,nx] and not reach[ny,nx]:
                reach[ny,nx]=True; q.append((ny,nx))
pad_full = pad_full | (outside & ~reach)
pad_mask = pad_full
base = hfill(base, pad_mask | over)
waterish = waterish & ~dilate(pad_mask,2) & ~dilate(plant_mask,1)
ripple_mask = erode4(water_anim) & ~dilate(pad_mask,2)

# ---- pads ----
pad_comps, _ = label(pad_mask)
padinfo=[]
for pix in pad_comps:
    ys, xs = pix[:,0], pix[:,1]
    ph = ((xs.mean()*11 + ys.mean()*5) % 32)/32.0
    k = int(rng.choice([1,1,2]))
    amp = rng.uniform(0.5, 0.9)
    padinfo.append((ys, xs, ph, k, amp))

def runs_of(mask, labarr):
    runs=[]
    for y in np.nonzero(mask.any(1))[0]:
        row = mask[y]; x=0; rl=[]
        while x<W:
            if row[x]:
                x2=x
                while x2+1<W and row[x2+1]: x2+=1
                rl.append((x,x2)); x=x2+1
            else: x+=1
        runs.append((int(y), rl))
    return runs
plant_runs = runs_of(plant_mask, unit_lab)
water_runs = runs_of(ripple_mask, None)

UC = []
for uid in range(len(uinfo)):
    ys, xs = units[uid]
    y1, hgt = uinfo[uid][0], uinfo[uid][1]
    j1, j2 = uinfo[uid][8], uinfo[uid][9]
    fy = (y1 - ys)/hgt
    ch_of = np.where(fy < j1, 0, np.where(fy < j2, 1, 2))
    chunks=[]
    for ch in range(3):
        sel = ch_of == ch
        cys, cxs = ys[sel], xs[sel]
        runs=[]
        if len(cys):
            for y in np.unique(cys):
                xr = np.sort(cxs[cys==y])
                a = xr[0]; prev = xr[0]
                for x in xr[1:]:
                    if x == prev+1: prev = x; continue
                    runs.append((int(y), int(a), int(prev))); a = x; prev = x
                runs.append((int(y), int(a), int(prev)))
        chunks.append((cys, cxs, runs))
    UC.append(chunks)

tt = np.arange(N)/N
chunkdx = np.zeros((len(uinfo),3), np.int32)
frames=[]
for t in range(N):
    fr = base.copy()
    for y, rl in water_runs:
        for (a,bb) in rl:
            c=(a+bb)//2
            dx = int(round(0.7*np.sin(TAU*(3*t/N) + 0.45*y) + 0.45*np.sin(TAU*(2*t/N) + 0.30*c)))
            if dx==0 or a+dx<0 or bb+dx>=W: continue
            if dx>0:
                fr[y, a+dx:bb+dx+1] = base[y, a:bb+1]
                fr[y, a:a+dx] = base[y, a-1] if a>0 else base[y, a]
            else:
                fr[y, a+dx:bb+dx+1] = base[y, a:bb+1]
                fr[y, bb+dx+1:bb+1] = base[y, bb+1] if bb+1<W else base[y, bb]
    ys, xs = np.nonzero(waterish)
    wv = np.sin(TAU*(3*t/N) + 0.50*ys + 0.35*xs + 1.3)
    sel = wv>0.86
    if sel.any():
        px = fr[ys[sel], xs[sel]].astype(np.int16) + np.array([14,18,16])
        fr[ys[sel], xs[sel]] = np.clip(px,0,255).astype(np.uint8)
    sel = wv<-0.92
    if sel.any():
        px = fr[ys[sel], xs[sel]].astype(np.int16) - np.array([8,10,10])
        fr[ys[sel], xs[sel]] = np.clip(px,0,255).astype(np.uint8)
    for (ys2, xs2, ph, k, amp) in padinfo:
        dx = int(round(amp*np.sin(TAU*(k*t/N+ph))))
        dy = int(round(amp*0.6*np.sin(TAU*(k*t/N+ph)+np.pi/2)))
        ny = ys2+dy; nx = xs2+dx
        ok = (ny>=0)&(ny<H)&(nx>=0)&(nx<W)
        fr[ny[ok], nx[ok]] = nat[ys2[ok], xs2[ok]]
    # tassels flick
    for (ys3, xs3, start, dur) in tass_info:
        if start <= t < start+dur:
            order = np.argsort(-xs3)
            for i in order:
                y, x = ys3[i], xs3[i]
                if x+1 < W: fr[y, x+1] = nat[y, x]
            # fill leftmost holes per row
            rows = np.unique(ys3)
            for y in rows:
                xr = xs3[ys3==y].min()
                fr[y, xr] = fr[y, xr-1] if xr>0 else fr[y, xr]
    # plants: per-leaf-unit articulated sway
    for uid, (y1, hgt, sz, k, ph, w2, ps2, ampT, j1, j2, f1, envA, envP, bias, maxdx, chunked) in enumerate(uinfo):
        th = TAU*(k*t/N + ph)
        sv = (1-w2)*np.sin(th) + w2*np.sin(2*th + ps2)
        env = 1 + envA*np.sin(TAU*tt[t] + envP)
        s2 = ampT*env*sv + bias
        d2 = int(max(-maxdx, min(maxdx, round(s2))))
        d1 = int(max(-maxdx, min(maxdx, round(s2*f1)))) if chunked else d2
        d0 = 0 if chunked else d2
        chunkdx[uid,0], chunkdx[uid,1], chunkdx[uid,2] = d0, d1, d2
    fr_bg = fr.copy()
    for uid in range(len(uinfo)):
        for ch in range(3):
            dx = chunkdx[uid, ch]
            cys, cxs, runs = UC[uid][ch]
            if len(cys)==0: continue
            if dx == 0:
                fr[cys, cxs] = nat[cys, cxs]
            else:
                for (y, a, bb) in runs:
                    if a+dx<0 or bb+dx>=W: continue
                    fr[y, a+dx:bb+dx+1] = nat[y, a:bb+1]
                    if dx>0:
                        fr[y, a:a+dx] = fr_bg[y, a-1] if a>0 else fr_bg[y, a]
                    else:
                        fr[y, bb+dx+1:bb+1] = fr_bg[y, bb+1] if bb+1<W else fr_bg[y, bb]
    frames.append(fr)
print('frames done')
np.save('frames.npy', np.stack(frames))

# reports
palm_chg = [int((np.abs(frames[t][290:470,0:110].astype(int)-frames[t-1][290:470,0:110].astype(int)).sum(2)>0).sum()) for t in range(1,N)]
print('palm region changed px per frame: min', min(palm_chg), 'max', max(palm_chg), 'mean', round(sum(palm_chg)/len(palm_chg),1))
allowed = quad | dilate(plant_mask,2) | dilate(tass,2)
worst = 0
for t in range(1,N):
    d = (np.abs(frames[t].astype(int)-frames[t-1].astype(int)).sum(2) > 0) & ~allowed
    worst = max(worst, int(d.sum()))
    if t <= 3 or d.sum()>0:
        ysx = np.nonzero(d)
        if len(ysx[0]): print('frame',t,'stray changed px', int(d.sum()), 'e.g.', list(zip(ysx[0][:6], ysx[1][:6])))
print('max stray changed px outside water/plants/tassels:', worst)
