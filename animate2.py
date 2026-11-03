import numpy as np
from PIL import Image
from collections import deque

nat = np.asarray(Image.open('native.png')).astype(np.uint8)
base = np.asarray(Image.open('base_inpainted.png')).astype(np.uint8)
M = np.load('masks.npy')
plant_mask, pad_mask, water_anim, quad = M[0], M[1], M[2], M[3]
H, W, _ = nat.shape
TAU = np.pi*2
N = 32

arr = nat.astype(np.float32)/255.0
cmax = arr.max(2); cmin = arr.min(2); df = cmax-cmin
r,g,b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
hue = np.zeros((H,W))
m=(cmax==r)&(df>0); hue[m]=((g-b)[m]/df[m])%6
m=(cmax==g)&(df>0); hue[m]=((b-r)[m]/df[m])+2
m=(cmax==b)&(df>0); hue[m]=((r-g)[m]/df[m])+4
hue*=60
sat = np.where(cmax>0, df/np.maximum(cmax,1e-6), 0)
green_all = (hue>=60)&(hue<=175)&(sat>0.25)&((nat[:,:,2].astype(int)-nat[:,:,0].astype(int))<40)
brown = (hue>=10)&(hue<=32)&(sat>0.35)&(nat[:,:,0]>nat[:,:,1])&(nat[:,:,1]>=nat[:,:,2])
waterish = water_anim & ~green_all & ~brown

def label(mask):
    lab = -np.ones((H,W), np.int32); comps=[]
    for y0 in range(H):
        for x0 in range(W):
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

plant_comps, plant_lab = label(plant_mask)
pad_comps, _ = label(pad_mask)
print('comps', len(plant_comps), 'pads', len(pad_comps))

rng = np.random.default_rng(77)
pinfo=[]   # y1, hgt, size, params...
for pix in plant_comps:
    ys, xs = pix[:,0], pix[:,1]
    y0, y1 = int(ys.min()), int(ys.max())
    hgt = max(y1-y0, 1); sz = len(pix)
    # speed: big plants slow, small ones quicker
    if sz > 600:   k = 1
    elif sz > 120: k = int(rng.choice([1,1,2]))
    elif sz > 30:  k = int(rng.choice([1,2,2]))
    else:          k = int(rng.choice([2,2,3]))
    ph   = rng.random()                      # main phase
    w2   = rng.uniform(0.10, 0.38)           # 2nd-harmonic mix (asymmetric waveform)
    ps2  = rng.uniform(0, TAU)               # 2nd-harmonic phase
    ampT = rng.uniform(0.85, 1.15)           # sway amplitude
    j1   = rng.uniform(0.30, 0.45)           # first bend joint (fraction of height)
    j2   = rng.uniform(0.62, 0.80)           # second bend joint
    f1   = rng.uniform(0.45, 0.65)           # mid-chunk sway factor
    envA = rng.uniform(0.10, 0.32)           # gust envelope depth
    envP = rng.uniform(0, TAU)               # gust phase
    bias = rng.uniform(-0.12, 0.12)          # slight lean asymmetry
    pinfo.append((y1, hgt, sz, k, ph, w2, ps2, ampT, j1, j2, f1, envA, envP, bias))

padinfo=[]
for i, pix in enumerate(pad_comps):
    ys, xs = pix[:,0], pix[:,1]
    ph = ((xs.mean()*11 + ys.mean()*5) % 32)/32.0
    k = int(rng.choice([1,1,2]))
    amp = rng.uniform(0.6, 1.0)
    padinfo.append((ys, xs, ph, k, amp))

def runs_of(mask):
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
plant_runs = runs_of(plant_mask)
water_runs = runs_of(water_anim)

# precompute per-comp per-frame gust envelope and harmonic tables
tt = np.arange(N)/N
chunkdx = np.zeros((len(pinfo),3), np.int32)
frames=[]
for t in range(N):
    fr = base.copy()
    # water ripple (unchanged, already organic)
    for y, rl in water_runs:
        for (a,bb) in rl:
            c=(a+bb)//2
            dx = int(round(0.7*np.sin(TAU*(2*t/N) + 0.45*y) + 0.45*np.sin(TAU*(t/N) + 0.30*c)))
            if dx==0 or a+dx<0 or bb+dx>=W: continue
            if dx>0:
                fr[y, a+dx:bb+dx+1] = base[y, a:bb+1]
                fr[y, a:a+dx] = base[y, a-1] if a>0 else base[y, a]
            else:
                fr[y, a+dx:bb+dx+1] = base[y, a:bb+1]
                fr[y, bb+dx+1:bb+1] = base[y, bb+1] if bb+1<W else base[y, bb]
    ys, xs = np.nonzero(waterish)
    wv = np.sin(TAU*(2*t/N) + 0.50*ys + 0.35*xs + 1.3)
    sel = wv>0.86
    if sel.any():
        px = fr[ys[sel], xs[sel]].astype(np.int16) + np.array([14,18,16])
        fr[ys[sel], xs[sel]] = np.clip(px,0,255).astype(np.uint8)
    sel = wv<-0.92
    if sel.any():
        px = fr[ys[sel], xs[sel]].astype(np.int16) - np.array([8,10,10])
        fr[ys[sel], xs[sel]] = np.clip(px,0,255).astype(np.uint8)
    # lily pads: individual bob speed/amplitude/phase
    for (ys2, xs2, ph, k, amp) in padinfo:
        dx = int(round(amp*np.sin(TAU*(k*t/N+ph))))
        dy = int(round(amp*0.85*np.sin(TAU*(k*t/N+ph)+np.pi/2)))
        if dx==0 and dy==0: continue
        ny = ys2+dy; nx = xs2+dx
        ok = (ny>=0)&(ny<H)&(nx>=0)&(nx<W)
        fr[ny[ok], nx[ok]] = nat[ys2[ok], xs2[ok]]
    # plants: articulated rigid sway (no per-row shear -> no tearing)
    for cid, (y1, hgt, sz, k, ph, w2, ps2, ampT, j1, j2, f1, envA, envP, bias) in enumerate(pinfo):
        th = TAU*(k*t/N + ph)
        sv = (1-w2)*np.sin(th) + w2*np.sin(2*th + ps2)
        env = 1 + envA*np.sin(TAU*tt[t] + envP)
        s2 = ampT*env*sv + bias
        d2 = int(max(-1, min(1, round(s2))))          # top chunk
        d1 = int(max(-1, min(1, round(s2*f1))))       # middle chunk
        chunkdx[cid, 0] = 0; chunkdx[cid, 1] = d1; chunkdx[cid, 2] = d2
    for y, rl in plant_runs:
        for (a,bb) in rl:
            cid = plant_lab[y, (a+bb)//2]
            y1, hgt = pinfo[cid][0], pinfo[cid][1]
            fy = (y1-y)/hgt
            j1 = pinfo[cid][8]; j2 = pinfo[cid][9]
            ch = 0 if fy < j1 else (1 if fy < j2 else 2)
            dx = chunkdx[cid, ch]
            if dx == 0: continue
            if a+dx<0 or bb+dx>=W: continue
            if dx>0:
                fr[y, a+dx:bb+dx+1] = nat[y, a:bb+1]
                fr[y, a:a+dx] = fr[y, a-1] if a>0 else fr[y, a]
            else:
                fr[y, a+dx:bb+dx+1] = nat[y, a:bb+1]
                fr[y, bb+dx+1:bb+1] = fr[y, bb+1] if bb+1<W else fr[y, bb]
    frames.append(fr)
print('frames done')
np.save('frames.npy', np.stack(frames))

# diversity report
for cid in [0, 5, 40, 100, 200]:
    if cid >= len(pinfo): continue
    y1, hgt, sz, k, ph, w2, ps2, ampT, j1, j2, f1, envA, envP, bias = pinfo[cid]
    trace=[]
    for t in range(N):
        th = TAU*(k*t/N + ph)
        sv = (1-w2)*np.sin(th) + w2*np.sin(2*th+ps2)
        env = 1 + envA*np.sin(TAU*tt[t] + envP)
        trace.append(int(max(-1,min(1,round(ampT*env*sv + bias)))))
    print(f'comp {cid} sz={sz} k={k} top-dx trace={trace}')
