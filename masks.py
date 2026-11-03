import numpy as np
from PIL import Image
from collections import deque

nat = np.asarray(Image.open('native.png')).astype(np.int32)
H, W, _ = nat.shape
R, G, B = nat[:,:,0], nat[:,:,1], nat[:,:,2]
arr = (nat/255.0).astype(np.float32)
cmax = arr.max(2); cmin = arr.min(2); df = cmax-cmin
r,g,b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
hue = np.zeros((H,W))
m=(cmax==r)&(df>0); hue[m]=((g-b)[m]/df[m])%6
m=(cmax==g)&(df>0); hue[m]=((b-r)[m]/df[m])+2
m=(cmax==b)&(df>0); hue[m]=((r-g)[m]/df[m])+4
hue*=60
sat = np.where(cmax>0, df/np.maximum(cmax,1e-6), 0)
val = cmax

green = (hue>=60)&(hue<=175)&(sat>0.25)&(val>0.12)&((B-R)<40)
# exclude floor tile palette sampled from pure-floor patches
patch = np.zeros((H,W), bool)
patch[242:278, 142:196] = True
patch[246:268, 96:140] = True
floor_cols = np.unique(nat[patch & green].reshape(-1,3), axis=0)
dif = nat.astype(np.int32)
excl = np.zeros((H,W), bool)
for fc in floor_cols:
    excl |= (np.abs(dif - fc[None,None,:]).sum(2) <= 12)
green &= ~excl

quad_pts = [(259,327),(327,362),(252,399),(184,362)]
def poly_mask(pts):
    ys, xs = np.mgrid[0:H,0:W]
    inside = np.zeros((H,W), bool)
    n=len(pts); j=n-1
    for i in range(n):
        xi,yi = pts[i]; xj,yj = pts[j]
        cond = ((yi>ys)!=(yj>ys)) & (xs < (xj-xi)*(ys-yi)/(yj-yi+1e-9)+xi)
        inside ^= cond
        j=i
    return inside
quad = poly_mask(quad_pts)
quad_er = quad.copy()
for _ in range(3):
    q2 = quad_er.copy()
    quad_er[1:,:] &= q2[:-1,:]; quad_er[:-1,:] &= q2[1:,:]
    quad_er[:,1:] &= quad_er[:,:-1]; quad_er[:,:-1] &= quad_er[:,1:]

lab = -np.ones((H,W), np.int32); comps=[]
for y0 in range(H):
    for x0 in range(W):
        if green[y0,x0] and lab[y0,x0]<0:
            cid=len(comps); q=deque([(y0,x0)]); lab[y0,x0]=cid; pix=[(y0,x0)]
            while q:
                y,x=q.popleft()
                for dy in(-1,0,1):
                    for dx in(-1,0,1):
                        ny,nx=y+dy,x+dx
                        if 0<=ny<H and 0<=nx<W and green[ny,nx] and lab[ny,nx]<0:
                            lab[ny,nx]=cid; q.append((ny,nx)); pix.append((ny,nx))
            comps.append(pix)
pad_mask = np.zeros((H,W), bool)
plant_green = np.zeros((H,W), bool)
for pix in comps:
    a2 = np.array(pix); ys_, xs_ = a2[:,0], a2[:,1]
    if len(pix)>=4 and quad[ys_,xs_].mean()>0.98 and quad_er[ys_,xs_].mean()>0.9:
        pad_mask[ys_,xs_] = True
    elif len(pix)>=4:
        plant_green[ys_,xs_] = True
pad_d = pad_mask.copy()
for _ in range(2):
    d2 = pad_d.copy()
    pad_d[1:,:] |= d2[:-1,:]; pad_d[:-1,:] |= d2[1:,:]; pad_d[:,1:] |= d2[:,:-1]; pad_d[:,:-1] |= d2[:,1:]
white = (sat<0.25)&(val>0.75)
pad_mask = pad_mask | (white & quad & pad_d)

pink = (hue>=290)&(hue<=350)&(sat>0.25)&(val>0.35)
orange = (hue>=15)&(hue<=45)&(sat>0.45)&(val>0.5)
flow_col = pink|white|orange
d = green.copy()
for _ in range(2):
    d2 = d.copy()
    d[1:,:] |= d2[:-1,:]; d[:-1,:] |= d2[1:,:]; d[:,1:] |= d2[:,:-1]; d[:,:-1] |= d2[:,1:]
flowers = flow_col & d
plant_mask = plant_green | (flowers & ~quad)

brown = (hue>=10)&(hue<=32)&(sat>0.35)&(val>0.2)&(val<0.75)&(R>G)&(G>=B)
water_anim = (quad | pad_mask) & ~brown

base = nat.copy().astype(np.uint8)
for y in range(H):
    row = pad_mask[y]; x=0
    while x<W:
        if row[x]:
            x2=x
            while x2+1<W and row[x2+1]: x2+=1
            lc = base[y, x-1] if x>0 else base[y, x2+1]
            rc = base[y, x2+1] if x2+1<W else base[y, x-1]
            for xx in range(x, x2+1):
                t=(xx-x+1)/(x2-x+2)
                base[y,xx] = (lc*(1-t)+rc*t).astype(np.uint8)
            x=x2+1
        else: x+=1
np.save('masks.npy', np.stack([plant_mask, pad_mask, water_anim, quad]))
Image.fromarray(base).save('base_inpainted.png')
print('masks: plant', plant_mask.sum(), 'pad', pad_mask.sum(), 'water', water_anim.sum())
