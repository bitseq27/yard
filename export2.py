import numpy as np, os
from PIL import Image
frames = np.load('frames.npy')
nat = np.asarray(Image.open('native.png'))
N,H,W,_ = frames.shape
srcpal = Image.fromarray(nat).quantize(colors=128, method=Image.MEDIANCUT)
q = [Image.fromarray(f).quantize(palette=srcpal, dither=Image.Dither.NONE) for f in frames]
q2 = [im.resize((W*2,H*2), Image.NEAREST) for im in q]
q2[0].save('pre.gif', save_all=True, append_images=q2[1:], duration=90, loop=0, optimize=True)
import subprocess
subprocess.run(['gifsicle','-O3','pre.gif','-o','courtyard_animated.gif'], check=True)
print('gif MB', round(os.path.getsize('courtyard_animated.gif')/1e6,2))
os.makedirs('mp4frames', exist_ok=True)
for i,im in enumerate(q2): im.save(f'mp4frames/f{i:02d}.png')
subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate','100/9','-i','mp4frames/f%02d.png',
                '-c:v','libx264','-pix_fmt','yuv420p','-crf','12','-preset','veryslow',
                '-movflags','+faststart','courtyard_animated.mp4'], check=True)
print('mp4 MB', round(os.path.getsize('courtyard_animated.mp4')/1e6,2))
# montage: 6 timesteps of the planter bed + right palm to eyeball independence
regs = [(180,140,340,260)]
for (x0,y0,x1,y1) in regs:
    ts = [0,6,12,18,24,30]
    cw, ch = x1-x0, y1-y0
    mont = Image.new('RGB', (cw*3*2, ch*2*2))
    for i,t in enumerate(ts):
        c = Image.fromarray(frames[t]).crop((x0,y0,x1,y1)).resize((cw*2,ch*2), Image.NEAREST)
        mont.paste(c, ((i%3)*cw*2, (i//3)*ch*2))
    mont.save('debug_motion.png')
