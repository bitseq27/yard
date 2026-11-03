import numpy as np
from PIL import Image
img = Image.open('uploads/iso_00080_.png').convert('RGB')
a = np.asarray(img).astype(np.uint8)
Image.fromarray(a[::4, ::4]).save('native.png')
print('native saved')
