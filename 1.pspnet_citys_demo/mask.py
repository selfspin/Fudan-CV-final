import os
import matplotlib.image as mpimg
import numpy as np

file_path = r'data'
save_path = r'out'
filelist = os.listdir(file_path)
for i in filelist:
    img_path = "data/" + i
    mask_path = "mask/" + i + ".png"
    save_path = "out/" + i
    base = mpimg.imread(img_path)
    mask = mpimg.imread(mask_path)
    w = 0.4
    out = w * base/255 + (1-w) * mask[..., :3]
    out = np.clip(out, 0, 1)
    mpimg.imsave(save_path, out)
    print(i + " finish")
