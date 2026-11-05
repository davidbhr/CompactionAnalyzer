"""Reproduce the grayscale images and selection masks from the supplied PNGs.

Run from any directory. The original source.png files are never modified.
These are display-derived examples, not recovered raw fluorescence intensities.
"""
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import tifffile
from scipy.ndimage import gaussian_filter

root = Path(__file__).resolve().parent
for case, point in [('spheroid', (1200, 910)), ('lamellipodia', (390, 325))]:
    image = imageio.imread(root/case/'source.png')[:, :, 0]  # Red display channel.
    yy, xx = np.indices(image.shape)
    radius = np.hypot(xx-point[0], yy-point[1])
    if case == 'spheroid':
        mask = (gaussian_filter(image.astype(float), 2) > 70) & (radius > 290) & (radius < 650)
    else:
        mask = (gaussian_filter(image.astype(float), 1) > 35) & (radius > 80) & (radius < 200)
    tifffile.imwrite(root/case/'orientation.tif', image, compression='deflate')
    tifffile.imwrite(root/case/'selection.tif', mask.astype(np.uint8)*255, compression='deflate')
