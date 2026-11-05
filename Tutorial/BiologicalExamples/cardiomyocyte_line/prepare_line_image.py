"""Reproduce the input for the whole-image cardiomyocyte line example.

Preparation record: rotate supplied siControl phalloidin image 1 by 20 degrees
counterclockwise, then crop 150 pixels from each side. This removes padded
corners before analysis; no segmentation or selection mask is used. Original
source TIFFs are retained unchanged. Coordinates refer to the cropped input.
"""
from pathlib import Path
import json
import numpy as np
import tifffile
from scipy.ndimage import rotate, distance_transform_edt


def prepare(folder=None):
    folder=Path(folder) if folder else Path(__file__).resolve().parent
    phalloidin=tifffile.imread(folder/'phalloidin_original.tif')
    if phalloidin.shape!=(1025,1025):
        raise ValueError('This preparation expects the supplied 1025 x 1025 image.')
    options=dict(angle=20.,reshape=False,mode='constant',cval=0,prefilter=False)
    rotated=rotate(phalloidin.astype(np.float32),order=1,**options)
    support=rotate(np.ones(phalloidin.shape,dtype=np.uint8),order=0,**options)>0
    margin=distance_transform_edt(np.pad(support,1))[1:-1,1:-1]
    crop=(slice(150,-150),)*2
    # Keep the whole input more than four tensor sigmas away from padding.
    assert margin[crop].min()>32
    image=rotated[crop].copy()
    tifffile.imwrite(folder/'orientation.tif',image,compression='deflate')
    metadata=dict(source_image=1,rotation_degrees_ccw=20.,pixel_size_um=1/6.411017,
                  image_interpolation='bilinear',crop_yx=[[150,875],[150,875]],
                  output_shape=list(image.shape),minimum_padding_distance_px=float(margin[crop].min()),
                  analysis_region='whole input image; no selection mask',
                  reference_line_xy=[[69,469],[655,255]])
    (folder/'preparation.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return image


if __name__=='__main__':
    prepare()
