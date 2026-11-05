"""Camera-sized inputs stay at their original resolution."""
from pathlib import Path
import sys

import numpy as np
import pytest
import tifffile
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import bridge


@pytest.mark.parametrize('suffix', ['.tif', '.png', '.jpg'])
def test_camera_size_and_rgb_channels(tmp_path, suffix):
    # 6 MP is above the old limit; RGB channels must not count as extra pixels.
    shape = (2000, 3000)
    path = tmp_path / ('camera' + suffix)
    if suffix == '.tif':
        tifffile.imwrite(path, np.zeros(shape, np.uint16))
    else:
        Image.new('RGB', (shape[1], shape[0]), (80, 100, 120)).save(path)
    array = bridge.read_image(path, path.name)
    assert array.shape == shape
    assert np.isfinite(array).all()


def test_twelve_megapixels_and_larger_rejected(tmp_path):
    path = tmp_path / 'camera.tif'
    tifffile.imwrite(path, np.zeros((3000, 4000), np.uint8))
    assert bridge.read_image(path, path.name).shape == (3000, 4000)
    tifffile.imwrite(path, np.zeros((3001, 4000), np.uint8))
    with pytest.raises(ValueError, match='12 million pixels'):
        bridge.read_image(path, path.name)


def test_stacks_are_not_flattened(tmp_path):
    path = tmp_path / 'stack.tif'
    tifffile.imwrite(path, np.zeros((5, 32, 32), np.uint16), photometric='minisblack')
    with pytest.raises(ValueError, match='TIFF stack'):
        bridge.read_image(path, path.name)
