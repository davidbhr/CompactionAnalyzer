"""Microscopy units and browser-to-desktop calibration regression checks."""
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest
import tifffile

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
from calibration import sigma_in_pixels, tiff_calibration


@pytest.mark.parametrize('unit,size', [('µm', .156), ('nm', 156.)])
def test_ome_size(tmp_path, unit, size):
    path = tmp_path / 'image.ome.tif'
    tifffile.imwrite(path, np.zeros((32, 32), np.uint16), ome=True,
                     metadata={'axes': 'YX', 'PhysicalSizeX': size, 'PhysicalSizeY': size,
                               'PhysicalSizeXUnit': unit, 'PhysicalSizeYUnit': unit})
    with tifffile.TiffFile(path) as handle:
        result = tiff_calibration(handle)
    assert result['pixel_size_um'] == pytest.approx(.156)
    assert result['source'] == 'OME-TIFF' and not result['assumed']


def test_imagej_and_display_dpi(tmp_path):
    image = np.zeros((32, 32), np.uint16)
    path = tmp_path / 'image.tif'
    tifffile.imwrite(path, image, imagej=True, resolution=(4., 4.), metadata={'unit': 'um'})
    with tifffile.TiffFile(path) as handle:
        assert tiff_calibration(handle)['pixel_size_um'] == .25
    tifffile.imwrite(path, image, resolution=(300, 300), resolutionunit='INCH')
    with tifffile.TiffFile(path) as handle:
        result = tiff_calibration(handle)
    assert result['assumed'] and result['pixel_size_um'] == 1.


@pytest.mark.parametrize('metadata', [
    {'PhysicalSizeX': .2, 'PhysicalSizeY': .4},
    {'PhysicalSizeX': .2},
    {},
])
def test_missing_or_non_square_calibration_is_marked(tmp_path, metadata):
    path = tmp_path / 'image.ome.tif'
    tifffile.imwrite(path, np.zeros((32, 32), np.uint16), ome=True, metadata=metadata)
    with tifffile.TiffFile(path) as handle:
        result = tiff_calibration(handle)
    assert result['assumed'] and result['pixel_size_um'] == 1.


def test_sigma_units_and_boundaries():
    assert sigma_in_pixels({'sigma': 7}) == 7
    for pixels in (.1, 7, 100):
        assert sigma_in_pixels({'sigma': pixels * .156, 'sigma_unit': 'µm',
                                'calibration': {'pixel_size_um': .156}}) == pytest.approx(pixels)


@pytest.mark.parametrize('size', [None, 0, -1, float('nan'), float('inf')])
def test_physical_sigma_rejects_invalid_size(size):
    with pytest.raises(ValueError, match='positive pixel size'):
        sigma_in_pixels({'sigma': 7, 'sigma_unit': 'µm', 'calibration': {'pixel_size_um': size}})


@pytest.mark.parametrize('settings', [{'sigma': 7, 'sigma_unit': 'mm'}, {'sigma': 0},
                                   {'sigma': 101}, {'sigma': float('nan')}])
def test_invalid_sigma(settings):
    with pytest.raises(ValueError):
        sigma_in_pixels(settings)


@pytest.mark.parametrize('mode', ['axis', 'point_orientation', 'line_orientation'])
def test_equivalent_units_and_export(mode):
    import bridge
    y, x = np.mgrid[:96, :96]
    image = np.sin((x + .4 * y) * .4) + .1 * np.random.default_rng(4).normal(size=x.shape)
    bridge.IMAGES.clear()
    bridge.IMAGES['orientation'] = {'array': image, 'name': 'orientation.tif'}
    request = dict(mode=mode, region='whole', sigma=4., sigma_unit='pixel', edge=10, blur=.5,
                   point=[48, 48], line=[[20, 20], [75, 60]],
                   calibration={'pixel_size_um': .5, 'source': 'manual', 'assumed': False})
    bridge.analyze(request)
    first = {key: np.array(value) for key, value in bridge.LAST['arrays'].items()}
    request.update(sigma=2., sigma_unit='µm')
    bridge.analyze(request)
    for key, value in first.items():
        np.testing.assert_allclose(bridge.LAST['arrays'][key], value, equal_nan=True)
    payload = dict(format='saenopy', scale=.5, calibration=request['calibration'])
    with zipfile.ZipFile(io.BytesIO(bridge.export_result(json.dumps(payload)))) as archive:
        assert json.loads(archive.read('calibration.json'))['pixel_size_um'] == .5
        with np.load(io.BytesIO(archive.read('measurement.saenopyOrientation')), allow_pickle=False) as result:
            assert result['pixel_size'] == .5
            assert result['orientation_parameters/sigma_tensor'] == 2.
            assert result['orientation_parameters/sigma_tensor_type'] == 'µm'
    with pytest.raises(ValueError, match='Run the analysis again'):
        bridge.export_result(json.dumps(dict(payload, calibration={'pixel_size_um': .25})))
