"""Numerical parity, selective returns and side effects of the direct API."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
import imageio.v2 as imageio

from CompactionAnalyzer import analyze_image
from CompactionAnalyzer import CompactionFunctions as cf
from CompactionAnalyzer.api import _mean
from CompactionAnalyzer.plotting import wrap_axis_angle


@pytest.fixture
def images():
    yy, xx = np.indices((128, 160), dtype=float)
    image = (10000 + 9000 * np.cos((xx + yy) * 2 * np.pi / 20)).astype(np.uint16)
    cell = np.zeros(image.shape, dtype=np.uint16)
    cell[42:86, 58:102] = 60000
    return image, cell


@pytest.mark.parametrize('mode', ['axis_orientation_only', 'axis_orientation_masked'])
@pytest.mark.parametrize('edge', [0, 10])
def test_axis_matches_pipeline_and_paths(images, tmp_path, mode, edge):
    image, cell = images
    f, m = tmp_path / 'fiber.tif', tmp_path / 'cell.tif'
    imageio.imwrite(f, image)
    imageio.imwrite(m, cell)
    masked = mode == 'axis_orientation_masked'
    reference = cf.StuctureAnalysisMain([str(f)], [str(m)] if masked else None,
        [str(tmp_path / 'reference')], analysis_mode=mode, sigma_tensor=4, edge=edge,
        plotting=False, SaveNumpy=False, SaveExcel=False, mode_saenopy=True)
    wanted = ['angle_x', 'angle_y', 'alignment_x', 'alignment_y', 'coherency',
              'mean_alignment_x', 'mean_alignment_y', 'mean_deviation_x',
              'mean_coherency', 'selection_mask']
    result = analyze_image(f, m if masked else None, mode=mode, outputs=wanted, sigma_tensor=4, edge=edge)
    assert list(result) == wanted
    mask = reference['validity_mask']
    np.testing.assert_array_equal(result['selection_mask'], mask)
    for output, key in [('angle_x', 'min_x'), ('angle_y', 'min_y'), ('coherency', 'coherency')]:
        np.testing.assert_allclose(result[output], np.where(mask, reference['arrays'][key], np.nan), atol=1e-12)
    for axis in ('x', 'y'):
        assert result[f'mean_alignment_{axis}'] == pytest.approx(reference['summary'][f'Alignment {axis}'])
        np.testing.assert_allclose(result[f'alignment_{axis}'], cf.axis_alignment(result[f'angle_{axis}']))
    assert result['mean_deviation_x'] == pytest.approx(np.nanmean(np.abs(result['angle_x'])))
    assert result['mean_coherency'] == pytest.approx(np.nanmean(result['coherency']))
    direct = analyze_image(image, cell if masked else None, mode=mode, outputs='angle_x', sigma_tensor=4, edge=edge)
    np.testing.assert_array_equal(direct, result['angle_x'])


def test_cell_matches_original_pipeline(images, tmp_path):
    image, cell = images
    f, m = tmp_path / 'fiber.tif', tmp_path / 'cell.tif'
    imageio.imwrite(f, image)
    imageio.imwrite(m, cell)
    p = dict(scale=.5, sigma_tensor=4, edge=10, seg_gaus1=1, seg_gaus2=None,
             segmention_min_area=20, regional_max_correction=False)
    reference = cf.StuctureAnalysisMain([str(f)], [str(m)], [str(tmp_path / 'reference')],
        plotting=False, SaveNumpy=False, SaveExcel=False, mode_saenopy=True, **p)
    result = analyze_image(image, cell, mode='full', outputs=['angle_x', 'angle_cell',
        'alignment_cell', 'mean_angle_cell', 'mean_alignment_cell', 'mean_coherency',
        'selection_mask', 'cell_mask'], **p)
    mask = result['selection_mask']
    np.testing.assert_allclose(result['angle_x'], np.where(mask, reference[0], np.nan))
    np.testing.assert_allclose(result['alignment_cell'], np.where(mask, reference[1], np.nan), atol=1e-12)
    assert result['mean_angle_cell'] == pytest.approx(reference[4]['Mean Angle'][0])
    assert result['mean_alignment_cell'] == pytest.approx(reference[4]['Orientation'][0])
    assert result['mean_coherency'] == pytest.approx(reference[4]['Mean Coherency'][0])
    np.testing.assert_array_equal(mask, ~result['cell_mask'])


def test_mask_sources_and_input_immutability(images, monkeypatch):
    image, second = images
    before = image.copy()
    same = analyze_image(image, mode='axis_orientation_masked', outputs=['angle_x', 'selection_mask'], sigma_tensor=4)
    explicit = analyze_image(image, image, mode='axis_orientation_masked', outputs='angle_x', sigma_tensor=4)
    np.testing.assert_array_equal(same['angle_x'], explicit)
    full_mask = np.zeros(image.shape, dtype=bool)
    full_mask[15:-15, 15:-15] = same['selection_mask']
    mask_before = full_mask.copy()
    monkeypatch.setattr(cf, 'create_axis_mask', lambda *a, **k: pytest.fail('Already segmented'))
    trusted = analyze_image(image, mask=full_mask, mode='axis_orientation_masked', outputs='angle_x', sigma_tensor=4)
    np.testing.assert_array_equal(trusted, explicit)
    np.testing.assert_array_equal(image, before)
    np.testing.assert_array_equal(full_mask, mask_before)


@pytest.mark.parametrize('mode', ['full', 'axis_orientation_only', 'axis_orientation_masked'])
def test_no_exports_and_single_tensor_call(images, tmp_path, monkeypatch, mode):
    image, cell = images
    options = dict(mode=mode)
    if mode != 'axis_orientation_only':
        options['mask'] = cell > 0
    monkeypatch.chdir(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail('Direct API must not invoke batch analysis or create exports')
    monkeypatch.setattr(cf, 'StuctureAnalysisMain', forbidden)
    monkeypatch.setattr(cf.os, 'makedirs', forbidden)
    monkeypatch.setattr(cf.np, 'save', forbidden)
    monkeypatch.setattr(cf.plt, 'figure', forbidden)
    original = cf.analyze_local
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(cf, 'analyze_local', counted)
    result = analyze_image(image, outputs=['mean_alignment_x', 'mean_angle_y', 'coherency'], sigma_tensor=4, **options)
    assert set(result) == {'mean_alignment_x', 'mean_angle_y', 'coherency'}
    assert calls == [1]
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(cf, 'analyze_local', forbidden)
    assert analyze_image(image, outputs='selection_mask', **options).dtype == bool


@pytest.mark.parametrize('value', [0, 13, 89, -89, -45, -90])
def test_known_directions_and_axial_means(value):
    yy, xx = np.indices((160, 160), dtype=float)
    theta = np.deg2rad(value)
    image = np.cos((xx * np.sin(theta) + yy * np.cos(theta)) * 2 * np.pi / 24)
    result = analyze_image(image, outputs=['mean_angle_x', 'mean_angle_y', 'mean_deviation_x'],
                           sigma_tensor=4, edge=30)
    assert abs(wrap_axis_angle(result['mean_angle_x'] - value)) < .2
    assert abs(wrap_axis_angle(result['mean_angle_y'] - value + 90)) < .2
    assert result['mean_deviation_x'] == pytest.approx(abs(value), abs=.2)
    assert abs(abs(_mean(np.array([89., -89.]), axial=True)) - 90) < 1e-10
    assert np.isnan(_mean(np.array([0., 90.]), axial=True))


def test_empty_flat_and_isotropic_are_not_false_alignment(images):
    image, _ = images
    empty = analyze_image(image, mask=np.zeros(image.shape, dtype=bool), mode='axis_orientation_masked',
                          outputs=['angle_x', 'mean_alignment_x', 'mean_coherency'], sigma_tensor=4)
    assert all(np.isnan(value).all() for value in empty.values())
    flat = analyze_image(np.ones((40, 40)), outputs=['mean_angle_x', 'mean_coherency'])
    assert all(np.isnan(value) for value in flat.values())


def test_cell_masks_distances_and_zero_crop(images, monkeypatch):
    image, cell = images
    mask = cell > 0
    monkeypatch.setattr(cf, 'segment_cell', lambda *a, **k: pytest.fail('Use the trusted mask'))
    result = analyze_image(image, mask=mask, mode='full', outputs=['angle_cell', 'alignment_cell',
        'selection_mask', 'cell_mask'], sigma_tensor=4, edge=0, min_dist=30, max_dist=45)
    yy, xx = np.indices(image.shape)
    distance = np.hypot(xx - 79.5, yy - 63.5)
    np.testing.assert_array_equal(result['selection_mask'], (~mask) & (distance >= 30) & (distance <= 45))
    assert result['angle_cell'].shape == image.shape
    assert np.isnan(result['alignment_cell'][~result['selection_mask']]).all()


@pytest.mark.parametrize('kwargs', [
    dict(mode='unknown'), dict(outputs='wrong'), dict(outputs=[]),
    dict(outputs=['angle_x', 'angle_x']), dict(outputs='angle_cell'),
    dict(mode='full'), dict(edge=-1), dict(edge=64), dict(edge=True),
    dict(sigma_tensor=0), dict(scale=0), dict(norm1=99, norm2=1),
    dict(max_dist=20), dict(mask_threshold=.5), dict(plotting=False),
    dict(mask=np.ones((128,160), dtype=bool)),
    dict(mode='axis_orientation_masked', mask=np.ones((128,160))),
    dict(mode='axis_orientation_masked', mask_image=np.ones((20,20))),
    dict(mode='full', mask=np.zeros((128,160), dtype=bool)),
])
def test_invalid_calls_fail_clearly(images, kwargs):
    with pytest.raises((ValueError, TypeError)):
        analyze_image(images[0], **kwargs)


def test_concurrent_calls_and_rgb(images):
    image, _ = images
    rgb = np.stack([image] * 3, axis=-1)
    sequential = [analyze_image(data, outputs='mean_alignment_x', sigma_tensor=4) for data in (image, rgb)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        parallel = list(pool.map(lambda data: analyze_image(data, outputs='mean_alignment_x', sigma_tensor=4), (image, rgb)))
    np.testing.assert_allclose(sequential, parallel)
    np.testing.assert_allclose(sequential[0], sequential[1], atol=1e-12)


@pytest.mark.parametrize('edge', [0, 10])
def test_point_uses_existing_radial_geometry(images, monkeypatch, edge):
    image, _ = images
    def tensor(image, **kwargs):
        vectors = np.zeros((*image.shape, 2))
        vectors[..., 0] = 1
        return np.ones(image.shape), None, vectors, None, None
    monkeypatch.setattr(cf, 'analyze_local', tensor)
    monkeypatch.setattr(cf, 'segment_cell', lambda *a, **k: pytest.fail('Point mode must not segment'))
    point = (80, 64)
    values = analyze_image(image, mode='point_orientation', reference_point=point,
        outputs=['alignment_point', 'angle_point', 'mean_alignment_point', 'angle_x', 'selection_mask'], edge=edge)
    angle, alignment = values['angle_point'], values['alignment_point']
    cx, cy = point[0]-edge, point[1]-edge
    assert np.isnan(angle[cy,cx])
    assert angle[cy,cx+1] == pytest.approx(0)
    assert angle[cy+1,cx] == pytest.approx(90)
    assert alignment[cy,cx+1] == pytest.approx(1)
    assert alignment[cy+1,cx] == pytest.approx(-1)
    assert alignment[cy+1,cx+1] == pytest.approx(0, abs=1e-14)
    assert values['mean_alignment_point'] == pytest.approx(np.nanmean(alignment))
    np.testing.assert_allclose(values['angle_x'], 0)
    assert values['selection_mask'].all()


def test_point_matches_cell_center_without_segmentation(images):
    image, _ = images
    mask = np.zeros(image.shape, dtype=bool)
    mask[64,80] = True
    cell = analyze_image(image, mask=mask, mode='full', ignore_cell_outline=True,
        outputs=['alignment_cell','angle_cell'], sigma_tensor=4, edge=10)
    point = analyze_image(image, mode='point_orientation', reference_point=(80,64),
        outputs=['alignment_point','angle_point'], sigma_tensor=4, edge=10)
    np.testing.assert_allclose(cell['alignment_cell'], point['alignment_point'], atol=1e-12)
    np.testing.assert_allclose(cell['angle_cell'], point['angle_point'], atol=1e-12)


@pytest.mark.parametrize('point', [None, (1,), (1,2,3), (-1,0), (160,0), (0,128), (np.nan,2)])
def test_point_must_be_inside_original_image(images, point):
    with pytest.raises(ValueError, match='reference_point'):
        analyze_image(images[0], mode='point_orientation', reference_point=point)


def test_point_distance_limits_and_mode_parameters(images):
    image, cell = images
    selection = analyze_image(image, mode='point_orientation', reference_point=(0,0),
        outputs='selection_mask', edge=10, min_dist=20, max_dist=50)
    yy,xx = np.indices(selection.shape)
    np.testing.assert_array_equal(selection, ((xx+10)**2+(yy+10)**2 >= 20**2) & ((xx+10)**2+(yy+10)**2 <= 50**2))
    for kwargs in [dict(mask_sigma=1), dict(seg_gaus1=1)]:
        with pytest.raises(ValueError):
            analyze_image(image, mode='point_orientation', reference_point=(80,64), **kwargs)
    with pytest.raises(ValueError, match='reference_point'):
        analyze_image(image, reference_point=(80,64))
    with pytest.raises(ValueError, match='Point-relative'):
        analyze_image(image, outputs='mean_angle_point')


@pytest.mark.parametrize('line,axis', [(((5, 8), (100, 8)), 'x'), (((20, 2), (20, 100)), 'y')])
@pytest.mark.parametrize('edge', [0, 10])
def test_line_matches_image_axes(images, line, axis, edge):
    image, _ = images
    values = analyze_image(image, mode='line_orientation', reference_line=line,
        outputs=['angle_line', 'deviation_line', 'alignment_line', 'mean_deviation_line',
                 'mean_alignment_line', 'mean_angle_line', 'angle_' + axis, 'alignment_' + axis],
        edge=edge, sigma_tensor=4)
    np.testing.assert_allclose(values['angle_line'], values['angle_' + axis], atol=1e-12)
    np.testing.assert_allclose(values['alignment_line'], values['alignment_' + axis], atol=1e-12)
    assert values['mean_deviation_line'] == pytest.approx(np.nanmean(np.abs(values['angle_line'])))
    assert values['mean_alignment_line'] == pytest.approx(np.nanmean(values['alignment_line']))
    assert values['mean_angle_line'] == pytest.approx(_mean(values['angle_line'], axial=True))


@pytest.mark.parametrize('line,expected', [(((10, 100), (100, 10)), -45),
                                         (((10, 10), (100, 100)), 45),
                                         (((100, 10), (10, 100)), -45)])
def test_line_direction_is_constant_and_endpoint_order_invariant(images, monkeypatch, line, expected):
    def horizontal(image, **kwargs):
        vectors = np.zeros((*image.shape, 2))
        vectors[..., 0] = 1
        return np.ones(image.shape), None, vectors, None, None
    monkeypatch.setattr(cf, 'analyze_local', horizontal)
    result = analyze_image(images[0], mode='line_orientation', reference_line=line,
        outputs=['angle_line', 'alignment_line', 'deviation_line', 'selection_mask'], edge=10)
    np.testing.assert_allclose(result['angle_line'], expected)
    np.testing.assert_allclose(result['deviation_line'], 45)
    np.testing.assert_allclose(result['alignment_line'], 0)
    assert result['selection_mask'].all()  # Not restricted to the segment or its vicinity.


@pytest.mark.parametrize('line', [None, (1, 2), ((1, 2),), ((1, 2), (1, 2)),
    ((-1, 0), (20, 20)), ((0, 0), (160, 128)), ((0, 0), (1, np.nan)), 'wrong'])
def test_line_rejects_invalid_points(images, line):
    with pytest.raises(ValueError, match='reference_line'):
        analyze_image(images[0], mode='line_orientation', reference_line=line)


def test_line_shared_tensor_no_side_effects_and_parameter_scope(images, tmp_path, monkeypatch):
    image, cell = images
    original = image.copy()
    calls = []
    tensor = cf.analyze_local
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cf, 'analyze_local', lambda *a, **k: (calls.append(1), tensor(*a, **k))[1])
    monkeypatch.setattr(cf, 'segment_cell', lambda *a, **k: pytest.fail('No line segmentation'))
    values = analyze_image(image, mode='line_orientation', reference_line=((10, 100), (100, 10)),
        outputs=['mean_alignment_line', 'angle_line', 'angle_x', 'mean_coherency'], sigma_tensor=4)
    assert calls == [1]
    np.testing.assert_array_equal(image, original)
    assert not list(tmp_path.iterdir())
    assert np.isfinite(values['mean_alignment_line'])
    for kwargs in [dict(min_dist=10), dict(max_dist=50), dict(mask_sigma=1),
                   dict(reference_point=(20, 20)), dict(outputs='angle_point')]:
        with pytest.raises(ValueError):
            analyze_image(image, mode='line_orientation', reference_line=((0, 0), (100, 100)), **kwargs)
    with pytest.raises(ValueError, match='reference_line'):
        analyze_image(image, reference_line=((0, 0), (100, 100)))
    with pytest.raises(ValueError, match='Line-relative'):
        analyze_image(image, outputs='angle_line')


@pytest.mark.parametrize('direction', [-67, -25, 13, 72])
def test_line_known_fibers_fractional_pixels_and_parallel_calls(direction):
    """Check the complete tensor calculation against known oblique fibers."""
    yy, xx = np.indices((180, 220), dtype=float)
    theta = np.deg2rad(direction)
    image = np.cos((xx*np.sin(theta) + yy*np.cos(theta)) * 2*np.pi/24)
    center = np.array([110.25, 90.75])
    along = np.array([np.cos(theta), -np.sin(theta)]) * 40.5
    line = np.stack([center-along, center+along])
    translated = line + [3.5, -6.25]
    perpendicular = np.array([np.sin(theta), np.cos(theta)]) * 40.5
    cross = np.stack([center-perpendicular, center+perpendicular])
    def call(reference):
        return analyze_image(image, mode='line_orientation', reference_line=reference,
            outputs=['mean_alignment_line', 'mean_deviation_line', 'angle_line'],
            edge=30, sigma_tensor=4)
    parallel = call(line)
    assert parallel['mean_alignment_line'] > .99
    assert parallel['mean_deviation_line'] < .2
    assert parallel['angle_line'].shape == (120, 160)
    with ThreadPoolExecutor(max_workers=3) as pool:
        reversed_line, shifted_line, orthogonal = list(pool.map(call, [line[::-1], translated, cross]))
    np.testing.assert_allclose(reversed_line['angle_line'], parallel['angle_line'], atol=1e-12)
    np.testing.assert_allclose(shifted_line['angle_line'], parallel['angle_line'], atol=1e-12)
    assert orthogonal['mean_alignment_line'] < -.99
    assert orthogonal['mean_deviation_line'] > 89.8


def test_point_fractional_original_coordinates_with_real_tensor():
    yy, xx = np.indices((128, 160), dtype=float)
    image = np.cos(yy * 2*np.pi/20)  # Horizontal fibers.
    point = (80.25, 63.75)
    values = analyze_image(image, mode='point_orientation', reference_point=point,
        outputs=['angle_point', 'alignment_point'], sigma_tensor=4, edge=20)
    # Expected unsigned angle to the radial vector, in original coordinates.
    expected = np.degrees(np.arctan2(np.abs(yy-point[1]), np.abs(xx-point[0])))[20:-20, 20:-20]
    np.testing.assert_allclose(values['angle_point'], expected, atol=1e-5)
    np.testing.assert_allclose(values['alignment_point'], 1-expected/45, atol=1e-7)
