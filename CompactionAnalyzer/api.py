"""Direct, in-memory analysis using the same calculations as the batch pipeline."""
import os
import numpy as np
import imageio.v2 as imageio
from skimage import color
from skimage.filters import gaussian

from . import CompactionFunctions as _cf
from .plotting import vectors_to_axis_angles, wrap_axis_angle, line_reference_angle


_DEFAULTS = dict(scale=1., sigma_tensor=None, edge=15, sigma_first_blur=.5,
                 norm1=1., norm2=99., mask_sigma=5., mask_threshold=.6, mask_invert=False,
                 segmention_thres=1., seg_gaus1=.5, seg_gaus2=100., seg_invert=False,
                 seg_iter=1, segmention_method='otsu', segmention_min_area=1000,
                 regional_max_correction=True, ignore_cell_outline=False,
                 min_dist=None, max_dist=None, shell_width=5.)
_CELL_PARAMETERS = {'segmention_thres', 'seg_gaus1', 'seg_gaus2', 'seg_invert',
                    'seg_iter', 'segmention_method', 'segmention_min_area',
                    'regional_max_correction', 'ignore_cell_outline', 'min_dist', 'max_dist'}
_MASK_PARAMETERS = {'mask_sigma', 'mask_threshold', 'mask_invert'}
_MAPS = {'angle_x', 'angle_y', 'deviation_x', 'deviation_y', 'alignment_x',
         'alignment_y', 'coherency', 'angle_cell', 'alignment_cell', 'angle_point', 'alignment_point',
         'angle_line', 'deviation_line', 'alignment_line'}
_OUTPUTS = _MAPS | {'mean_' + name for name in _MAPS} | {'selection_mask', 'cell_mask', 'distance_profile'}
_MODES = ('full', 'axis_orientation_only', 'axis_orientation_masked', 'point_orientation', 'line_orientation')


def _image(value, name):
    if isinstance(value, (str, os.PathLike)):
        value = imageio.imread(value)
    image = np.asarray(value)
    if image.ndim == 3 and image.shape[-1] == 3:
        image = color.rgb2gray(image)
    if image.ndim != 2 or min(image.shape) < 2:
        raise ValueError(f'{name} must be one 2D grayscale or RGB image, not a stack')
    if image.dtype.kind not in 'buif' or not np.isfinite(image).all():
        raise ValueError(f'{name} must contain finite real pixel values')
    return image.astype(float, copy=False)


def _positive(value, name, zero=False):
    if not np.isscalar(value) or not np.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f'{name} must be finite and {"nonnegative" if zero else "positive"}')


def _normalize(image, parameters):
    # A flat percentile range has no usable contrast. Avoid division by zero;
    # the tensor will correctly report undefined orientation/coherency.
    low, high = np.percentile(image, [parameters['norm1'], parameters['norm2']])
    if high <= low:
        return np.zeros(image.shape)
    return _cf.normalize(image, parameters['norm1'], parameters['norm2'])


def _mean(values, axial=False):
    values = values[np.isfinite(values)]
    if not values.size:
        return float('nan')
    if not axial:
        return float(values.mean())
    phase = np.deg2rad(2 * values)
    x, y = np.cos(phase).mean(), np.sin(phase).mean()
    if np.hypot(x, y) <= 1e-12:
        return float('nan')  # No unique mean direction, e.g. equal orthogonal fibers.
    return float(wrap_axis_angle(np.rad2deg(np.arctan2(y, x)) / 2))


def analyze_image(image, mask_image=None, *, mode='axis_orientation_only',
                  outputs='mean_alignment_x', mask=None, reference_point=None,
                  reference_line=None, **parameters):
    """Return requested scalars/maps from one image, without writing any files.

    image, mask_image : ndarray or path
        One grayscale/RGB orientation image and an optional registered image.
        In full mode the second image supplies cell segmentation. In masked
        axis mode it supplies the selection; omitting it uses the first image.
        Point/line modes optionally use it to select the analysis area.
    mode : 'full', 'axis_orientation_only', 'axis_orientation_masked', 'point_orientation', 'line_orientation'
        The three batch modes plus orientation to a chosen point or line. Full mode
        requires a real cell image or explicit cell mask; no dummy cell is used.
    outputs : str or iterable of str
        A string returns that value directly; an iterable returns a dictionary.
        Maps: angle_x/y, deviation_x/y, alignment_x/y, coherency, angle_cell,
        alignment_cell. Prefix any with mean_ for a scalar. selection_mask is
        the analyzed region; cell_mask is the segmented cell (full mode only).
        Cell-relative outputs require full mode. mean_angle_x/y are axial
        circular means; mean_deviation_x/y are mean absolute deviations.
        distance_profile (point mode only) returns 1D arrays with ring distances,
        mean_alignment_point, mean_angle_point, mean_coherency and pixel counts.
    mask : 2D boolean ndarray, optional
        Trusted mask in original image coordinates, instead of mask_image.
        In masked axis and point/line modes True selects pixels. In full mode True marks the
        cell; its centroid defines the reference and statistics exclude it.
    reference_point : (x, y), optional
        Required in point_orientation mode, in uncropped pixel coordinates
        (zero-based x right, y down; fractional pixels are accepted).
        An optional mask selects pixels without moving the point. Outputs angle_point
        and alignment_point use the same radial geometry as the cell center.
        The angle at the reference point itself is undefined.
    reference_line : ((x1, y1), (x2, y2)), optional
        Required in line_orientation mode. Two distinct points inside the
        original image, in the same zero-based pixel convention as
        reference_point, define a fixed reference axis, independent of their
        order or separation. angle_line is the signed axial angle in [-90, 90);
        deviation_line is its absolute value; alignment_line is +1 parallel
        and -1 perpendicular. An optional mask selects pixels without changing the line.
    **parameters
        Existing batch names/units: scale in µm/pixel, sigma_tensor and blur
        sigmas in pixels, edge in pixels, norm1/norm2 in percentiles. Default
        sigma_tensor=7/scale. Mask parameters: mask_sigma/threshold/invert.
        Full mode also accepts the original automatic segmentation parameters,
        ignore_cell_outline and min_dist/max_dist (pixels from the cell center).
        Point mode supports distance_profile with shell_width in pixels (default
        5). Rings start at the reference point; distance_px is the ring midpoint,
        distance_um is distance_px * scale. Empty rings have NaN means.
        Unknown, export-related or mode-inapplicable parameters are rejected.

    Returned maps are edge-cropped, with NaN outside the selected region or
    where undefined. Masks are boolean. Means weight finite pixels equally;
    an empty/undefined mean is NaN. The input arrays are never modified.
    No plots, output directories, temporary files, Excel, CSV or YAML are made.
    See docs/direct_api.md for the output table and examples.
    """
    if mode not in _MODES:
        raise ValueError(f'mode must be one of {_MODES}')
    single = isinstance(outputs, str)
    names = (outputs,) if single else tuple(outputs)
    if not names or any(not isinstance(name, str) or name not in _OUTPUTS for name in names):
        raise ValueError('outputs must contain names from: ' + ', '.join(sorted(_OUTPUTS)))
    if len(set(names)) != len(names):
        raise ValueError('outputs must not contain duplicate names')
    if mode != 'point_orientation' and ('distance_profile' in names or 'shell_width' in parameters):
        raise ValueError('distance_profile and shell_width require mode="point_orientation"')
    if mode != 'full' and any('cell' in name for name in names):
        raise ValueError('Cell-relative outputs require mode="full"')
    if mode != 'point_orientation' and any('point' in name for name in names):
        raise ValueError('Point-relative outputs require mode="point_orientation"')
    if mode != 'point_orientation' and reference_point is not None:
        raise ValueError('reference_point requires mode="point_orientation"')
    if mode != 'line_orientation' and any('line' in name for name in names):
        raise ValueError('Line-relative outputs require mode="line_orientation"')
    if mode != 'line_orientation' and reference_line is not None:
        raise ValueError('reference_line requires mode="line_orientation"')
    unknown = set(parameters) - _DEFAULTS.keys()
    if unknown:
        raise TypeError('Unknown parameters: ' + ', '.join(sorted(unknown)))
    invalid = set(parameters) & (_MASK_PARAMETERS if mode == 'full' else _CELL_PARAMETERS)
    if mode == 'point_orientation':
        invalid -= {'min_dist', 'max_dist'}
    if mode == 'axis_orientation_only' or (mode in ('point_orientation', 'line_orientation') and mask_image is None):
        invalid |= set(parameters) & _MASK_PARAMETERS
    if invalid:
        raise ValueError(f'Parameters not applicable to {mode}: ' + ', '.join(sorted(invalid)))
    if mask is not None and mask_image is not None:
        raise ValueError('Supply either mask or mask_image, not both')
    if mode == 'axis_orientation_only' and (mask is not None or mask_image is not None):
        raise ValueError('This mode takes one orientation image without a mask')
    if mode == 'full' and mask is None and mask_image is None:
        raise ValueError('Full mode requires a cell image in mask_image or a boolean cell mask')
    if mask is not None and set(parameters) & (_MASK_PARAMETERS | (_CELL_PARAMETERS - {'min_dist', 'max_dist', 'ignore_cell_outline'})):
        raise ValueError('Segmentation parameters do not apply to an already supplied boolean mask')

    p = dict(_DEFAULTS, **parameters)
    _positive(p['scale'], 'scale')
    _positive(p['shell_width'], 'shell_width')
    if p['sigma_tensor'] is None:
        p['sigma_tensor'] = 7 / p['scale']
    _positive(p['sigma_tensor'], 'sigma_tensor')
    _positive(p['sigma_first_blur'], 'sigma_first_blur', zero=True)
    if not 0 <= p['norm1'] < p['norm2'] <= 100:
        raise ValueError('Require 0 <= norm1 < norm2 <= 100')
    for name in ('min_dist', 'max_dist'):
        if p[name] is not None:
            _positive(p[name], name, zero=True)
    if p['min_dist'] is not None and p['max_dist'] is not None and p['min_dist'] > p['max_dist']:
        raise ValueError('min_dist must not exceed max_dist')

    original = _image(image, 'image')
    line_angle = line_reference_angle(reference_line, original.shape) if mode == 'line_orientation' else None
    edge = p['edge']
    if isinstance(edge, (bool, np.bool_)) or not isinstance(edge, (int, np.integer)) or edge < 0 or 2 * edge >= min(original.shape):
        raise ValueError('edge must be a nonnegative integer leaving a nonempty image')
    crop = (slice(edge, -edge),) * 2 if edge else (slice(None),) * 2
    source = original if mask_image is None else _image(mask_image, 'mask_image')
    if source.shape != original.shape:
        raise ValueError('image and mask_image must have identical pixel grids/shapes')
    cell_mask = None
    center = None
    if mask is not None:
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != original.shape:
            raise ValueError('mask must be boolean and match the uncropped image shape')
        mask = mask.copy()
    if mode == 'point_orientation':
        try:
            point = np.asarray(reference_point, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError('reference_point must be a finite (x, y) inside the image') from error
        if (point.shape != (2,) or not np.isfinite(point).all()
                or not 0 <= point[0] < original.shape[1] or not 0 <= point[1] < original.shape[0]):
            raise ValueError('reference_point must be a finite (x, y) inside the image')
        center = tuple(point - edge)
        distance, _, _, _ = _cf._cell_reference_coordinates(original[crop].shape, center)
        selected = np.ones(distance.shape, dtype=bool)
        if p['min_dist'] is not None:
            selected &= distance >= p['min_dist']
        if p['max_dist'] is not None:
            selected &= distance <= p['max_dist']
    elif mode == 'full':
        if mask is not None:
            if not mask.any():
                raise ValueError('A cell mask must contain at least one pixel')
            yy, xx = np.nonzero(mask)
            center = (float(xx.mean()) - edge, float(yy.mean()) - edge)
            cell_mask = mask
        else:
            for name in ('segmention_thres', 'seg_gaus1'):
                _positive(p[name], name, zero=name == 'seg_gaus1')
            if p['seg_gaus2'] is not None:
                _positive(p['seg_gaus2'], 'seg_gaus2', zero=True)
            if p['segmention_method'] not in ('otsu', 'yen', 'entropy'):
                raise ValueError('segmention_method must be otsu, yen or entropy')
            normalized = _normalize(source, p)
            if np.ptp(normalized) == 0:
                raise ValueError('The cell image has no usable contrast; supply a cell mask instead')
            try:
                segmented = _cf.segment_cell(normalized, thres=p['segmention_thres'],
                    seg_gaus1=p['seg_gaus1'], seg_gaus2=p['seg_gaus2'], seg_invert=p['seg_invert'],
                    seg_iter=p['seg_iter'], segmention_method=p['segmention_method'],
                    segmention_min_area=p['segmention_min_area'], regional_max_correction=p['regional_max_correction'],
                    show_segmentation=False)
            except ValueError as error:
                raise ValueError('Cell segmentation failed; adjust parameters or supply a boolean cell mask') from error
            cell_mask = segmented['mask']
            center = tuple(value - edge for value in segmented['centroid'])
        selected = ~cell_mask[crop]
        if p['min_dist'] is not None or p['max_dist'] is not None:
            yy, xx = np.indices(selected.shape)
            distance = np.hypot(xx - center[0], yy - center[1])
            if p['min_dist'] is not None:
                selected &= distance >= p['min_dist']
            if p['max_dist'] is not None:
                selected &= distance <= p['max_dist']
    elif mode == 'axis_orientation_masked':
        selected = mask[crop] if mask is not None else _cf.create_axis_mask(
            source[crop], p['mask_sigma'], p['mask_threshold'], p['mask_invert'])
    else:
        selected = np.ones(original[crop].shape, dtype=bool)

    if mode in ('point_orientation', 'line_orientation') and (mask is not None or mask_image is not None):
        region = mask[crop] if mask is not None else _cf.create_axis_mask(
            source[crop], p['mask_sigma'], p['mask_threshold'], p['mask_invert'])
        selected &= region

    cache = {'selection_mask': selected.copy()}
    if cell_mask is not None:
        cache['cell_mask'] = cell_mask[crop].copy()
    if any(name not in cache for name in names):
        filtered = gaussian(_normalize(original, p), sigma=p['sigma_first_blur'])
        if cell_mask is not None and not p['ignore_cell_outline']:
            filtered[cell_mask] = np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            coherency, _, minimum, _, _ = _cf.analyze_local(filtered, sigma=p['sigma_tensor'], size=0, filter_type='gaussian')
        coherency, minimum = coherency[crop], minimum[crop]
        cache['coherency'] = coherency

    def field(name):
        if name not in cache:
            if name in ('angle_x', 'angle_y'):
                cache['angle_x'], cache['angle_y'] = vectors_to_axis_angles(minimum)
            elif name == 'angle_line':
                cache[name] = wrap_axis_angle(field('angle_x') - line_angle)
            elif name == 'deviation_line':
                cache[name] = np.abs(field('angle_line'))
            elif name == 'alignment_line':
                cache[name] = _cf.axis_alignment(field('angle_line'))
            elif name in ('angle_cell', 'alignment_cell', 'angle_point', 'alignment_point'):
                _, _, dx, dy = _cf._cell_reference_coordinates(minimum.shape[:2], center)
                with np.errstate(invalid='ignore'):
                    suffix = 'point' if mode == 'point_orientation' else 'cell'
                    cache['angle_' + suffix], cache['alignment_' + suffix] = _cf._cell_orientation_fields(minimum, dx, dy)
            elif name.startswith('deviation_'):
                cache[name] = np.abs(field('angle_' + name[-1]))
            elif name.startswith('alignment_'):
                cache[name] = _cf.axis_alignment(field('angle_' + name[-1]))
        return cache[name]

    result = {}
    for name in names:
        if name == 'distance_profile':
            from .distance import point_distance_profile
            result[name] = point_distance_profile(field('alignment_point'), field('coherency'),
                center, mask=selected, shell_width=p['shell_width'], scale=p['scale'])
        elif name in ('selection_mask', 'cell_mask'):
            result[name] = cache[name]
        elif name.startswith('mean_'):
            result[name] = _mean(field(name[5:])[selected], axial=name in ('mean_angle_x', 'mean_angle_y', 'mean_angle_line'))
        else:
            result[name] = np.where(selected, field(name), np.nan)
    return result[names[0]] if single else result
