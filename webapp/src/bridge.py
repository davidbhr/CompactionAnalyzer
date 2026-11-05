"""Browser I/O and plots. All measurements call the existing public Python API."""
import base64
import csv
import io
import json
import time
import zipfile

import numpy as np
from PIL import Image
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from CompactionAnalyzer import analyze_image
from calibration import assumed_calibration, tiff_calibration, sigma_in_pixels

IMAGES = {}
LAST = None
MAX_PIXELS = 12_000_000
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelcolor': '#475569', 'text.color': '#172b43',
                     'xtick.color': '#64748b', 'ytick.color': '#64748b',
                     'figure.facecolor': 'white', 'axes.facecolor': 'white'})


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def gray_preview(array):
    low, high = np.percentile(array, [1, 99])
    return np.clip((array - low) / (high - low), 0, 1) if high > low else np.zeros(array.shape)


def png(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format='PNG')
    return base64.b64encode(stream.getvalue()).decode()


def read_image(path, filename, return_calibration=False):
    # Inspect dimensions before decoding; stacks are deliberately not flattened.
    calibration = assumed_calibration()
    if filename.lower().endswith(('.tif', '.tiff')):
        with tifffile.TiffFile(path) as handle:
            series = handle.series[0]
            shape = series.shape
            if len(handle.pages) != 1:
                raise ValueError('Choose one 2D image, not a TIFF stack. Export one plane first.')
            if len(shape) not in (2, 3) or np.prod(shape[:2]) > MAX_PIXELS:
                raise ValueError('Use one 2D image with at most 12 million pixels (for example 4000 × 3000). For larger images, use the desktop GUI or Python API.')
            array = series.asarray()
            calibration = tiff_calibration(handle)
    else:
        with Image.open(path) as handle:
            if handle.width * handle.height > MAX_PIXELS:
                raise ValueError('Use an image with at most 12 million pixels (for example 4000 × 3000). For larger images, use the desktop GUI or Python API.')
            if getattr(handle, 'n_frames', 1) != 1:
                raise ValueError('Choose one still image, not an image stack or animation.')
            if handle.mode in ('P', 'RGBA', 'LA'):
                handle = handle.convert('RGB')
            array = np.asarray(handle)
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        array = rgb2gray(array[..., :3])
    if array.ndim != 2 or min(array.shape) < 16:
        raise ValueError('Choose a 2D grayscale or RGB image, at least 16 pixels on each side.')
    if array.dtype.kind not in 'buif' or not np.isfinite(array).all():
        raise ValueError('The image must contain finite numeric pixel values.')
    array = array.astype(float, copy=False)
    return (array, calibration) if return_calibration else array


def request_arguments(request):
    if 'orientation' not in IMAGES:
        raise ValueError('Load an orientation image first.')
    mode = request['mode']
    region = request.get('region', 'whole')
    image = IMAGES['orientation']['array']
    second = None
    if mode == 'full' or region == 'second':
        if 'mask' not in IMAGES:
            raise ValueError('Load the cell image.' if mode == 'full' else 'Load the second image for the mask.')
        second = IMAGES['mask']['array']
    elif region == 'same':
        second = image
    api_mode = ('axis_orientation_only' if region == 'whole' else 'axis_orientation_masked') if mode == 'axis' else mode
    parameters = dict(sigma_tensor=sigma_in_pixels(request), edge=int(request['edge']),
                      sigma_first_blur=float(request['blur']))
    if mode == 'full':
        parameters.update(segmention_thres=float(request['threshold']),
                          seg_gaus1=float(request['cell_blur']), seg_gaus2=float(request['background_blur']),
                          seg_invert=bool(request.get('invert', False)))
    elif region != 'whole':
        parameters.update(mask_threshold=float(request['threshold']),
                          mask_sigma=float(request['mask_blur']), mask_invert=bool(request.get('invert', False)))
    if mode == 'point_orientation':
        parameters['reference_point'] = request.get('point')
    elif mode == 'line_orientation':
        parameters['reference_line'] = request.get('line')
    return image, second, api_mode, parameters


def mask_preview(request):
    image, second, mode, parameters = request_arguments(request)
    # A mask preview does not depend on an unfinished point/line selection.
    if mode in ('point_orientation', 'line_orientation'):
        parameters.pop('reference_point', None)
        parameters.pop('reference_line', None)
        mode = 'axis_orientation_only' if second is None else 'axis_orientation_masked'
    key = 'cell_mask' if mode == 'full' else 'selection_mask'
    selected = analyze_image(image, second, mode=mode, outputs=key, **parameters)
    edge = parameters['edge']
    overlay = np.zeros((*image.shape, 4), dtype=np.uint8)
    rgba = np.zeros((*selected.shape, 4), dtype=np.uint8)
    rgba[selected] = (237, 95, 111, 82)
    target = (slice(edge, -edge),) * 2 if edge else (slice(None),) * 2
    overlay[target] = rgba
    return {'overlay': png(overlay), 'pixels': int(selected.sum()), 'kind': key}


def fig_bytes(fig):
    stream = io.BytesIO()
    fig.savefig(stream, format='png', dpi=145, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    return stream.getvalue()


def map_figure(image, values, title, limits, cmap, request, *, cell_mask=None):
    fig, ax = plt.subplots(figsize=(4.4, 4.0), layout='constrained')
    edge = int(request['edge'])
    crop = (slice(edge, -edge),) * 2 if edge else (slice(None),) * 2
    extent = (-.5 + edge, image.shape[1] - .5 - edge, image.shape[0] - .5 - edge, -.5 + edge)
    ax.imshow(gray_preview(image)[crop], cmap='gray', vmin=0, vmax=1, extent=extent)
    artist = ax.imshow(values, cmap=cmap, vmin=limits[0], vmax=limits[1], alpha=.72, extent=extent)
    # Keep the colourbar at full opacity while the image remains visible underneath.
    scalar = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(*limits), cmap=cmap)
    bar = fig.colorbar(scalar, ax=ax, shrink=.76, pad=.035)
    bar.set_ticks([limits[0], (limits[0] + limits[1]) / 2, limits[1]])
    bar.outline.set_visible(False)
    if cell_mask is not None and cell_mask.any() and not cell_mask.all():
        xx = np.arange(cell_mask.shape[1]) + edge
        yy = np.arange(cell_mask.shape[0]) + edge
        ax.contour(xx, yy, cell_mask, levels=[.5], colors=['#f59aa6'], linewidths=.7)
    if request['mode'] == 'point_orientation':
        ax.plot(*request['point'], '+', color='#ffe28a', ms=13, mew=1.7)
    elif request['mode'] == 'line_orientation':
        first, second = request['line']
        ax.axline(first, second, color='#ffe28a', lw=1.3, ls='--')
        ax.plot(*np.array([first, second]).T, 'o', color='#ffe28a', ms=4)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    ax.set_title(title, loc='left', fontsize=11, pad=10)
    ax.set_axis_off()
    return fig_bytes(fig)


def analyze(request):
    global LAST
    LAST = None
    start = time.perf_counter()
    image, second, mode, parameters = request_arguments(request)
    suffix = {'full': 'cell', 'point_orientation': 'point', 'line_orientation': 'line'}.get(mode)
    outputs = ['angle_x', 'angle_y', 'alignment_x', 'alignment_y', 'coherency', 'selection_mask',
               'mean_alignment_x', 'mean_alignment_y', 'mean_deviation_x', 'mean_deviation_y', 'mean_coherency']
    if suffix:
        outputs += ['angle_' + suffix, 'alignment_' + suffix, 'mean_alignment_' + suffix]
    if mode == 'full':
        outputs += ['cell_mask']
    values = analyze_image(image, second, mode=mode, outputs=outputs, **parameters)
    if not np.isfinite(values['coherency']).any():
        raise ValueError('No valid orientations were found. Check image contrast, crop and mask threshold.')
    summaries = {name: float(value) for name, value in values.items() if name.startswith('mean_')}
    if suffix:
        data = np.abs(values['angle_' + suffix])
        summaries['mean_deviation_' + suffix] = float(np.nanmean(data))
    names = [('angle_x', 'Angle to X (°)', (-90, 90), 'twilight_shifted'),
             ('angle_y', 'Angle to Y (°)', (-90, 90), 'twilight_shifted'),
             ('alignment_x', 'Alignment to X', (-1, 1), 'coolwarm'),
             ('alignment_y', 'Alignment to Y', (-1, 1), 'coolwarm'),
             ('coherency', 'Coherency', (0, 1), 'viridis')]
    if suffix:
        label = {'cell': 'cell center', 'point': 'point', 'line': 'line'}[suffix]
        names += [('angle_' + suffix, 'Angle to ' + label + ' (°)',
                   (-90, 90) if suffix == 'line' else (0, 90),
                   'twilight_shifted' if suffix == 'line' else 'magma'),
                  ('alignment_' + suffix, 'Alignment to ' + label, (-1, 1), 'coolwarm')]
    plots = {name: map_figure(image, values[name], title, limits, cmap, request,
                             cell_mask=values.get('cell_mask')) for name, title, limits, cmap in names}
    hist_key = 'angle_' + (suffix or 'x')
    angles = values[hist_key]
    bins = np.arange(-90, 95, 5) if suffix in (None, 'line') else np.arange(0, 95, 5)
    counts, edges = np.histogram(angles[np.isfinite(angles)], bins=bins)
    probabilities = counts / counts.sum() if counts.sum() else counts.astype(float)
    fig, ax = plt.subplots(figsize=(5.4, 3.0), layout='constrained')
    ax.bar(edges[:-1], probabilities, width=np.diff(edges), align='edge',
           color='#16857a' if suffix is None else '#6c5aa5', edgecolor='white', linewidth=.6)
    ax.set(xlabel='Angle to ' + ({'cell': 'cell center', 'point': 'point', 'line': 'line'}.get(suffix, 'X axis')) + ' (°)',
           ylabel='Probability per bin', xlim=(bins[0], bins[-1]), ylim=(0, max(.05, probabilities.max() * 1.15)))
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
    plots['histogram'] = fig_bytes(fig)
    metadata = {'analysis_mode': mode, 'image': IMAGES['orientation']['name'],
                'mask_image': IMAGES.get('mask', {}).get('name') if second is not None and request.get('region') != 'same' else None,
                'mask_source': 'cell_image' if mode == 'full' else request.get('region', 'whole'),
                'parameters': parameters, 'input_shape_yx': list(image.shape),
                'calibration': request.get('calibration', assumed_calibration()),
                'sigma_input': {'value': float(request['sigma']), 'unit': request.get('sigma_unit', 'pixel')},
                'map_shape_yx': list(values['coherency'].shape),
                'coordinates': 'Original zero-based pixels: x right, y down. Maps are edge-cropped.',
                'statistics': 'Unweighted means over valid selected pixels. Alignment = 1 - abs(angle)/45.',
                'histogram': 'Probability per 5-degree bin; probabilities sum to 1.',
                'summary': clean(summaries)}
    values['histogram_edges_deg'] = edges
    values['histogram_probability'] = probabilities
    LAST = {'arrays': values, 'plots': plots, 'metadata': metadata}
    return {'summary': clean(summaries), 'plots': {name: base64.b64encode(data).decode() for name, data in plots.items()},
            'pixels': int(np.isfinite(angles).sum()), 'shape': list(image.shape),
            'seconds': round(time.perf_counter() - start, 2), 'suffix': suffix, 'metadata': clean(metadata)}


def handle_request(raw):
    global LAST
    request = json.loads(raw)
    command = request['command']
    if command == 'load':
        LAST = None
        slot = request['slot']
        if slot not in ('orientation', 'mask'):
            raise ValueError('Invalid image slot')
        # A failed replacement clears the previous image, preventing stale analysis.
        IMAGES.pop(slot, None)
        array, calibration = read_image('/tmp/incoming-image', request['name'], return_calibration=True)
        IMAGES[slot] = {'array': array, 'name': request['name'], 'calibration': calibration}
        value = {'preview': png((gray_preview(array) * 255).astype('uint8')),
                 'width': array.shape[1], 'height': array.shape[0], 'calibration': calibration}
    elif command == 'clear':
        IMAGES.pop(request['slot'], None)
        LAST = None
        value = {}
    elif command == 'mask':
        value = mask_preview(request)
    elif command == 'analyze':
        value = analyze(request)
    else:
        raise ValueError('Unknown browser request')
    return json.dumps(clean(value), allow_nan=False)


def export_result(raw):
    request = json.loads(raw)
    if LAST is None:
        raise ValueError('Run an analysis first.')
    calibration = request.get('calibration', LAST['metadata'].get('calibration', assumed_calibration()))
    previous = LAST['metadata'].get('calibration', {})
    if (LAST['metadata'].get('sigma_input', {}).get('unit') == 'µm'
            and calibration.get('pixel_size_um') != previous.get('pixel_size_um')):
        raise ValueError('Pixel size changed. Run the analysis again for sigma in µm.')
    LAST['metadata']['calibration'] = calibration
    if request.get('format') == 'saenopy':
        from saenopy_export import export_project
        return export_project(LAST, IMAGES, request.get('scale'))
    arrays = io.BytesIO()
    np.savez_compressed(arrays, **LAST['arrays'], metadata_json=json.dumps(clean(LAST['metadata'])))
    if request.get('format') == 'npz':
        return arrays.getvalue()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('arrays.npz', arrays.getvalue())
        archive.writestr('parameters.json', json.dumps(clean(LAST['metadata']), indent=2))
        table = io.StringIO()
        writer = csv.writer(table)
        writer.writerow(['measurement', 'value'])
        writer.writerows(LAST['metadata']['summary'].items())
        archive.writestr('summary.csv', table.getvalue())
        for name, data in LAST['plots'].items():
            archive.writestr('plots/' + name + '.png', data)
        archive.writestr('README.txt', 'CompactionAnalyzer browser results\n\n'
            'Open arrays with: data = numpy.load("arrays.npz", allow_pickle=False)\n'
            'Maps use the original resolution after the configured edge crop.\n'
            'NaN marks pixels outside the selection or undefined orientations.\n'
            'See parameters.json for mode, original point/line coordinates and settings.\n'
            'Angles are in degrees. Histograms are probabilities per bin (sum = 1).\n')
    return stream.getvalue()
