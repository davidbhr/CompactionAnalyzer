"""Compare saved orientation measurements without Qt or their source images.

Read the same saved columns as Saenopy's orientation PlottingWindow. Each file
contributes one value/distribution; group means and SEM weight images equally.
No tensor analysis is rerun. NumPy containers are read without pickle.
"""
import base64
import csv
import hashlib
import io
import json
import math
import textwrap
import zipfile

import numpy as np
import matplotlib.pyplot as plt
from CompactionAnalyzer.plotting import convert_axis_reference, wrap_axis_angle

MODES = {'full', 'axis_orientation_only', 'axis_orientation_masked',
         'point_orientation', 'line_orientation'}
# Saved column names deliberately match the desktop GUI.
METRICS = {
    'cell_alignment': ('Cell-relative alignment', 'Orientation', 'full', (-1, 1)),
    'cell_deviation': ('Angular deviation to cell centre (°)', 'Mean Angle', 'full', (0, 90)),
    'alignment_x': ('Alignment to X axis', 'Alignment x', None, (-1, 1)),
    'alignment_y': ('Alignment to Y axis', 'Alignment y', None, (-1, 1)),
    'deviation_x': ('Angular deviation to X axis (°)', 'Mean absolute deviation x', None, (0, 90)),
    'deviation_y': ('Angular deviation to Y axis (°)', 'Mean absolute deviation y', None, (0, 90)),
    'coherency': ('Coherency', 'Mean Coherency', None, (0, 1)),
    'point_alignment': ('Alignment towards point', 'Alignment point', 'point_orientation', (-1, 1)),
    'point_deviation': ('Angular deviation to point (°)', 'Mean angle point', 'point_orientation', (0, 90)),
    'line_alignment': ('Alignment to reference line', 'Alignment line', 'line_orientation', (-1, 1)),
    'line_deviation': ('Angular deviation to reference line (°)', 'Mean deviation line', 'line_orientation', (0, 90)),
    'histogram_x': ('Angle histogram: X axis', None, None, (-90, 90)),
    'histogram_y': ('Angle histogram: Y axis', None, None, (-90, 90)),
    'histogram_line': ('Angle histogram: reference line', None, 'line_orientation', (-90, 90)),
    'histogram_point': ('Angular deviation histogram: point', None, 'point_orientation', (0, 90)),
    'histogram_cell': ('Angular deviation histogram: cell centre', None, 'full', (0, 90)),
}
COLORS = ['#397f75', '#8864ad', '#cc8750', '#4a84b6', '#bb6976', '#798647']


class SavedResult:
    """Lazy reader: avoid loading large vector/image arrays we never use."""
    def __init__(self, path):
        self.archive = zipfile.ZipFile(path)
        entries = self.archive.infolist()
        if len(entries) > 4096 or sum(e.file_size for e in entries) > 768 * 1024**2:
            self.archive.close()
            raise ValueError('This result is too large for the browser comparison.')
        self.names = {e.filename[:-4] for e in entries if e.filename.endswith('.npy')}
        self.cache = {}

    def close(self):
        self.archive.close()

    def get(self, key, default=None):
        if key not in self.names:
            return default
        if key not in self.cache:
            info = self.archive.getinfo(key + '.npy')
            if info.file_size > 160 * 1024**2:
                raise ValueError('A saved array is too large for the browser comparison.')
            with self.archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError('Unsupported NumPy array version.')
                if dtype.hasobject or dtype.kind not in 'biufSU':
                    raise ValueError('Only numeric and text result arrays are supported; pickle is not allowed.')
                size = math.prod(shape) * dtype.itemsize
                if size > 160 * 1024**2 or size > info.file_size - stream.tell():
                    raise ValueError('Invalid or oversized saved array.')
            with self.archive.open(info) as stream:
                value = np.load(stream, allow_pickle=False)
            if value.ndim == 0:
                value = value.item()
                if isinstance(value, str) and value == '__NONE__':
                    value = None
            self.cache[key] = value
        return self.cache[key]

    def vector(self, key):
        value = self.get(key)
        if isinstance(value, str) and value in ('tuple', 'list'):
            return [self.get(key + '/0'), self.get(key + '/1')]
        return value


def scalar(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def selected_area(saved, mode, shape):
    mask = saved.get('validity_mask')
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
    elif mode == 'full':
        cell = saved.get('segmentation/mask')
        if cell is None:
            return None
        edge = int(saved.get('orientation_parameters/edge', 0) or 0)
        if edge < 0:
            raise ValueError('Invalid saved edge crop.')
        crop = slice(edge, -edge) if edge else slice(None)
        mask = ~np.asarray(cell, dtype=bool)[crop, crop]
        if mask.shape != shape:
            return None
        maximum = saved.get('orientation_parameters/max_dist')
        minimum = saved.get('orientation_parameters/min_dist')
        if maximum or minimum:
            center = saved.vector('segmentation/centroid')
            if center is None:
                return None
            yy, xx = np.indices(shape)
            distance = np.hypot(xx - center[0] + edge, yy - center[1] + edge)
            if maximum:
                mask &= distance <= maximum
            if minimum:
                mask &= distance >= minimum
    elif mode == 'axis_orientation_masked':
        return None
    else:
        mask = np.ones(shape, dtype=bool)
    if mask.shape != shape:
        raise ValueError('The saved mask and angle maps have different shapes.')
    return mask


def distribution(angles, mask, limits=(-90, 90)):
    angles = np.asarray(angles)
    valid = np.isfinite(angles)
    if mask is not None:
        valid &= mask
    bins = np.arange(limits[0], limits[1] + 5, 5, dtype=float)
    counts = np.histogram(angles[valid], bins=bins)[0]
    return {'edges': bins.tolist(), 'counts': counts.tolist()} if counts.sum() else None


def read_measurement(path, name):
    saved = SavedResult(path)
    try:
        if saved.get('___save_name__') != 'ResultOrientation':
            raise ValueError('Choose a .saenopyOrientation result, not a traction-force project.')
        if str(saved.get('___save_version__')) not in ('1.0', '1.1'):
            raise ValueError('This orientation file version is not supported yet.')
        mode = saved.get('analysis_mode', 'full')
        if mode not in MODES:
            raise ValueError('Unknown saved orientation mode.')
        prefix = 'results_total/0/'
        row = {key[len(prefix):]: saved.get(key) for key in saved.names if key.startswith(prefix)}
        if not row:
            raise ValueError('This measurement has no results. Run the image analysis first.')
        histograms = {}
        # Current files already contain these counts. Read them without images.
        histogram = saved.get('axis_histogram')
        if histogram is not None:
            histogram = np.asarray(histogram, dtype=float)
            if histogram.shape != (36, 4) or not np.isfinite(histogram).all():
                raise ValueError('Invalid saved X/Y histogram.')
            edges = np.arange(-90, 95, 5, dtype=float)
            if (not np.allclose(histogram[:, 0], edges[:-1]) or
                    not np.allclose(histogram[:, 1], edges[1:]) or np.any(histogram[:, 2:] < 0)):
                raise ValueError('The saved histogram does not use the supported 5° bins.')
            for i, axis in enumerate(('x', 'y')):
                if histogram[:, i + 2].sum():
                    histograms['histogram_' + axis] = {'edges': edges.tolist(), 'counts': histogram[:, i + 2].tolist()}
        x = saved.get('angle_to_x_map') if mode in ('full', 'line_orientation') or histogram is None else None
        if x is not None:
            x = np.asarray(x, dtype=float)
            if x.ndim != 2:
                raise ValueError('Expected a two-dimensional saved angle map.')
            mask = selected_area(saved, mode, x.shape)
            if mask is not None:
                y = saved.get('angle_to_y_map')
                y = convert_axis_reference(x, 'y') if y is None else np.asarray(y)
                if y.shape != x.shape:
                    raise ValueError('The saved angle maps have different shapes.')
                valid = mask & np.isfinite(x) & np.isfinite(y)
                for axis, angles in (('x', x), ('y', y)):
                    if valid.any():
                        deviation = float(np.abs(angles[valid]).mean())
                        row['Alignment ' + axis] = 1 - deviation / 45
                        row['Mean absolute deviation ' + axis] = deviation
                    hist = distribution(angles, valid)
                    if hist:
                        histograms['histogram_' + axis] = hist
            if mode == 'line_orientation':
                reference = scalar(row.get('Reference line angle (deg)'))
                if reference is not None:
                    hist = distribution(wrap_axis_angle(x - reference), mask)
                    if hist:
                        histograms['histogram_line'] = hist
        for axis in ('x', 'y'):
            alignment = scalar(row.get('Alignment ' + axis))
            if alignment is not None:
                row['Mean absolute deviation ' + axis] = 45 * (1 - alignment)
        if mode != 'full' and scalar(row.get('Mean Coherency')) is None:
            coherence = saved.get('coherence_map')
            if coherence is not None:
                coherence = np.asarray(coherence)
                valid = np.isfinite(coherence)
                mask = selected_area(saved, mode, coherence.shape)
                if mask is not None:
                    valid &= mask
                    row['Mean Coherency'] = float(coherence[valid].mean()) if valid.any() else None
        if mode in ('full', 'point_orientation'):
            alignment_map = saved.get('orientation_map')
            if alignment_map is not None:
                alignment_map = np.asarray(alignment_map)
                mask = selected_area(saved, mode, alignment_map.shape)
                if mask is not None:
                    hist = distribution(45 * (1 - np.clip(alignment_map, -1, 1)), mask, (0, 90))
                    if hist:
                        histograms['histogram_cell' if mode == 'full' else 'histogram_point'] = hist
        values = {}
        for key, (_, column, family, _) in METRICS.items():
            if column and (family is None or family == mode):
                value = scalar(row.get(column))
                if value is not None:
                    values[key] = value
        if not values and not histograms:
            raise ValueError('This file contains no supported summary values or angle distributions.')
        with open(path, 'rb') as stream:
            fingerprint = hashlib.file_digest(stream, 'sha256').hexdigest()
        return {'id': fingerprint, 'name': name, 'mode': mode, 'values': values, 'histograms': histograms}
    finally:
        saved.close()


def csv_text(rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        # Names are user input, so don't create spreadsheet formulas on export.
        writer.writerow({key: ("'" + value if isinstance(value, str) and value.startswith(('=', '+', '-', '@', '\t', '\r')) else value)
                         for key, value in row.items()})
    return stream.getvalue()


def plot_comparison(request):
    metric = request['metric']
    if metric not in METRICS:
        raise ValueError('Choose a supported measurement.')
    title, _, _, limits = METRICS[metric]
    is_histogram = metric.startswith('histogram_')
    groups = request['groups']
    if not 1 <= len(groups) <= 20:
        raise ValueError('Use between one and twenty conditions.')
    raw, summary, excluded, seen = [], [], 0, set()
    total_images = 0
    fig, ax = plt.subplots(figsize=(max(7.2, min(14, len(groups) * .9)), 4.8))
    try:
        labels, positions = [], []
        for index, group in enumerate(groups):
            name = str(group['name'])[:100]
            color = COLORS[index % len(COLORS)]
            values = []
            edges = None
            for item in group['files']:
                if item['id'] in seen:
                    raise ValueError('The same measurement was assigned more than once.')
                seen.add(item['id'])
                if is_histogram:
                    histogram = item['histograms'].get(metric)
                    if histogram is None:
                        excluded += 1
                        continue
                    counts = np.asarray(histogram['counts'], dtype=float)
                    this_edges = np.asarray(histogram['edges'], dtype=float)
                    if counts.ndim != 1 or len(this_edges) != len(counts) + 1 or not np.isfinite(counts).all() or np.any(counts < 0) or counts.sum() <= 0:
                        raise ValueError('Invalid histogram counts.')
                    if edges is not None and not np.array_equal(edges, this_edges):
                        raise ValueError('Histogram bins differ between measurements.')
                    edges = this_edges
                    fractions = counts / counts.sum()
                    values.append(fractions)
                    raw.extend({'condition': name, 'file': item['name'], 'measurement_id': item['id'], 'mode': item['mode'],
                                'bin_start_deg': float(a), 'bin_end_deg': float(b), 'probability': float(p)}
                               for a, b, p in zip(edges[:-1], edges[1:], fractions))
                else:
                    value = scalar(item['values'].get(metric))
                    if value is None:
                        excluded += 1
                        continue
                    values.append(value)
                    raw.append({'condition': name, 'file': item['name'], 'measurement_id': item['id'], 'mode': item['mode'], 'value': value})
            n = len(values)
            if not n:
                continue
            total_images += n
            values = np.asarray(values)
            mean = values.mean(axis=0)
            sem = values.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
            if is_histogram:
                centers = (edges[:-1] + edges[1:]) / 2
                ax.stairs(mean, edges, color=color, label=f'{name} (n={n})', linewidth=1.8)
                ax.fill_between(centers, np.maximum(0, mean - sem), mean + sem, step='mid', color=color, alpha=.16)
                summary.extend({'condition': name, 'n': n, 'bin_start_deg': float(a), 'bin_end_deg': float(b),
                                'mean_probability': float(m), 'sem': float(s) if n > 1 else None}
                               for a, b, m, s in zip(edges[:-1], edges[1:], mean, sem))
            else:
                position = len(positions)
                positions.append(position)
                labels.append(textwrap.fill(name, 18) + f'\n(n={n})')
                ax.bar(position, mean, color=color, alpha=.7, width=.58,
                       yerr=float(sem) if n > 1 else None, capsize=4)
                offsets = np.linspace(-.16, .16, n) if n > 1 else np.zeros(1)
                ax.scatter(position + offsets, values, s=24, color=color, edgecolor='white', linewidth=.6, zorder=3)
                summary.append({'condition': name, 'n': n, 'mean': float(mean), 'sem': float(sem) if n > 1 else None})
        if not summary:
            raise ValueError('No selected measurements contain this metric.')
        if is_histogram:
            ax.set_xlim(limits)
            ax.set_ylim(bottom=0)
            ax.set_xlabel(title.replace('Angle histogram: ', 'Orientation angle to ') + ' (°)' if 'deviation' not in title else title.replace('histogram: ', 'to ') + ' (°)')
            ax.set_ylabel('Probability per 5° bin')
            legend = ax.legend(frameon=False)
            for text in legend.get_texts():
                text.set_parse_math(False)
        else:
            ax.set_ylim(limits)
            ax.set_xticks(positions, labels, parse_math=False)
            ax.set_ylabel(title)
            ax.axhline(0, color='#aab6bb', linewidth=.6)
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        png, svg = io.BytesIO(), io.StringIO()
        fig.savefig(png, format='png', dpi=170)
        fig.savefig(svg, format='svg')
        raw_fields = ['condition', 'file', 'measurement_id', 'mode', 'bin_start_deg', 'bin_end_deg', 'probability'] if is_histogram else ['condition', 'file', 'measurement_id', 'mode', 'value']
        summary_fields = ['condition', 'n', 'bin_start_deg', 'bin_end_deg', 'mean_probability', 'sem'] if is_histogram else ['condition', 'n', 'mean', 'sem']
        return {'png': base64.b64encode(png.getvalue()).decode(), 'svg': svg.getvalue(),
                'raw_csv': csv_text(raw, raw_fields), 'summary_csv': csv_text(summary, summary_fields),
                'summary': summary, 'excluded': excluded,
                'n': total_images,
                'metric': metric, 'title': title}
    finally:
        plt.close(fig)


def import_request(request_json):
    request = json.loads(request_json)
    return json.dumps(read_measurement('/tmp/incoming-image', request['name']), allow_nan=False)


def comparison_request(request_json):
    return json.dumps(plot_comparison(json.loads(request_json)), allow_nan=False)
