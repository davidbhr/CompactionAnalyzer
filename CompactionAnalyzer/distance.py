"""Distance-shell summaries shared by the direct API and saved GUI results."""
import numpy as np


def point_distance_profile(alignment, coherency, center, *, mask=None,
                           shell_width=5., scale=1.):
    """Average finite selected pixels in rings around a point on the map grid.

    ``center`` is (x, y) on these already cropped maps. Shells start at zero
    and use [start, end) bounds in pixels. Empty shells have NaN means and zero
    counts; an empty selection returns empty arrays. No intensity/coherency
    weighting is applied. Coherency has its own count because it can be defined
    where the reference-relative angle is not (including the point itself).
    """
    alignment = np.asarray(alignment, dtype=float)
    coherency = np.asarray(coherency, dtype=float)
    if alignment.ndim != 2 or alignment.shape != coherency.shape or not alignment.size:
        raise ValueError('alignment and coherency must be matching nonempty 2D maps')
    center = np.asarray(center, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError('center must be a finite (x, y) on the cropped map grid')
    for name, value in [('shell_width', shell_width), ('scale', scale)]:
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be positive and finite')
    selected = np.ones(alignment.shape, dtype=bool) if mask is None else np.asarray(mask)
    if selected.dtype != bool or selected.shape != alignment.shape:
        raise ValueError('mask must be boolean and match the maps')
    yy, xx = np.indices(alignment.shape)
    radius = np.hypot(xx-center[0], yy-center[1])[selected]
    with np.errstate(over='ignore'):
        extent = radius.max()/shell_width if radius.size else -1
    if not np.isfinite(extent) or extent >= 100000:
        raise ValueError('shell_width produces too many shells; increase it')
    count = int(np.floor(extent))+1
    bins = np.floor(radius/shell_width).astype(np.int64)
    left = np.arange(count, dtype=float)*shell_width
    profile = dict(distance_start_px=left, distance_end_px=left+shell_width,
                   distance_px=left+shell_width/2,
                   distance_um=(left+shell_width/2)*scale)

    def average(values):
        values = values[selected]
        valid = np.isfinite(values)
        n = np.bincount(bins[valid], minlength=count)
        total = np.bincount(bins[valid], weights=values[valid], minlength=count)
        mean = np.full(count, np.nan)
        np.divide(total, n, out=mean, where=n>0)
        return mean, n

    profile['mean_alignment_point'], profile['valid_pixels'] = average(alignment)
    profile['mean_angle_point'] = 45*(1-profile['mean_alignment_point'])
    profile['mean_coherency'], profile['coherency_pixels'] = average(coherency)
    return profile
