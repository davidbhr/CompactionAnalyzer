# Direct Python API

`analyze_image` analyzes one orientation image and returns only the requested
numbers or arrays. It reads NumPy arrays or image paths and creates no output
folders, temporary images, plots, Excel files, CSV files or parameter files.

```python
from CompactionAnalyzer import analyze_image

# One requested statistic -> one float, in degrees.
deviation = analyze_image(
    "orientation.tif",
    mode="axis_orientation_only",  # Default: whole-image X/Y orientation.
    outputs="mean_deviation_x",
    scale=0.48, sigma_tensor=3 / 0.48, edge=10,
)

# Several requested values -> a dictionary with exactly these keys.
result = analyze_image(
    "orientation.tif", "alpha_actinin.tif",
    mode="axis_orientation_masked",
    outputs=["mean_alignment_x", "mean_coherency", "angle_x"],
    scale=0.48, sigma_tensor=3 / 0.48, edge=10,
    mask_sigma=5, mask_threshold=0.5,
)
alignment = result["mean_alignment_x"]  # float, -1 to +1
angle_map = result["angle_x"]          # 2D NumPy array, degrees
```

Paths can also be `pathlib.Path` objects. For NumPy inputs, pass the arrays in
the same positions. Inputs must be one finite 2D grayscale or RGB image;
split z/t stacks into individual images before calling the function. A second
image must have the same shape and registration as the orientation image.
The API does not register channels or infer pixel size from file metadata.

## Cell-relative orientation

For **cell-relative analysis**, the second image supplies the existing cell
segmentation and its centroid. The API computes the requested cell-relative
maps/statistics without the batch pipeline's distance-shell or angular-sector
tables. It requires a real cell reference; it does not create an artificial cell.

```python
cell_result = analyze_image(
    "collagen.tif", "cell.tif", mode="full",
    outputs=["mean_alignment_cell", "mean_angle_cell"],
    scale=0.318, sigma_tensor=7 / 0.318, edge=40,
    seg_gaus1=6, seg_gaus2=80,
)
```

Here the collagen image supplies the fiber directions; the cell image supplies
the segmentation and reference center.

## Choose the measurement

If `mode` is omitted, `analyze_image` uses `axis_orientation_only`.
It does not infer the mode from the number of images. Use `full` for a cell
reference, or `axis_orientation_masked` when the second image only selects an area.
The batch function `StuctureAnalysisMain` instead defaults to `full`.

| `mode` | Required input | Region and reference |
|---|---|---|
| `axis_orientation_only` (default) | Orientation image | Whole cropped image; X/Y reference axes |
| `axis_orientation_masked` | Orientation image; optional second mask-source image | Pixels inside the intensity mask; X/Y reference axes |
| `full` | Orientation image and cell image, or an explicit cell mask | Pixels outside the cell; reference direction towards its centroid |
| `point_orientation` | Orientation image and `reference_point=(x, y)` | Whole cropped image or optional mask; radial reference towards the selected point |
| `line_orientation` | Orientation image and `reference_line=((x1, y1), (x2, y2))` | Whole cropped image or optional mask; one fixed reference direction parallel to the line |

For a **same-image mask**, select `axis_orientation_masked` and omit the
second image. Masking uses the original image after edge cropping, with the
existing Gaussian/threshold calculation. The mask selects pixels after the
tensor calculation, so it introduces no artificial fiber edges.

```python
coherency = analyze_image(
    "phalloidin.tif",
    mode="axis_orientation_masked", outputs="mean_coherency",
    scale=0.48, sigma_tensor=3 / 0.48, edge=10,
    mask_threshold=0.5,
)
```

## Orientation towards a point

Set a reference in zero-based pixels of the original, uncropped image.
`(0, 0)` is the top-left pixel, X increases to the right and Y downwards.
Integer and fractional coordinates are accepted. It must lie inside the input image, but may lie in the edge crop.
This mode uses the cell-center geometry without segmentation or a dummy cell.

```python
point_result = analyze_image(
    "orientation.tif", mode="point_orientation", reference_point=(320, 240),
    outputs=["mean_alignment_point", "mean_angle_point", "alignment_point"],
    sigma_tensor=6, edge=10,
)
```

Point-relative alignment is +1 radial, 0 at 45 degrees and -1 tangential.
The reference pixel itself has no radial direction and returns NaN for
point-relative outputs. X/Y statistics and coherency are still available there
when defined. Optional `min_dist` and `max_dist` select distances from the point
in pixels. A mask can further restrict the selected area.

### Distance profiles around a point

Request `distance_profile` to average in concentric rings around the selected
point. `shell_width` is in pixels (default 5). The mask and any `min_dist` /
`max_dist` limits still apply. This does not repeat the tensor calculation.

```python
profile = analyze_image(
    "orientation.tif", mode="point_orientation", reference_point=(320, 240),
    mask=selected_area, sigma_tensor=6, edge=10,
    shell_width=10, scale=0.5, outputs="distance_profile",
)
print(profile["distance_um"], profile["mean_alignment_point"])
```

The returned dictionary contains equally sized one-dimensional arrays:

| Key | Meaning |
|---|---|
| `distance_start_px`, `distance_end_px` | Ring boundaries in pixels; each ring includes its start and excludes its end |
| `distance_px`, `distance_um` | Ring midpoint in pixels and midpoint multiplied by `scale` |
| `mean_alignment_point` | Mean radial alignment, from −1 tangential to +1 radial |
| `mean_angle_point` | Mean absolute angular deviation towards the point, in degrees |
| `mean_coherency` | Mean local coherency, independent of the reference direction |
| `valid_pixels`, `coherency_pixels` | Number of finite selected values for the orientation and coherency means, respectively |

Rings start at distance zero from the point, not at the mask boundary. The
profile ends after the farthest selected pixel. Empty rings have NaN means
and zero counts; a completely empty selection returns empty arrays. Each
finite selected pixel has equal weight. Coherency has a separate count because
it can be defined where the reference-relative angle is undefined.

For uncalibrated images, use `distance_px`; the default `scale=1` does not
establish a physical calibration. To compare images, use matching ring widths
in the chosen distance unit, then average the image means within each ring.
`distance_profile` and `shell_width` are only available in point mode.

## Orientation relative to a line

Two distinct points in original image coordinates define a fixed reference axis.
Use the same zero-based X-right, Y-down pixel convention as the point reference; fractional coordinates are accepted. Both endpoints
must be inside the image; they may lie in the edge crop. The reference extends
across the whole image, independent of segment length or position.

```python
line_result = analyze_image(
    "orientation.tif", mode="line_orientation",
    reference_line=((80, 240), (560, 120)),
    outputs=["mean_alignment_line", "mean_deviation_line", "angle_line"],
    sigma_tensor=6, edge=10,
)
```

The line's angle is `atan2(-(y2-y1), x2-x1)` in degrees. The fiber angle relative
to it is `wrap180(angle_x - line_angle)`, with the same `[-90, 90)` convention
as X/Y. Reversing the endpoints changes neither the angles nor the alignment.
`alignment_line = 1 - abs(angle_line) / 45`: +1 parallel, 0 at 45°, −1 perpendicular.
The mean angle is axial/circular; mean deviation averages absolute angles.
X/Y outputs and coherency are also available. No additional tensor calculation
is needed. A mask can restrict the selected area; line mode does not use distance limits.

## Optional masks for point and line references

Both reference modes accept `mask=selected_area` for a boolean array or
`mask_image="mask.tif"` for an image to segment. Pass the orientation image as
`mask_image` for a same-image mask. Omit both to analyze the whole cropped image.
The mask selects pixels after the tensor calculation; the reference point or
line stays unchanged and may lie outside the mask. For a point, any distance
limits are intersected with the mask.

```python
line_in_mask = analyze_image(
    "orientation.tif", mask_image="mask.tif", mode="line_orientation",
    reference_line=((80, 240), (560, 120)), outputs="mean_deviation_line",
    sigma_tensor=6, edge=10, mask_sigma=5, mask_threshold=0.5,
)
```

The GUI offers **Whole image**, **Mask from the same image** and
**Mask from a second image** for both reference types.

## Use an existing mask

Pass a **boolean array in original, uncropped image coordinates** through
`mask=` to skip segmentation. Use either `mask` or `mask_image`, not both.
An integer label image must first be converted to the intended boolean selection.

```python
# orientation and selected_area are arrays already held by the caller.
alignment_map = analyze_image(
    orientation, mask=selected_area,
    mode="axis_orientation_masked", outputs="alignment_x",
    sigma_tensor=6, edge=10,
)
```

In masked axis and point/line modes, `True` marks the area to include. In `full` mode, `True`
marks the cell itself: its centroid defines the reference, and statistics use
the surrounding pixels. A supplied cell mask must be nonempty; select the
intended cell before passing a mask containing multiple objects. Segmentation
parameters are rejected with a supplied boolean mask, since they would have
no effect. The function does not change the input images or masks.

## Available outputs

Use the name in the first column for a cropped 2D map. Add `mean_` to request
its image mean as a Python float. A string returns its value directly; an
iterable, even with one name, returns a dictionary.

| Map | Mean | Meaning |
|---|---|---|
| `angle_x`, `angle_y` | `mean_angle_x`, `mean_angle_y` | Signed axial angle to X/Y, in degrees in `[-90, 90)`; the mean uses circular statistics with 180-degree symmetry |
| `deviation_x`, `deviation_y` | `mean_deviation_x`, `mean_deviation_y` | Absolute angular deviation to X/Y, from 0 to 90 degrees; arithmetic mean of absolute deviations |
| `alignment_x`, `alignment_y` | `mean_alignment_x`, `mean_alignment_y` | Existing linear score `1 - abs(angle) / 45`: +1 parallel, 0 at 45 degrees, -1 perpendicular |
| `coherency` | `mean_coherency` | Local directional consistency from 0 to 1, independent of reference axis |
| `angle_cell` | `mean_angle_cell` | Unsigned angular deviation from the direction towards the cell centroid, from 0 to 90 degrees; full mode only |
| `alignment_cell` | `mean_alignment_cell` | +1 radial, -1 perpendicular to the cell-center direction; full mode only |
| `angle_point` | `mean_angle_point` | Unsigned angular deviation towards the reference point, 0 to 90 degrees; point mode only |
| `alignment_point` | `mean_alignment_point` | +1 radial, -1 tangential to the reference point; point mode only |
| `angle_line` | `mean_angle_line` | Signed axial angle relative to the selected line; circular mean, line mode only |
| `deviation_line` | `mean_deviation_line` | Absolute angular deviation to the line, from 0 to 90 degrees; arithmetic mean |
| `alignment_line` | `mean_alignment_line` | +1 parallel, −1 perpendicular to the line; line mode only |

`selection_mask` returns the boolean region used for the statistics, after
edge cropping and any cell-distance limits. It describes the selected region,
not whether every pixel has a defined orientation. `cell_mask` returns the
cropped cell mask in full mode. Masks have no `mean_` output.

Returned numeric maps have NaN outside the selected region and wherever the
requested quantity is undefined. Means give each finite selected pixel equal
weight; no coherency or intensity weighting is applied. An empty selection or
undefined quantity returns NaN. A circular mean with no unique direction, such
as equal horizontal and vertical contributions, also returns NaN. The pair
+89 and -89 degrees averages to the vertical direction, not to 0 degrees.
Coherency can still be valid at a pixel whose orientation is undefined.

The map shape is `(height - 2*edge, width - 2*edge)`; `edge=0` preserves the full
shape. Positive X angles slope upwards to the right, and
`angle_y = wrap180(angle_x - 90)`. The output is a fiber direction, so no
additional gradient-to-fiber correction is needed.

## Parameters and shared calculations

Parameter names and units follow `StuctureAnalysisMain`, including its
historical `segmention_*` spellings:

| Parameters | Defaults / units |
|---|---|
| `scale`, `sigma_tensor` | 1 µm/pixel; tensor sigma defaults to `7 / scale` pixels. Supply pixel size and tensor scale for the acquisition. |
| `edge`, `sigma_first_blur` | 15 pixels cropped on each side; initial Gaussian sigma 0.5 pixels |
| `norm1`, `norm2` | Normalization percentiles 1 and 99 |
| `mask_sigma`, `mask_threshold`, `mask_invert` | Masked axis mode: 5 pixels, 0.6, False; same defaults as the batch Python API |
| `segmention_thres`, `seg_gaus1`, `seg_gaus2` | Full mode: 1.0, 0.5 pixels, 100 pixels; `seg_gaus2=None` uses one Gaussian |
| `seg_invert`, `seg_iter`, `segmention_method` | Full mode: False, 1, `"otsu"`; `"yen"` and `"entropy"` also supported |
| `segmention_min_area`, `regional_max_correction` | Full mode: 1000 pixels, True |
| `ignore_cell_outline` | Full mode: False. True retains signal inside the cell for tensor calculation; summary pixels still exclude the cell, as in the batch pipeline. |
| `min_dist`, `max_dist` | Full and point modes: None. Optional distance limits from the cell centroid or reference point in pixels; both limits apply consistently to all returned maps and means. |

Unknown parameters, cell outputs in axis modes and options for the wrong mode
raise an error. Export flags such as `SaveNumpy` or `plotting` are not accepted:
this API never exports files. Point-distance shells are available as described
above. Weighted means, cell-distance shells, sectors, batch export and interactive
segmentation remain available through `StuctureAnalysisMain`.

The function reuses the existing normalization, Gaussian filters, structure
tensor, mask/cell segmentation, axis conversion and alignment functions.
Cell-reference geometry and angle calculation are shared with the original
pipeline. One call computes the tensor once, even when several outputs are
requested; a mask-only request skips it entirely. The entry point has no mutable
per-call global state and can be called concurrently on independent images.
It does not start a worker pool automatically; callers control concurrency and
memory use. An executable example is in
[`Tutorial/Example_DirectAPI.py`](../Tutorial/Example_DirectAPI.py).
