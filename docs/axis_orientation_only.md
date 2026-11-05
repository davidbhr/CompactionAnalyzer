# CompactionAnalyzer: axis orientation with or without a validity mask

## Focused changes

Only three existing package modules change:

- `CompactionFunctions.py`: trailing mode/mask parameters, an early return for the
  two axis modes and shared NumPy output helpers. Full preprocessing, segmentation
  and statistics remain in their original locations. The two Excel-summary
  readers use portable path joins and report missing input tables explicitly.
- `plotting.py`: helpers for axis conversion, loading and plotting. All 17
  original plotting functions are unchanged. New figures reuse `plot_fancy_overlay`.
- `StructureTensor.py`: recover an undefined maximum eigenvector from a defined
  minimum vector for axis-aligned structures, analogous to the existing minimum
  recovery. If both directions are undefined, they remain NaN.

The mode needs no additional runtime dependencies. The tutorial update corrects
the existing YAML dependency name to `PyYAML`, fixes standalone script imports
and makes example paths relative to their scripts. Examples keep their direct
parameter/call style, with clearer input checks and comments. The main README
covers the existing full workflow as well as the new mode. No release-version change is included. GUI integration is implemented in the
companion Saenopy checkout; installing only this package does not update an
existing standalone Saenopy application. General fixes to the original
full-mode `edge=0` handling are outside this PR; the reduced mode
does handle zero cropping correctly.

## Usage

`analysis_mode="full"` remains the default. Existing positional calls retain
their meaning. For images without a cell channel:

```python
from CompactionAnalyzer.CompactionFunctions import StuctureAnalysisMain

scale = 0.155981  # micrometres per pixel; use your acquisition's value
StuctureAnalysisMain(
    fiber_list=["fiber.tif"], cell_list=None, out_list=["out_axis/fiber"],
    scale=scale, sigma_tensor=int(5 / scale), edge=int(5 / scale),
    analysis_mode="axis_orientation_only", plotting=False, SaveNumpy=True,
)
```

Use one 2D image per entry. Split t/z stacks before analysis. The reduced mode
uses the same normalization, Gaussian and structure-tensor functions as full.
Common maps match full mode when it uses the same fiber parameters without a
cell mask or distance limits (for example, `cell_list=None`). A full analysis
that excludes a cell region intentionally analyzes a different image mask.
It skips dummy cells, segmentation, cell centres, radial/sector/shell statistics
and Excel tables. Axis angles are still computed. `SaveExcel` has no effect.
Cell-distance limits and the original full-mode segmentation options require
full mode. The masked axis mode uses the separate `mask_sigma`,
`mask_threshold` and `mask_invert` options.

`plotting` and `SaveNumpy` are independent. Use separate output directories for
full and reduced runs: existing files are not deleted automatically.

## NumPy output and angle convention

All three modes save these angle/vector files under `NumpyArrays`, without reducing precision:

| File | Meaning |
|---|---|
| `Angles_towards_x_axis.npy` | Fiber / minimum eigenvector angle to x |
| `Angles_towards_y_axis.npy` | Fiber / minimum eigenvector angle to y |
| `Vector_min_ax0.npy`, `Vector_min_ax1.npy` | Minimum vector components [x, y down] |
| `Vector_max_ax0.npy`, `Vector_max_ax1.npy` | Maximum vector components [x, y down] |
| `CoherencyMap.npy`, `FiberImageCrop.npy` | Existing coherency and normalized crop |
| `AngleMap_NoReference.npy` | Compatibility alias of the minimum x-angle map |

These are nine files. All modes additionally save `Alignment_x_axis.npy` and
`Alignment_y_axis.npy`. Full retains six additional cell-related arrays (17
files total), while unmasked axis mode saves 11. Masked axis mode saves a 12th
file, the boolean `ValidityMask.npy`; its alignment maps contain NaN outside
the mask. Raw angle/vector files keep their original values. These axis scores
are distinct from the full mode's radial `OrientationMap.npy`.

Angles are axial: directions differing by 180 degrees are equivalent. Values
are degrees in `[-90, 90)`. For x, horizontal is 0, vertical is -90, and a fiber
sloping up to the right is +45. Stored vector components use y pointing down.
The x formula matches the previous `AngleMap_NoReference` calculation.

The minimum eigenvector already describes the fiber direction. The maximum
describes the perpendicular gradient direction. The y convention is
`wrap180(angle_x - 90)`. `legacy_flip` explicitly reproduces the former reader's
`90 - angle_x`, including its handling of +90. It is a different convention:

| Stored x | New x | New y | Old reader (`legacy_flip`) |
|---|---|---|---|
| +30 degrees | +30 | -60 | +60 |

Never silently switch references when comparing with previous summaries.

### Which files contain the same information?

`Vector_min_ax0.npy` and `Vector_min_ax1.npy` retain their historical names
and meanings: the x and downward-y components of the unit fiber eigenvector.
`Vector_max_ax0.npy` and `Vector_max_ax1.npy` retain the corresponding maximum
eigenvector components. This vector gives the perpendicular, dominant gradient
direction of the local structure tensor, not the raw intensity-gradient vector
or its magnitude. Both vectors keep their existing names, values and meaning.
Only the additional maximum-eigenvector angle maps are omitted.

The angle files are another representation of those directions. They are
signed axial angles in degrees, not vector components and not absolute
deviations. Use `abs(angle)` for the unsigned deviation from the chosen axis.
The minimum-eigenvector angles already describe fibers; no further 90-degree
gradient-to-fiber correction is needed.

Several files are deliberately redundant in the current export:

- `AngleMap_NoReference.npy` is an exact copy of `Angles_towards_x_axis.npy`
  under its original name, retained for existing scripts.
- The fiber y-angle is `wrap_axis_angle(angle_x - 90)`. Both axis files are
  kept for direct use in external readers.
- `Alignment_x_axis.npy` and `Alignment_y_axis.npy` are derived scores:
  `1 - abs(angle) / 45`, from +1 (parallel) to -1 (perpendicular). They are not
  angle maps. Y alignment is the negative of X alignment; masked exports use
  NaN outside `ValidityMask.npy`.

For example, a horizontal fiber has fiber angles x=0 and y=-90 degrees;
its gradient direction has x=-90 and y=0 degrees. Fiber alignment is +1
to X and -1 to Y.

For the current axis summaries and plots, the fiber x-angle, coherency,
normalized fiber crop and optional validity mask contain the necessary image
data. Additional angle maps and alignment scores can be reconstructed. A
reconstructed unit vector represents the same axial orientation but need not
retain the original eigenvector's arbitrary sign. Preserve the vector files
when a downstream reader needs their exact components.

New analyses retain both minimum and maximum vectors and add fiber angles to
the x and y axes. Only the two `Angles_max_evec_towards_*_axis.npy` maps and
their two angle figures are omitted. There is no separate diagnostic export.
Existing files are left in place; use a new output directory for the smaller
export. The loader still accepts earlier results containing maximum-vector
angle maps and derives missing fiber y angles from x. Coherency, the image crop
and the selected mask are separate information and cannot be reconstructed
from fiber angles alone.

## Reading and plotting

```python
from CompactionAnalyzer.plotting import load_axis_arrays, convert_axis_reference

arrays = load_axis_arrays("out_axis/fiber/NumpyArrays")
angle_x, angle_y = arrays["min_x"], arrays["min_y"]
old_reader_values = convert_axis_reference(angle_x, "legacy_flip")
```

The loader accepts old `AngleMap_NoReference.npy` outputs and derives missing
y maps. Maximum maps are available only when the saved angles or vectors exist.
Loading uses `allow_pickle=False`. Inconsistent old/new x maps or x/y maps raise
an error, so files left over from another analysis cannot silently disagree.

With `plotting=True`, the package retains its overlay style, layer order,
transparency, colorbar and scalebar. `Angles_towards_x_axis.png` replaces
`Angle_NoReference_overlay*.png` with unchanged rendering. The y-axis figure
and `CoherencyMap.png` use the same style. Other full plots remain. Saved-data
plots also show only the fiber directions, even for earlier folders that
contain maximum-vector arrays.

`plot_saved_axis_orientation(folder, scale=...)` plots saved maps in the package
style. Its background contrast can differ from an analysis-time plot because
`FiberImageCrop.npy` is normalized twice, as before. Analysis-time plots use the
once-normalized image. Angle values do not change.

### Minimal actinin-mask example

[`Example_ActininMask.py`](../Tutorial/Example_ActininMask.py) reads saved Phall
orientation maps and selects pixels using the matching raw actinin channel.
This can focus an analysis of mixed cultures on actinin-positive,
cardiomyocyte-containing areas. It is an intensity selection, not automatic
cell-type classification or identification of individual cells.

The required actinin data are not bundled. Set `numpy_folder`, `actinin_file`,
`output_folder`, `scale` and `edge` for your own experiment. The image and maps
must belong to the same position and z plane (or the same MAX projection), with
the same pixel grid and crop. Shape mismatches raise an error.

The mask retains the local reader's calculation: crop the original actinin
image, smooth with Gaussian sigma=5, then compare it with
`threshold_otsu(filtered * 0.6)`. The Phall channel supplies the fiber angles;
the actinin channel only selects pixels. No tensor analysis or cell
segmentation is rerun and the input NumPy files are not changed.

The example saves `ActininMask.npy` and two
`Angles_towards_{x,y}_axis_actinin_masked.npy` matrices. They contain the original
angle values in degrees, with NaN outside the mask or where orientation was
undefined. With `plotting=True`, it also saves x/y overlays using the existing
package function and an actinin/mask comparison for checking the selection.
`plotting=False` skips those images and still saves the masked arrays.

The example is self-contained and does not import a separate batch reader.
Use `convert_axis_reference(angle_x, "legacy_flip")` only when reproducing an
existing `90 - angle_x` analysis. Grouping t/z/MAX results into experiment
summaries is a separate step.

These angles are axial, so later averaging must account for the wrap at
-90/+90 degrees: +89 and -89 are almost parallel, although their arithmetic
mean is 0. The reader returns pixel values without choosing statistical weights
or pooling separate experiments, z planes and MAX projections.

## Masked mode, statistics and GUI integration

`analysis_mode="axis_orientation_masked"` uses paired entries in `fiber_list`
and `cell_list`. The second channel must have the same shape and registration.
[`Example_AxisOrientationMasked.py`](../Tutorial/Example_AxisOrientationMasked.py)
is a direct single-pair example; the post-hoc example above remains available.

The public `create_axis_mask(cropped_raw_channel, sigma=5, threshold=0.6,
invert=False)` function reproduces the reader mask and retains all selected
regions. No largest-cell selection, background correction, dilation or radial
analysis is performed. Mask parameters are independent of full-mode cell
segmentation settings. The fiber tensor is computed before applying the mask.

`axis_alignment(angles)` uses `1 - abs(wrap_axis_angle(angles))/45`, the original
package's linear convention, not `cos(2*theta)`. `summarize_axis_orientation`
returns a summary dictionary and a 36-row histogram table (left edge, right
edge, count x, count y). NaN angles are excluded in both axes. The unweighted
means give each valid pixel one vote; optional weighted means have explicit
column names. Zero valid pixels or zero total weight yield NaN.

With `SaveNumpy=True`, both axis modes write those two tables as
`results_axis.csv` and `histogram_axis.csv`, alongside the NumPy maps. With
`plotting=True`, both save `Histogram_axis_orientation.png`; masked mode adds
`ValidityMask.png` and `Channels_angles_towards_{x,y}_axis.png`. The paired
channels use a 25% `gist_rainbow` angle overlay. Output flags are independent;
`parameters.yml` is always written. Use separate output folders for comparisons.

`load_axis_arrays` includes `validity_mask` when available; saved-data overlays
use it automatically. For custom statistics, explicitly apply this mask to
the raw angles. The reader's historical `90-angle` transform remains an opt-in
reference conversion and is never silently used by the new modes.

For Saenopy, `mode_saenopy=True` accepts exactly one image in axis modes and
returns a dictionary with `arrays`, `validity_mask`, `summary`, `histogram`.
The full-mode seven-item return tuple remains unchanged. The companion GUI
adds the initial cell/axis choice, an optional whole-image checkbox, mask
preview, both angle maps, axis alignment and histogram views. Old result files
load as full mode. A preview does not change previously analyzed results: run
orientation analysis again after changing mask settings. Group plots average
the per-image alignment means, so images (not their pixel counts) have equal
weight. No radial-distance curves are invented for axis measurements.

## Validation

The external development suite passed 94 targeted checks (86 package/reader/example
checks and 8 companion-GUI checks). It covers the package,
documentation examples and local readers/launchers. Comparisons against the
original package and reader checked common arrays, known fiber directions,
constant images, output switches, bypassed cell operations, masks and images
with and without a cell channel. The original reader's positional colorbar
labels were adapted for current Matplotlib solely for that image comparison.
The renamed x plot and retained full plots matched the original PNG pixels.
Additional checks covered relative/moved input paths, inconsistent legacy
aliases and both eigenvectors against NumPy's independent symmetric eigensolver.
A wheel built from this checkout also imported and ran the reduced analysis
successfully; its dependency metadata contains `PyYAML`.
The actinin tutorial was also compared with the separate local reader: masks
and selected x/y values match, input arrays remain unchanged, and the plotting
switch, zero crop and mismatched-image checks behave as documented.
The full README's Python blocks were checked in sequence, and existing example
inputs/analysis arguments matched the earlier scripts. The projection template
was checked on synthetic slices, the window sweep excludes stale result folders,
and the updated Excel readers preserve the earlier summary values. These
tutorial orchestration checks use workspace fixtures instead of rerunning large
analysis batches in the checkout.

All 88 common NumPy arrays from the eight bundled RGB stress-fiber images were
also exactly equal between full and reduced mode. The standalone reader accepts
those RGB images using the same grayscale conversion as analysis. Development
tests, temporary data and verification logs are kept outside this repository.

The bundled cell/collagen and empty-gel examples completed for all four images
each, including their Excel summaries. Both stress-fiber modes completed for
all eight images, and the collagen-only axis example for all four channels.
The window-size example completed all 29 settings and its final plot. The
projection template was checked against known projections for three synthetic
channels. Generated verification results are not retained in the repository.

Before adding axis summaries, an indicative local timing on one bundled 1024x1024 RGB image gave medians of
8.95 s (`full`) and 1.12 s (`axis_orientation_only`) over three runs with the
same fiber settings. Plotting, NumPy saving and Excel saving were disabled in
both modes to measure computation. This is not a general speed guarantee;
image size, tensor window, hardware and exports affect the actual runtime.

## Bundled examples

From the repository root, run `python Tutorial/Example_AxisOrientation.py` for
all eight stress-fiber images. The example retains scale=0.48, sigma_tensor=3/scale,
edge=10, first blur=0.5 and normalization percentiles 1/99 from the existing
StressFibers script. Results go to the ignored `Tutorial/AxisOrientation_output`.
The full StressFibers script remains available for comparison.

`Tutorial/Example_CollagenAxisOrientation.py` demonstrates the same
mode on the four collagen channels, without using the associated cell images.
It retains its previous fiber parameters and exposes `analysis_mode` explicitly.
The cell/collagen and empty-gel examples retain full mode and their Excel summaries.

The stress-fiber example needs no actinin mask or second image. The separate
actinin-mask tutorial demonstrates selection of already analyzed pixel values;
it requires the user's own matching channels and analysis settings.
