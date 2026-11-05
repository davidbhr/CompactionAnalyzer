# Biological orientation examples

Run [Example_BiologicalOrientation.py](../Example_BiologicalOrientation.py) from
the repository root. The image files are bundled here. The release ZIP contains
the same script next to this folder and prepared Saenopy results.

| Case | Orientation image | Selection / reference | Pixel size |
|---|---|---|---|
| `cardiomyocyte` | Phalloidin-stained cardiomyocytes | Registered α-actinin image | 0.1559815 µm/px |
| `spheroid` | Spheroid, red display channel shown in grayscale | Peripheral selection; point (1200, 910) | Approx. 1 µm/px from the embedded bar |
| `cardiomyocyte_line` ([script](../Example_CardiomyocyteLine.py)) | Phalloidin-stained cardiomyocytes, prepared field | Whole image; line from (69, 469) to (655, 255) | 0.1559815 µm/px |
| `lamellipodia` | Mouse fibroblast, red display channel | Peripheral selection; point (390, 325) | Unknown; pixel units |

Point coordinates are zero-based (x, y) in the complete, uncropped input image,
with x to the right and y down. No images were resized. The illustrated points
were selected visually and can be changed in the API or GUI.

The two cardiomyocyte TIFFs are the supplied `_phall_5.tif` and `_actn_5.tif`
files, copied without changes. Both are pixel-identical to frame 5 of the original
eight-frame stacks. The stacks specify `unit=micron` and 6.411017 pixels per unit,
giving 1/6.411017 µm/pixel. Their shared image dimensions are 1025 × 1025 pixels.
Phalloidin supplies the orientation; α-actinin supplies only the selection mask.
The mask uses Gaussian sigma 8 pixels and an Otsu threshold factor of 0.65. It selects
α-actinin-positive regions to exclude fibroblast regions outside the mask, but
does not distinguish overlapping cells in a projection.

The point examples are derived from rendered PNGs, not the original microscopy
intensities. The spheroid is a pseudocolored figure; extracting its red display channel
does not undo the colormap. The fibroblast image is a red/blue composite; the red display
channel isolates the peripheral structures used here. These demonstrate the
workflow. Use raw channels for quantitative comparisons between samples.

The spheroid mask selects red intensity above 70 after Gaussian smoothing
(sigma 2 px), within 290–650 px of the point. This excludes the core and the
scale annotation. The lamellipodia mask selects intensity above 35 after smoothing
(sigma 1 px), within 80–200 px of the selected cell point. This excludes the
central nuclear region and neighboring cells. These are illustrative selections;
inspect them and adapt them to the structures of interest.

The spheroid figure contains a 200-µm bar spanning 200 pixels (x=2038…2237).
This provides an approximate calibration for the PNG-derived example. The
fibroblast figure has no physical calibration: its API analysis uses sigma 3 pixels
and returns angles and dimensionless alignment. No cell-size assumption is used.
Exporting it to Saenopy requires a measured `--pixel-size` in µm/pixel.

[prepare_point_images.py](prepare_point_images.py) reproduces both grayscale
images and masks from the bundled `source.png` files.
[provenance.json](provenance.json) records original names, source hashes and
preparation steps.


## Cardiomyocytes relative to a line

[Example_CardiomyocyteLine.py](../Example_CardiomyocyteLine.py) analyzes the
whole phalloidin image without a selection mask. Both source TIFFs are retained
in `cardiomyocyte_line`; the α-actinin channel is not used for this example.

For reproducibility, [prepare_line_image.py](cardiomyocyte_line/prepare_line_image.py)
records the preparation: the source image is rotated 20° counterclockwise on
the original 1025 × 1025 canvas with bilinear interpolation, then cropped to
rows and columns 150:875. The resulting input is 725 × 725 pixels, with no
padded corners. Cropping does not rescale the pixels. Pixel size remains 0.1559815 µm.
The entire input lies more than 32 pixels from the padded rotation boundary.

The endpoints (69, 469) and (655, 255) refer to this complete input, before its
15-pixel analysis crop. All 695 × 695 analyzed pixels are selected. The saved
GUI result has no mask-image channel. Reversing or shortening the reference
line along the same axis leaves the result unchanged.

Rotation resamples intensities. The original source images and preparation
metadata are retained for analyses that do not need this preparation.
