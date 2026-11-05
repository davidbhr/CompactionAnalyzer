# Browser examples

These single-image examples come from the CompactionAnalyzer tutorials.
TIFFs retain their original pixel values and resolution; only lossless compression is applied.

- Cell: Tutorial/ExampleCells/Cell_1 (collagen C003, calcein C004), 0.318 µm/pixel.
- Phalloidin: Tutorial/StressFibers/controlzoom1.5, first prepared field, 0.48 µm/pixel.
- Cardiomyocyte: Tutorial/BiologicalExamples/cardiomyocyte_05, 1/6.411017 µm/pixel.
- Line: Tutorial/BiologicalExamples/cardiomyocyte_line, same calibration; prepared rotated/cropped example.
- Point: Tutorial/Example_SyntheticReferences.make_image("point", seed=0), synthetic pixel coordinates.

Browser controls use pixel units. Presets convert physical filter widths to pixels where needed.
The point mask selects an annulus. The line preset analyzes the whole image.
