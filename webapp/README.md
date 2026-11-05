# CompactionAnalyzer in the browser

A small interface for analyzing images and comparing saved measurements. Select a cell, X/Y axes,
a point or a line as the reference. General, point and line analysis can use
the whole image, a mask from the same image, or a mask from a second image.

The application calls the existing `CompactionAnalyzer.analyze_image` API.
Python runs in a Web Worker through Pyodide. Input images stay in the browser;
there is no image upload endpoint or Python server. Packages are downloaded
from jsDelivr and PyPI when the engine first loads.

## Try it locally

From the repository root:

```bash
python webapp/build.py
python -m http.server 8000 --directory webapp/dist --bind 127.0.0.1
```

Open <http://127.0.0.1:8000>. The build uses only Python's standard library.
An internet connection is needed to load Pyodide and its dependencies.
Opening `index.html` as a `file://` URL does not work because workers and
Python files are loaded using HTTP.

1. Load a single TIFF, PNG or JPEG, or select one of the five examples.
2. Choose the reference and, where applicable, a second image.
3. For a point, click once in the input image. For a line, click twice.
   Coordinates can also be entered directly. Reset reference starts over.
4. Set the tensor window, pixel size and mask threshold directly above the image.
   Mask controls are disabled until a mask source is available. Open **More settings**
   for additional parameters, then select **Analyze**.
5. Save individual PNG plots, a NumPy archive, all results, or a Saenopy project.

Changing an input or analysis parameter invalidates the displayed result.
The threshold slider refreshes the mask after a short delay. Calculation can
be cancelled; this restarts the worker and clears the loaded images.

## Compare saved measurements

Open **Compare data** to use the website for group analysis on its own.
Add conditions, then drop evaluated `.saenopyOrientation` files onto each
condition or use its **choose files** button. Rename conditions or exclude
individual files with their checkboxes. No source images, installation or
new pixel calibration are needed. Files are read locally in the browser.

Choose a measurement and select **Plot**. Bars show the mean, SEM and each
image's value. Histograms normalize each image to a total probability of 1
before averaging across images. Both plots weight images equally, as in
Saenopy Data Analysis. Missing metrics are excluded and counted; they do not
become zero values. With one image, the SEM is undefined and no error bar is
drawn. Alignment axes stay between −1 and +1; coherency stays between 0 and 1.

Download PNG or SVG figures and CSV tables of individual values and group
summaries. **Load example comparison** shows three images each of horizontal,
vertical and crossing synthetic fibers. The example contains compact saved
summaries and histograms from the same evaluated tutorial measurements.

The reader supports orientation file versions 1.0 and 1.1, uses saved statistics
and derives missing supported histograms from saved maps. It does not rerun
orientation analysis or require Qt. It reads NumPy data without pickle and
accepts at most 200 files, 20 conditions and 128 MB per file. Duplicate files
are skipped. Conditions remain in browser memory until the page is reloaded.

## Publish on GitHub Pages

Changes to the website or analysis code on `master` build the website automatically.
Publishing remains manual: use **Run workflow** to update the public website.

1. Commit and push `webapp/`, `.github/workflows/browser-demo-pages.yml`, and
   the current `CompactionAnalyzer` source, including `api.py` and `distance.py`.
   The workflow must be present on the default branch for GitHub to show its
   manual **Run workflow** button.
2. In the repository, select **Settings → Pages → Source: GitHub Actions**.
3. Under **Actions**, open **Publish browser demo to GitHub Pages** and run it
   on the branch containing the desired analysis code.
4. GitHub displays the deployed URL in the completed workflow. For this project
   the normal project address is `https://davidbhr.github.io/CompactionAnalyzer/`.

`build.py` packages the current analysis modules without changing them. Each
build records their SHA-256 hashes in `build-info.json`. Browser source files
are in `src/`; the generated `dist/` folder is ignored by Git. Assets use
relative URLs, so the demo also works below a repository subpath.

GitHub's setup instructions:
<https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages>

## Results and limits

- One 2D image at a time, up to 12 million pixels (for example 4000 × 3000) and
  128 MB per input file. Analysis and export retain the original resolution.
  Large images need more browser memory; use the desktop GUI if memory runs out.
  Images are not resized for calculation. Stacks are rejected.
- RGB images are converted to grayscale using the same conversion as the API.
- Tensor σ can use pixels or µm. Other analysis lengths use pixels; angles use
  degrees. Point and line coordinates refer to the original image, before the edge crop.
- The selected area controls which pixels contribute to maps and summaries.
  In cell-relative mode, the segmented cell is excluded and its center is the
  reference. General/point/line masks select the included area.
- Histograms report probability per 5-degree bin; their values sum to 1.
  X/Y analysis displays the X-axis histogram. Line angles are signed axial
  angles; point and cell deviations range from 0 to 90 degrees.
- Mean values weight valid pixels equally, matching the direct API. This is
  not the separate intensity/coherency-weighted batch statistic.
- Exported maps are edge-cropped with NaN outside valid selected pixels.
  Point/line geometry, input shape and settings are retained in the metadata.
- Batch image analysis and distance plots remain in the desktop/Python workflows.

```python
import json
import numpy as np

result = np.load('compaction_arrays.npz', allow_pickle=False)
coherency = result['coherency']
angle_x = result['angle_x']
settings = json.loads(str(result['metadata_json']))
```

The complete ZIP also includes PNG figures, `summary.csv` and `parameters.json`.
The browser interface and bundled analysis code use the repository's GPL v3 license.

## Continue in Saenopy

**Save Saenopy project ↓** downloads a ZIP containing `measurement.saenopyOrientation` and
the corresponding grayscale images. Extract the entire ZIP, then drag the
measurement file into the updated Saenopy Orientation GUI. Keep the `data/`
folder beside the result. The project uses relative image paths and the
`ResultOrientation` 1.1 format, including axis maps, masks and references.
Keep this project for later comparisons: the extracted `.saenopyOrientation`
file can also be dropped into **Compare data** in the browser (up to 512 MB per
result file). The source images are only needed to reopen the image analysis in
the desktop GUI.

Pixel size is read from OME-TIFF or ImageJ TIFF microscopy metadata when available.
The biological presets contain their calibration. Otherwise, the field starts at
**1 µm/px**, highlighted as an assumed value that you can replace. Ordinary image
DPI is not used as a microscopy calibration. The assumption is also recorded in
the downloads; check it before interpreting physical distances in Saenopy.

Enter the tensor window σ in **px or µm** using the unit selector beside it.
Switching units converts the current value. Changing the pixel size while using
µm requires a new analysis. Angular measurements do not need a physical scale.
A blank or invalid pixel size prevents Saenopy export and shows a message beside
the download buttons; PNG, NumPy and **All results** exports still work for an
analysis made with σ in pixels.

For cell-relative projects the exporter calls the existing full workflow to
include its weighted statistics and distance shells. This takes additional
time; the browser's displayed means remain the unweighted API statistics.
Point projects include distance profiles calculated from the saved maps.
The desktop GUI itself is not embedded in the website.
