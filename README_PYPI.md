# CompactionAnalyzer — Structure Tensor-Based Image Orientation Analysis in Python

Measure local fiber orientation and angular deviation relative to X/Y axes, reference points and lines. Coherency measures the strength of local alignment independently of its direction. Analyze whole images or select regions with a mask from the same image or a second channel.

The cell-relative mode measures fiber alignment and matrix compaction around cells as indirect markers of cell contractility. Other applications include actin stress fibers, cardiomyocytes and aligned tissue structures.

Use the **Python API** in your own programs, the **Saenopy GUI** for batch processing, or the **browser app** for analysis without installation.

[Try it online](https://davidbhr.github.io/CompactionAnalyzer/) · [Download the Saenopy GUI](https://github.com/rgerum/saenopy/releases) · [Documentation and examples](https://github.com/davidbhr/CompactionAnalyzer#readme)

## Use it from Python

```bash
python -m pip install CompactionAnalyzer
```

Pass an image path or a NumPy array to `analyze_image` and select the results you need:

```python
from CompactionAnalyzer import analyze_image

result = analyze_image(
    "fibers.tif",  # Replace with your image path.
    mode="axis_orientation_only",
    outputs=["mean_alignment_x", "mean_deviation_x", "mean_coherency"],
    sigma_tensor=5,  # Pixels.
    edge=15,
)
print(result)
```

The API returns numbers or NumPy maps directly, without writing output files. It also supports cell-relative analysis, optional masks, and point or line references specified in pixel coordinates. See the [API reference](https://github.com/davidbhr/CompactionAnalyzer/blob/master/docs/direct_api.md) for modes, outputs and parameters.

## Use a graphical interface

- **[Saenopy GUI](https://github.com/rgerum/saenopy/releases):** Open **Orientation** to analyze images or process multiple image pairs as a batch. A standalone Windows EXE is available; no separate Python installation is needed. The GUI is distributed through Saenopy, separately from this Python package.
- **[Browser app](https://davidbhr.github.io/CompactionAnalyzer/):** Load an example or your own images, choose a reference, inspect the results and download plots or arrays. Image analysis runs locally in your browser, one image or image pair at a time.

Both interfaces can compare completed measurements across conditions, including mean alignment, angular deviation, coherency and angle histograms.

## Citation

If you use this work, please cite [Fiber alignment in 3D collagen networks as a biophysical marker for cell contractility](https://doi.org/10.1016/j.matbio.2023.11.004), Böhringer et al., *Matrix Biology* (2023).
