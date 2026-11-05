# Maintainer notes

Release coordination and development notes, moved from the user documentation.

This checkout prepares **CompactionAnalyzer 1.0.8**, including the whole-image and masked x/y orientation workflows. The companion **Saenopy 1.0.9** requires `CompactionAnalyzer>=1.0.8,<2.0.0` and installs a compatible version automatically. Publish CompactionAnalyzer 1.0.8 to PyPI before publishing Saenopy 1.0.9. A source commit alone does not publish the package; use the existing release/publish workflow.

To use the changes and examples in **this checkout**, run this command from the repository root, in the environment where you will run the analysis:

```bash
python -m pip install -e .
```

The example scripts also select the package from this checkout. If an IDE such as Spyder has already imported another version, restart its Python kernel before switching checkouts. The public function name `StuctureAnalysisMain` and historical parameter spellings such as `segmention_thres` are retained for compatibility.

After both prepared versions have been published, `python -m pip install --upgrade saenopy` installs the updated GUI and its compatible CompactionAnalyzer dependency.

**Option B (without a Python installation):** standalone applications are available on the [Saenopy releases page](https://github.com/rgerum/saenopy/releases).

**Graphical interface:** CompactionAnalyzer is integrated into the Orientation workflow of [Saenopy](https://saenopy.readthedocs.io/en/latest/). See the [interface documentation](https://saenopy.readthedocs.io/en/latest/2.5d_orientation/interface.html) for loading measurements and examples. The companion Saenopy update offers **Relative to Cell**, **Relative to X and Y Axes** and **Relative to a Point or Line**, with three axis regions: whole image, a mask from the same image, or a mask from a second image. These new controls require both updated Python checkouts; previously published standalone applications do not include them yet.

The whole-image and phalloidin GUI packages are prepared for release attachment;
until uploaded, open their extracted result files through **Existing Measurement**.

The direct `analyze_image` API is available in the CompactionAnalyzer 1.0.8 source checkout.
