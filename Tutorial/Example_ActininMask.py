import os
import sys

# Use the package from this checkout, as in the other tutorial scripts.
tutorial_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(tutorial_folder))

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu
from CompactionAnalyzer.plotting import load_axis_arrays, plot_axis_orientation_arrays


# Supply your own paired images/results; ActininMaskData is not bundled.
# Analyze the Phall channel first with SaveNumpy=True (full or axis mode).
# The actinin image must show the SAME position and z plane / MAX projection.
input_folder = os.path.join(tutorial_folder, "ActininMaskData")
numpy_folder = os.path.join(input_folder, "sample_phall", "NumpyArrays")
actinin_file = os.path.join(input_folder, "sample_actn.tif")
output_folder = os.path.join(tutorial_folder, "ActininMask_output")

scale = 0.155981                # micrometres per pixel; match the original analysis
edge = int(5 / scale)           # same crop as the saved NumPy maps
mask_sigma = 5                  # pixels; settings from the original local reader
mask_threshold_factor = 0.6
plotting = True                 # False skips figures; selected arrays are still saved
dpi = 200


def main():
    arrays = load_axis_arrays(numpy_folder)  # Also accepts old AngleMap_NoReference.
    actinin = plt.imread(actinin_file)
    if actinin.ndim == 3 and actinin.shape[-1] == 3:
        actinin = rgb2gray(actinin)
    if actinin.ndim != 2 or edge < 0 or 2*edge >= min(actinin.shape):
        raise ValueError("Use one 2D actinin image and the original analysis crop")
    crop = slice(edge, -edge) if edge else slice(None)
    actinin = actinin[crop, crop]
    if actinin.shape != arrays["min_x"].shape:
        raise ValueError("Actinin image and angle maps must have the same cropped shape")

    # Select actinin-positive pixels to focus on cardiomyocyte-containing areas.
    # This is an intensity mask, not automatic identification of individual cells.
    # Crop BEFORE smoothing, and use the raw channel, exactly as in the reader.
    filtered = gaussian(actinin, sigma=mask_sigma)
    mask = filtered > threshold_otsu(filtered * mask_threshold_factor)

    # These are FIBER angles in degrees, not vector components. No 90-angle flip:
    # x: horizontal=0, vertical=-90; y uses wrap180(x-90). Missing angles stay NaN.
    angle_x = np.where(mask, arrays["min_x"], np.nan)
    angle_y = np.where(mask, arrays["min_y"], np.nan)
    os.makedirs(output_folder, exist_ok=True)
    np.save(os.path.join(output_folder, "ActininMask.npy"), mask)
    np.save(os.path.join(output_folder, "Angles_towards_x_axis_actinin_masked.npy"), angle_x)
    np.save(os.path.join(output_folder, "Angles_towards_y_axis_actinin_masked.npy"), angle_y)

    # For later histograms/statistics, select valid pixels. Do not simply average
    # signed angles across the -90/+90 boundary; +89 and -89 are almost parallel.
    selected_x = angle_x[np.isfinite(angle_x)]
    selected_y = angle_y[np.isfinite(angle_y)]
    print(f"Selected {selected_x.size} pixels with a defined fiber direction")

    if plotting:
        # Reuse the existing package overlay; only the displayed pixels change.
        plot_arrays = {"min_x": angle_x, "min_y": angle_y, "fiber": arrays["fiber"]}
        plot_axis_orientation_arrays(plot_arrays, os.path.join(output_folder, "Figures"),
                                     scale=scale, dpi=dpi, include_coherency=False)
        # Check the mask on each dataset before using its selected angle values.
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(actinin, cmap="gray"); axes[0].set_title("Actinin")
        axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Selected pixels")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, "ActininMask.png"), dpi=dpi)
        plt.close(fig)


if __name__ == "__main__":
    main()
