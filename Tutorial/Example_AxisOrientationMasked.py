import os
import sys

tutorial_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(tutorial_folder))
from CompactionAnalyzer.CompactionFunctions import StuctureAnalysisMain


# Supply two registered channels from the SAME position and z / MAX image.
# These cardiomyocyte images are not bundled; no private data paths are needed.
input_folder = os.path.join(tutorial_folder, "ActininMaskData")
fiber_file = os.path.join(input_folder, "sample_phall.tif")
mask_file = os.path.join(input_folder, "sample_actn.tif")
output_folder = os.path.join(tutorial_folder, "AxisOrientationMasked_output")
# For a cloud-synchronized checkout, choose an output directory outside it.

scale = 0.155981             # micrometres per pixel; adapt to your acquisition
sigma_tensor = int(5/scale)  # pixels; choose for the fiber structures
edge = int(5/scale)
mask_sigma = 5              # pixels; applied to the cropped raw actinin channel
mask_threshold = 0.6        # threshold_otsu(blurred_actinin * mask_threshold)
plotting = True             # False still saves NumPy arrays and small CSV tables


def main():
    for filename in (fiber_file, mask_file):
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Supply your own matching Phall/actinin images: {filename}")
    StuctureAnalysisMain(
        fiber_list=[fiber_file], cell_list=[mask_file], out_list=[output_folder],
        scale=scale, sigma_tensor=sigma_tensor, edge=edge,
        sigma_first_blur=0.5, norm1=1, norm2=99,
        analysis_mode="axis_orientation_masked",
        mask_sigma=mask_sigma, mask_threshold=mask_threshold, mask_invert=False,
        plotting=plotting, SaveNumpy=True, SaveExcel=False)
    # cell_list supplies a VALIDITY channel here, not a radial cell reference.
    # All actinin-positive regions are retained. This is intensity selection,
    # not automatic identification of individual cardiomyocytes.
    # Raw angle maps remain unchanged; ValidityMask.npy selects valid pixels.
    # Alignment_x_axis.npy / Alignment_y_axis.npy contain NaN outside the mask.
    # results_axis.csv contains equal-pixel and explicitly named weighted means.


if __name__ == "__main__":
    main()
