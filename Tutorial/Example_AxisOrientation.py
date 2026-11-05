import os
import sys

# Run from any working directory and use the package from this checkout.
tutorial_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(tutorial_folder))
from CompactionAnalyzer.CompactionFunctions import *


# Bundled stress-fiber images: orientation to x/y, without a cell channel.
# The scale and fiber parameters match the existing StressFibers example.
fiber_list_string = os.path.join(tutorial_folder, "StressFibers", "controlzoom1.5", "Experiment_0*_ch00*.tif")
cell_list_string = None
output_folder = os.path.join(tutorial_folder, "AxisOrientation_output")
fiber_list, cell_list, out_list = generate_lists(fiber_list_string, cell_list_string, output_main=output_folder)
# Lists are naturally sorted; check matching channels when adapting the patterns.
if not fiber_list:
    raise FileNotFoundError(f"No fiber images match: {fiber_list_string}")
if cell_list is not None and len(cell_list) != len(fiber_list):
    raise ValueError("The fiber and cell patterns must select matching image pairs")

scale = 0.48                   # micrometres per pixel
sigma_tensor = 3/scale          # same tensor window as the full stress-fiber example
edge = 10
sigma_first_blur = 0.5
norm1, norm2 = 1, 99
plotting = True                 # False skips figures; NumPy output is independent
dpi = 200
SaveNumpy = True

StuctureAnalysisMain(fiber_list=fiber_list,
                     cell_list=cell_list,
                     out_list=out_list,
                     scale=scale,
                     sigma_tensor=sigma_tensor,
                     edge=edge,
                     sigma_first_blur=sigma_first_blur,
                     norm1=norm1, norm2=norm2,
                     plotting=plotting, dpi=dpi,
                     SaveNumpy=SaveNumpy,
                     analysis_mode="axis_orientation_only")

# The minimum eigenvector is the fiber direction; the maximum is its gradient.
# Both modes save identical x/y filenames. AngleMap_NoReference.npy remains an
# alias of Angles_towards_x_axis.npy for existing readers.
# No Excel/cell-centre results are produced, so do not call the cell summaries.
