import os
import sys

# Resolve inputs relative to this script and use the package from this checkout.
tutorial_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(tutorial_folder))

import glob as glob
from tqdm import tqdm
from natsort import natsorted
from CompactionAnalyzer.CompactionFunctions import max_projection
from CompactionAnalyzer.utilities import load_stack



# Template for your own 2D TIFF slices; these raw stacks are not bundled.
# Example input: data/Day1/cell1_frames/z01_C003.tif, z02_C003.tif, ...
folder_pattern = os.path.join(tutorial_folder, "..", "data", "Day*", "*frames")
output_folder = os.path.join(tutorial_folder, "Max_all")
channels = ["C003", "C004", "C001"]  # Adjust to the channels present in your data.
folders = natsorted([path for path in glob.glob(folder_pattern) if os.path.isdir(path)])
if not folders:
    raise FileNotFoundError(f"No frame folders found; set folder_pattern for your data: {folder_pattern}")

#loop through these folders
for f in tqdm(folders):
    # Read each channel separately, using only TIFF images.
    for channel in channels:
        img_list = natsorted([path for path in glob.glob(os.path.join(f, "*"+channel+"*"))
                             if os.path.isfile(path) and os.path.splitext(path)[1].lower() in (".tif", ".tiff")])
        if not img_list:
            raise FileNotFoundError(f"No TIFF slices for channel {channel} in {f}; check channels")
        stack = load_stack(img_list)
        # As before, max_projection rescales each projection to 8-bit (0..255).
        # This template does not preserve the original absolute intensity scale.
        maxproj = max_projection(stack)
        outfolder = os.path.join(output_folder, os.path.basename(os.path.dirname(f)), os.path.basename(f))
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
        maxproj.save(os.path.join(outfolder, channel + ".tif"))








