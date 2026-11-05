import os
import sys

# Resolve inputs relative to this script and use the package from this checkout.
tutorial_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(tutorial_folder))

import numpy as np
import matplotlib.pyplot as plt
from CompactionAnalyzer.CompactionFunctions import *
import pandas as pd
from tqdm import tqdm

# Compare tensor windows on the same bundled fiber/cell pair.
# Fiber and segmentation settings remain fixed throughout the sweep.


fiber_list_string =  os.path.join(tutorial_folder, "DetermineWindowSize", "Cell_1", "C003.tif")
cell_list_string =  os.path.join(tutorial_folder, "DetermineWindowSize", "Cell_1", "C004.tif")  # ExampleCell

  


# Set analysis parameters
scale =  0.318                # imagescale as um per pixel
edge = 40                       # Cut off border pixels where orientations may be unreliable.
segmention_thres =1              # Multiplier for the cell-segmentation threshold.
seg_gaus1, seg_gaus2 =1.5,100     # Cell-image filters for local contrast enhancement.
sigma_first_blur  = 0.5         # slight first bluring of whole image before using structure tensor
show_segmentation = False
segmention_method="otsu"               #  use "otsu" , "yen"  or "entropy" as segmentation method
regional_max_correction = True
seg_iter = 1

# specify which windowsizes should be tested (list of values in µm)
sigma_list = np.arange(1.0,30,1)   ## alternative in the style of [1,2,3,4,5]


## main output folder
output_folder = os.path.join(tutorial_folder, "DetermineWindowSize_Output")
#create output folder if not existing
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
    

# Keep the result paths from THIS sweep, so older Sigma* folders are not mixed in.
result_files = []
# loop through tensor windows
for sigma in tqdm(sigma_list): 
    # Create outputfolder
    output_sub = os.path.join(output_folder, rf"Sigma{str(sigma).zfill(3)}")     # subpath to store results
    fiber_list,cell_list, out_list = generate_lists(fiber_list_string, cell_list_string, output_main =output_sub)
    if len(fiber_list) != 1 or len(cell_list) != 1:
        raise ValueError("The window-size example needs exactly one matching fiber/cell pair")
    
    sigma_tensor = sigma/scale          # sigma of applied gauss filter / window for structure tensor analysis in px
                                                         
    # Start the structure analysis with the above specified parameters
    StuctureAnalysisMain(fiber_list=fiber_list,
                         cell_list=cell_list, 
                         out_list=out_list,
                         scale=scale, 
                         sigma_tensor = sigma_tensor , 
                         edge = edge , 
                         segmention_thres =segmention_thres , 
                         seg_gaus1=  seg_gaus1, 
                         seg_gaus2 = seg_gaus2 ,
                         show_segmentation = show_segmentation ,    
                         sigma_first_blur  = sigma_first_blur , 
                         regional_max_correction = regional_max_correction,
                         seg_iter =seg_iter,
                         segmention_method=segmention_method,
                         SaveNumpy = False  ,
                         plotting = True,
                         dpi = 100
                        )
    result_files.append(os.path.join(out_list[0], "results_total.xlsx"))


### plot results
# Read one scalar orientation value for each window actually tested above.
sigmas = list(sigma_list)
orientation = [float(pd.read_excel(path)["Orientation (weighted by intensity and coherency)"].iloc[0])
               for path in result_files]


fig = plt.figure(figsize=(6,4))
plt.plot(sigmas,orientation, "o-")
plt.ylabel("Orientation", fontsize=12)
plt.xlabel("Tensor sigma (μm)", fontsize=12)
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(output_folder,"Results.png"),dpi=500)
plt.show()


