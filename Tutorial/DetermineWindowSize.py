import os
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
import imageio 
import glob as glob
from natsort import natsorted
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from CompactionAnalyzer.CompactionFunctions import *
import pandas as pd
from tqdm import tqdm

# load in the maxprojection data of cell and fiber to do the test 

#### see parameter file for segmentation 


fiber_list_string =  r"DetermineWindowSize\Cell_1\C003.tif"
cell_list_string =  r"DetermineWindowSize\Cell_1\C004.tif"  # ExampleCell

  


# Set analysis arameters 
scale =  0.318                # imagescale as um per pixel
edge = 40                       # Cutt of pixels at the edge since values at the border cannot be trusted
segmention_thres =1# for cell segemetntion, thres 1 equals normal otsu threshold , change to detect different percentage of bright pixel
seg_gaus1, seg_gaus2 =1.5,100     # 2 gaus filters used for local contrast enhancement for segementation
sigma_first_blur  = 0.5         # slight first bluring of whole image before using structure tensor
show_segmentation = False
segmention_method="otsu"               #  use "otsu" , "yen"  or "entropy" as segmentation method
regional_max_correction = True
seg_iter = 1

# specify which windowsizes should be tested (list of values in µm)
sigma_list = np.arange(1.0,30,1)   ## alternative in the style of [1,2,3,4,5]


## main output folder
output_folder = r"DetermineWindowSize_Output"
#create output folder if not existing
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
    

# loop through experiments 
for sigma in tqdm(sigma_list): 
    # Create outputfolder
    output_sub = os.path.join(output_folder, rf"Sigma{str(sigma).zfill(3)}")     # subpath to store results
    fiber_list,cell_list, out_list = generate_lists(fiber_list_string, cell_list_string, output_main =output_sub)
    
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
                         SaveNumpy = False  ,
                         plotting = True,
                         dpi = 100
                        )


### plot results
#read in all creates result folder
result_folders = natsorted(glob.glob(os.path.join(output_folder, "Sigma*")))

## extract sigma from folder and read in the main orientation for each subfolder
sigmas = [float(os.path.basename(i).split("Sigma")[1]) for i in result_folders]
orientation = [pd.read_excel(glob.glob(os.path.join(i,"*","results_total.xlsx"))[0])["Orientation (weighted by intensity and coherency)"]  for i in result_folders]


fig = plt.figure(figsize=(6,4))
plt.plot(sigmas,orientation, "o-")
plt.ylabel("Orientation", fontsize=12)
plt.xlabel("Windowsize (μm)", fontsize=12)
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(output_folder,"Results.png"),dpi=500)
plt.show()


