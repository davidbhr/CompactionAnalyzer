# Full-mode reference for four empty gels: the center comes from a dummy cell.
# For axis maps without that artificial reference, see Example_AxisOrientation.py.
import os
import sys

# Resolve inputs relative to this script and use the package from this checkout.
tutorial_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(tutorial_folder))

from CompactionAnalyzer.CompactionFunctions import *


# maxprojection Data
# read in list of cells and list of fibers to evaluate
#  glob.glob is used for individual list of paths [] from these strings
fiber_list_string =  os.path.join(tutorial_folder, "EmptyGel", "*", "*C003*.tif")
cell_list_string = None  # No real cell image; full mode supplies a dummy center.


# Generate input and output lists automatically
# fiber_list, cell_list and out_list can also be created manually
# as e.g. out_list=["output/conditionx/cell1", "output/conditiony/cell2"] etc...
output_folder = os.path.join(tutorial_folder, "EmptyGel_output") # base path to store results
fiber_list,cell_list, out_list = generate_lists(fiber_list_string, cell_list_string, output_main =output_folder)
# Lists are naturally sorted; check matching channels when adapting the patterns.
if not fiber_list:
    raise FileNotFoundError(f"No fiber images match: {fiber_list_string}")
if cell_list is not None and len(cell_list) != len(fiber_list):
    raise ValueError("The fiber and cell patterns must select matching image pairs")


# Set Parameters
scale =  0.249                  # imagescale as um per pixel
sigma_tensor = 3/scale          # sigma of applied gauss filter / window for structure tensor analysis in px
                                # should be in the order of the objects to analyze !!
                                # 3 um for these empty-gel images
edge = 40                       # Cut off pixels at the edge since values at the border cannot be trusted
segmention_thres = 1.0          # for cell segmentation, thres 1 equals normal otsu threshold , change to detect different percentage of bright pixel
seg_gaus1, seg_gaus2 = 8,80     # 2 gaus filters used for local contrast enhancement for segmentation
show_segmentation = False        # display the segmentation output to test parameters - script will not run further
sigma_first_blur  = 1         # slight first bluring of whole image before using structure tensor
angle_sections = 5              # size of angle sections in degree
shell_width =  5/scale          # pixel width of distance shells (px-value=um-value/scale)
manual_segmention = False       # manual segmentation of mask by click cell outline
plotting = True                 # creates and saves plots additionally to excel files
dpi = 200                       # resolution of plots to be stored
SaveNumpy = False                # saves numpy arrays for later analysis - might create lots of data
norm1,norm2 = 1,99              # contrast spreading for input images  by setting all values below norm1-percentile to zero and
                                # all values above norm2-percentile to 1
seg_invert=False                # if segmentation is inverted (True) dark objects are detected instead of bright ones
seg_iter = 1                    # repetition of closing and dilation steps for segmentation
segmention_method="otsu"               #  use "otsu" or "yen"  as segmentation method
load_segmentation = False        # True loads a full-size segmentation dictionary from path_seg.
path_seg = None                  # Dictionary with mask/radius/centroid, not the cropped segmention.npy.
regional_max_correction = True


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
                     angle_sections = angle_sections,
                     shell_width = shell_width    ,
                     manual_segmention = manual_segmention,
                     plotting = plotting,
                     dpi = dpi,
                     SaveNumpy = SaveNumpy ,
                     norm1=norm1,
                     norm2 = norm2,
                     seg_invert=seg_invert,
                     seg_iter=seg_iter,
                     segmention_method=segmention_method,
                     load_segmentation=load_segmentation,
                     regional_max_correction=regional_max_correction,
                     path_seg=path_seg)


# Summarize Data for all cells in subfolders of analysis output
SummarizeResultsTotal(data=rf"{output_folder}", output_folder= os.path.join(output_folder, "Combine_Set1"))
SummarizeResultsDistance(data=rf"{output_folder}", output_folder= os.path.join(output_folder, "Combine_Set1"))
