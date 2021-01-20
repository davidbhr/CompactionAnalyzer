from CompactionAnalyzer.CompactionFunctions import *


# maxprojection Data
# read in list of cells and list of fibers to evaluate 
#  glob.glob is used for individual list of paths [] from these strings  
fiber_list_string =  r"EmptyGel\*\*C003*.tif"
cell_list_string =  r"EmptyGel\*\Cell.tif"   # ExampleCell


# Generate input and output listt automatically
# fiber_list, cell_list and out_list can also be created manual 
# as e.g. out_list=["output/conditionx/cell1", "output/conditiony/cell2"] etc...
output_folder = "EmptyGel_output" # base path to store results
fiber_list,cell_list, out_list = generate_lists(fiber_list_string, cell_list_string, output_main =output_folder)


# Set Parameters 
scale =  0.249                  # imagescale as um per pixel
sigma_tensor = 7/scale          # sigma of applied gauss filter / window for structure tensor analysis in px
                                # should be in the order of the objects to analyze !! 
                                # 7 um for collagen 
edge = 40                       # Cutt of pixels at the edge since values at the border cannot be trusted
segmention_thres = 1.0          # for cell segemetntion, thres 1 equals normal otsu threshold , change to detect different percentage of bright pixel
seg_gaus1, seg_gaus2 = 8,80     # 2 gaus filters used for local contrast enhancement for segementation
show_segmentation = False        # display the segmentation output to test parameters - script wont run further
sigma_first_blur  = 0.5         # slight first bluring of whole image before using structure tensor
angle_sections = 5              # size of angle sections in degree 
shell_width =  5/scale          # pixel width of distance shells (px-value=um-value/scale)
manual_segmention = False       # manual segmentation of mask by click cell outline
plotting = True                 # creates and saves plots additionally to excel files 
dpi = 200                       # resolution of plots to be stored
SaveNumpy = True                # saves numpy arrays for later analysis - might create lots of data
norm1,norm2 = 1,99              # contrast spreading for input images  by setting all values below norm1-percentile to zero and
                                # all values above norm2-percentile to 1
seg_invert=False                # if segmentation is inverted (True) dark objects are detected inseated of bright ones
seg_iter = 1                    # repetition of closing and dilation steps for segmentation      
segmention_method="otsu"               #  use "otsu" or "yen"  as segmentation method
load_segmentation = False        # if true enter the path of the segementation math in path_seg to
path_seg = None                  # load in a saved.segmetnion.npy 

                      
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
                     path_seg=path_seg)


# Summarize Data for all cells in subfolders of analysis output
SummarizeResultsTotal(data="EmptyGel_output", output_folder= "EmptyGel_output\Combine_Set1")
#SummarizeResultsDistance(data="EmptyGel_output", output_folder= "EmptyGel_output\Combine_Set1")







    