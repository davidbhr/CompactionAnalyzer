from CompactionAnalyzer.CompactionFunctions import *


# maxprojection Data
# read in list of cells and list of fibers to evaluate 
#  glob.glob is used for individual list of paths [] from these strings  
fiber_list_string =  r"..\Tutorial\*\Fiber.tif"
cell_list_string =  r"..\Tutorial\*\Cell.tif"   # ExampleCell


# Generate input and output listt automatically
# fiber_list, cell_list and out_list can also be created manual 
# as e.g. out_list=["output/conditionx/cell1", "output/conditiony/cell2"] etc...
output_folder = "Analysis_output" # base path to store results
fiber_list,cell_list, out_list = generate_lists(fiber_list_string, cell_list_string, output_main =output_folder)


# Set Parameters 
scale =  0.318                  # imagescale as um per pixel
sigma_tensor = 7/scale          # sigma of applied gauss filter / window for structure tensor analysis in px
                                # should be in the order of the objects to analyze !! 
                                # 7 um for collagen 
edge = 40                       # Cutt of pixels at the edge since values at the border cannot be trusted
segmention_thres = 1.0          # for cell segemetntion, thres 1 equals normal otsu threshold , user also can specify gaus1 + gaus2 in segmentation if needed
seg_gaus1, seg_gaus2 = 8,80     # 2 gaus filters used for local contrast enhancement
show_segmentation = False       # display the segmentation ooutput
sigma_first_blur  = 0.5         # slight first bluring of whole image before using structure tensor
angle_sections = 5              # size of angle sections in degree 
shell_width =  5/scale          # pixel width of distance shells (px-value=um-value/scale)
manual_segmention = False       # manual segmentation of mask by click cell outline
plotting = True                 # creates and saves plots additionally to excel files 
dpi = 200                       # resolution of plots to be stored
SaveNumpy = True                # saves numpy arrays for later analysis - might create lots of data
norm1,norm2 = 1,99              # contrast spreading for input images  by setting all values below norm1-percentile to zero and
                                # all values above norm2-percentile to 1
                         
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
                     norm2 = norm2)

import numpy as np    
import glob as glob  
import pandas as pd
import os
gfhf

data= "Analysis_output"

def SummarizeResultsTotal(data, output_folder= None):
    if not output_folder:
        output_folder=os.path.join(data,"CombinedFiles")
     #create output folder if not existing
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)    
    
    list_total = glob.glob(data+"\\**\\results_total.xlsx", recursive=True)

    
    
    # initialize a combining result total dictionary for all cells
    results_total_combined = { 'Path':[], 'Mean Angle (weighted by coherency)': [], 'Mean Angle (weighted by intensity and coherency)': [], 
               'Orientation (weighted by coherency)': [], 'Orientation (weighted by intensity and coherency)': [], 
               'Overall weighted Oriantation (mean all cells)': [],
               'Overall weighted Oriantation (std all cells)': []} 
    
    for i, path in enumerate(list_total):
       results_total_combined['Orientation (weighted by coherency)'].append(float(pd.read_excel(list_total[i])['Orientation (weighted by coherency)']))
       results_total_combined['Orientation (weighted by intensity and coherency)'].append(float(pd.read_excel(list_total[i])['Orientation (weighted by intensity and coherency)']))
       results_total_combined['Mean Angle (weighted by coherency)'].append(float(pd.read_excel(list_total[i])['Mean Angle (weighted by coherency)']))
       results_total_combined['Mean Angle (weighted by intensity and coherency)'].append(float(pd.read_excel(list_total[i])['Mean Angle (weighted by intensity and coherency)']))
       results_total_combined['Path'].append(str(list_total[i]))
       
    ovreall_orientation = np.nanmean(results_total_combined['Orientation (weighted by intensity and coherency)'])  
    ovreall_orientation_std = np.nanstd(results_total_combined['Orientation (weighted by intensity and coherency)'])  
    results_total_combined['Overall weighted Oriantation (mean all cells)'].extend([ovreall_orientation]*len(list_total))
    results_total_combined['Overall weighted Oriantation (std all cells)'].extend([ovreall_orientation_std]*len(list_total))
    
        
     # create excel sheet with results for angle analysis       
    excel_total_combined =  pd.DataFrame.from_dict(results_total_combined)
    excel_total_combined.to_excel(os.path.join(output_folder,"results_total_combined.xlsx"))
    
    
    return results_total_combined

from CompactionAnalyzer.plotting import *

def SummarizeResultsDistance(data, output_folder= None, dpi=200):

    if not output_folder:
        output_folder=os.path.join(data,"CombinedFiles")
     #create output folder if not existing
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)   
    
    list_distance = glob.glob(data+"\\**\\results_distance.xlsx", recursive=True)    
     # initialize a combining distance result dictionary for all cells
    results_total_distance = { 'Path':[], 'Shell_mid (µm)': [], 'Intensity': [],
                              'Intensity Norm': [], 'Orientation': [],
                              'Angle': []} 

    # find maximum length (different distances in images depending on cell positon - but all should have same scaling)       
    find_max =  [np.array(pd.read_excel(list_distance[i])['Shell_mid (µm)']) for i, path in enumerate(list_distance)]
    max_length = np.max([len(i) for i in find_max])
    
    # make matrix according to longest cell and fill rest with np.nan 
    distances = np.empty([len(list_distance),max_length])
    distances[:] = np.nan
    # fill the matrix with data
    for n,i in enumerate(list_distance):
        length = len(pd.read_excel(list_distance[n])['Shell_mid (µm)'])
        distances[n,:length] = pd.read_excel(list_distance[n])['Shell_mid (µm)']  
    results_total_distance['Shell_mid (µm)'].extend(np.nanmean(distances, axis=0)  )
    
    
     # make matrix according to longest cell and fill rest with np.nan 
    intensity = np.empty([len(list_distance),max_length])
    intensity[:] = np.nan
    # fill the matrix with data
    for n,i in enumerate(list_distance):
        length = len(pd.read_excel(list_distance[n])['Intensity (individual)'])
        intensity[n,:length] = pd.read_excel(list_distance[n])['Intensity (individual)']  
    results_total_distance['Intensity'].extend(np.nanmean(intensity, axis=0)  )
    
    
    
    
    
    
    #plot intensity and orientation averaged over all cells over distance 
    plot_distance(results_total_distance,string_plot = "Intensity",
              path_png=os.path.join(output_folder,"Intensity_allcells.png"),dpi=dpi)
    
    
    
    
    
    
    # results_total_distance['Intensity'].append((pd.read_excel(list_distance[i])['Intensity (individual)']))
    # results_total_distance['Intensity Norm'].append((pd.read_excel(list_distance[i])['Intensity Norm (individual)']))
    # results_total_distance['Orientation'].append((pd.read_excel(list_distance[i])['Orientation (individual)']))
    # results_total_distance['Path'].append((list_distance[i]))
        



    
    # to do function that reads in all and make excel file of all main values
    
    # then another excel file and plot with mean over distance, (and maybe mean over angle ? )
    